
import os
import monai
from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, SmartCacheDataset, load_decathlon_datalist
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    Resized,
    LoadImaged,
    Spacingd,
    ToMetaTensord,
)
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torchvision import transforms
import torchio as tio
import pickle
import random
from collections import defaultdict
import functools
from functools import reduce
from PIL import ImageFilter, ImageOps
from monai.utils.type_conversion import convert_to_tensor
from torch.utils.data._utils.collate import default_collate 
from data.transforms import transformsFuncd, FlipJitterd, MinMaxNormalized
from data.sampler import StratifiedBatchSampler
from data.mri_dataset import MultiChannelDataset

import statistics
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import argparse
import sys
sys.path.append('../')
from data.mri_dataset import MonaiDataset
# labels = ['NC', 'MCI', 'DE', 'AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def minmax_normalize(x):
    eps = torch.finfo(torch.float32).eps if isinstance(x, torch.Tensor) else np.finfo(np.float32).eps
    return (x - x.min()) / (x.max() - x.min() + eps)


def pklsave(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def pkload(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
    
def init_fn(worker):
    M = 2**32 - 1
    seed = torch.LongTensor(1).random_().item()
    seed = (seed + worker) % M
    np.random.seed(seed)
    random.seed(seed)

def get_all_coords(stride):
    return torch.tensor(
        np.stack([v.reshape(-1) for v in
            np.meshgrid(
                    *[stride//2 + np.arange(0, s, stride) for s in _shape],
                    indexing='ij')],
            -1), dtype=torch.int16)

_zero = torch.tensor([0])

def gen_feats():
    x, y, z = 240, 240, 155
    feats = np.stack(
            np.meshgrid(
                np.arange(x), np.arange(y), np.arange(z),
                indexing='ij'), -1).astype('float32')
    shape = np.array([x, y, z])
    feats -= shape/2.0
    feats /= shape

    return feats

def collate_handle_corrupted(samples_list, dataset, dtype=torch.half, labels=None):
    # ic(type(samples_list[0]))
    if isinstance(samples_list[0], list):
        # samples_list = list(reduce(lambda x,y: x + y, samples_list, []))
        samples_list = [s for s in samples_list if s is not None]
        # samples_list = [s for sample in samples_list for s in sample if s is not None]
    # ic(len(samples_list))
    orig_len = len(samples_list)
    # for the loss to be consistent, we drop samples with NaN values in any of their corresponding crops
    for i, s in enumerate(samples_list):
        # ic(s is None)
        if s is None:
            continue
        if isinstance(s, dict):
            if 'global_crops' in s.keys() and 'local_crops' in s.keys():
                print(len(s['global_crops']),len(s['local_crops']))
                for c in s['global_crops'] + s['local_crops']:
                    # ic(c.size(), torch.isnan(c).any())
                    if torch.isnan(c).any() or c.shape[0] != 1:
                        samples_list[i] = None
                        # ic(i, 'removed sample')
                        
            elif 'image' in s.keys():
                for c in s['image']:
                    if torch.isnan(c).any() or c.shape[0] != 1:
                        samples_list[i] = None
                        # ic(i, 'removed sample')
        
        elif isinstance(s, torch.Tensor):
            if torch.isnan(s).any() or s.shape[0] != 1:
                samples_list[i] = None
                # ic(i, 'removed sample')
                break
    samples_list = list(filter(lambda x: x is not None, samples_list))
    # ic(len(samples_list))

    if len(samples_list) == 0:
        # return None
        ic('recursive call')
        return collate_handle_corrupted([dataset[random.randint(0, len(dataset)-1)] for _ in range(orig_len)], dataset, labels=labels)

    if isinstance(samples_list[0], torch.Tensor):
        samples_list = [s for s in samples_list if not torch.isnan(s).any()]
        collated_images = torch.stack([convert_to_tensor(s) for s in samples_list])
        return {"image": collated_images}

    if "image" in samples_list[0]:
        samples_list = [s for s in samples_list if not torch.isnan(s["image"]).any()]
        # print('samples list: ', len(samples_list))
        collated_images = torch.stack([convert_to_tensor(s["image"]) for s in samples_list])
        collated_labels = {k: torch.Tensor([s["label"][k] for s in samples_list]) for k in labels}
        return {"image": collated_images,
                "label": collated_labels}
        # return {"image": torch.stack([s["image"] for s in samples_list]).to(dtype)}

    global_crops_list = [crop for s in samples_list for crop in s["global_crops"] if (not torch.isnan(crop).any() and crop.shape[0]==1)]
    local_crops_list = [crop for s in samples_list for crop in s["local_crops"] if (not torch.isnan(crop).any() and crop.shape[0]==1)]

    # ic(len(global_crops_list), len(local_crops_list))


    if len(global_crops_list) > 0:
        assert len(set([crop.shape[0] for crop in global_crops_list])) == 1
        collated_global_crops = torch.stack(global_crops_list).to(dtype)
    else:
        collated_global_crops = None
    if len(local_crops_list) > 0:
        assert len(set([crop.shape[0] for crop in local_crops_list])) == 1
        collated_local_crops = torch.stack(local_crops_list).to(dtype)
    else:
        collated_local_crops = None

    # B = len(collated_global_crops)
    # N = n_tokens

    # n_samples_masked = int(B * mask_probability)
    # probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    # upperbound = 0
    # masks_list = []

    # for i in range(0, n_samples_masked):
    #     prob_min = probs[i]
    #     prob_max = probs[i + 1]
    #     masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
    #     upperbound += int(N * prob_max)
    # for i in range(n_samples_masked, B):
    #     masks_list.append(torch.BoolTensor(mask_generator(0)))
    
    # random.shuffle(masks_list)
    # collated_masks = torch.stack(masks_list).flatten(1)
    # mask_indices_list = collated_masks.flatten().nonzero().flatten()

    # masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    return {
        "collated_global_crops": collated_global_crops,
        "collated_local_crops": collated_local_crops,
        # "collated_masks": collated_masks,
        # "mask_indices_list": mask_indices_list,
        # "masks_weight": masks_weight,
        # "upperbound": upperbound,
        # "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }

def monai_collate_singles(samples_list, dataset, dtype=torch.half, return_dict=False, labels=None, multilabel=False):
    orig_len = len(samples_list)
    for s in samples_list:
        ic(s is None)
        if isinstance(s, tuple):
            fname, img = s
            if s is None or img is None or img["image"] is None:
                samples_list.remove(s)
        else:
            if s is None or s["image"] is None:
                samples_list.remove(s)

    samples_list = [s for s in samples_list if s is not None and not torch.isnan(s["image"]).any()]
    diff = orig_len - len(samples_list)
    ic(diff)
    if diff > 0:
        ic('recursive call')  
        return monai_collate_singles(samples_list + [dataset[random.randint(0, len(dataset)-1)] for _ in range(diff)], dataset, return_dict=return_dict, labels=labels, multilabel=multilabel)

    if return_dict:
        collated_dict = {"image": torch.stack([convert_to_tensor(s["image"]) for s in samples_list])}
        if labels:
            if multilabel:
                collated_dict["label"] = {k: torch.Tensor([s["label"][k] for s in samples_list]) for k in labels}
            else:
                collated_dict["label"] = torch.LongTensor([s["label"] for s in samples_list])
        return collated_dict
    
    else:
        if isinstance(samples_list[0], tuple):
            # return fnames, imgs
            fnames_list = [s[0] for s in samples_list]
            imgs_list = [convert_to_tensor(s[1]["image"]) for s in samples_list]
            return fnames_list, torch.stack(imgs_list)
        return torch.stack([convert_to_tensor(s["image"]) for s in samples_list])
    
    

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head, embed_dim):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        backbone.conv3d_transpose = torch.nn.Identity()
        backbone.conv3d_transpose_1 = torch.nn.Identity()
        self.backbone = backbone
        self.head = head
        self.embed_dim = embed_dim

    def forward(self, x, is_training=True):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            ic(start_idx, end_idx)
            ic(torch.stack(x[start_idx: end_idx]).size())
            ic(type(self.backbone))
            _out = self.backbone(torch.stack(x[start_idx: end_idx]))
            # _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            ic(type(_out))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            ic(_out.size())
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        ic(output.view(output.shape[0], -1, self.embed_dim).size())
        return self.head(output.view(output.shape[0], -1, self.embed_dim))

class RandomResizedCrop3D(object):
    def __init__(self, size, scale, p):
        self.size = (size,size,size) if isinstance(size, int) else size
        self.Resize3D = tio.transforms.Resize(self.size)
        self.prob = p
        self.scale = scale
    
    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img
        
        target_scale = random.randrange(**self.scale)
        ic(target_scale)
        target_shape = img.shape * target_scale
        ic(target_shape)

        Crop3D = tio.transforms.CropOrPad(target_shape)
        img = Crop3D(img)
        ic(img.size())
        img = self.Resize3D(img)

        return img


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class Solarization3D(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p, threshold=128):
        self.p = p
        self.threshold = threshold

    def solarize(self, img):
        img[img > self.threshold] *= -1
        img[img < 0] += 255
        return img

    def __call__(self, img):
        if random.random() < self.p:
            return self.solarize(img)
        else:
            return img
        
class Solarization3Dd(Solarization3D):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p, threshold=128, keys=["image"]):
        super().__init__(p, threshold=threshold)
        self.keys = keys

    def solarize(self, img):
        img[img > self.threshold] *= -1
        img[img < 0] += 255
        return img

    def __call__(self, img):
        if random.random() < self.p:
            for k in self.keys:
                img[k] = self.solarize(img[k])
            return img
        else:
            return img
        
def rand_bbox(size, lam, minW=10, minH=10, minZ=10):
    W = size[-3]
    H = size[-2]
    Z = size[-1]
    ic(W,H,Z)

    cut_rat = np.cbrt(1.0 - lam)
    ic(cut_rat)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)
    cut_z = np.int32(Z * cut_rat)
    ic(cut_w, cut_h, cut_z)
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    cz = np.random.randint(Z)

    ic(cx, cy, cz)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbz1 = np.clip(cz - cut_z // 2, 0, Z)
    bbx2 = np.clip(cx + cut_w // 2, minW, W)
    bby2 = np.clip(cy + cut_h // 2, minH, H)
    bbz2 = np.clip(cz + cut_z // 2, minZ, Z)

    return bbx1, bby1, bbz1, bbx2, bby2, bbz2

def monai_collate(samples_list, dataset, dtype=torch.half):
    ic(type(samples_list[0]), len(samples_list))
    orig_len = len(samples_list)
    ic(orig_len, type(samples_list[0]))
    for i, sample in enumerate(samples_list):
        if sample is None:
            continue
        if isinstance(sample[0], dict):
            sample = [s["image"] for s in sample]
        if isinstance(sample[0], list):
            sample = list(reduce(lambda x,y: x + y, sample, []))
        sample_len = len(sample)
        sample = [s for s in sample if not (s is None or torch.isnan(s).any() or s.shape[0] != 1)]
        ic(sample_len, len(sample))
        if len(sample) < sample_len:
            samples_list[i] = None
        else:
            samples_list[i] = sample

    samples_list = [s for s in samples_list if s is not None]
    ic(len(samples_list))
    diff = orig_len - len(samples_list)
    ic(diff)
    if diff > 0:
        ic('recursive call')  
        return monai_collate(samples_list + [dataset[random.randint(0, len(dataset)-1)] for _ in range(diff)], dataset)

    for s in samples_list:
        ic(type(s))
        if isinstance(s, list):
            ic(len(s), [si.size() for si in s])
        elif isinstance(s, torch.Tensor):
            ic(s.size())
    if isinstance(samples_list[0], list):
        samples_list = list(reduce(lambda x,y: x + y, samples_list, []))
    ic(len(samples_list))
    return torch.stack([convert_to_tensor(s) for s in samples_list])

def monai_collate_singles(samples_list, dataset, dtype=torch.half, return_dict=False, labels=None, multilabel=False):
    orig_len = len(samples_list)
    for s in samples_list:
        # ic(type(s))
        if s is None:
            samples_list.remove(s)
        elif isinstance(s, tuple):
            fname, img = s
            if s is None or img is None or img["image"] is None:
                samples_list.remove(s)
        else:
            if s is None or s["image"] is None:
                samples_list.remove(s)

    samples_list = [s for s in samples_list if not (s is None or torch.isnan(s["image"]).any() or s["image"].shape[0] != 1)]
    diff = orig_len - len(samples_list)
    ic(diff)
    if diff > 0:
        ic('recursive call')  
        return monai_collate_singles(samples_list + [dataset[random.randint(0, len(dataset)-1)] for _ in range(diff)], dataset, return_dict=return_dict, labels=labels, multilabel=multilabel)

    if return_dict:
        collated_dict = {"image": torch.stack([convert_to_tensor(s["image"]) for s in samples_list])}
        if labels:
            if multilabel:
                collated_dict["label"] = {k: torch.Tensor([s["label"][k] for s in samples_list]) for k in labels}
            else:
                collated_dict["label"] = torch.Tensor([s["label"] for s in samples_list])
        return collated_dict
    
    else:
        if isinstance(samples_list[0], tuple):
            # return fnames, imgs
            fnames_list = [s[0] for s in samples_list]
            imgs_list = [convert_to_tensor(s[1]["image"]) for s in samples_list]
            return fnames_list, torch.stack(imgs_list)
        return torch.stack([convert_to_tensor(s["image"]) for s in samples_list])


def get_loader(train_list, val_list=None, num_workers=4,
                a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0,
                roi_x=96, roi_y=96, roi_z=96, sw_batch_size=2,
                batch_size=2, distributed=False, cache_dataset=False, smartcache_dataset=False):

    
    train_transforms = transformsFuncd("train", roi_x, keys=["image"])
    val_transforms = transformsFuncd("val", roi_x, keys=["image"])

    if cache_dataset:
        print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(data=train_list, transform=train_transforms, cache_rate=0.5, num_workers=num_workers)
    elif smartcache_dataset:
        print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=train_list,
            transform=train_transforms,
            replace_rate=1.0,
            cache_num=2 * batch_size * sw_batch_size,
        )
    else:
        print("Using generic dataset")
        train_ds = MonaiDataset(data=train_list, transform=train_transforms)
    collate_fn = functools.partial(monai_collate_singles, dataset=train_ds, return_dict=True)
    val_ds = MonaiDataset(data=val_list, transform=val_transforms)

    if distributed:
        train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        val_sampler = DistributedSampler(dataset=val_ds, even_divisible=True, shuffle=False, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    else:
        train_sampler = None
        val_sampler = None
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler, drop_last=True, collate_fn=collate_fn
    )

    collate_fn = functools.partial(monai_collate_singles, dataset=val_ds, return_dict=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, sampler=val_sampler, shuffle=False, drop_last=True, collate_fn=collate_fn
                            )

    return train_loader, val_loader


def get_sampler(dataset, dname, sampling, labels, batch_size, smote_strategy=None, seed=52, image_size=128, modal=None, transformsFunc=None, cfg=None):
    data_list = dataset.data if not isinstance(dataset, list) else dataset
    DATASETCLS = type(dataset) 
    if sampling == 'weighted':
        weights, counts = dataset.get_sample_weights()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=len(data_list), replacement=True)
    elif sampling == 'SMOTE':
        suffix = 'notmaj' if smote_strategy != 'minority' else 'min'
        if isinstance(dataset, MultiChannelDataset) and os.path.exists(f"./data/{'_'.join(dataset.modal)}_{dname}_{'_'.join(labels)}_{suffix}_X_sm.npy"):
            X_sm = np.load(f"./data/{'_'.join(dataset.modal)}_{dname}_{'_'.join(labels)}_{suffix}_X_sm.npy")
            y_sm = list(np.load(f"./data/{'_'.join(dataset.modal)}_{dname}_{'_'.join(labels)}_{suffix}_y_sm.npy"))
            masks_sm = np.load(f"./data/{'_'.join(dataset.modal)}_{dname}_{'_'.join(labels)}_{suffix}_masks_sm.npy")

        elif not isinstance(dataset, MultiChannelDataset) and os.path.exists(f"./data/{dname}_{'_'.join(labels)}_{suffix}_X_sm.npy"):
            X_sm = np.load(f"./data/{dname}_{'_'.join(labels)}_{suffix}_X_sm.npy")
            y_sm = list(np.load(f"./data/{dname}_{'_'.join(labels)}_{suffix}_y_sm.npy"))
            masks_sm = np.load(f"./data/{dname}_{'_'.join(labels)}_{suffix}_masks_sm.npy")
        else:    
            ic("Running SMOTE")
            smote = SMOTE(sampling_strategy=smote_strategy.replace('-',' ').replace('_',' '), random_state=seed + dist.get_rank())
            raw_mris = []
            y = []
            keys = dataset.modal if not isinstance(dataset, MonaiDataset) else ["image"]
            transform = transformsFuncd('val', image_size, keys=keys)
            masks = []
            for d in tqdm(data_list):
                y.append(d['label'])
                if not isinstance(dataset, MonaiDataset):
                    mask = []
                    v = []
                    for idx, mod in enumerate(dataset.modal):
                        d = transform(d)
                        if not mod in d or d[mod] is None or torch.isnan(d[mod]).any():
                            v.append(torch.zeros((1, image_size, image_size, image_size)))
                            mask.append(False)
                        else:
                            # ic(visit[mod].size())
                            v.append(visit[mod])
                            mask.append(True)
                    stacked_v = torch.stack(v)
                    raw_mris.append(stacked_v)
                    masks.append(torch.LongTensor(mask))
                else:
                    raw_mris.append(transform(d)['image']) # val transforms here because we want to resample based on original MRI not a randomly augmented version

                
            ic(len(raw_mris), len(y), len(masks))
            X = torch.cat(raw_mris).numpy().reshape(-1,len(dataset.modal)*image_size*image_size*image_size)
            ic(X.shape)
            # y = [d['label'] for d in data_list]
            X_sm, y_sm = smote.fit_resample(X, y)
            ic(X_sm.shape, len(y_sm))

            
            if not isinstance(dataset, MonaiDataset):
                masks = torch.stack(masks)
                num_synthetic_samples = len(y_sm) - len(y)
                # generate random masks for the synthetic samples
                synthetic_masks = np.random.choice([0,1], size=(num_synthetic_samples, len(dataset.modal)))
                # synthetic_masks = np.tile(masks.mean(axis=0), (num_synthetic_samples, len(dataset.modal))).round().astype(int) 
                ic(synthetic_masks.shape)
                masks_sm = np.vstack([masks, synthetic_masks])
                ic(masks_sm.shape)
                np.save(f"./data/{'_'.join(dataset.modal)}_{dname}_{'_'.join(labels)}_{suffix}_masks_sm.npy", masks_sm)
                np.save(f"./data/{'_'.join(dataset.modal)}_{dname}_{'_'.join(labels)}_{suffix}_X_sm.npy", X_sm)
                np.save(f"./data/{'_'.join(dataset.modal)}_{dname}_{'_'.join(labels)}_{suffix}_y_sm.npy", np.asarray(y_sm))
            else:
                np.save(f"./data/{dname}_{'_'.join(labels)}_{suffix}_X_sm.npy", X_sm)
                np.save(f"./data/{dname}_{'_'.join(labels)}_{suffix}_y_sm.npy", np.asarray(y_sm))
                
        if not isinstance(dataset, MonaiDataset):
            X_sm = torch.from_numpy(X_sm)
            masks_sm = torch.from_numpy(masks_sm)
            # y_sm = torch.LongTensor(y_sm)
            ic(X_sm.size(), masks_sm.size(), len(y_sm))
            data_list_sm = [{'visit':X_sm[idx,...].reshape(len(dataset.modal),image_size,image_size,image_size), 'label':l, 'mask': masks_sm[idx]} for (idx,l) in enumerate(y_sm)]
        else:
            data_list_sm = [{'image':X_sm[idx,...].reshape(image_size,image_size,image_size)[None,...], 'label':l} for (idx,l) in enumerate(y_sm)]

        keys = dataset.modal if not isinstance(dataset, MonaiDataset) else ["image"]
        del dataset

        sm_transforms = Compose([ToMetaTensord(keys=keys, allow_missing_keys=True),
                                 FlipJitterd(keys=keys, allow_missing_keys=True),
                                 monai.transforms.RandGaussianSmoothd(keys=keys, prob=0.5, allow_missing_keys=True),
                                 MinMaxNormalized(keys=keys),
                                 ])
       
        dataset = MonaiDataset(data_list_sm, transform=sm_transforms)
        ic(dataset.get_sample_weights()[1])
        sampler = DistributedSampler(
            dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
        )
        # sampler = StratifiedBatchSampler(torch.tensor(y_sm, dtype=int), batch_size=batch_size, shuffle=True)
    elif sampling == 'over':
        ros = RandomOverSampler(random_state=seed + dist.get_rank())
        X = torch.Tensor([i for i in range(len(data_list))]).unsqueeze(1) # to avoid loading data here, we just need the indices
        y = [d['label'] for d in data_list]
        X_ros, y_ros = ros.fit_resample(X, y)
        # retrieve the filenames from the generated indices 
        print(sorted(Counter(y_ros).items()))
        X_ros = X_ros.astype(int)
        ic(X_ros.shape, X_ros[4], y_ros[4])
        ic(data_list[int(X_ros[4])])
        data_list_ros = [data_list[int(X_ros[i])] for i in range(X_ros.shape[0])]
        ic(data_list_ros[0])
        if not isinstance(dataset, MonaiDataset):
            dataset = DATASETCLS(data_list_ros, cfg.name, modal=cfg.modal, acq_type=['2D', '3D'], labels=cfg.labels, img_size=cfg.img_size, stripped=cfg.stripped, apply_mask=cfg.apply_mask, random_masking=cfg.random_masking, transform=dataset.transform)
        else:
            dataset = MonaiDataset(data_list_ros, transform=transformsFunc)
        sampler = DistributedSampler(
            dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
        )
        # train_sampler = StratifiedBatchSampler(torch.tensor(y_ros, dtype=int), batch_size=batch_size, shuffle=True)
    elif sampling == 'under':
        rus = RandomUnderSampler(random_state=seed + dist.get_rank())
        X = torch.Tensor([i for i in range(len(data_list))]).unsqueeze(1) # to avoid loading data here, we just need the indices
        y = [d['label'] for d in data_list]
        X_rus, y_rus = rus.fit_resample(X, y)
        # retrieve the filenames from the generated indices 
        print(sorted(Counter(y_rus).items()))
        X_rus = X_rus.astype(int)
        ic(X_rus.shape, X_rus[4], y_rus[4])
        ic(data_list[int(X_rus[4])])
        data_list_rus = [data_list[int(X_rus[i])] for i in range(X_rus.shape[0])]
        if not isinstance(dataset, MonaiDataset):
            dataset = DATASETCLS(data_list_rus, dname, transform=transformsFunc, img_size=image_size, labels=labels, modal=modal, random_masking=dataset.random_masking, mask_prob=dataset.mask_prob, apply_mask=dataset.apply_mask)
        else:
            dataset = MonaiDataset(data_list_rus, transform=transformsFunc)
        # train_sampler = DistributedSampler(
        #     train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
        # )
        weights, counts = dataset.get_sample_weights()
        print(weights,counts)
        # sampler = StratifiedBatchSampler(torch.tensor(y_rus, dtype=int), batch_size=batch_size, shuffle=True)
        sampler = DistributedSampler(
            dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
        )
    elif sampling == 'stratified':
        y = [d["label"] for d in data_list]
        sampler = StratifiedBatchSampler(torch.tensor(y, dtype=int), batch_size=batch_size, shuffle=True)
    else:
        sampler = DistributedSampler(
            dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
        )

    return sampler, dataset


def patient_level_split(dataset, split_ratio, criterion="majority", seed=42):
    patient_labels = defaultdict(list)
    # get patient labels (aggregate by mode for example)
    
    df = pd.DataFrame(dataset.data)

    patient_labels = df.groupby("ID")["label"].agg(lambda x: x.mode()[0])

    patients_df = pd.DataFrame({'patient_ID': patient_labels.index, 'patient_label': patient_labels.values})
    ic(patients_df.iloc[:2,:])
    # for datum in dataset.data:
    #     patient_labels['patient_ID'].append(datum['ID'])
    #     patient_labels['patient_label'].append(datum['label'])

    # for k,v in patient_labels.items(): 
    #     if criterion in ["majority", "mode"]:
    #         patient_labels[k] = statistics.mode(v)
    #     else:
    #         raise NotImplementedError

    train_pts, tmp_pts = train_test_split(patients_df, stratify=patients_df['patient_label'], test_size=1-split_ratio, random_state=seed)
    val_pts, test_pts = train_test_split(tmp_pts, stratify=tmp_pts['patient_label'], test_size=0.5, random_state=seed)

    train_data = [datum for datum in dataset.data if datum['ID'] in list(train_pts['patient_ID'])]
    val_data = [datum for datum in dataset.data if datum['ID'] in list(val_pts['patient_ID'])]
    test_data = [datum for datum in dataset.data if datum['ID'] in list(test_pts['patient_ID'])]

    ic(len(train_data), len(val_data), len(test_data))

    return train_data, val_data, test_data
    
