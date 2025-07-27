
import os
import io
import glob
import numpy as np
import nibabel as nib
import monai
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler
import json
import pandas as pd
import copy
import time
import datetime
from tqdm import tqdm
import logging
import csv
import math
import itertools

from utils import ops
from utils.data_utils import rand_bbox
from config import MyConfig
from omegaconf import OmegaConf, ListConfig
import wandb
import tempfile

from models.rfnet import Model as RFNetOrig
from models.rfnet import Model2 as RFNet
from models.mmFormer import Modelv2 as mmFormer
from models.mmFormer import Model as mmFormerOrig
from models.ims2trans import Model as IMS2Trans
from models.ims2trans_orig import Model as IMS2TransOrig
from models.m2ftrans import Model as M2FTrans
from models.m2ftrans import Model2 as M2FTransv2
from models.vision_transformer import SwinUNETRBase
from models.lora_vit import LoRA_ViT_timm
from monai.networks.nets import SwinUNETR

from optimizers.lr_scheduler import WarmupCosineSchedule
from timm.scheduler import CosineLRScheduler
from data.transforms import MinMaxNormalized, transformsFuncd
from data.brats_datasets import Brats_loadall, Brats_loadall_test
from data.isles_dataset import Isles_loadall, Isles_loadall_test
from data.brats_datasets_nii import MultiEpochsDataLoader
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from utils.dist_utils import is_main_process, reduce_tensor
from utils.data_utils import monai_collate_singles, get_sampler, patient_level_split
from utils.stat_utils import get_metrics
from utils.visualization_utils import write_scores
from timm.utils import AverageMeter
from sklearn.model_selection import train_test_split


from losses.simclr import ContrastiveLoss
from losses.SEG_loss import SoftmaxWeightedLoss, dynamic_bce_loss, DiceLoss, GeneralizedDiceLoss, test_dice_hd95_softmax
from losses.focal_loss import FocalLoss

from icecream import ic, install
install()
ic.configureOutput(includeContext=True)

class Trainer:
    def __init__(self, cfg: MyConfig) -> None:

        self.cfg = cfg
        self.debug = self.cfg.train.debug

        self.use_ddp = (cfg.hardware.multi_gpus == "ddp") and torch.cuda.is_available()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0)) if self.use_ddp else 0
        self.global_rank = int(os.environ.get("RANK", 0)) if self.use_ddp else 0
        self.acc_metric = None

        if self.use_ddp:
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = ops.get_device(self.cfg.hardware.device)

        self.shuffle_all = "SHUFFLE_ALL"


        self.seed = ops.default(self.cfg.train.seed, np.random.randint(1000, 1000000))

        # auto set eval batch size to maximize GPU memory usage
        if not self.cfg.eval.batch_size:
            if "depthwise" not in self.cfg.model.arch:
                ## bs=512, takes 12 GB memory
                gpu_mem = ops.get_gpu_mem(return_total_mem=True)
                eval_batch_size = int(512 * gpu_mem / 14)
                # round to the nearest power of 2
                eval_batch_size = 2 ** int(np.log2(eval_batch_size))
            else:
                eval_batch_size = 128  ## too large will cause error

            self.cfg.eval.batch_size = eval_batch_size
            print(f"self.cfg.eval.batch_size: {self.cfg.eval.batch_size}")

        ic(self.cfg.train.logdir)
        ic(self.cfg.region_prompt.embed_module)
        ic(self.cfg.modal_prompt.embed_module)
        if self.cfg.train.logdir is None:
            self.cfg.train.logdir = f"checkpoints/{self.cfg.model.arch}dim{self.cfg.model.transformer_dims}_roi{self.cfg.model.img_size}ps{self.cfg.model.patch_size}_{self.cfg.train.criterion}"
            if self.cfg.model.region_prompt:
                self.cfg.train.logdir += f"AnatPrompting{self.cfg.region_prompt.embed_module.type}{self.cfg.region_prompt.version}"
            if self.cfg.model.modal_prompt:
                self.cfg.train.logdir += f"ModalPrompting{self.cfg.modal_prompt.embed_module.type}"

            self.cfg.train.logdir += f"_{self.cfg.model.fusion_type}Fusion_{self.cfg.dataset.name}"
            self.cfg.train.logdir += f"_{self.cfg.dataset.labels}" if self.cfg.dataset.labels else ""
            self.cfg.train.logdir += f"_{'_'.join(self.cfg.dataset.modal)}"
            if self.cfg.dataset.inflate:
                self.cfg.train.logdir += "_inflate"

            self.cfg.train.logdir += f"_augment{self.cfg.train.augment}_a{self.cfg.train.cont_alpha}b{self.cfg.train.cont_beta}/"
        
        

        if self.cfg.train.resume_train and self.cfg.train.resume_model is None:
            self.cfg.train.resume_model = self.cfg.train.logdir + f"model{self.cfg.eval.suffix}.pt"
        
        if not self.cfg.eval.save_path:
            if self.cfg.test and self.cfg.train.resume_model:
                self.cfg.eval.save_path = os.path.dirname(self.cfg.train.resume_model)
            else:
                self.cfg.eval.save_path = self.cfg.train.logdir
        
        ic(self.cfg.train.resume_train, self.cfg.train.resume_model, self.cfg.train.logdir, self.cfg.eval.save_path)        
        # os.makedirs(self.cfg.eval.save_path, exist_ok=True)
            
        self.use_amp = not self.cfg.train.noamp
        self.cfg.dataset.load_segmentations = ops.default(self.cfg.dataset.load_segmentations, self.cfg.model.region_prompt or "RegionModalMix" in self.cfg.train.augment)
        self._init_logger()
        self._build_model()
        self._build_dataset()
        # ic(wandb.config)
        ic(self.cfg)
        self._build_optimizer()
        self._build_scheduler()
        self._init_loss()
        # exit()
        self.model.cuda(self.device)
        
        for p in self.model.parameters():
            ic(p.device)
            break
        
        if self.use_ddp:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DistributedDataParallel(self.model, device_ids=[self.device],
                                        output_device=self.device,)

                
        self.best_val, self.best_balacc, self.best_f1 = 1e8, 0, 0

        self.start_epoch = 0
        ic(self.cfg.train.resume_train, self.cfg.train.resume_model, self.cfg.train.logdir, self.cfg.eval.save_path)
        if self.cfg.train.resume_model:
            resume_path = self.cfg.train.resume_model
            self._load_model(resume_path)

        

    def _forward_model_SEG(
        self,
        x,
        mask,
        seg_label=None,
        train=True,
        **kwargs,
    ):
        """
        forward step, depending on the type of model.
        @param x
        @param mask
        @param seg_label
        """
        augment = self.cfg.train.augment
        model = self.model.module if self.use_ddp else self.model
        ic(type(model))
        if isinstance(model, IMS2Trans) or isinstance(model, mmFormer) or isinstance(model, mmFormerOrig):
            preds = model(x, mask, features_only=False, is_training=train)
        elif isinstance(model, IMS2TransOrig):
            preds = model(x, mask)
        elif isinstance(model, RFNetOrig) or isinstance(model, RFNet):
            preds = model(x, mask, is_training=train)
        
        elif isinstance(model, M2FTrans) or isinstance(model, M2FTransv2):
            preds = model(x, mask)
        else:
            raise NotImplementedError
                

        return preds

    def _compute_loss_SEG(
        self,
        preds,
        target,
        epoch,
        x=None
    ):
        """
        Function to compute loss for the task of SEGMENTATION.

        """
        augment = self.cfg.train.augment
        

        CELoss = dynamic_bce_loss if self.cfg.model.num_cls == 1 else SoftmaxWeightedLoss(self.cfg.model.num_cls)
        DSCLoss = GeneralizedDiceLoss if self.cfg.train.criterion == 'GeneralizedDSC' else DiceLoss(num_cls=self.cfg.model.num_cls)
        loss_dict = {}
        if self.cfg.model.arch in ['mmFormer', 'RFNet', 'M2FTrans', 'M2FTransv2']:
            if isinstance(target, tuple):
                target = target[0]

            ic(torch.isnan(target).any())
            fuse_pred, sep_preds, prm_preds = preds
            
            loss_dict['fuse_cross_loss'] = CELoss(fuse_pred, target)
            loss_dict['fuse_dice_loss'] = DSCLoss(fuse_pred, target)
            loss_dict['fuse_loss'] = loss_dict['fuse_cross_loss'] + loss_dict['fuse_dice_loss']

            loss_dict['sep_cross_loss'] = torch.zeros(1).cuda(self.device).float()
            loss_dict['sep_dice_loss'] = torch.zeros(1).cuda(self.device).float()
            for sep_pred in sep_preds:
                ic(sep_pred.size())
                loss_dict['sep_cross_loss'] += CELoss(sep_pred, target)
                loss_dict['sep_dice_loss'] += DSCLoss(sep_pred, target)
            loss_dict['sep_loss'] = loss_dict['sep_cross_loss'] + loss_dict['sep_dice_loss']

            loss_dict['prm_cross_loss'] = torch.zeros(1).cuda(self.device).float()
            loss_dict['prm_dice_loss'] = torch.zeros(1).cuda(self.device).float()
            for prm_pred in prm_preds:
                loss_dict['prm_cross_loss'] += CELoss(prm_pred, target)
                loss_dict['prm_dice_loss'] += DSCLoss(prm_pred, target)
            loss_dict['prm_loss'] = loss_dict['prm_cross_loss'] + loss_dict['prm_dice_loss']

            if epoch < self.cfg.train.region_fusion_start_epoch:
                loss_dict['total'] = loss_dict['fuse_loss'] * 0.0+ loss_dict['sep_loss'] + loss_dict['prm_loss']
            else:
                loss_dict['total'] = loss_dict['fuse_loss'] + loss_dict['sep_loss'] + loss_dict['prm_loss']
            
        elif 'IMS2Trans' in self.cfg.model.arch:
            fuse_pred, dis_preds, prm_preds = preds
            dis_preds, dis_target = dis_preds
            ic(torch.isnan(fuse_pred).any())
            cont = ContrastiveLoss(batch_size=dis_target.size(0))
            # dis_preds1, dis_preds2 = torch.chunk(dis_preds, 2, dim=0)
            # print(dis_target.size())
            
            if '3DMMCutMix' in augment:
                target_a, target_b = target
                ic(self.lam)
                loss_dict['fuse_cross_loss'] = CELoss(
                    fuse_pred, target_a) * self.lam + CELoss(
                    fuse_pred, target_b) * (
                    1.0 - self.lam
                )
                loss_dict['fuse_dice_loss'] = DSCLoss(
                    fuse_pred, target_a) * self.lam + DSCLoss(
                fuse_pred, target_b) * (
                    1.0 - self.lam
                )
            else:
                loss_dict['fuse_cross_loss'] = CELoss(fuse_pred, target)
                loss_dict['fuse_dice_loss'] = DSCLoss(fuse_pred, target)
            loss_dict['fuse_loss'] = loss_dict['fuse_cross_loss'] + loss_dict['fuse_dice_loss']

            loss_dict['dis_fdc_loss'] = torch.zeros(1).cuda(self.device).float()
            dis_lambda = 0.1  # FDC
            for dis_pred in dis_preds:
                ic(dis_pred.shape, dis_target.shape)
                ic(torch.isnan(dis_pred).any(), torch.isnan(dis_target).any())
                loss_dict['dis_fdc_loss'] += cont(dis_pred, dis_target)
            loss_dict['dis_loss'] = dis_lambda * loss_dict['dis_fdc_loss']

            loss_dict['prm_cross_loss'] = torch.zeros(1).cuda(self.device).float()
            loss_dict['prm_dice_loss'] = torch.zeros(1).cuda(self.device).float()
            for prm_pred in prm_preds:
                ic(torch.isnan(prm_pred).any())
                if '3DMMCutMix' in self.cfg.train.augment:
                    loss_dict['prm_cross_loss'] += CELoss(
                        prm_pred, target_a) * self.lam + CELoss(prm_pred, target_b) * (
                        1.0 - self.lam
                    )
                    loss_dict['prm_dice_loss'] += DSCLoss(prm_pred, target_a) * self.lam
                    + DSCLoss(prm_pred, target_b) * (
                        1.0 - self.lam
                    )
                else:
                    loss_dict['prm_cross_loss'] += CELoss(prm_pred, target)
                    loss_dict['prm_dice_loss'] += DSCLoss(prm_pred, target)

            loss_dict['prm_loss'] = loss_dict['prm_cross_loss'] + loss_dict['prm_dice_loss']

            if epoch < self.cfg.train.region_fusion_start_epoch:
                loss_dict['total'] = loss_dict['fuse_loss'] * 0.0 + loss_dict['dis_loss'] + loss_dict['prm_loss']
            else:
                loss_dict['total'] = loss_dict['fuse_loss'] + loss_dict['dis_loss'] + loss_dict['prm_loss']
            
        return loss_dict

    def _save_checkpoints_helper(self, save_dict, suffix):
        file_name = os.path.join(self.cfg.train.logdir, f'model{suffix}.pt')
        
        if not self.cfg.logging.wandb.sweep_id:
            torch.save(save_dict, file_name)
        else:
            with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
                torch.save(save_dict, tmp_file.name) 
                tmpfile_path = tmp_file.name
                sweep_id = self.cfg.logging.wandb.sweep_id.split("/")[-1]
                artifact = wandb.Artifact(name=f"model{suffix}_{sweep_id}", type="model")
                artifact.add_file(tmpfile_path, name=f"model{suffix}_{sweep_id}.pt")
                key = "val_loss" if suffix == '_best' else "balacc" if suffix == '_bestbalacc' else "F1"
                artifact.metadata = {suffix[1:]: save_dict[key]}
                wandb.log_artifact(artifact)

    def _save_checkpoints(self, save_dict):
        epoch = save_dict['epoch']
        val_loss = save_dict['val_loss']
        balacc = save_dict['balacc']
        f1 = save_dict['F1']
        self._save_checkpoints_helper(save_dict, "")            

        if val_loss < self.best_val:
            self.best_val = val_loss
            self._save_checkpoints_helper(save_dict, "_best")            
            print('Saved best loss model!')

        if balacc > self.best_balacc:
            self.best_balacc = balacc
            self._save_checkpoints_helper(save_dict, "_bestbalacc")
            print('Saved best balacc model!')

        if f1 > self.best_f1:
            self.best_f1 = f1
            self._save_checkpoints_helper(save_dict, "_bestf1")
            print('Saved best F1 model!')


    ## training loop
    def train(self):
        print("########### training ###########")
        start = time.time()
        torch.set_grad_enabled(True)
        
        for epoch in range(self.start_epoch, self.cfg.train.num_epochs):
            self.model.is_training = True
            self.model.train()
            
            if (self.cfg.dataset.name.lower() in ['brats2018', 'brats2020', 'isles2022', 'gliomapost']):
                self.train_one_epoch_SEG(epoch)
            else:
                raise NotImplementedError
            
            
            ##########model save
            save_dict = {
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
                }
            
            if (epoch + 1) % 50 == 0 or (epoch >= (self.cfg.train.num_epochs-10)):
                suffix = f"_{epoch+1}" if epoch != self.cfg.train.num_epochs - 1 else "_last"
                self._save_checkpoints_helper(save_dict, suffix)

            self._save_checkpoints_helper(save_dict, "_last")

        msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
        print(msg)


    def _backward(self, loss, epoch, batch_idx):
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if ((batch_idx + 1)% self.cfg.train.accum_iter == 0) or ((batch_idx + 1) == len(self.train_loader)):
                if self.cfg.train.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    # param_norms = clip_gradients(model, args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.train.max_grad_norm)
                    
                # cancel_gradients_last_layer(epoch, model, args.freeze_last_layer)
                
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if self.cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.max_grad_norm)

            if ((batch_idx + 1)% self.cfg.train.accum_iter == 0) or ((batch_idx + 1) == len(self.train_loader)):
                self.optimizer.step()
                           
                self.optimizer.zero_grad(set_to_none=True)

    def train_one_epoch_SEG(self, epoch):
        train_loss = AverageMeter()
        batch_time = AverageMeter()
        start = time.time()
        end = time.time()
        iter_per_epoch = self.cfg.train.iter_per_epoch
        train_iter = iter(self.train_loader)
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step(epoch)
            step_lr = self.scheduler.get_last_lr()[0]

        # step_lr = self.optimizer.param_groups[0]["lr"]


        for batch_idx in range(iter_per_epoch):
            step = (batch_idx + 1) + epoch*iter_per_epoch
            
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                data = next(train_iter)
            
            with torch.cuda.amp.autocast(self.scaler is not None):
                x, target, mask = data[:3]
                x = x.cuda(self.device, non_blocking=True)
                target = target.cuda(self.device, non_blocking=True)
                mask = mask.cuda(self.device, non_blocking=True)

                
                if self.cfg.dataset.load_segmentations:
                    seg_label = data[3]
                    # ic(seg_label)
                    seg_label = seg_label.cuda(self.device, non_blocking=True)
                    ic(seg_label.size())
                else:
                    seg_label = None ## to make model forward call easier

                assert not torch.isnan(x).any(), "Input data contains NaNs"
                assert not torch.isnan(target).any(), "Labels contain NaNs"
                assert not torch.isnan(mask).any(), "Labels contain NaNs"
                
                if self.cfg.train.augment != None and self.cfg.train.augment != "none":
                    #print("Augmentation is: ", args.augment)
                    new_x, new_target, new_mask = self.init_augment(x, target, mask, seg_label=seg_label)
                    
                else:
                    new_x, new_target, new_mask = x, target, mask

                ic(new_x.shape, new_mask.shape)
                # exit()

                ic(new_x.min(), new_x.max(), new_x.dtype)
                ic(new_mask.size(), new_mask.min(), new_mask.max(), new_mask.dtype)
                preds = self._forward_model_SEG(new_x, new_mask, seg_label=seg_label, train=True)

                dist.barrier()

                loss_dict = self._compute_loss_SEG(preds, new_target, epoch)
                ic(loss_dict)
        
            self._backward(loss_dict['total'], epoch, batch_idx)
            
            torch.cuda.synchronize()
            total_loss_t = reduce_tensor(loss_dict['total'])
            train_loss.update(total_loss_t.item(), x.size(0))

            if is_main_process() and self.cfg.logging.wandb.use_wandb:
                wandb.log({k:v.item() for k,v in loss_dict.items()}, step=step)

            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (iter_per_epoch - batch_idx)
            
            batch_time.update(time.time() - end)
            end = time.time()
            print(
                f"Train: [{epoch}/{self.cfg.train.num_epochs}][{batch_idx}/{iter_per_epoch}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {step_lr:.8f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {train_loss.val:.4f} ({train_loss.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )
        
        print(' * Train Loss: {:.4f}'.format(train_loss.avg))
        

    def validate_SEG_BRATS(self, data_loader):
        self.model.is_training = False
        self.model.eval()
        ###modality missing mask
        masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
                [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
                [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
                [True, True, True, True]]
        mask_name = ['t2', 't1ce', 't1', 'flair', 
                    't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
                    'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
                    'flairt1cet1t2']
        
        # valid_masks = [[False, False, False, True],
        #         [False, True, False, True], [True, True, False, True],
        #         [True, True, True, True]]
        # valid_mask_name = ['t2', 't1cet2', 'flairt1cet2',
        #             'flairt1cet1t2']
        
        # masks = [comb for comb in list(itertools.product([0,1], repeat=len(self.cfg.dataset.modal))) if sum(comb) > 0]
        # mask_name = []
        # print(self.cfg.dataset.modal)
        # for mask in masks:
        #     print(mask)
        #     mask_str = ''.join([mod for i,mod in enumerate(self.cfg.dataset.modal) if mask[i]==1])
        #     mask_name.append(mask_str)
        # print(mask_name)
        # mask = mask[::-1]
        # mask_name = mask_name[::-1]

        test_dice_score = AverageMeter()
        test_hd95_score = AverageMeter()
        csv_name = os.path.join(self.cfg.train.logdir,f"eval{self.cfg.dataset.test_name}_ignoreRC{self.cfg.eval.suffix}.csv")
        with torch.no_grad():
            file = open(csv_name, "a+")
            csv_writer = csv.writer(file)
            csv_writer.writerow(['WT Dice', 'TC Dice', 'ET Dice','ETPro Dice', 'NCR/NET Dice', 'ED Dice', 'WT HD95', 'TC HD95', 'ET HD95', 'ETPro HD95', 'NCR/NET HD95', 'ED HD95'])
            file.close()
            # data = next(iter(data_loader))

            for i, mask in enumerate(masks[::-1]):
                logging.info('{}'.format(mask_name[::-1][i]))
                file = open(csv_name, "a+")
                csv_writer = csv.writer(file)
                csv_writer.writerow([mask_name[::-1][i]])
                file.close()
                dice_score, hd95_score = test_dice_hd95_softmax(
                    data_loader,
                    self.model,
                    dataname = self.cfg.dataset.test_name,
                    feature_mask = mask,
                    mask_name = mask_name[::-1][i],
                    csv_name=csv_name,
                    augmentation=self.cfg.train.augment)
                test_dice_score.update(dice_score)
                test_hd95_score.update(hd95_score)

            logging.info('Avg Dice scores: {}'.format(test_dice_score.avg))
            logging.info('Avg HD95 scores: {}'.format(test_hd95_score.avg))


    def init_3DMMCutMix(self, augmented_input, new_target):
        self.lam = np.random.beta(1.0, 1.0)
        rand_index = torch.randperm(augmented_input.size()[0]).cuda(self.device, non_blocking=True) # assuming the random index refers to the sample
        target_a = new_target
        target_b = new_target[rand_index]

        bbx1, bby1, bbz1, bbx2, bby2, bbz2 = rand_bbox(augmented_input.size(), self.lam)



        augmented_input[:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2] = augmented_input[
            rand_index, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2
        ]

        self.lam = 1 - (
            (bbx2 - bbx1)
            * (bby2 - bby1)
            * (bbz2 - bbz1)
            / (augmented_input.size()[-1] * augmented_input.size()[-2] * augmented_input.size()[-3])
        )

        return augmented_input, (target_a, target_b)
    
    def init_RegionModalMix(self, new_x, new_mask, seg_label):
        B = new_x.size(0)//2
        ic(new_mask, new_x.size())
        for i in range(B):
            # first get mask of this sample, and make sure the generated random numbers are only for present modalities
            # present_indices = torch.nonzero(new_mask[i] == 1).squeeze()
            # rand_modidx = present_indices[torch.randint(0,len(present_indices),(len(self.labelmap),))]

            rand_modidx = [torch.randperm(new_x.size()[1]).cuda(self.device, non_blocking=True) for i in range(len(self.labelmap))]
            orig_modidx = torch.Tensor([i for i in range(len(self.cfg.dataset.modal))]).long()
            # ic(modmix.size())
            alpha = 1.0
            for enum, region in enumerate(self.labelmap):
                if int(region) in [0]:
                    continue
                ic(seg_label[i].size(), rand_modidx[enum])
                regidx = torch.where(seg_label[i,0] == int(region))
                ic(len(regidx))
                if len(regidx[0]) == 0:
                    continue # skip empty region
                
                
                # ic(new_x[i,rand_modidx[enum]].size(), (seg_label[i,rand_modidx[enum]] == int(region)).size())
                
                # new_x[i+B][regidx1] = (new_x[i+B][regidx1] * (1-alpha)) + new_x[i][regidx2] * alpha
                
                for c, mod in enumerate(orig_modidx):
                    commpixels = (seg_label[i,c] == seg_label[i,rand_modidx[enum][c]])
            
                    regidx1 = torch.where((seg_label[i,c] * commpixels) == int(region))
                    regidx2 = torch.where((seg_label[i,rand_modidx[enum][c]] * commpixels) == int(region))
                    ic(len(regidx1[0]), len(regidx2[0]))

                    new_x[i+B,c][regidx1] = (new_x[i+B,c][regidx1] * (1-alpha)) + new_x[i, rand_modidx[enum][c]][regidx2] * alpha

                    # new_x[i+B,c][regidx1] = (new_x[i+B,c][regidx1] * (1-alpha)) + new_x[i, rand_modidx[enum][c]][regidx1] * alpha

                    # new_x[i+B,mod,regidx[0], regidx[1], regidx[2]] = (new_x[i+B,mod, regidx[0], regidx[1], regidx[2]] * (1-alpha)) + vals * alpha
                new_mask[i+B,:] = torch.logical_or(new_mask[i+B,:], new_mask[i+B,rand_modidx[enum]])

        # roi = self.cfg.model.img_size
        # for modidx, mod in enumerate(self.cfg.dataset.modal):
        #     for j in range(B):
        #         seg_nifti = nib.Nifti1Image(seg_label[j].detach().cpu().view(-1,roi,roi,roi)[modidx].float().numpy(), affine=np.eye(4))
        #         nib.save(seg_nifti, f"./seg{j}_{mod}.nii")
                
        # #         # if modidx == len(args.modal) - 1:
        # #         #     modmixnifti = nib.Nifti1Image(new_x[j,modidx+1].detach().cpu().squeeze(0).numpy(), affine=np.eye(4))
        # #         #     nib.save(modmixnifti, f"./modmix{j}.nii") 
                
        #         newx1_nifti = nib.Nifti1Image(new_x[j].detach().cpu().view(-1,roi,roi,roi)[modidx].numpy(), affine=np.eye(4))
        #         nib.save(newx1_nifti, f"./new_x1-{j}_{mod}.nii")
                
        #         newx2_nifti = nib.Nifti1Image(new_x[B+j].detach().cpu().view(-1,roi,roi,roi)[modidx].numpy(), affine=np.eye(4))
        #         nib.save(newx2_nifti, f"./new_x2-{j}_{mod}.nii")

        # exit()

        
        return new_x, new_mask

    def init_augment(self, _input, target, mask, seg_label=None):
        his_input = _input
        his_mask = mask
        his_target = target

        new_input = torch.cat([his_input, _input], dim=0).cuda(self.device, non_blocking=True)
        new_mask = torch.cat([his_mask, mask], dim=0).cuda(self.device, non_blocking=True)
        new_target = torch.cat([his_target, target], dim=0).cuda(self.device, non_blocking=True)

        if '3DMMCutMix' in self.cfg.train.augment:
            new_input, new_target = self.init_3DMMCutMix(new_input, new_target)
        elif "RegionModalMix" in self.cfg.train.augment:
            new_input, new_mask = self.init_RegionModalMix(new_input, new_mask, seg_label)
        
        # new_x1, new_x2 = torch.chunk(new_input, 2, dim=0)
        # ic(new_x1.size(), new_x2.size())
        # ic(torch.mean(_input), torch.min(_input), torch.max(_input))
        # ic(torch.mean(new_x1), torch.min(new_x1), torch.max(new_x1))
        # ic(torch.mean(new_x2), torch.min(new_x2), torch.max(new_x2))
        # ic(torch.unique(seg_label), torch.min(seg_label), torch.max(seg_label))
        # # assert not torch.isnan(new_x1).any(), "Original sample contains NaNs!!"
        # # assert not torch.isnan(new_x2).any(), "Augmented sample contains NaNs!!"
        # # assert not torch.isnan(new_mask).any(), "Mask contains NaNs!!"
        # roi = self.cfg.dataset.img_size
        # for modidx, mod in enumerate(self.cfg.dataset.modal):
        #     for j in range(seg_label.size(0)):
        #         seg_nifti = nib.Nifti1Image(seg_label[j].float().detach().cpu().view(-1,roi,roi,roi)[modidx].numpy(), affine=np.eye(4))
        #         nib.save(seg_nifti, f"./figures/fig2/seg{j}_{mod}.nii")
                
        # #         # if modidx == len(args.modal) - 1:
        # #         #     modmixnifti = nib.Nifti1Image(new_x[j,modidx+1].detach().cpu().squeeze(0).numpy(), affine=np.eye(4))
        # #         #     nib.save(modmixnifti, f"./modmix{j}.nii") 
        #         newx1_nifti = nib.Nifti1Image(new_x1[j].detach().cpu().view(-1,roi,roi,roi)[modidx].numpy(), affine=np.eye(4))
        #         nib.save(newx1_nifti, f"./figures/fig2/new_x1-{j}_{mod}.nii")
                
        #         newx2_nifti = nib.Nifti1Image(new_x2[j].detach().cpu().view(-1,roi,roi,roi)[modidx].numpy(), affine=np.eye(4))
        #         nib.save(newx2_nifti, f"./figures/fig2/new_x2-{j}_{mod}.nii")

        # ic(torch.equal(new_x1,_input))
        # ic(torch.equal(new_x2,_input))

        # exit()        
        
        return new_input, new_target, new_mask

    def _init_logger(self):
        if dist.get_rank() == 0 and self.cfg.logging.wandb.use_wandb:
            # config = wandb.config
            
            if self.cfg.logging.wandb.run_name is None:
                self.cfg.logging.wandb.run_name = os.path.basename(self.cfg.train.logdir)

            wandb.init( 
                    # set the wandb project where this run will be logged
                    project=self.cfg.logging.wandb.project_name,
                    entity=self.cfg.logging.wandb.entity,
                    save_code=True,
                    name=self.cfg.logging.wandb.run_name,
                    settings=wandb.Settings(start_method="thread"),
                )
            ic(wandb.config)
            
            dotlist = []
            for param,val in wandb.config.items():
                ic(param,val)
                if "." in param:
                    dotlist.append(f"{param}={val}")
                else:
                    if os.path.exists(f"configs/{param}/{val}.yaml"):
                        override_cfg = OmegaConf.load(f"configs/{param}/{val}.yaml")
                        OmegaConf.update(self.cfg, param, override_cfg, force_add=True)
                    else:
                        OmegaConf.update(self.cfg, param, val, force_add=True)
                    
            self.cfg = OmegaConf.merge(self.cfg,OmegaConf.from_dotlist(dotlist))
            config = OmegaConf.to_container(
                    self.cfg, resolve=True, throw_on_missing=True
                )
            wandb.config.update(config)

            if not self.cfg.logging.wandb.sweep_id:
                wandb.run.name = self.cfg.logging.wandb.run_name
            else:
                self.cfg.train.logdir = os.path.join("checkpoints", wandb.run.name)
            # wandb.run.save()
            wandb.run.log_code(self.cfg.logging.wandb.log_code)

        if self.cfg.train.debug:
            ic.enable()
        else:
            ic.disable()
        
        ## create any directories needed for logging/saving checkpoints.
        os.makedirs(self.cfg.train.logdir, exist_ok=True)
        
        # for key, value in wandb.config.items():
        #     if hasattr(args, key):
        #         setattr(args, key, value)


    def _build_model(self):

        self.cfg.model.modal = self.cfg.dataset.modal
        ## force add "img_size" to model
        if "img_size" not in self.cfg["model"]:
            OmegaConf.update(self.cfg, "model.img_size", [self.cfg.dataset.img_size], force_add=True)
        if not self.cfg.model.num_cls:
            self.cfg.model.num_cls = 2 if len(self.cfg.model.labels) == 1 else len(self.cfg.model.labels)

        
        if 'RFNet' in self.cfg.model.arch:
            model = RFNetOrig(self.cfg.model) if 'orig' in self.cfg.model.arch.lower() else RFNet(self.cfg.model)
        
        elif 'M2FTrans' in self.cfg.model.arch:
            model = M2FTrans(self.cfg.model) if not 'v2' in self.cfg.model.arch else M2FTransv2(self.cfg.model)


        elif 'IMS2Trans' in self.cfg.model.arch:
            if 'orig' in self.cfg.model.arch.lower():
                model = IMS2TransOrig(
                    img_size=self.cfg.model.img_size,
                    )
            else:
                model = IMS2Trans(
                    self.cfg.model,
                    in_channels=1,
                    normalize=True,
                    use_checkpoint=True,
                )
        
        elif 'mmFormer' in self.cfg.model.arch:
            
            model = mmFormerOrig(self.cfg.model) if 'orig' in self.cfg.model.arch.lower() else mmFormer(self.cfg.model)

        elif self.cfg.model.arch == 'SwinUNETR':
            model = SwinUNETR(
                    in_channels=len(self.cfg.model.modal),
                    out_channels=3,     #just for loading the Brats ckpt
                    img_size=(self.cfg.model.img_size, self.cfg.model.roi_y, self.cfg.model.roi_z),
                    feature_size=48,
                    use_checkpoint=False,
                    use_v2=False,
                    drop_rate=self.cfg.model.drop_rate,
                    dropout_path_rate=self.cfg.model.drop_rate,
                    attn_drop_rate=self.cfg.model.drop_rate,
                )
            in_dim = 16 * model.swinViT.embed_dim
            pretrained_pth = "/projectnb/ivc-ml/dlteif/pretrained_models/model_swinunetr_BRATS21.pt"
            # pretrained_pth = "/projectnb/ivc-ml/dlteif/pretrained_models/model_swinvit.pt"
            model_dict = torch.load(pretrained_pth, map_location="cpu")
            model_dict["state_dict"] = {k.replace("swinViT.", "module.").replace('.linear', '.fc'): v for k, v in model_dict["state_dict"].items()}
            ic(model_dict["state_dict"].keys())
            model.load_from(model_dict)
            
            model_dict["state_dict"] = {k.replace('module.', 'swinViT.').replace('.fc', '.linear'): v for k, v in model_dict["state_dict"].items()}
            model.load_state_dict(model_dict["state_dict"])
            if self.cfg.model.lora > 0:
                model.swinViT = LoRA_ViT_timm(vit_model=model.swinViT, r=4, alpha=1, num_classes=0)

            model = SwinUNETRBase(model)

        else:
            pass
        
        print(model)

        self.model = model
        
        
    def _build_optimizer(self):
        params = list(self.model.parameters())
    
        print("Trainable params: ", sum(p.numel() for p in params if p.requires_grad))
        print("Total params: ", sum(p.numel() for p in params))
        # exit()
        if "betas" in self.cfg.optimizer.params:
            betas = self.cfg.optimizer.params.betas
            if isinstance(betas, list):  # Convert to tuple if needed
                self.cfg.optimizer.params.betas = tuple(betas)

            print(type(self.cfg.optimizer.params.betas))

        if self.cfg.optimizer.name == "adam":
            self.optimizer = optim.Adam(params=params, **self.cfg.optimizer.params)
        elif self.cfg.optimizer.name == "adagrad":
            self.optimizer = optim.Adagrad(params=params, **self.cfg.optimizer.params)
        elif self.cfg.optimizer.name == "adamw":
            self.optimizer = optim.AdamW(params=params, **self.cfg.optimizer.params)

        elif self.cfg.optimizer.name == "sgd":
            self.optimizer = optim.SGD(params=params, lr=self.cfg.optimizer.params.lr, momentum=self.cfg.optimizer.params.momentum, weight_decay=self.cfg.optimizer.params.weight_decay)
        else:
            raise NotImplementedError
        
        # for group in self.optimizer.param_groups:
        #     ic(group.keys())

        for group in self.optimizer.param_groups:
            if "betas" not in group:
                group["betas"] = self.cfg.optimizer.params.betas
        
    def _build_scheduler(self):
        if self.cfg.scheduler:
            if self.cfg.scheduler.name == "warmup_cosine":
                # self.scheduler = WarmupCosineSchedule(self.optimizer, warmup_steps=self.cfg.model.warmup_steps, t_total=self.cfg.model.num_epochs)
                self.scheduler = WarmupCosineSchedule(self.optimizer, **self.cfg.scheduler.params)
            elif self.cfg.scheduler.name == "cosine":
                self.scheduler = CosineLRScheduler(self.optimizer, **self.cfg.scheduler.params)
            elif self.cfg.scheduler.name == "poly":
                lr = self.cfg.optimizer.params.lr
                def lambdas(epoch):
                    return np.power(1 - np.float32(epoch)/np.float32(self.cfg.train.num_epochs), 0.9)

                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambdas)
            elif self.cfg.scheduler.name == 'warmup_poly':
                def lambdas(epoch):
                    warmup = self.cfg.scheduler.warmup_steps
                    lr = self.cfg.optimizer.params.lr
                    if epoch < warmup:
                        now_lr = np.float32(epoch)/np.float32(100.0)
                    else:
                        now_lr = np.power(1 - (np.float32(epoch) - np.float32(100.0))/(np.float32(self.cfg.train.num_epochs)-np.float32(100.0)), 0.9)

                    return now_lr

                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambdas)
        else:
            self.scheduler = None
        

    def _build_dataset(self):
        ic(self.cfg.dataset.modal)
        if self.cfg.dataset.load_segmentations or 'RegionModalMix' in self.cfg.train.augment:
            self.labelmap = json.load(open("./data/synthseg_labelmapComplete.json", "r"))
            # self.labelmap = json.load(open("./data/synthseg_labelmap.json", "r"))
        
        if self.cfg.dataset.name.lower() in ['brats2018', 'brats2020', 'isles2022', 'gliomapost']:
            self._build_dataset_npy()
            return
  
    def _build_dataset_npy(self):
        roi = self.cfg.dataset.img_size
        
        if self.cfg.dataset.load_segmentations:
            train_transforms = f'Compose([RandCrop3D(({roi},{roi},{roi})), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64, np.int64)),])'
            # train_transforms = f'Compose([RandCrop3D(({roi},{roi},{roi})), NumpyType((np.float32, np.int64, np.int64)),])'
            test_transforms = 'Compose([NumpyType((np.float32, np.int64, np.int64)),])'
        else:
            train_transforms = f'Compose([RandCrop3D(({roi},{roi},{roi})), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
            test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
        
        if 'brats' in self.cfg.dataset.name.lower():
            root = f'/projectnb/ivc-ml/dlteif/BraTS2020/{self.cfg.dataset.name}_Training_none_npy/'
            self.train_dset = Brats_loadall(root=root, transforms=train_transforms, num_cls=self.cfg.model.num_cls, train_file=self.cfg.dataset.train_path, load_segmentations=self.cfg.dataset.load_segmentations, normalization=False)
            self.test_dset = Brats_loadall_test(transforms=test_transforms, root=root, test_file=self.cfg.dataset.test_path, normalization=False)
        elif 'gliomapost' in self.cfg.dataset.name.lower():
            root = f'/projectnb/ivc-ml/dlteif/MU-Glioma-Post/MU-Glioma-Post_npy/'
            if self.cfg.dataset.split_ratio == 0.0 and not self.cfg.dataset.train_path:
                self.train_dset = None
                self.test_dset = Brats_loadall_test(transforms=test_transforms, root=root, test_file=self.cfg.dataset.test_path, normalization=False)
            else:
                self.train_dset = Brats_loadall(root=root, transforms=train_transforms, num_cls=self.cfg.model.num_cls, train_file=self.cfg.dataset.train_path, load_segmentations=self.cfg.dataset.load_segmentations, normalization=False)
                self.test_dset = Brats_loadall_test(transforms=test_transforms, root=root, test_file=self.cfg.dataset.test_path, normalization=False)
        elif 'isles' in self.cfg.dataset.name.lower():
            root = f'/projectnb/ivc-ml/dlteif/ISLES-2022/ISLES2022_npy_Zscorenorm/'
            train_list, test_list = self._split_stratify_isles(root)
            self.train_dset = Isles_loadall(train_list, root=root, transforms=train_transforms, num_cls=self.cfg.model.num_cls, load_segmentations=self.cfg.dataset.load_segmentations, normalization=False)
            self.test_dset = Isles_loadall_test(test_list, transforms=test_transforms, root=root, normalization=False)

        if self.train_dset:
            if self.use_ddp:
                train_sampler = DistributedSampler(
                    self.train_dset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False
                )
            else:
                train_sampler = RandomSampler(
                    self.train_dset, replacement=False
                )
            self.train_loader = MultiEpochsDataLoader(
                dataset=self.train_dset,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.hardware.num_workers,
                sampler=train_sampler,
                pin_memory=True)
        else:
            self.train_loader = None
        
        if self.use_ddp:
            test_sampler = DistributedSampler(
                self.test_dset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False
            )
        else:
            test_sampler = RandomSampler(
                self.test_dset, replacement=False
            )

        
        self.test_loader = MultiEpochsDataLoader(
            dataset=self.test_dset,
            batch_size=self.cfg.eval.batch_size,
            num_workers=0, #self.cfg.hardware.num_workers,
            # sampler=val_sampler,
            pin_memory=True,
            shuffle=False)


    def _init_loss(self):
        if not self.cfg.dataset.name.lower() in ['brats2018', 'brats2020', 'isles2022', 'gliomapost']:
            if self.cfg.test or self.cfg.train.criterion == 'CE':
                self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.cfg.train.label_smoothing)
            elif self.cfg.train.criterion == 'wCE':
                weights, _ = self.train_dset.get_sample_weights()
                self.criterion = torch.nn.CrossEntropyLoss(weight=weights.cuda(self.device, non_blocking=True), label_smoothing=self.cfg.train.label_smoothing)
            elif self.cfg.train.criterion == 'focal':
                weights, _ = self.train_dset.get_sample_weights()
                self.criterion = FocalLoss(alpha=weights.cuda(self.device, non_blocking=True), gamma=2)
            else:
                raise NotImplementedError
            # self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.cfg.train.label_smoothing)

        if self.cfg.train.noamp:
            self.scaler = None
        else:
            self.scaler = GradScaler()
        

    def _get_best_artifact(self, path):
        if ":v" in path:
            artifact_dir = "artifacts/" + path.split("/")[-1]
            if not os.path.exists(artifact_dir):
                run = wandb.init()
                artifact = run.use_artifact(path, type='model')
                artifact_dir = artifact.download()
        
            ic(artifact_dir)
        else:
            api = wandb.Api()

            artifact_versions = api.artifact_versions("model", path)
            best_metric = 0
            best_version = 0
            key = self.cfg.eval.suffix[1:] #"val_loss" if self.cfg.eval.suffix == '_best' else "balacc" if self.cfg.eval.suffix == '_bestbalacc' else "F1"
            for artifact in artifact_versions:
                print(artifact.metadata)
                if key != "val_loss":
                    if artifact.metadata[key] > best_metric:
                        best_metric = artifact.metadata[key]
                        best_version = artifact
                else:
                    if key not in artifact.metadata:
                        key = "best"
                    if artifact.metadata[key] < best_metric:
                        best_metric = artifact.metadata[key]
                        best_version = artifact
                
            ic(best_metric)
            artifact_dir = best_version.download()
        path = os.path.join(artifact_dir, os.path.basename(artifact_dir).split(":")[0] + ".pt")
        print(path)
        return path

    def _load_model(self, path):
        if self.cfg.eval.wandb_artifact:
            path = self._get_best_artifact(path)
            self.cfg.train.resume_model = path
            
        checkpoint = torch.load(path, map_location=f'cuda:{self.cfg.hardware.gpu}')
        # checkpoint = torch.load(path)
        print('best epoch: {}'.format(checkpoint['epoch']))
        if self.cfg.train.checkpoint_key in ['student', 'teacher']:
            state_dict = {k.replace('module.backbone.', ''):v for k,v in checkpoint[self.cfg.train.checkpoint_key].items() if 'backbone' in k}
            self.model.load_state_dict(state_dict, strict=False)
        else:
            state_dict = {k.replace('module.', ''):v for k,v in checkpoint[self.cfg.train.checkpoint_key].items()}
            ic(isinstance(self.model, DistributedDataParallel))
            if isinstance(self.model, DistributedDataParallel):
                state_dict = {f'module.{k}':v for k,v in checkpoint[self.cfg.train.checkpoint_key].items()}
                
            
            ic(state_dict.keys())
            ic(self.model.state_dict().keys())
            # workarounds for some of the checkpoints saved with extra keys by mistake
            if self.cfg.model.fusion_type != 'attn':
                state_dict = {k:v for k,v in state_dict.items() if 'fusionBlock' not in k}

            # state_dict = {k.replace('mpm_conv_encoder', 'modalPromptModule'):v for k,v in state_dict.items()}
            if not (self.cfg.dataset.name.lower() in ['brats2018', 'brats2020', 'isles2022', 'gliomapost']):
                state_dict = {k:v for k,v in state_dict.items() if not ('decoder_fuse' in k or 'decoder_sep' in k)}
            
                
            # strict = False if self.cfg.model.arch == 'mmFormer' else True
            # self.model.modalEncoder = None
            self.model.load_state_dict(state_dict, strict=True)
        
            
        if 'epoch' in checkpoint and self.cfg.train.start_epoch is None:
            self.start_epoch = checkpoint['epoch'] + 1
        
        if 'optimizer' in checkpoint and not self.cfg.test:
            # if args.fusion_type != 'attn':
            #     checkpoint['optimizer']['param_groups'] = [p for p in checkpoint['optimizer']['param_groups'] if 'fusionBlock' not in p}
            # ic(checkpoint['optimizer']['param_groups'])
            # ic(optimizer.state_dict()['param_groups'])
            # ic(state)
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print('Loaded optimizer state dict successfully.')
            except:
                pass
        if 'balacc' in checkpoint:
            self.best_balacc = checkpoint['balacc']
            print("best balacc: ", self.best_balacc)
        if 'F1' in checkpoint:
            self.best_f1 = checkpoint['F1']
            print("best f1: ", self.best_f1)

        if 'val_loss' in checkpoint:
            self.best_val = checkpoint['val_loss']
            print("best val loss: ", self.best_val)
    
        ic(self.best_val, self.best_balacc, self.best_f1)
        # exit()



