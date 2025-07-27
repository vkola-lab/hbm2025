#coding=utf-8
import logging
import os
import random
import numpy as np
import monai
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from models.anatprompting_module import Framework
from utils.logging_utils import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from utils.dist_utils import init_distributed_mode
from losses.SEG_loss import dice_loss, softmax_output_dice_class4, cal_hd95
from medpy.metric import hd95
from config import MyConfig
from trainer import Trainer
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)


def test_softmax_visualize(
        data,
        model,
        dataname,
        feature_mask=None,
        mask_name=None,
        writer = None,
        save_path = None,
        ):

    model.eval()
    vals_dice_evaluation = AverageMeter()
    vals_hd95_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    patch_size = model.img_size if not isinstance(model.img_size, tuple) else model.img_size[0]
    one_tensor = torch.ones(1, patch_size, patch_size, patch_size).float().cuda()

    if dataname in ['BRATS2020', 'BRATS2018']:
        num_cls = 4
        class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'

    sw_batch_size = 2 if isinstance(model, Framework) else 1
    inferer = monai.inferers.SlidingWindowInferer(patch_size, sw_batch_size=sw_batch_size, mode='gaussian')
    names = [data[-1]]

    np.save(os.path.join(save_path, f"{names[0]}_vol.npy"), data[0].numpy())
    np.save(os.path.join(save_path, f"{names[0]}_seg.npy"), data[1].numpy())

    target = data[1].cuda().unsqueeze(0)
    x = data[0].cuda().unsqueeze(0)
    
    
    if isinstance(model, Framework):
        seg_labels = data[-2].cuda().unsqueeze(0)
        ic(seg_labels.size())
    else:
        seg_labels = None
    if feature_mask is not None:
        mask = torch.from_numpy(np.array(feature_mask)).unsqueeze(0)
        if len(names) > 1:
            mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1) 
    else:
        mask = data[2].unsqueeze(0)
    mask = mask.cuda()
    ic(x.size(), target.size(), mask.size(), names)
    ## pad to patch size first.
    # pad_transform = monai.transforms.SpatialPad(spatial_size=(x.size(1),patch_size,patch_size,patch_size), method='symmetric')
    # x = pad_transform(x)

    ic(x.size())
    _, _, H, W, Z = x.size()
    if isinstance(model, DistributedDataParallel):
        model.module.is_training=False
    else:
        model.is_training = False
            
    if isinstance(model, Framework):
        kwargs = {"train": False}
        def _callback(model, inp, mask, **kwargs):
            ic(inp.size())
            _x = torch.chunk(inp, 2, dim=0)
            ic(len(_x))
            inputs, seg_labels = _x[0], _x[1]
            ic(inputs.size(), seg_labels.size())
            kwargs['seg_labels'] = seg_labels
            return model.forward_SEG(inputs, mask, **kwargs)
        
        x_input = torch.cat((x, seg_labels), dim=0)
        ic(x_input.size())
        pred = inferer(x_input, lambda inp: _callback(model, inp, mask, **kwargs))
    else:
        pred = inferer(x, model, mask)

    pred = torch.argmax(pred, dim=1)
    pred_npy = pred.cpu().detach().numpy()
    ic(x.shape, x.min(), x.max(), target.shape)
        
    if dataname.lower() in ['brats2021','brats2020', 'brats2018']:
        scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
        ic(scores_separate, scores_evaluation)
        scores_hd95 = np.array(cal_hd95(pred[0].cpu().numpy(), target[0].cpu().numpy()))
        ic(scores_hd95)

    for k, name in enumerate(names):
        msg = 'Subject {}/{}'.format((k+1), len(names))
        msg += '{:>20}, '.format(name)

        vals_separate.update(scores_separate[k])
        vals_dice_evaluation.update(scores_evaluation[k])
        vals_hd95_evaluation.update(scores_hd95)
        msg += 'DSC: '
        msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
        msg += ', HD95: '
        msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_hd95)])
        logging.info(msg)
        save_dict = {'pred': pred_npy[0],
                 'filename': names[0],
                 'scores': msg
                 }
        np.save(os.path.join(save_path, f"{mask_name}_pred_{names[0]}.npy"), save_dict)


    logging.info(msg)
    model.train()


###modality missing mask
masks_test = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks_test))
mask_name = ['t2', 't1ce', 't1', 'flair',
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
print (masks_torch.int())

# masks_valid = [[False, False, True, False],
#             [False, True, True, False],
#             [True, True, False, True],
#             [True, True, True, True]]
masks_valid = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
# t1,t1cet1,flairticet2,flairt1cet1t2
masks_valid_torch = torch.from_numpy(np.array(masks_valid))
masks_valid_array = np.array(masks_valid)
masks_all = [True, True, True, True]
masks_all_torch = torch.from_numpy(np.array(masks_all))
# mask_name_valid = ['t1',
#                 't1cet1',
#                 'flairt1cet2',
#                 'flairt1cet1t2']
mask_name_valid = ['t2', 't1c', 't1', 'flair',
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
print (masks_valid_torch.int())

@hydra.main(version_base=None, config_path="/projectnb/ivc-ml/dlteif/multimodalMRI/configs", config_name="cfg_SEG")
def main(cfg: MyConfig) -> None:
    use_ddp = (cfg.hardware.multi_gpus == "ddp") and torch.cuda.is_available()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False #
        torch.autograd.set_detect_anomaly(False)

        if not cfg.hardware.master_port:
            cfg.hardware.master_port = 29435 + random.randint(0, 20000)
        init_distributed_mode(cfg.hardware)


    num_gpus = torch.cuda.device_count()
    cudnn.deterministic = True
    if "num_gpus" not in cfg["hardware"]:
        OmegaConf.update(cfg, "hardware.num_gpus", num_gpus, force_add=True)

    trainer = Trainer(cfg)
    from data.transforms import (Compose, Pad, NumpyType, PadToShape)
    import nibabel as nib
    itemidx = 8
    data = list(trainer.test_loader.dataset.__getitem__(itemidx))
    pad_dim = max(*data[0].size())
    # pad_transform = monai.transforms.Compose([
    #     # monai.transforms.CropForeground(),
    #     monai.transforms.SpatialPad(spatial_size=(pad_dim,)*3, method = 'symmetric', mode='minimum')
    # ])
    ic(data[0].size(), data[1].size())
    # x = np.zeros((data[0].size(0),pad_dim,pad_dim,pad_dim))

    
    # data[0] = pad_transform(data[0])
    # data[1] = pad_transform(data[1].unsqueeze(0)).squeeze(0)
    
    
    _, D,H,W = data[0].size()
    pad_list = [pad_dim, pad_dim, pad_dim]
    # ic(x.size(), pad_dim)
    # trainer.test_loader.dataset.transforms = eval(f'Compose([PadToShape({pad_list}),NumpyType((np.float32, np.int64)),])')
    # ic(trainer.test_loader.dataset.transforms)
    # data = trainer.test_loader.dataset.__getitem__(itemidx)
    ic(data[0].size(), data[1].size())
    
    # np.save('./paddedX.npy', data[0].numpy())
    # np.save('./paddedY.npy', data[1].numpy())
    # exit()
    # data = tuple(data)
    #########Visualize Evaluate
   
    test_score = AverageMeter()
    writer_visualize = SummaryWriter(log_dir=trainer.cfg.train.logdir)
    visualize_step = 0
    with torch.no_grad():
        logging.info('###########visualize model###########')
        # for idx, data in enumerate(tqdm(test_loader)):
        for i, mask in enumerate(masks_test[::-1]):
            logging.info('{}'.format(mask_name[::-1][i]))
            test_softmax_visualize(
                data,
                trainer.model,
                dataname = trainer.cfg.dataset.test_name,
                feature_mask = mask,
                mask_name = mask_name[::-1][i],
                writer = writer_visualize,
                save_path = trainer.cfg.train.logdir)
                


    # if args.resume is not None:
    #     checkpoint = torch.load(args.resume)
    #     pretrained_dict = checkpoint['state_dict']
    #     model_dict = model.state_dict()
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     logging.info('pretrained_dict: {}'.format(pretrained_dict))
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict)
    #     logging.info('load ok')


if __name__ == '__main__':
    main()