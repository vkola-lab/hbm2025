import monai.inferers
import torch.nn.functional as F
import torch
import monai
import logging
import torch.nn as nn
from utils.logging_utils import AverageMeter
import numpy as np
import nibabel as nib
from torch.nn.parallel import DistributedDataParallel
import time
from tqdm import tqdm
from medpy.metric import hd95
import csv
import os
from models.anatprompting_module import Framework
from utils.dist_utils import is_main_process

__all__ = ['sigmoid_dice_loss','softmax_dice_loss','GeneralizedDiceLoss','FocalLoss', 'dice_loss']

cross_entropy = F.cross_entropy

def dynamic_bce_loss(output, target):
    B = target.size(0)
    target = target.float()
    # compute pos_weight over batch
    num_pos = target.sum()
    num_neg = target.numel() - num_pos
    
    pos_weight = (num_neg / (num_pos + 1e-6)).clamp(max=1e4)

    loss = F.binary_cross_entropy_with_logits(
        output, target, pos_weight=pos_weight
    )

    return torch.mean(loss) 

def binary_dice_loss(output, target, eps=1e-7): # soft dice loss
    output = output.float()
    target = target.float()
    intersection = torch.sum(output * target)
    union = torch.sum(output) + torch.sum(target)
    dice = (2. * intersection + eps) / (union + eps)
    return 1.0 - dice

def multiclass_dice_loss(output, target, num_cls=5, eps=1e-7):
    output = output.float()
    target = target.float()
    dice = 0.0
    for i in range(num_cls):
        intersection = torch.sum(output[:,i] * target[:,i])
        union = torch.sum(output[:,i]) + torch.sum(target[:,i])
        dice += (2.0 * intersection + eps) / (union + eps)  # eps added to numerator in cases of highly imbalanced data when intersection is near zero.
    return 1.0 - dice / num_cls


class DiceLoss:
    def __init__(self, num_cls=5, eps=1e-7):
        self.num_cls = num_cls
        self.eps = eps

    def __call__(self, output, target):
        if self.num_cls == 1:
            return binary_dice_loss(output, target, eps=self.eps)
        
        return multiclass_dice_loss(output, target, num_cls=self.num_cls, eps=self.eps)



class SoftmaxWeightedLoss:
    def __init__(self, num_cls=5):
        self.num_cls = num_cls

    def __call__(self, output, target):
        if self.num_cls == 1:
            raise ValueError("softmax_weighted_loss is not valid for num_cls=1. Use BCEWithLogitsLoss instead.")
        
        target = target.float()
        B, _, H, W, Z = output.size()
        ic(target.size(), output.size())
        for i in range(self.num_cls):
            outputi = output[:, i, :, :, :]
            targeti = target[:, i, :, :, :]
            # ic(outputi.size(), targeti.size())
            weighted = 1.0 - (torch.sum(targeti, (1,2,3)) * 1.0 / torch.sum(target, (1,2,3,4)))
            # ic(weighted.size())
            weighted = torch.reshape(weighted, (-1,1,1,1)).repeat(1,H,W,Z)
            # ic(weighted.size())
            if i == 0:
                cross_loss = -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
            else:
                cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        cross_loss = torch.mean(cross_loss)
        return cross_loss
            
def softmax_loss(output, target, num_cls=5):
    target = target.float()
    _, _, H, W, Z = output.size()
    for i in range(num_cls):
        outputi = output[:, i, :, :, :]
        targeti = target[:, i, :, :, :]
        if i == 0:
            cross_loss = -1.0 * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        else:
            cross_loss += -1.0 * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss)
    return cross_loss

def FocalLoss(output, target, alpha=0.25, gamma=2.0):
    target[target == 4] = 3 # label [4] -> [3]
    # target = expand_target(target, n_class=output.size()[1]) # [N,H,W,D] -> [N,4,H,W,D]
    if output.dim() > 2:
        output = output.view(output.size(0), output.size(1), -1)  # N,C,H,W,D => N,C,H*W*D
        output = output.transpose(1, 2)  # N,C,H*W*D => N,H*W*D,C
        output = output.contiguous().view(-1, output.size(2))  # N,H*W*D,C => N*H*W*D,C
    if target.dim() == 5:
        target = target.contiguous().view(target.size(0), target.size(1), -1)
        target = target.transpose(1, 2)
        target = target.contiguous().view(-1, target.size(2))
    if target.dim() == 4:
        target = target.view(-1) # N*H*W*D
    # compute the negative likelyhood
    logpt = -F.cross_entropy(output, target)
    pt = torch.exp(logpt)
    # compute the loss
    loss = -((1 - pt) ** gamma) * logpt
    # return loss.sum()
    return loss.mean()

def sigmoid_dice_loss(output, target,alpha=1e-5):
    # output: [-1,3,H,W,T]
    # target: [-1,H,W,T] noted that it includes 0,1,2,4 here
    loss1 = dice(output[:,0,...],(target==1).float(),eps=alpha)
    loss2 = dice(output[:,1,...],(target==2).float(),eps=alpha)
    loss3 = dice(output[:,2,...],(target == 4).float(),eps=alpha)
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(1-loss1.data, 1-loss2.data, 1-loss3.data))
    return loss1+loss2+loss3


def softmax_dice_loss(output, target,eps=1e-5): #
    # output : [bsize,c,H,W,D]
    # target : [bsize,H,W,D]
    loss1 = dice(output[:,1,...],(target==1).float())
    loss2 = dice(output[:,2,...],(target==2).float())
    loss3 = dice(output[:,3,...],(target==4).float())
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(1-loss1.data, 1-loss2.data, 1-loss3.data))

    return loss1+loss2+loss3


# Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
class GeneralizedDiceLoss:
    def __init__(self, num_cls=3, eps=1e-5, weight_type='square'):
        """
            Generalized Dice : 'Generalized dice overlap as a deep learning loss function for highly unbalanced segmentations'
        """
        self.num_cls = num_cls
        self.eps = eps
        self.weight_type = weight_type


    def __call__(self, output, target): # Generalized dice loss
        

        # target = target.float()
        if target.dim() == 4:
            target[target == 4] = 3 # label [4] -> [3]
            target = expand_target(target, n_class=output.size()[1]) # [N,H,W,D] -> [N,4，H,W,D]

        output = flatten(output)[1:,...] # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
        target = flatten(target)[1:,...] # [class, N*H*W*D]

        target_sum = target.sum(-1) # sub_class_voxels [3,1] -> 3个voxels
        if self.weight_type == 'square':
            class_weights = 1. / (target_sum * target_sum + self.eps)
        elif self.weight_type == 'identity':
            class_weights = 1. / (target_sum + self.eps)
        elif self.weight_type == 'sqrt':
            class_weights = 1. / (torch.sqrt(target_sum) + self.eps)
        else:
            raise ValueError('Check out the weight_type :', self.weight_type)

        # print(class_weights)
        intersect = (output * target).sum(-1)
        intersect_sum = (intersect * class_weights).sum()
        denominator = (output + target).sum(-1)
        denominator_sum = (denominator * class_weights).sum() + self.eps

        loss1 = 2*intersect[0] / (denominator[0] + self.eps)
        loss2 = 2*intersect[1] / (denominator[1] + self.eps)
        loss3 = 2*intersect[2] / (denominator[2] + self.eps)
        #logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(loss1.data, loss2.data, loss3.data))

        return 1 - 2. * intersect_sum / denominator_sum #, [loss1.data, loss2.data, loss3.data]


def expand_target(x, n_class,mode='softmax'):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
        """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        xx[:,1,:,:,:] = (x == 1)
        xx[:,2,:,:,:] = (x == 2)
        xx[:,3,:,:,:] = (x == 3)
    if mode.lower() == 'sigmoid':
        xx[:,0,:,:,:] = (x == 1)
        xx[:,1,:,:,:] = (x == 2)
        xx[:,2,:,:,:] = (x == 3)
    return xx.to(x.device)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)


def dice_coef_metric_per_classes(probabilities: np.ndarray,
                                    truth: np.ndarray,
                                    treshold: float = 0.5,
                                    eps: float = 1e-9,
                                    classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:
    """
    Calculate Dice score for data batch and for each class.
    Params:
        probobilities: model outputs after activation function.
        truth: model targets.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        classes: list with name classes.
        Returns: dict with dice scores for each class.
    """
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= treshold).astype(np.float32)
    assert(predictions.shape == truth.shape)

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = 2.0 * (truth_ * prediction).sum()
            union = truth_.sum() + prediction.sum()
            if truth_.sum() == 0 and prediction.sum() == 0:
                 scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)
                
    return scores

patch_size = 128

def softmax_output_dice_class4(output, target):
    eps = 1e-8
    #######label1: NCR/NET########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    ncr_net_dice = intersect1 / denominator1

    #######label2: ED########
    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / denominator2

    #######label3: ET########
    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    enhancing_dice = intersect3 / denominator3

    ####post processing:
    if torch.sum(o3) < 500:
       o4 = o3 * 0.0
    else:
       o4 = o3
    t4 = t3
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    enhancing_dice_postpro = intersect4 / denominator4

    o_whole = o1 + o2 + o3 
    t_whole = t1 + t2 + t3 
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat(
        (
            torch.unsqueeze(ncr_net_dice, 1), 
            torch.unsqueeze(edema_dice, 1), 
            torch.unsqueeze(enhancing_dice, 1),
        ),
          dim=1)
    dice_evaluate = torch.cat(
        (
            torch.unsqueeze(dice_whole, 1), 
            torch.unsqueeze(dice_core, 1), 
            torch.unsqueeze(enhancing_dice, 1), 
            torch.unsqueeze(enhancing_dice_postpro, 1),
        ), 
        dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()

def softmax_output_dice_class4_masked(output, target):
    eps = 1e-8

    # Create a mask to ignore voxels labeled 4 (RC) in target [MU-Glioma-Post dataset]
    valid_mask = (target != 4)

    # Define one-hot class indicators and apply mask
    o1 = ((output == 1) & valid_mask).float()
    t1 = ((target == 1) & valid_mask).float()
    o2 = ((output == 2) & valid_mask).float()
    t2 = ((target == 2) & valid_mask).float()
    o3 = ((output == 3) & valid_mask).float()
    t3 = ((target == 3) & valid_mask).float()

    # Dice for each class
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denom1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    ncr_net_dice = intersect1 / denom1

    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denom2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / denom2

    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denom3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    enhancing_dice = intersect3 / denom3

    # Postprocessing threshold
    if torch.sum(o3) < 500:
        o4 = o3 * 0.0
    else:
        o4 = o3
    t4 = t3
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denom4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    enhancing_dice_postpro = intersect4 / denom4

    # WT and TC
    o_whole = o1 + o2 + o3
    t_whole = t1 + t2 + t3
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denom_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denom_whole

    o_core = o1 + o3
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denom_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denom_core

    dice_separate = torch.cat(
        (ncr_net_dice.unsqueeze(1), edema_dice.unsqueeze(1), enhancing_dice.unsqueeze(1)),
        dim=1
    )
    dice_evaluate = torch.cat(
        (dice_whole.unsqueeze(1), dice_core.unsqueeze(1), enhancing_dice.unsqueeze(1), enhancing_dice_postpro.unsqueeze(1)),
        dim=1
    )

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()


def softmax_output_dice_class5(output, target):
    eps = 1e-8
    #######label1########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    necrosis_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    non_enhancing_dice = intersect3 / denominator3

    o4 = (output == 4).float()
    t4 = (target == 4).float()
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    enhancing_dice = intersect4 / denominator4

    ####post processing:
    if torch.sum(o4) < 500:
        o5 = o4 * 0
    else:
        o5 = o4
    t5 = t4
    intersect5 = torch.sum(2 * (o5 * t5), dim=(1,2,3)) + eps
    denominator5 = torch.sum(o5, dim=(1,2,3)) + torch.sum(t5, dim=(1,2,3)) + eps
    enhancing_dice_postpro = intersect5 / denominator5

    o_whole = o1 + o2 + o3 + o4
    t_whole = t1 + t2 + t3 + t4
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3 + o4
    t_core = t1 + t3 + t4
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat((torch.unsqueeze(necrosis_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(non_enhancing_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()

def compute_BraTS_HD95(ref, pred):
    """
    ref and gt are binary integer numpy.ndarray s
    spacing is assumed to be (1, 1, 1)
    :param ref:
    :param pred:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)
    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 1.0
            # follow ACN and SMU-Net
            # return 373.12866
            # follow nnUNet
    elif num_pred == 0 and num_ref != 0:
        return 1.0
        # follow ACN and SMU-Net
        # return 373.12866
        # follow in nnUNet
    else:
        return hd95(pred, ref, (1, 1, 1))

def cal_hd95(output, target):
     # whole tumor
    mask_gt = (target != 0).astype(int)
    mask_pred = (output != 0).astype(int)
    hd95_whole = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # tumor core
    mask_gt = ((target == 1) | (target ==3)).astype(int)
    mask_pred = ((output == 1) | (output ==3)).astype(int)
    hd95_core = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # enhancing
    mask_gt = (target == 3).astype(int)
    mask_pred = (output == 3).astype(int)
    hd95_enh = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    mask_gt = (target == 3).astype(int)
    if np.sum((output == 3).astype(int)) < 500:
       mask_pred = (output == 3).astype(int) * 0
    else:
       mask_pred = (output == 3).astype(int)
    hd95_enhpro = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # ncr_net
    mask_gt = (target == 1).astype(int)
    mask_pred = (output == 1).astype(int)
    hd95_ncr_net = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # edema
    mask_gt = (target == 2).astype(int)
    mask_pred = (output == 2).astype(int)
    hd95_edema = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    hd95_separate = (hd95_ncr_net, hd95_edema) 

    hd95_evaluate = (hd95_whole, hd95_core, hd95_enh, hd95_enhpro)

    return np.array(hd95_separate), np.array(hd95_evaluate)


def cal_hd95_ignore_rc(output, target):
    # Create mask to ignore RC voxels (label 4 in target)
    rc_mask = (target != 4)

    def mask_out_rc(gt_mask, pred_mask):
        gt_mask = gt_mask * rc_mask
        pred_mask = pred_mask * rc_mask
        return gt_mask.astype(int), pred_mask.astype(int)

    # Whole tumor
    gt_mask = (target != 0).astype(int)
    pred_mask = (output != 0).astype(int)
    gt_mask, pred_mask = mask_out_rc(gt_mask, pred_mask)
    hd95_whole = compute_BraTS_HD95(gt_mask, pred_mask)

    # Tumor core
    gt_mask = ((target == 1) | (target == 3)).astype(int)
    pred_mask = ((output == 1) | (output == 3)).astype(int)
    gt_mask, pred_mask = mask_out_rc(gt_mask, pred_mask)
    hd95_core = compute_BraTS_HD95(gt_mask, pred_mask)

    # Enhancing tumor
    gt_mask = (target == 3).astype(int)
    pred_mask = (output == 3).astype(int)
    gt_mask, pred_mask = mask_out_rc(gt_mask, pred_mask)
    hd95_enh = compute_BraTS_HD95(gt_mask, pred_mask)

    # Enhancing with postprocessing threshold
    gt_mask = (target == 3).astype(int)
    if np.sum((output == 3).astype(int)) < 500:
        pred_mask = np.zeros_like(output)
    else:
        pred_mask = (output == 3).astype(int)
    gt_mask, pred_mask = mask_out_rc(gt_mask, pred_mask)
    hd95_enhpro = compute_BraTS_HD95(gt_mask, pred_mask)

    # NCR/NET
    gt_mask = (target == 1).astype(int)
    pred_mask = (output == 1).astype(int)
    gt_mask, pred_mask = mask_out_rc(gt_mask, pred_mask)
    hd95_ncr_net = compute_BraTS_HD95(gt_mask, pred_mask)

    # Edema
    gt_mask = (target == 2).astype(int)
    pred_mask = (output == 2).astype(int)
    gt_mask, pred_mask = mask_out_rc(gt_mask, pred_mask)
    hd95_edema = compute_BraTS_HD95(gt_mask, pred_mask)

    hd95_separate = (hd95_ncr_net, hd95_edema)
    hd95_evaluate = (hd95_whole, hd95_core, hd95_enh, hd95_enhpro)

    return np.array(hd95_separate), np.array(hd95_evaluate)



def binary_hd95(pred, target):
    from medpy.metric.binary import hd95
    pred = pred.cpu().numpy().astype(bool)
    target = target.cpu().numpy().astype(bool)
    try:
        return hd95(pred, target)
    except:
        return 0.0


def get_sliding_window_indices(H, W, Z, patch_size, min_overlap=0.5):
    """
    Generate sliding window start indices for an image of shape (H, W, Z).
    
    Args:
        H, W, Z: Image dimensions
        patch_size: Size of the sliding window patch (can be smaller than H, W, Z)
        overlap: Fraction of overlap between adjacent patches (default 0.5)
    
    Returns:
        Lists of start indices for each dimension
    """

    def compute_indices(dim_size):
        if patch_size >= dim_size:
            return [0]  # Only one window covering the entire dimension
        # dynamically compute stride
        stride = max(1, int(patch_size * (1 - min_overlap)))    # at least 1 pixel step.
        idx_list = list(range(0, dim_size - patch_size + 1, stride))
        
        # Ensure last patch always covers the end
        if idx_list[-1] + patch_size < dim_size:
            idx_list.append(dim_size - patch_size)

        return idx_list

    h_idx_list = compute_indices(H)
    w_idx_list = compute_indices(W)
    z_idx_list = compute_indices(Z)

    return h_idx_list, w_idx_list, z_idx_list


def test_dice_hd95_softmax(
        test_loader,
        model,
        dataname = 'BraTS2020',
        feature_mask=None,
        mask_name=None,
        csv_name=None,
        augmentation='none',
        ):

    # H, W, T = 240, 240, 155
    model.eval()
    vals_dice_evaluation = AverageMeter()
    vals_hd95_separate = AverageMeter()
    vals_hd95_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    if isinstance(model, DistributedDataParallel):
        model.module.is_training=False
        img_size = model.module.img_size
    else:
        img_size = model.img_size
        model.is_training = False

    patch_size = img_size if not isinstance(img_size, tuple) else img_size[0]
    one_tensor = torch.ones(1, patch_size, patch_size, patch_size).float().cuda()

    if dataname.lower() in ['brats2021', 'brats2020', 'brats2018', 'gliomapost']:
        num_cls = 4
        class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'
    elif dataname.lower() == 'brats2015':
        num_cls = 5
        class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'
    elif dataname.lower() == 'isles2022':
        num_cls = 1
        class_evaluation = 'lesion'
        class_separate = 'lesion'
        
    

    for i, data in enumerate(test_loader):
        target = data[1].cuda()
        x = data[0].cuda() 
        if len(x.size()) > 5:
            x = x.squeeze(2)
        names = data[-1]
        if isinstance(model, Framework):
            seg_labels = data[-2].cuda().squeeze(2)
            ic(seg_labels.size())
        else:
            seg_labels = None
        if feature_mask is not None:
            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        else:
            mask = data[2]
        mask = mask.cuda()
        ic(x.size(), target.size(), names)
        ## pad to patch size first.
        # pad_transform = monai.transforms.SpatialPad(spatial_size=(x.size(1),patch_size,patch_size,patch_size), method='symmetric')
        # x = pad_transform(x)
        ic(x.size())
        B, _, H, W, Z = x.size()

        # to avoid errors at last batch of size < batch_size
        sw_batch_size = 2 * B if isinstance(model, Framework) else B
        inferer = monai.inferers.SlidingWindowInferer(patch_size, sw_batch_size=sw_batch_size, mode='gaussian')
        # if H == patch_size:
        #     pred = model(x, mask)
        # else:
            #########get h_ind, w_ind, z_ind for sliding windows
            # h_idx_list, w_idx_list, z_idx_list = get_sliding_window_indices(H, W, Z, patch_size, 0.5)

            # ic(h_idx_list, w_idx_list, z_idx_list)
            # #####compute calculation times for each pixel in sliding windows
            # weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
            # for h in h_idx_list:
            #     for w in w_idx_list:
            #         for z in z_idx_list:
            #             ic(weight1[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size].size(), one_tensor.size())
            #             weight1[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += one_tensor
            # weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

            # #####evaluation
            # pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()
            
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
            
        #     for h in h_idx_list:
        #         for w in w_idx_list:
        #             for z in z_idx_list:
        #                 x_input = x[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size]
        #                 ic(x_input.size())
        #                 pred_part = model(x_input, mask)
        #                 ic(pred_part.size())
        #                 pred[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += pred_part
        #     pred = pred / weight
        #     b = time.time()
        #     pred = pred[:, :, :H, :W, :Z]

        ic(pred.shape, pred.min(), pred.max())
        pred = torch.argmax(pred, dim=1)
        pred_npy = pred.cpu().detach().numpy()
        # _x = x[0,0,:,:,:].cpu().detach().numpy()
        ic(x.shape, x.min(), x.max(), target.shape)

        if dataname.lower() in ['brats2021','brats2020', 'brats2018']:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
            ic(scores_separate, scores_evaluation)
            hd95_scores = [cal_hd95(pred[k].cpu().numpy(), target[k].cpu().numpy()) for k in range(len(names))]
            hd95_separate = [score[0] for score in hd95_scores]
            hd95_evaluation = [score[1] for score in hd95_scores] 
            ic(hd95_separate, hd95_evaluation)

        elif dataname.lower() == 'gliomapost':
            scores_separate, scores_evaluation = softmax_output_dice_class4_masked(pred, target)
            ic(scores_separate, scores_evaluation)
            hd95_scores = [cal_hd95_ignore_rc(pred[k].cpu().numpy(), target[k].cpu().numpy()) for k in range(len(names))]
            hd95_separate = [score[0] for score in hd95_scores]
            hd95_evaluation = [score[1] for score in hd95_scores] 
            ic(hd95_separate, hd95_evaluation)

        elif dataname.lower() == 'isles2022':
            scores_separate = [binary_dice_loss(pred[k], target[k]) for k in range(len(names))]
            scores_evaluation = scores_separate
            hd95_separate = [binary_hd95(pred[k], target[k]) for k in range(len(names))]
            hd95_evaluation = hd95_separate
            ic(scores_separate, hd95_separate)

        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i+1), len(test_loader), (k+1), len(names))
            msg += '{:>20}, '.format(name)

            vals_separate.update(scores_separate[k])
            vals_dice_evaluation.update(scores_evaluation[k])
            vals_hd95_separate.update(hd95_separate[k])
            vals_hd95_evaluation.update(hd95_evaluation[k])
            msg += 'DSC: '
            
            if is_main_process():
                if num_cls > 1:
                    ic(type(class_evaluation), class_evaluation, type(scores_evaluation[k]), type(hd95_evaluation), hd95_evaluation)
                    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
                    msg += ', ' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])
                    msg += ', HD95: '
                    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, hd95_evaluation[k])])
                    msg += ', ' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, hd95_separate[k])])
                else:
                    msg += 'DSC: {}: {:.4f}'.format(class_evaluation, scores_evaluation[k])
                    msg += ', HD95: '
                    msg += '{}: {:.4f}'.format(class_evaluation, hd95_evaluation[k])
                
            
            logging.info(msg)
            # save_dict = {'pred': pred_npy[0],
            #          'filename': names[0],
            #          'scores': msg
            #          }
            # np.save(os.pa th.join(os.path.dirname(csv_name), f"{mask_name}_pred_{names[0]}.npy"), save_dict)
            # return vals_dice_evaluation.avg, vals_hd95_evaluation.avg
            # exit()
            if csv_name and is_main_process():
                file = open(csv_name, "a+")
                csv_writer = csv.writer(file)
                if num_cls > 1:
                    csv_writer.writerow([scores_evaluation[k][0], scores_evaluation[k][1], scores_evaluation[k][2],scores_evaluation[k][3], scores_separate[k][0], scores_separate[k][1],\
                    hd95_evaluation[k][0], hd95_evaluation[k][1], hd95_evaluation[k][2], hd95_evaluation[k][3],\
                    hd95_separate[k][0], hd95_separate[k][1]])
                #     csv_writer.writerow([scores_evaluation[k][0], scores_evaluation[k][1], scores_evaluation[k][2],scores_evaluation[k][3],\
                # scores_hd95[0], scores_hd95[1], scores_hd95[2], scores_hd95[3]])

                else:
                    csv_writer.writerow([scores_evaluation[k], hd95_evaluation[k]])
                file.close()

    msg = 'Average scores:'
    msg += 'DSC: ' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_dice_evaluation.avg)])
    msg += ', ' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_separate.avg)])
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_hd95_evaluation.avg)])
    msg += ', HD95: ' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_hd95_separate.avg)])
    if is_main_process():
        print (msg)
        logging.info(msg)
    model.train()
    return vals_dice_evaluation.avg, vals_hd95_evaluation.avg