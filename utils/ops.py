# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import numpy as np
import torch
from numpy.random import randint
import torch.distributed as dist
import math
from enum import Enum
import torch.distributed as dist

import time
from typing import Callable, Union, List, Tuple, Any, Dict


############################################
########## Device, GPU, mem, Cuda ##########

def get_machine_name():
    import socket

    machine_name = socket.gethostname()
    return machine_name


def get_device(cuda_device=None, verbose=True):
    cuda = default(cuda_device, "cuda")
    device = torch.device(f"{cuda_device}" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("device: ", device)
    return device


def get_gpu_mem(cuda="cuda:0", return_total_mem=False):
    free, total = torch.cuda.mem_get_info(device=cuda)
    free_gb, total_gb = free / 1024**3, total / 1024**3
    use_gb = total_gb - free_gb
    out = f"used/avail mem: {use_gb:.1f}/{total_gb:.1f} GB"
    if return_total_mem:
        return total_gb
    else:
        return out


def get_gpu_mem_all() -> None:
    ## get all gpu available
    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        free_gb = get_gpu_mem(cuda=f"cuda:{i}")
        print(f"\tdevice: {i+1}/{n_gpus}, avail mem: {free_gb}GB")


def gpu_mem_report_details():
    import humanize, psutil, GPUtil

    print("CPU RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available))
    gpu_list = GPUtil.getGPUs()
    for i, gpu in enumerate(gpu_list):
        print(
            "GPU {:d} ... Mem Used: {:.0f}MB\t Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%".format(
                i,
                gpu.memoryTotal - gpu.memoryFree,
                gpu.memoryFree,
                gpu.memoryTotal,
                gpu.memoryUtil * 100,
            )
        )


def gpu_mem_report(device: Union[int, List, torch.device, None] = None, msg=None):
    def get_mem_msg(cuda):
        free, total = torch.cuda.mem_get_info(device=cuda)
        free_gb, total_gb = free / 1024**3, total / 1024**3
        used_gb = total_gb - free_gb
        msg_1 = f"Device {cuda} - {torch.cuda.get_device_name(cuda)}"
        msg_2 = f"Mem used: {used_gb:.2f} GB; free/total: {free_gb:.2f}/{total_gb:.2f} GB\n"
        return msg_1, msg_2

    def ensure_list(x: Union[int, List]):
        if isinstance(x, List):
            return x
        else:
            return [x]

    if not torch.cuda.is_available():  # skip if gpu is not available
        return None
    if device is None:
        device = range(torch.cuda.device_count())
    else:
        device = ensure_list(device)

    if msg is not None:
        print(msg)

    for cuda in device:
        msg_1, msg_2 = get_mem_msg(cuda)
        print(msg_1, "\n", msg_2, "------")


def move_to_cuda(sample, device):
    def _move_to_cuda(tensor):
        return tensor.to(device)

    return apply_to_sample(_move_to_cuda, sample)


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)



def patch_rand_drop(args, x, x_rep=None, max_drop=0.3, max_block_sz=0.25, tolr=0.05):
    c, h, w, z = x.size()
    n_drop_pix = np.random.uniform(0, max_drop) * h * w * z
    mx_blk_height = int(h * max_block_sz)
    mx_blk_width = int(w * max_block_sz)
    mx_blk_slices = int(z * max_block_sz)
    tolr = (int(tolr * h), int(tolr * w), int(tolr * z))
    total_pix = 0
    while total_pix < n_drop_pix:
        rnd_r = randint(0, h - tolr[0])
        rnd_c = randint(0, w - tolr[1])
        rnd_s = randint(0, z - tolr[2])
        rnd_h = min(randint(tolr[0], mx_blk_height) + rnd_r, h)
        rnd_w = min(randint(tolr[1], mx_blk_width) + rnd_c, w)
        rnd_z = min(randint(tolr[2], mx_blk_slices) + rnd_s, z)
        if x_rep is None:
            x_uninitialized = torch.empty(
                (c, rnd_h - rnd_r, rnd_w - rnd_c, rnd_z - rnd_s), dtype=x.dtype, device=dist.get_rank()
            ).normal_()
            x_uninitialized = (x_uninitialized - torch.min(x_uninitialized)) / (
                torch.max(x_uninitialized) - torch.min(x_uninitialized)
            )
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_uninitialized
        else:
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_rep[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z]
        total_pix = total_pix + (rnd_h - rnd_r) * (rnd_w - rnd_c) * (rnd_z - rnd_s)
    return x


def rot_rand(args, x_s):
    img_n = x_s.size()[0]
    ic(x_s.size(), img_n)
    x_aug = x_s.detach().clone()
    device = x_s.device
    x_rot = torch.zeros(img_n).long().to(device)
    ic(x_rot.size())
    for i in range(img_n):
        x = x_s[i]
        orientation = np.random.randint(0, 4)
        if orientation == 0:
            pass
        elif orientation == 1:
            x = x.rot90(1, (2, 3))
        elif orientation == 2:
            x = x.rot90(2, (2, 3))
        elif orientation == 3:
            x = x.rot90(3, (2, 3))
        x_aug[i] = x
        x_rot[i] = orientation
    return x_aug, x_rot


def aug_rand(args, samples):
    img_n = samples.size()[0]
    x_aug = samples.detach().clone()
    for i in range(img_n):
        x_aug[i] = patch_rand_drop(args, x_aug[i])
        idx_rnd = randint(0, img_n)
        if idx_rnd != i:
            x_aug[i] = patch_rand_drop(args, x_aug[i], x_aug[idx_rnd])
    return x_aug


class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        """
        get all values from Enum
        """
        return list(map(lambda c: c.value, cls))

def pairwise_distance_v2(proxies, x, squared=False):
    if squared:
        return (torch.cdist(x, proxies, p=2)) ** 2
    else:
        return torch.cdist(x, proxies, p=2)

def exists(val):
    return val is not None

def default(val, default):
    return val if exists(val) else default

def set_seeds(seed=2022):
    seed = seed + dist.get_rank()
    random.seed(seed)
    np.random.seed(seed )
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # ensures reproducibility

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}

    for key, value in kwargs.items():
        ic(key)
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]