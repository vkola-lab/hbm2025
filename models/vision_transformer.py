# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import torch
import torch.nn as nn

import random
from collections import Counter
from typing import Tuple, Optional
from utils.ops import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class PPTAttention(Attention):
    """
    Copy from https://github.com/xjwu1024/PPT/blob/main/ppt_deit.py
    Which seems originally from https://github.com/adaptivetokensampling/ATS/blob/main/libs/models/transformers/ats_block.py

    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
     - Return the tokens scores
    """

    # copy and refine from the ATS
    @staticmethod
    def score_assignment_step(attn, v):
        """
        Token Score Assignment Step.
        :param attn: attention matrix
        :param v: values
        :return: sorted significance scores and their corresponding indices ## where is sorted????
        """

        B, H, _, _ = attn.shape
        C = v.shape[3] * H
        v_norm = torch.linalg.norm(
            v.transpose(1, 2).reshape(B, attn.shape[2], C), ord=2, dim=2
        )  # value norm of size [B x T]
        import numpy as np

        np.save(
            "/projectnb/ivc-ml/dlteif/channel_adaptive_v2_new_channels/outputs/attn_vnorm/channelvit8_vnorm.npy",
            v_norm.detach().cpu().numpy(),
        )

        significance_score = attn[:, :, 0].sum(dim=1)  # attention weights of CLS token of size [B x T]
        np.save(
            "/projectnb/ivc-ml/chaupham/channel_adaptive_v2_new_channels/outputs/attn_vnorm/channelvit8_attn.npy",
            significance_score.detach().cpu().numpy(),
        )

        significance_score = significance_score * v_norm  # [B x T]

        np.save(
            "/projectnb/ivc-ml/dlteif/channel_adaptive_v2_new_channels/outputs/attn_vnorm/channelvit8_attnvnorm.npy",
            significance_score.detach().cpu().numpy(),
        )

        significance_score = significance_score[:, 1:]  # [B x T-1]

        significance_score = significance_score / significance_score.sum(dim=1, keepdim=True)  # [B x T-1]

        return significance_score

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # , size: torch.Tensor = Optional[None]
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        # if size is not None:
        #     attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        scores = self.score_assignment_step(attn, v)  # [B, N-1]
        largest = True
        constant = 9999 if largest else -9999
        # confirm cls tokens are reserved
        cls_score = constant * torch.ones(scores.shape[0], 1).to(scores.device)
        scores = torch.cat([cls_score, scores], dim=-1)  # [B, N]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k, scores as well here
        # return x, k.mean(1), scores
        return x, scores


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class BlockV2(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PPTAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_attention=False, pruning_method=None, nc: int = 0):
        y, scores = self.attn(self.norm1(x))
        # print(scores.shape)
        # print(torch.sort(scores))
        counter = None
        if pruning_method is not None:
            B, chw, d = x.shape
            num_channels = random.randint(1, nc)
            num_tokens = num_channels * (chw // nc) + 1

            ## keep top num_tokens_kept tokens
            if pruning_method == "token_pruning":
                _, indices = torch.topk(scores, num_tokens, dim=1, largest=True)
                # print("----indices", indices)
                # indices: torch.Size([6, 272])
                # y: torch.Size([6, 289, 384])
                ## get the corresponding tokens along 2nd dim
                idx = torch.arange(y.size(0)).unsqueeze(1)

                y = y[idx, indices]
                x = x[idx, indices]
                indices_flat = indices.flatten()
                counter = Counter(indices_flat.tolist())

            elif pruning_method == "channel_pruning":
                ## sum scores for each channels
                HW = chw // nc
                scores = scores[:, 1:].view(B, nc, HW).sum(dim=[0, -1])
                ## keep top num_channels channels
                _, indices = torch.topk(scores, num_channels, largest=True)
                ## add 0 to indices to keep the cls token
                drops = [True]  ## make sure the first token ([CLS]) is not dropped
                for i in range(nc):
                    if i in indices:
                        tmp = [True] * HW
                    else:
                        tmp = [False] * HW
                    drops.extend(tmp)
                drops = torch.tensor(drops, device=x.device)
                # print("------drops", drops)

                y = y[:, drops, :]
                x = x[:, drops, :]
                counter = Counter(indices.tolist())

            else:
                raise ValueError("Invalid pruning method")

        if return_attention:
            ## TODO: this is not attn but scores
            return scores
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.training:
            return x, counter
        else:
            return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class PatchEmbed3D(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=nn.LayerNorm):
        super().__init__()
        num_patches = (img_size // patch_size)**3 * in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, 
                            embed_dim,
                            kernel_size=patch_size,
                            stride=patch_size)

        ic(self.proj)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x)
        x = x.permute(0, 2, 3, 4, 1)  # Change to [B, D, H, W, C] for LayerNorm
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)  # Back to [B, C, D, H, W]
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed3D(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, d, w, h, c):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        d0 = d // self.patch_embed.patch_size

        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0, d0 = w0 + 0.1, h0 + 0.1, d0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(np.cbrt(N)), int(np.cbrt(N)), int(np.cbrt(N)), dim
            ).permute(0, 4, 1, 2, 3),
            scale_factor=(w0 / (np.cbrt(N)), h0 / (np.cbrt(N)), d0 / (np.cbrt(N))),
            mode="trilinear",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-3]
            and int(h0) == patch_pos_embed.shape[-2]
            and int(d0) == patch_pos_embed.shape[-1]
        )

        ic(patch_pos_embed.size())

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).view(1, 1, -1, dim)
        ic(patch_pos_embed.size())

        # create copies of the positional embeddings for each channel
        patch_pos_embed = patch_pos_embed.expand(1, c, -1, dim).reshape(1, -1, dim)
        ic(patch_pos_embed.size())

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, d, w, h = x.shape
        ic(x.shape)
        x = self.patch_embed(x)  # patch linear embedding
        ic(x.size())
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        ic(x.size())

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, d, w, h, nc)

        return self.pos_drop(x)

    def forward(self, x):
        ic(x.size())
        x = self.prepare_tokens(x)
        ic(x.size())
        for blk in self.blocks:
            # ic(blk)
            x = blk(x)
            ic(x.size())
        x = self.norm(x)
        ic(x.size())
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class SwinUNETRBase(torch.nn.Module):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.swinunetr = model

    def forward(self, x_in):
        ic(x_in.size(), x_in.min(), x_in.max(), torch.isnan(x_in).any())
        hidden_states_out = self.swinunetr.swinViT(x_in, normalize=self.swinunetr.normalize)
        ic(h.size() for h in hidden_states_out)
        enc0 = self.swinunetr.encoder1(x_in)
        enc1 = self.swinunetr.encoder2(hidden_states_out[0])
        enc2 = self.swinunetr.encoder3(hidden_states_out[1])
        enc3 = self.swinunetr.encoder4(hidden_states_out[2])
        dec4 = self.swinunetr.encoder10(hidden_states_out[4])
        ic(enc0.size(), enc1.size(), enc2.size(), enc3.size(), dec4.size())
        ic(dec4.min(), dec4.max(), torch.isnan(dec4).any())
        return dec4
