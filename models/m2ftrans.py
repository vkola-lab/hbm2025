import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.layers import (general_conv3d, normalization, prm_generator, prm_fusion,
                    prm_generator_laststage, region_aware_modal_fusion, fusion_postnorm)
from torch.nn.init import constant_, xavier_uniform_

# from visualizer import get_local

######### mask ######

def mask_gen_fusion(Batchsize, NumHead, patches, NumClass, mask):
    attn_shape = (patches*(NumClass+1), patches*(NumClass+1))
    bs = mask.size(0)
    mask_shape = (bs, NumHead, patches*(NumClass+1), patches*(NumClass+1))
    self_mask_batch = torch.zeros(mask_shape)
    for j in range(bs):
        self_mask = np.zeros(attn_shape)
        for i in range(NumClass):
            self_mask[patches*i:patches*(i+1),patches*i:patches*(i+1)] = 1
        self_mask[patches*NumClass:patches*(NumClass+1),:] = 1
        for i in range(NumClass):
            if mask[j][i] == 0:
                self_mask[patches*NumClass:patches*(NumClass+1),patches*i:patches*(i+1)] = 0
        self_mask = torch.from_numpy(self_mask)
        self_mask = torch.unsqueeze(self_mask, 0).repeat(NumHead,1,1)
        self_mask_batch[j] = self_mask

    return self_mask_batch == 1


def mask_gen_cross4(Batchsize, K, C, mask):
    ic(mask.size())
    bs, mods = mask.size()
    attn_shape = (bs, K, C)
    self_mask = np.ones(attn_shape)
    ic(self_mask.shape)
    for j in range(bs):
        for i in range(mods):
            if mask[j][i] == 0:
                self_mask[j:j+1,:,(C//mods)*i:(C//mods)*(i+1)] = 0

    self_mask = torch.from_numpy(self_mask)

    return self_mask == 1

##############

###### M2FTrans blocks #######

def nchwd2nlc2nchwd(module, x):
    B, C, H, W, D = x.shape
    x = x.flatten(2).transpose(1, 2)
    x = module(x)
    x = x.transpose(1, 2).reshape(B, C, H, W, D).contiguous()
    return x

class DepthWiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthWiseConvBlock, self).__init__()
        mid_channels = in_channels
        self.conv1 = nn.Conv3d(in_channels,
                               mid_channels,
                               1, 1)
        layer_norm = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = layer_norm(mid_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv3d(mid_channels,
                               mid_channels,
                               3, 1, 1, groups=mid_channels)
        self.norm2 = layer_norm(mid_channels)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv3d(mid_channels,
                               out_channels,
                               1, 1)
        self.norm3 = layer_norm(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = nchwd2nlc2nchwd(self.norm1, x)
        x = self.act1(x)

        x = self.conv2(x)
        x = nchwd2nlc2nchwd(self.norm2, x)
        x = self.act2(x)

        x = self.conv3(x)
        x = nchwd2nlc2nchwd(self.norm3, x)
        return x

class GroupConvBlock(nn.Module):
    def __init__(self,
                 embed_dims=8,
                 expand_ratio=4,
                 proj_drop=0.):
        super(GroupConvBlock, self).__init__()
        self.pwconv1 = nn.Conv3d(embed_dims,
                                 embed_dims * expand_ratio,
                                 1, 1)
        layer_norm = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = layer_norm(embed_dims * expand_ratio)
        self.act1 = nn.GELU()
        self.dwconv = nn.Conv3d(embed_dims * expand_ratio,
                                embed_dims * expand_ratio,
                                3, 1, 1, groups=embed_dims)
        self.norm2 = layer_norm(embed_dims * expand_ratio)
        self.act2 = nn.GELU()
        self.pwconv2 = nn.Conv3d(embed_dims * expand_ratio,
                                 embed_dims,
                                 1, 1)
        self.norm3 = layer_norm(embed_dims)
        self.final_act = nn.GELU()
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, identity=None):
        input = x
        x = self.pwconv1(x)
        x = nchwd2nlc2nchwd(self.norm1, x)
        x = self.act1(x)

        x = self.dwconv(x)
        x = nchwd2nlc2nchwd(self.norm2, x)
        x = self.act2(x)

        x = self.pwconv2(x)
        x = nchwd2nlc2nchwd(self.norm3, x)

        if identity is None:
            x = input + self.proj_drop(x)
        else:
            x = identity + self.proj_drop(x)

        x = self.final_act(x)

        return x

class AttentionLayer(nn.Module):
    def __init__(self,
                 kv_dim=8,
                 query_dim=4,
                 attn_drop=0.,
                 proj_drop=0.):
        super(AttentionLayer, self).__init__()
        self.attn_drop = nn.Dropout(attn_drop)
        self.query_map = DepthWiseConvBlock(query_dim, query_dim)
        self.key_map = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map = DepthWiseConvBlock(kv_dim, kv_dim)
        self.out_project = DepthWiseConvBlock(query_dim, query_dim)

        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, query, key, value):
        """x: B, C, H, W, D"""
        identity = query
        qb, qc, qh, qw, qd = query.shape
        query = self.query_map(query).flatten(2)
        key = self.key_map(key).flatten(2)
        value = self.value_map(value).flatten(2)

        attn = (query @ key.transpose(-2, -1)) * (query.shape[-1]) ** -0.5
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ value
        x = x.reshape(qb, qc, qh, qw, qd)
        x = self.out_project(x)
        return identity + self.proj_drop(x)


class CrossBlock(nn.Module):
    def __init__(self,
                 feature_channels=8,
                 num_classes=4,
                 expand_ratio=4,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 ffn_feature_maps=True):
        super(CrossBlock, self).__init__()
        self.ffn_feature_maps = ffn_feature_maps

        self.cross_attn = AttentionLayer(kv_dim=feature_channels,
                                         query_dim=num_classes,
                                         attn_drop=attn_drop_rate,
                                         proj_drop=drop_rate)

        if ffn_feature_maps:
            self.ffn2 = GroupConvBlock(embed_dims=feature_channels,
                                       expand_ratio=expand_ratio)
        self.ffn1 = GroupConvBlock(embed_dims=num_classes,
                                   expand_ratio=expand_ratio)

    def forward(self, kernels, feature_maps):
        kernels = self.cross_attn(query=kernels,
                                  key=feature_maps,
                                  value=feature_maps)

        kernels = self.ffn1(kernels, identity=kernels)

        if self.ffn_feature_maps:
            feature_maps = self.ffn2(feature_maps, identity=feature_maps)

        return kernels, feature_maps

class ResBlock(nn.Module):
    def __init__(self, in_channels=4, channels=4):
        super(ResBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_channels, eps=1e-6)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv3d(in_channels, channels, 3, 1, 1)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv3d(channels, channels, 3, 1, 1)
        if channels != in_channels:
            self.identity_map = nn.Conv3d(in_channels, channels, 1, 1, 0)
        else:
            self.identity_map = nn.Identity()

    def forward(self, x):
        # refer to paper
        # Identity Mapping in Deep Residual Networks
        out = nchwd2nlc2nchwd(self.norm1, x)
        out = self.act1(out)
        out = self.conv1(out)
        out = nchwd2nlc2nchwd(self.norm2, out)
        out = self.act2(out)
        out = self.conv2(out)
        out = out + self.identity_map(x)

        return out

class MultiMaskCrossBlock(nn.Module):
    def __init__(self,
                 feature_channels=8*16,
                 num_classes=8*16,
                 expand_ratio=4,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 ffn_feature_maps=True):
        super(MultiMaskCrossBlock, self).__init__()
        self.ffn_feature_maps = ffn_feature_maps

        self.cross_attn = MultiMaskAttentionLayer(kv_dim=feature_channels,
                                         query_dim=num_classes,
                                         attn_drop=attn_drop_rate,
                                         proj_drop=drop_rate)

        if ffn_feature_maps:
            self.ffn2 = GroupConvBlock(embed_dims=feature_channels,
                                       expand_ratio=expand_ratio)
        self.ffn1 = GroupConvBlock(embed_dims=num_classes,
                                   expand_ratio=expand_ratio)

    def forward(self, kernels, feature_maps, mask):
        flair, t1ce, t1, t2 = feature_maps
        kernels = self.cross_attn(query = kernels,
                                  key = feature_maps,
                                  value = feature_maps,
                                  mask = mask)

        kernels = self.ffn1(kernels, identity=kernels)

        if self.ffn_feature_maps:
            flair = self.ffn2(flair, identity=flair)
            t1ce = self.ffn2(t1ce, identity=t1ce)
            t1 = self.ffn2(t1, identity=t1)
            t2 = self.ffn2(t2, identity=t2)
            feature_maps = (flair, t1ce, t1, t2)

        return kernels, feature_maps


class MultiMaskCrossBlock2(nn.Module):
    def __init__(self,
                 modal,
                 feature_channels=8*16,
                 num_classes=8*16,
                 expand_ratio=4,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 ffn_feature_maps=True):
        super(MultiMaskCrossBlock2, self).__init__()
        self.ffn_feature_maps = ffn_feature_maps

        self.cross_attn = MultiMaskAttentionLayer2(modal,
                                         kv_dim=feature_channels,
                                         query_dim=num_classes,
                                         attn_drop=attn_drop_rate,
                                         proj_drop=drop_rate)

        if ffn_feature_maps:
            self.ffn2 = GroupConvBlock(embed_dims=feature_channels,
                                       expand_ratio=expand_ratio)
        self.ffn1 = GroupConvBlock(embed_dims=num_classes,
                                   expand_ratio=expand_ratio)

    def forward(self, kernels, feature_maps, mask):
        ic(len(feature_maps))
        C = len(feature_maps)
        kernels = self.cross_attn(query = kernels,
                                  key = feature_maps,
                                  value = feature_maps,
                                  mask = mask)

        kernels = self.ffn1(kernels, identity=kernels)

        if self.ffn_feature_maps:
            flair = self.ffn2(flair, identity=flair)
            t1ce = self.ffn2(t1ce, identity=t1ce)
            t1 = self.ffn2(t1, identity=t1)
            t2 = self.ffn2(t2, identity=t2)
            feature_maps = (flair, t1ce, t1, t2)

        return kernels, feature_maps

class MultiMaskAttentionLayer(nn.Module):
    def __init__(self,
                 kv_dim=8,
                 query_dim=4,
                 attn_drop=0.,
                 proj_drop=0.):
        super(MultiMaskAttentionLayer, self).__init__()
        self.attn_drop = nn.Dropout(attn_drop)
        self.query_map = DepthWiseConvBlock(query_dim, query_dim)
        self.key_map_flair = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map_flair = DepthWiseConvBlock(kv_dim, kv_dim)
        self.key_map_t1ce = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map_t1ce = DepthWiseConvBlock(kv_dim, kv_dim)
        self.key_map_t1 = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map_t1 = DepthWiseConvBlock(kv_dim, kv_dim)
        self.key_map_t2 = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map_t2 = DepthWiseConvBlock(kv_dim, kv_dim)
        self.out_project = DepthWiseConvBlock(query_dim, query_dim)

        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, query, key, value, mask):
        """x: B, C, H, W, D"""
        identity = query
        flair, t1ce, t1, t2 = key
        qb, qc, qh, qw, qd = query.shape
        query = self.query_map(query).flatten(2)
        key_flair = self.key_map_flair(flair).flatten(2)
        value_flair = self.value_map_flair(flair).flatten(2)
        key_t1ce = self.key_map_t1ce(t1ce).flatten(2)
        value_t1ce = self.value_map_t1ce(t1ce).flatten(2)
        key_t1 = self.key_map_t1(t1).flatten(2)
        value_t1 = self.value_map_t1(t1).flatten(2)
        key_t2 = self.key_map_t2(t2).flatten(2)
        value_t2 = self.value_map_t2(t2).flatten(2)

        key = torch.cat((key_flair, key_t1ce, key_t1, key_t2), dim=1)
        value = torch.cat((value_flair, value_t1ce, value_t1, value_t2), dim=1)

        kb, kc, kl = key.shape

        attn = (query @ key.transpose(-2, -1)) * (query.shape[-1]) ** -0.5
        self_mask = mask_gen_cross4(qb, qc, kc, mask).cuda(non_blocking=True)
        attn = attn.masked_fill(self_mask==0, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ value
        x = x.reshape(qb, qc, qh, qw, qd)
        x = self.out_project(x)
        return identity + self.proj_drop(x)

class MultiMaskAttentionLayer2(nn.Module):
    def __init__(self,
                 modal,
                 kv_dim=8,
                 query_dim=4,
                 attn_drop=0.,
                 proj_drop=0.):
        super(MultiMaskAttentionLayer2, self).__init__()
        self.modal = modal
        self.attn_drop = nn.Dropout(attn_drop)
        self.query_map = DepthWiseConvBlock(query_dim, query_dim)
        self.key_map_mod = nn.ModuleDict()
        self.value_map_mod = nn.ModuleDict()
        for mod in self.modal:
            self.key_map_mod[mod] = DepthWiseConvBlock(kv_dim, kv_dim)
            self.value_map_mod[mod] = DepthWiseConvBlock(kv_dim, kv_dim)
        
        self.out_project = DepthWiseConvBlock(query_dim, query_dim)

        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, query, key, value, mask):
        """x: B, C, H, W, D"""
        identity = query
        
        qb, qc, qh, qw, qd = query.shape
        query = self.query_map(query).flatten(2)
        
        value = tuple(
            self.value_map_mod[mod](k).flatten(2)
            for mod, k in zip(self.modal, key)
        )
        key = tuple(
            self.key_map_mod[mod](k).flatten(2)
            for mod, k in zip(self.modal, key)
        )

        key = torch.cat(key, dim=1)
        value = torch.cat(value, dim=1)

        kb, kc, kl = key.shape

        attn = (query @ key.transpose(-2, -1)) * (query.shape[-1]) ** -0.5
        self_mask = mask_gen_cross4(qb, qc, kc, mask).cuda(non_blocking=True)
        attn = attn.masked_fill(self_mask==0, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ value
        x = x.reshape(qb, qc, qh, qw, qd)
        x = self.out_project(x)
        return identity + self.proj_drop(x)


###############

class MultiCrossToken(nn.Module):
    def __init__(
            self,
            image_h=128,
            image_w=128,
            image_d=128,
            h_stride=16,
            w_stride=16,
            d_stride=16,
            num_layers=2,
            mlp_ratio=4,
            drop_rate=0.1,
            attn_drop_rate=0.0,
            interpolate_mode='trilinear',
            channel=8*16):
        super(MultiCrossToken, self).__init__()

        self.channels = channel
        self.H = image_h // h_stride
        self.W = image_w // w_stride
        self.D = image_d // d_stride
        self.interpolate_mode = interpolate_mode
        self.layers = nn.ModuleList([
            MultiMaskCrossBlock(feature_channels=self.channels,
                                    num_classes=self.channels,
                                    expand_ratio=mlp_ratio,
                                    drop_rate=drop_rate,
                                    attn_drop_rate=attn_drop_rate,
                                    ffn_feature_maps=i != num_layers - 1,
                                    ) for i in range(num_layers)])

    def forward(self, inputs, kernels, mask):
        feature_maps = inputs
        for layer in self.layers:
            kernels, feature_maps = layer(kernels, feature_maps, mask)

        return kernels

class MultiCrossToken2(nn.Module):
    def __init__(
            self,
            modal,
            image_h=128,
            image_w=128,
            image_d=128,
            h_stride=16,
            w_stride=16,
            d_stride=16,
            num_layers=2,
            mlp_ratio=4,
            drop_rate=0.1,
            attn_drop_rate=0.0,
            interpolate_mode='trilinear',
            channel=8*16):
        super(MultiCrossToken2, self).__init__()

        self.channels = channel
        self.H = image_h // h_stride
        self.W = image_w // w_stride
        self.D = image_d // d_stride
        self.interpolate_mode = interpolate_mode
        self.layers = nn.ModuleList([
            MultiMaskCrossBlock2(modal=modal,
                                    feature_channels=self.channels,
                                      num_classes=self.channels,
                                      expand_ratio=mlp_ratio,
                                      drop_rate=drop_rate,
                                      attn_drop_rate=attn_drop_rate,
                                      ffn_feature_maps=i != num_layers - 1,
                                      ) for i in range(num_layers)])

    def forward(self, inputs, kernels, mask):
        feature_maps = inputs
        for layer in self.layers:
            kernels, feature_maps = layer(kernels, feature_maps, mask)

        return kernels

class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg
        self.e1_c1 = general_conv3d(1, cfg.conv_dims, pad_type='reflect')
        self.e1_c2 = general_conv3d(cfg.conv_dims, cfg.conv_dims, pad_type='reflect')
        self.e1_c3 = general_conv3d(cfg.conv_dims, cfg.conv_dims, pad_type='reflect')

        self.e2_c1 = general_conv3d(cfg.conv_dims, cfg.conv_dims*2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d(cfg.conv_dims*2, cfg.conv_dims*2, pad_type='reflect')
        self.e2_c3 = general_conv3d(cfg.conv_dims*2, cfg.conv_dims*2, pad_type='reflect')

        self.e3_c1 = general_conv3d(cfg.conv_dims*2, cfg.conv_dims*4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d(cfg.conv_dims*4, cfg.conv_dims*4, pad_type='reflect')
        self.e3_c3 = general_conv3d(cfg.conv_dims*4, cfg.conv_dims*4, pad_type='reflect')

        self.e4_c1 = general_conv3d(cfg.conv_dims*4, cfg.conv_dims*8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d(cfg.conv_dims*8, cfg.conv_dims*8, pad_type='reflect')
        self.e4_c3 = general_conv3d(cfg.conv_dims*8, cfg.conv_dims*8, pad_type='reflect')

        self.e5_c1 = general_conv3d(cfg.conv_dims*8, cfg.conv_dims*16, stride=2, pad_type='reflect')
        self.e5_c2 = general_conv3d(cfg.conv_dims*16, cfg.conv_dims*16, pad_type='reflect')
        self.e5_c3 = general_conv3d(cfg.conv_dims*16, cfg.conv_dims*16, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))
        ic(x1.size(), x2.size(), x3.size(), x4.size(), x5.size())
        return x1, x2, x3, x4, x5

class Decoder_sep(nn.Module):
    def __init__(self, cfg):
        super(Decoder_sep, self).__init__()
        self.cfg = cfg
        num_cls = cfg.num_cls
        self.d4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d4_c1 = general_conv3d(cfg.conv_dims*16, cfg.conv_dims*8, pad_type='reflect')
        self.d4_c2 = general_conv3d(cfg.conv_dims*16, cfg.conv_dims*8, pad_type='reflect')
        self.d4_out = general_conv3d(cfg.conv_dims*8, cfg.conv_dims*8, k_size=1, padding=0, pad_type='reflect')

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d(cfg.conv_dims*8, cfg.conv_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d(cfg.conv_dims*8, cfg.conv_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d(cfg.conv_dims*4, cfg.conv_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d(cfg.conv_dims*4, cfg.conv_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d(cfg.conv_dims*4, cfg.conv_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d(cfg.conv_dims*2, cfg.conv_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d(cfg.conv_dims*2, cfg.conv_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(cfg.conv_dims*2, cfg.conv_dims, pad_type='reflect')
        self.d1_out = general_conv3d(cfg.conv_dims, cfg.conv_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=cfg.conv_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):

        de_x5 = self.d4_c1(self.d4(x5))
        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))

        de_x4 = self.d3_c1(self.d3(de_x4))
        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))

        de_x3 = self.d2_c1(self.d2(de_x3))
        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))

        de_x2 = self.d1_c1(self.d1(de_x2))
        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred


class Decoder_fusion(nn.Module):
    def __init__(self, cfg):
        super(Decoder_fusion, self).__init__()
        self.cfg = cfg
        num_cls = cfg.num_cls
        self.d5_c2 = general_conv3d(cfg.conv_dims*32, cfg.conv_dims*16, pad_type='reflect')
        self.d5_out = general_conv3d(cfg.conv_dims*16, cfg.conv_dims*16, k_size=1, padding=0, pad_type='reflect')

        if 'v2' in cfg.arch.lower():
            self.CT5 = MultiCrossToken2(cfg.modal, h_stride=16, w_stride=16, d_stride=16, channel=cfg.conv_dims*16)
            self.CT4 = MultiCrossToken2(cfg.modal, h_stride=8, w_stride=8, d_stride=8, channel=cfg.conv_dims*8)
        else:
            self.CT5 = MultiCrossToken(h_stride=16, w_stride=16, d_stride=16, channel=cfg.conv_dims*16)
            self.CT4 = MultiCrossToken(h_stride=8, w_stride=8, d_stride=8, channel=cfg.conv_dims*8)
        # self.CT3 = MultiCrossToken(h_stride=4, w_stride=4, d_stride=4, channel=cfg.conv_dims*4)
        # self.CT2 = MultiCrossToken(h_stride=2, w_stride=2, d_stride=2, channel=cfg.conv_dims*2)
        # self.CT1 = MultiCrossToken(h_stride=1, w_stride=1, d_stride=1, channel=cfg.conv_dims*1)

        self.d4_c1 = general_conv3d(cfg.conv_dims*16, cfg.conv_dims*8, pad_type='reflect')
        self.d4_c2 = general_conv3d(cfg.conv_dims*16, cfg.conv_dims*8, pad_type='reflect')
        self.d4_out = general_conv3d(cfg.conv_dims*8, cfg.conv_dims*8, k_size=1, padding=0, pad_type='reflect')

        self.d3_c1 = general_conv3d(cfg.conv_dims*8, cfg.conv_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d(cfg.conv_dims*8, cfg.conv_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d(cfg.conv_dims*4, cfg.conv_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d(cfg.conv_dims*4, cfg.conv_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d(cfg.conv_dims*4, cfg.conv_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d(cfg.conv_dims*2, cfg.conv_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d(cfg.conv_dims*2, cfg.conv_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(cfg.conv_dims*2, cfg.conv_dims, pad_type='reflect')
        self.d1_out = general_conv3d(cfg.conv_dims, cfg.conv_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=cfg.conv_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)

        # self.RFM5 = fusion_postnorm(in_channel=cfg.conv_dims*16, num_cls=num_cls)
        # self.RFM4 = fusion_postnorm(in_channel=cfg.conv_dims*8, num_cls=num_cls)
        self.RFM3 = fusion_postnorm(in_channel=cfg.conv_dims*4, num_cls=len(cfg.modal))
        self.RFM2 = fusion_postnorm(in_channel=cfg.conv_dims*2, num_cls=len(cfg.modal))
        self.RFM1 = fusion_postnorm(in_channel=cfg.conv_dims*1, num_cls=len(cfg.modal))

        self.prm_fusion5 = prm_fusion(in_channel=cfg.conv_dims*16, basic_dim=cfg.conv_dims, num_cls=num_cls)
        self.prm_fusion4 = prm_fusion(in_channel=cfg.conv_dims*8, basic_dim=cfg.conv_dims, num_cls=num_cls)
        self.prm_fusion3 = prm_fusion(in_channel=cfg.conv_dims*4, basic_dim=cfg.conv_dims, num_cls=num_cls)
        self.prm_fusion2 = prm_fusion(in_channel=cfg.conv_dims*2, basic_dim=cfg.conv_dims, num_cls=num_cls)
        self.prm_fusion1 = prm_fusion(in_channel=cfg.conv_dims*1, basic_dim=cfg.conv_dims, num_cls=num_cls)


    def forward(self, dx1, dx2, dx3, dx4, dx5, fusion, mask):
        mask = mask.bool()
        prm_pred5 = self.prm_fusion5(fusion)
        de_x5 = self.CT5(dx5, fusion, mask)
        de_x5 = torch.cat((de_x5, fusion), dim=1)
        de_x5 = self.d5_out(self.d5_c2(de_x5))
        de_x5 = self.d4_c1(self.up2(de_x5))

        prm_pred4 = self.prm_fusion4(de_x5)
        de_x4 = self.CT4(dx4, de_x5, mask)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        prm_pred3 = self.prm_fusion3(de_x4)
        de_x3 = self.RFM3(dx3, mask)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        prm_pred2 = self.prm_fusion2(de_x3)
        de_x2 = self.RFM2(dx2, mask)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        prm_pred1 = self.prm_fusion1(de_x2)
        de_x1 = self.RFM1(dx1, mask)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (prm_pred1, self.up2(prm_pred2), self.up4(prm_pred3), self.up8(prm_pred4), self.up16(prm_pred5))

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class MaskedResidual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, mask):
        y, attn = self.fn(x, mask)
        return y + x, attn


class MaskedPreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x, mask):
        x = self.norm(x)
        x, attn = self.fn(x, mask)
        return self.dropout(x), attn


class MaskedAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0, num_class=4
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_class = num_class

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    # @get_local('attn')
    def forward(self, x, mask):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        self_mask = mask_gen_fusion(B, self.num_heads, N // (self.num_class+1), self.num_class, mask).to(attn.device)
        attn = attn.masked_fill(self_mask==0, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class MaskedTransformer(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1, n_levels=1, n_points=4, num_class=4):
        super(MaskedTransformer, self).__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                MaskedResidual(
                    MaskedPreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        MaskedAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate, num_class=num_class),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )

        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)


    def forward(self, x, mask):
        attn_list=[]
        for j in range(self.depth):
            x, attn = self.cross_attention_list[j](x, mask)
            attn_list.append(attn.detach())
            x = self.cross_ffn_list[j](x)
        return x, attn_list


class Bottleneck(nn.Module):
    def __init__(self, cfg):
        super(Bottleneck, self).__init__()
        self.cfg = cfg
        self.trans_bottle = MaskedTransformer(embedding_dim=cfg.conv_dims*16, depth=cfg.depth, heads=cfg.num_heads, mlp_dim=cfg.mlp_dim)
        self.num_cls = len(cfg.modal)

    def forward(self, x, mask, fusion, pos):
        flair, t1ce, t1, t2 = x
        embed_flair = flair.flatten(2).transpose(1, 2).contiguous()
        embed_t1ce = t1ce.flatten(2).transpose(1, 2).contiguous()
        embed_t1 = t1.flatten(2).transpose(1, 2).contiguous()
        embed_t2 = t2.flatten(2).transpose(1, 2).contiguous()
        ic(embed_flair.size(), embed_t1ce.size(), embed_t1.size(), embed_t1.size(), embed_t2.size(), fusion.size())
        embed_cat = torch.cat((embed_flair, embed_t1ce, embed_t1, embed_t2, fusion), dim=1)
        embed_cat = embed_cat + pos
        embed_cat_trans, attn = self.trans_bottle(embed_cat, mask)
        flair_trans, t1ce_trans, t1_trans, t2_trans, fusion_trans = torch.chunk(embed_cat_trans, self.num_cls+1, dim=1)

        return flair_trans, t1ce_trans, t1_trans, t2_trans, fusion_trans, attn


class Bottleneck2(nn.Module):
    def __init__(self, cfg):
        super(Bottleneck2, self).__init__()
        self.cfg = cfg
        self.num_cls = len(cfg.modal)
        self.trans_bottle = MaskedTransformer(embedding_dim=cfg.conv_dims*16, depth=cfg.depth, heads=cfg.num_heads, mlp_dim=cfg.mlp_dim, num_class=self.num_cls)

    def forward(self, x, mask, fusion, pos):
        embed_mod = [x_mod.flatten(2).transpose(1, 2).contiguous() for x_mod in x]
        embed_cat = embed_mod + [fusion]
        ic([emb.size() for emb in embed_cat])
        embed_cat = torch.cat(embed_cat, dim=1)
        embed_cat = embed_cat + pos
        ic(embed_cat.size())
        embed_cat_trans, attn = self.trans_bottle(embed_cat, mask)
        embed_trans = torch.chunk(embed_cat_trans, self.num_cls+1, dim=1)
        fusion_trans = embed_trans[-1]
        return embed_trans[:-1], fusion_trans, attn

class Weight_Attention(nn.Module):
    def __init__(self, cfg):
        super(Weight_Attention, self).__init__()
        self.cfg = cfg
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.num_modals = len(cfg.modal)
        self.img_size = cfg.img_size
        # self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, de_x1, de_x2, de_x3, de_x4, de_x5, attn):

        flair_tra, t1ce_tra, t1_tra, t2_tra = de_x5
        flair_x4, t1ce_x4, t1_x4, t2_x4 = de_x4
        flair_x3, t1ce_x3, t1_x3, t2_x3 = de_x3
        flair_x2, t1ce_x2, t1_x2, t2_x2 = de_x2
        flair_x1, t1ce_x1, t1_x1, t2_x1 = de_x1


        attn_0 = attn[0]
        attn_fusion = attn_0[:, :, ((self.img_size//16) * (self.img_size//16) * (self.img_size//16))*4 :, :]
        attn_flair, attn_t1ce, attn_t1, attn_t2, attn_self = torch.chunk(attn_fusion, self.num_modals+1, dim=-1)

        attn_flair = torch.sum(torch.sum(attn_flair, dim=1), dim=-2).reshape(flair_tra.size(0), (self.img_size//16), (self.img_size//16), (self.img_size//16)).unsqueeze(dim=1)
        attn_t1ce = torch.sum(torch.sum(attn_t1ce, dim=1), dim=-2).reshape(flair_tra.size(0), (self.img_size//16), (self.img_size//16), (self.img_size//16)).unsqueeze(dim=1)
        attn_t1 = torch.sum(torch.sum(attn_t1, dim=1), dim=-2).reshape(flair_tra.size(0), (self.img_size//16), (self.img_size//16), (self.img_size//16)).unsqueeze(dim=1)
        attn_t2 = torch.sum(torch.sum(attn_t2, dim=1), dim=-2).reshape(flair_tra.size(0), (self.img_size//16), (self.img_size//16), (self.img_size//16)).unsqueeze(dim=1)


        dex5 = (flair_tra*(attn_flair), t1ce_tra*(attn_t1ce), t1_tra*(attn_t1), t2_tra*(attn_t2))

        attn_flair, attn_t1ce, attn_t1, attn_t2 = self.upsample(attn_flair), self.upsample(attn_t1ce), self.upsample(attn_t1), self.upsample(attn_t2)
        dex4 = (flair_x4*(attn_flair), t1ce_x4*(attn_t1ce), t1_x4*(attn_t1), t2_x4*(attn_t2))

        attn_flair, attn_t1ce, attn_t1, attn_t2 = self.upsample(attn_flair), self.upsample(attn_t1ce), self.upsample(attn_t1), self.upsample(attn_t2)
        dex3 = (flair_x3*(attn_flair), t1ce_x3*(attn_t1ce), t1_x3*(attn_t1), t2_x3*(attn_t2))

        attn_flair, attn_t1ce, attn_t1, attn_t2 = self.upsample(attn_flair), self.upsample(attn_t1ce), self.upsample(attn_t1), self.upsample(attn_t2)
        dex2 = (flair_x2*(attn_flair), t1ce_x2*(attn_t1ce), t1_x2*(attn_t1), t2_x2*(attn_t2))

        attn_flair, attn_t1ce, attn_t1, attn_t2 = self.upsample(attn_flair), self.upsample(attn_t1ce), self.upsample(attn_t1), self.upsample(attn_t2)
        dex1 = (flair_x1*(attn_flair), t1ce_x1*(attn_t1ce), t1_x1*(attn_t1), t2_x1*(attn_t2))

        return dex1, dex2, dex3, dex4, dex5
    
class Weight_Attention2(nn.Module):
    def __init__(self, cfg):
        super(Weight_Attention2, self).__init__()
        self.cfg = cfg
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.num_modals = len(cfg.modal)
        self.img_size = cfg.img_size
        # self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, de_x1, de_x2, de_x3, de_x4, de_x5, attn):

        de_x5 = list(de_x5)
        de_x4 = list(de_x4)
        de_x3 = list(de_x3)
        de_x2 = list(de_x2)
        de_x1 = list(de_x1)

        attn_0 = attn[0]
        attn_fusion = attn_0[:, :, ((self.img_size//16) * (self.img_size//16) * (self.img_size//16)*self.num_modals):, :]
        ic(attn_fusion.size())
        attn_chunks = torch.chunk(attn_fusion, self.num_modals+1, dim=-1)
        mod_attn, attn_self = list(attn_chunks[:self.num_modals]), attn_chunks[-1]
        
        dex1, dex2, dex3, dex4, dex5 = [], [], [], [], []
        for idx in range(self.num_modals):
            mod_attn[idx] = torch.sum(torch.sum(mod_attn[idx], dim=1), dim=-2).reshape(de_x5[idx].size(0), (self.img_size//16), (self.img_size//16), (self.img_size//16)).unsqueeze(dim=1)
            ic(mod_attn[idx].size(), de_x5[idx].size())
            dex5.append(de_x5[idx]*mod_attn[idx])

            mod_attn[idx] = self.upsample(mod_attn[idx])
            dex4.append(de_x4[idx]*mod_attn[idx])

            mod_attn[idx] = self.upsample(mod_attn[idx])
            dex3.append(de_x3[idx]*mod_attn[idx])

            mod_attn[idx] = self.upsample(mod_attn[idx])
            dex2.append(de_x2[idx]*mod_attn[idx])

            mod_attn[idx] = self.upsample(mod_attn[idx])
            dex1.append(de_x1[idx]*mod_attn[idx])

        return tuple(dex1), tuple(dex2), tuple(dex3), tuple(dex4), tuple(dex5)

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.img_size = cfg.img_size
        self.num_cls = cfg.num_cls
        self.num_modals = len(cfg.modal)
        self.modal = cfg.modal
        self.flair_encoder = Encoder(cfg)
        self.t1ce_encoder = Encoder(cfg)
        self.t1_encoder = Encoder(cfg)
        self.t2_encoder = Encoder(cfg)
        self.Bottleneck = Bottleneck(cfg)
        self.decoder_fusion = Decoder_fusion(cfg)
        self.decoder_sep = Decoder_sep(cfg)
        self.weight_attention = Weight_Attention(cfg)

        self.pos = nn.Parameter(torch.zeros(1, ((self.img_size//16) * (self.img_size//16) * (self.img_size//16))*5, cfg.conv_dims*16))
        self.fusion = nn.Parameter(nn.init.normal_(torch.zeros(1, ((self.img_size//16) * (self.img_size//16) * (self.img_size//16)), cfg.conv_dims*16), mean=0.0, std=1.0))

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #

    def forward(self, x, mask, features_only=False):
        #extract feature from different layers
        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4, :, :, :])

        x_bottle = (flair_x5, t1ce_x5, t1_x5, t2_x5)

        B = x.size(0)
        fusion = torch.tile(self.fusion, [B, 1, 1])

        flair_trans, t1ce_trans, t1_trans, t2_trans, fusion_trans, attn = self.Bottleneck(x_bottle, mask, fusion, self.pos)

        flair_tra = flair_trans.view(x.size(0), (self.img_size//16), (self.img_size//16), (self.img_size//16), self.cfg.conv_dims*16).permute(0, 4, 1, 2, 3).contiguous()
        t1ce_tra = t1ce_trans.view(x.size(0), (self.img_size//16), (self.img_size//16), (self.img_size//16), self.cfg.conv_dims*16).permute(0, 4, 1, 2, 3).contiguous()
        t1_tra = t1_trans.view(x.size(0), (self.img_size//16), (self.img_size//16), (self.img_size//16), self.cfg.conv_dims*16).permute(0, 4, 1, 2, 3).contiguous()
        t2_tra = t2_trans.view(x.size(0), (self.img_size//16), (self.img_size//16), (self.img_size//16), self.cfg.conv_dims*16).permute(0, 4, 1, 2, 3).contiguous()
        fusion_tra = fusion_trans.view(x.size(0), (self.img_size//16), (self.img_size//16), (self.img_size//16), self.cfg.conv_dims*16).permute(0, 4, 1, 2, 3).contiguous()


        de_x5 = (flair_tra, t1ce_tra, t1_tra, t2_tra)
        de_x4 = (flair_x4, t1ce_x4, t1_x4, t2_x4)
        de_x3 = (flair_x3, t1ce_x3, t1_x3, t2_x3)
        de_x2 = (flair_x2, t1ce_x2, t1_x2, t2_x2)
        de_x1 = (flair_x1, t1ce_x1, t1_x1, t2_x1)

        de_x1, de_x2, de_x3, de_x4, de_x5 = self.weight_attention(de_x1, de_x2, de_x3, de_x4, de_x5, attn)

        # de_x5 = torch.stack(de_x5, dim=1)
        # de_x4 = torch.stack(de_x4, dim=1)
        de_x3 = torch.stack(de_x3, dim=1)
        de_x2 = torch.stack(de_x2, dim=1)
        de_x1 = torch.stack(de_x1, dim=1)

        
        if features_only:
            return de_x1, de_x2, de_x3, de_x4, de_x5, fusion_tra

        fuse_pred, prm_preds = self.decoder_fusion(de_x1, de_x2, de_x3, de_x4, de_x5, fusion_tra, mask)

        if self.is_training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)
            return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), prm_preds
        return fuse_pred



class Model2(nn.Module):
    def __init__(self, cfg):
        super(Model2, self).__init__()
        self.cfg = cfg
        self.img_size = cfg.img_size
        self.num_cls = cfg.num_cls
        self.num_modals = len(cfg.modal)
        self.modal = cfg.modal
        self.modal_encoders = nn.ModuleDict({mod: Encoder(cfg) for mod in self.modal})
        
        self.Bottleneck = Bottleneck2(cfg)
        self.decoder_fusion = Decoder_fusion(cfg) if not cfg.classifier else None
        self.decoder_sep = Decoder_sep(cfg) if not cfg.classifier else None
        self.weight_attention = Weight_Attention2(cfg)

        self.pos = nn.Parameter(torch.zeros(1, ((self.img_size//16) * (self.img_size//16) * (self.img_size//16))*(self.num_modals+1), cfg.conv_dims*16))
        self.fusion = nn.Parameter(nn.init.normal_(torch.zeros(1, ((self.img_size//16) * (self.img_size//16) * (self.img_size//16)), cfg.conv_dims*16), mean=0.0, std=1.0))

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #

        dummy_input = torch.randn(1, self.num_modals, 1, self.img_size, self.img_size, self.img_size)
        mask = torch.randn(1, self.num_modals)
        feats = self.forward(dummy_input, mask, features_only=True)
        fusion_tra = feats[-1]
        self.in_features = list(torch.squeeze(fusion_tra).size())

    def forward(self, x, mask, features_only=False):
        #extract feature from different layers
        if len(x.size()) <= 5:
            x = torch.unsqueeze(x, dim=2)
        ic(x.size(), mask.size())
        mod_x1, mod_x2, mod_x3, mod_x4, mod_x5 = {}, {}, {}, {}, {}

        for idx, mod in enumerate(self.modal): 
            mod_x1[mod], mod_x2[mod], mod_x3[mod], mod_x4[mod], mod_x5[mod] = self.modal_encoders[mod](x[:, idx, :, :, :, :])
        

        x_bottle = tuple(mod_x5.values())

        B = x.size(0)
        ic(self.fusion.size())
        fusion = torch.tile(self.fusion, [B, 1, 1])

        mod_trans, fusion_trans, attn = self.Bottleneck(x_bottle, mask, fusion, self.pos)

        mod_trans = [m_trans.view(x.size(0), (self.img_size//16), (self.img_size//16), (self.img_size//16), self.cfg.conv_dims*16).permute(0, 4, 1, 2, 3).contiguous() for m_trans in mod_trans]

        fusion_tra = fusion_trans.view(x.size(0), (self.img_size//16), (self.img_size//16), (self.img_size//16), self.cfg.conv_dims*16).permute(0, 4, 1, 2, 3).contiguous()

        de_x5 = tuple(mod_trans)
        de_x4 = tuple(mod_x4.values())
        de_x3 = tuple(mod_x3.values())
        de_x2 = tuple(mod_x2.values())
        de_x1 = tuple(mod_x1.values())

        de_x1, de_x2, de_x3, de_x4, de_x5 = self.weight_attention(de_x1, de_x2, de_x3, de_x4, de_x5, attn)

        # de_x5 = torch.stack(de_x5, dim=1)
        # de_x4 = torch.stack(de_x4, dim=1)
        de_x3 = torch.stack(de_x3, dim=1)
        de_x2 = torch.stack(de_x2, dim=1)
        de_x1 = torch.stack(de_x1, dim=1)

        if features_only:
            ic(de_x1.size(), de_x2.size(), de_x3.size())
            ic([f.size() for f in de_x4])
            ic([f.size() for f in de_x5])
            ic(fusion_tra.size())
            return tuple(mod_x5[mod] for mod in self.modal), de_x5, fusion_tra

        fuse_pred, prm_preds = self.decoder_fusion(de_x1, de_x2, de_x3, de_x4, de_x5, fusion_tra, mask)

        mod_preds = {}
        if self.is_training:
            for mod in self.modal:
                mod_preds[mod] = self.decoder_sep(mod_x1[mod], mod_x2[mod], mod_x3[mod], mod_x4[mod], mod_x5[mod])
            
            
            return fuse_pred, (mod_preds[mod] for mod in self.modal), prm_preds
        return fuse_pred
