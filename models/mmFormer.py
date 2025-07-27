import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import torch
import math
from models.layers import general_conv3d_prenorm, fusion_prenorm
from utils.ops import trunc_normal_

class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()

        self.e1_c1 = nn.Conv3d(in_channels=1, out_channels=cfg.conv_dims, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=True)
        self.e1_c2 = general_conv3d_prenorm(cfg.conv_dims, cfg.conv_dims, pad_type='reflect')
        self.e1_c3 = general_conv3d_prenorm(cfg.conv_dims, cfg.conv_dims, pad_type='reflect')

        self.e2_c1 = general_conv3d_prenorm(cfg.conv_dims, cfg.conv_dims*2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d_prenorm(cfg.conv_dims*2, cfg.conv_dims*2, pad_type='reflect')
        self.e2_c3 = general_conv3d_prenorm(cfg.conv_dims*2, cfg.conv_dims*2, pad_type='reflect')

        self.e3_c1 = general_conv3d_prenorm(cfg.conv_dims*2, cfg.conv_dims*4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d_prenorm(cfg.conv_dims*4, cfg.conv_dims*4, pad_type='reflect')
        self.e3_c3 = general_conv3d_prenorm(cfg.conv_dims*4, cfg.conv_dims*4, pad_type='reflect')

        self.e4_c1 = general_conv3d_prenorm(cfg.conv_dims*4, cfg.conv_dims*8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d_prenorm(cfg.conv_dims*8, cfg.conv_dims*8, pad_type='reflect')
        self.e4_c3 = general_conv3d_prenorm(cfg.conv_dims*8, cfg.conv_dims*8, pad_type='reflect')

        self.e5_c1 = general_conv3d_prenorm(cfg.conv_dims*8, cfg.conv_dims*16, stride=2, pad_type='reflect')
        self.e5_c2 = general_conv3d_prenorm(cfg.conv_dims*16, cfg.conv_dims*16, pad_type='reflect')
        self.e5_c3 = general_conv3d_prenorm(cfg.conv_dims*16, cfg.conv_dims*16, pad_type='reflect')

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

        return x1, x2, x3, x4, x5

class Decoder_sep(nn.Module):
    def __init__(self, cfg):
        super(Decoder_sep, self).__init__()
        num_cls = cfg.num_cls
        self.d4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d4_c1 = general_conv3d_prenorm(cfg.conv_dims*16, cfg.conv_dims*8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm(cfg.conv_dims*16, cfg.conv_dims*8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(cfg.conv_dims*8, cfg.conv_dims*8, k_size=1, padding=0, pad_type='reflect')

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d_prenorm(cfg.conv_dims*8, cfg.conv_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm(cfg.conv_dims*8, cfg.conv_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(cfg.conv_dims*4, cfg.conv_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d_prenorm(cfg.conv_dims*4, cfg.conv_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(cfg.conv_dims*4, cfg.conv_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(cfg.conv_dims*2, cfg.conv_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d_prenorm(cfg.conv_dims*2, cfg.conv_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(cfg.conv_dims*2, cfg.conv_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(cfg.conv_dims, cfg.conv_dims, k_size=1, padding=0, pad_type='reflect')

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

class Decoder_fuse(nn.Module):
    def __init__(self, cfg):
        super(Decoder_fuse, self).__init__()
        num_cls = cfg.num_cls
        self.d4_c1 = general_conv3d_prenorm(cfg.conv_dims*16, cfg.conv_dims*8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm(cfg.conv_dims*16, cfg.conv_dims*8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(cfg.conv_dims*8, cfg.conv_dims*8, k_size=1, padding=0, pad_type='reflect')

        self.d3_c1 = general_conv3d_prenorm(cfg.conv_dims*8, cfg.conv_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm(cfg.conv_dims*8, cfg.conv_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(cfg.conv_dims*4, cfg.conv_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d_prenorm(cfg.conv_dims*4, cfg.conv_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(cfg.conv_dims*4, cfg.conv_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(cfg.conv_dims*2, cfg.conv_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d_prenorm(cfg.conv_dims*2, cfg.conv_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(cfg.conv_dims*2, cfg.conv_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(cfg.conv_dims, cfg.conv_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_d4 = nn.Conv3d(in_channels=cfg.conv_dims*16, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d3 = nn.Conv3d(in_channels=cfg.conv_dims*8, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d2 = nn.Conv3d(in_channels=cfg.conv_dims*4, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv3d(in_channels=cfg.conv_dims*2, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_layer = nn.Conv3d(in_channels=cfg.conv_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)

        num_cls = len(cfg.modal)
        self.RFM5 = fusion_prenorm(in_channel=cfg.conv_dims*16, num_cls=num_cls)
        self.RFM4 = fusion_prenorm(in_channel=cfg.conv_dims*8, num_cls=num_cls)
        self.RFM3 = fusion_prenorm(in_channel=cfg.conv_dims*4, num_cls=num_cls)
        self.RFM2 = fusion_prenorm(in_channel=cfg.conv_dims*2, num_cls=num_cls)
        self.RFM1 = fusion_prenorm(in_channel=cfg.conv_dims*1, num_cls=num_cls)


    def forward(self, x1, x2, x3, x4, x5):
        ic(x5.size())
        de_x5 = self.RFM5(x5)
        pred4 = self.softmax(self.seg_d4(de_x5))
        de_x5 = self.d4_c1(self.up2(de_x5))
        ic(pred4.size())

        de_x4 = self.RFM4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        pred3 = self.softmax(self.seg_d3(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))
        ic(pred3.size())

        de_x3 = self.RFM3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        pred2 = self.softmax(self.seg_d2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))
        ic(pred2.size())

        de_x2 = self.RFM2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        pred1 = self.softmax(self.seg_d1(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))
        ic(pred1.size())

        de_x1 = self.RFM1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)
        ic(pred.size())

        return pred, (self.up2(pred1), self.up4(pred2), self.up8(pred3), self.up16(pred4))


class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # ic(x.size())
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
        # ic(q.size(), k.size(), v.size())
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # ic(attn.size())

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        # ic(x.size())
        x = self.proj_drop(x)
        return x


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
        # ic(x.size())
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1, n_levels=1, n_points=4):
        super(Transformer, self).__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
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


    def forward(self, x, pos):
        ic(x.size(), pos.size())
        for j in range(self.depth):
            # ic(j)
            x = x + pos
            x = self.cross_attention_list[j](x)
            ic(x.size())
            x = self.cross_ffn_list[j](x)
            ic(x.size())
        return x


class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()
    
    def forward(self, x, mask):
        # ic(x.size(), mask.size())
        mask = mask.view(*mask.size(), 1, 1, 1, 1).bool()
        # ic(mask.size())
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        # ic(mask.device, x.device, y.device)
        y[mask.expand_as(x)] = x[mask.expand_as(x)]
        x = y.view(B, -1, H, W, Z)
        return x


class Modelv2(nn.Module):
    def __init__(self, cfg):
        super(Modelv2, self).__init__()
        self.cfg = cfg
        self.modal = cfg.modal
        self.patch_size = cfg.patch_size
        self.img_size = cfg.img_size
        self.complete_modalities = ["T1", "T2", "FLAIR"] if 'complete' in cfg.arch else None
        init_modal = self.modal if self.complete_modalities is None else self.complete_modalities
        self.num_modals = len(init_modal)
        self.modal_encoders = nn.ModuleDict({mod: Encoder(cfg) for mod in init_modal})

        ########### IntraFormer
        self.modal_encode_conv = nn.ModuleDict({mod: nn.Conv3d(cfg.conv_dims*16, cfg.transformer_dims, kernel_size=1, stride=1, padding=0) for mod in init_modal})
        
        self.modal_decode_conv = nn.ModuleDict({mod: nn.Conv3d(cfg.transformer_dims, cfg.conv_dims*16, kernel_size=1, stride=1, padding=0) for mod in init_modal})

        self.modal_pos = nn.ParameterDict({mod: nn.Parameter(torch.zeros(1, self.patch_size**3, cfg.transformer_dims)) for mod in init_modal})

        self.modal_transformers = nn.ModuleDict({mod: Transformer(embedding_dim=cfg.transformer_dims, depth=cfg.depth, heads=cfg.num_heads, mlp_dim=cfg.mlp_dim, dropout_rate=cfg.dropout_rate) for mod in init_modal})
        ########### IntraFormer

        ########### InterFormer
        self.multimodal_transformer = Transformer(embedding_dim=cfg.transformer_dims, depth=cfg.depth, heads=cfg.num_heads, mlp_dim=cfg.mlp_dim, n_levels=self.num_modals, dropout_rate=cfg.dropout_rate)
        self.multimodal_decode_conv = nn.Conv3d(cfg.transformer_dims*self.num_modals, cfg.conv_dims*16*self.num_modals, kernel_size=1, padding=0)
        ########### InterFormer

        self.masker = MaskModal()

        self.decoder_fuse = Decoder_fuse(cfg) if not cfg.classifier else None
        self.decoder_sep = Decoder_sep(cfg) if not cfg.classifier else None

        self.is_training = False
        for mod in self.modal:
            trunc_normal_(self.modal_pos[mod])
        self.apply(self._init_weights)

        dummy_input = torch.randn(1, self.num_modals, 1, self.img_size, self.img_size, self.img_size)
        mask = torch.randn(1, self.num_modals)
        feats = self.forward(dummy_input, mask, features_only=True)
        x5_inter = feats[-2]
        self.in_features = list(torch.squeeze(x5_inter).size())
        # ic(self.in_features)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        # trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask, features_only=False, is_training=False):
        #extract feature from different layers
        if len(x.size()) <= 5:
            x = torch.unsqueeze(x, dim=2)
        # ic(x.size())
        mod_x1, mod_x2, mod_x3, mod_x4, mod_x5 = {}, {}, {}, {}, {}
        mod_token_x5 = {}
        mod_intra_token_x5, mod_intra_x5 = {}, {}
        mod_pred = {}
        for idx, mod in enumerate(self.modal):
            ic(x[:,idx,:,:,:,:].size())
            mod_x1[mod], mod_x2[mod], mod_x3[mod], mod_x4[mod], mod_x5[mod] = self.modal_encoders[mod](x[:, idx, :, :, :, :])
            ic(mod_x1[mod].size(), mod_x2[mod].size(), mod_x3[mod].size(), mod_x4[mod].size(), mod_x5[mod].size())
            ########### IntraFormer
            mod_token_x5[mod] = self.modal_encode_conv[mod](mod_x5[mod]).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.cfg.transformer_dims)
            ic(mod_token_x5[mod].size(), self.modal_pos[mod].size())
            mod_intra_token_x5[mod] = self.modal_transformers[mod](mod_token_x5[mod], self.modal_pos[mod])
            ic(mod_intra_token_x5[mod].size())
            
            #----- Tokenize modality-specific transformer outputs to feed multi-modal transformer
            mod_intra_x5[mod] = mod_intra_token_x5[mod].view(x.size(0), self.patch_size, self.patch_size, self.patch_size, self.cfg.transformer_dims).permute(0, 4, 1, 2, 3).contiguous()
            # ic(mod_intra_x5[mod].size())
            if is_training and self.decoder_sep:
                #----- Upsample modality-specific encoder outputs
                mod_pred[mod] = self.decoder_sep(mod_x1[mod], mod_x2[mod], mod_x3[mod], mod_x4[mod], mod_x5[mod])
                ic(mod_pred[mod].size())
            ########### IntraFormer

        x1 = self.masker(torch.stack(list(mod_x1.values()), dim=1), mask) #BxMxCxHWZ
        x2 = self.masker(torch.stack(list(mod_x2.values()), dim=1), mask)
        x3 = self.masker(torch.stack(list(mod_x3.values()), dim=1), mask)
        x4 = self.masker(torch.stack(list(mod_x4.values()), dim=1), mask)
        x5 = self.masker(torch.stack(list(mod_x5.values()), dim=1), mask)
        x5_intra = self.masker(torch.stack(list(mod_intra_x5.values()), dim=1), mask)

        ########### InterFormer
        mod_intra_x5 = torch.chunk(x5_intra, self.num_modals, dim=1) 
        # ic(type(mod_intra_x5), len(mod_intra_x5))
        # ic(mod_intra_x5[0].size())
        multimodal_token_x5 = torch.cat([inp.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.cfg.transformer_dims) for inp in mod_intra_x5], dim=1)
        ic(multimodal_token_x5.size())

        multimodal_pos = torch.cat(list(self.modal_pos.values()), dim=1)
        multimodal_inter_token_x5 = self.multimodal_transformer(multimodal_token_x5, multimodal_pos)
        ic(multimodal_inter_token_x5.size())
        # ic(multimodal_inter_token_x5.view(multimodal_inter_token_x5.size(0), self.patch_size, self.patch_size, self.patch_size, self.cfg.transformer_dims*self.num_modals).permute(0, 4, 1, 2, 3).contiguous().size())
        multimodal_inter_x5 = self.multimodal_decode_conv(multimodal_inter_token_x5.view(multimodal_inter_token_x5.size(0), self.patch_size, self.patch_size, self.patch_size, self.cfg.transformer_dims*self.num_modals).permute(0, 4, 1, 2, 3).contiguous())
        x5_inter = multimodal_inter_x5
        ic(x1.size(), x2.size(), x3.size(), x4.size(), x5.size())
        ic(x5_inter.size())
        if features_only:
            return x1, x2, x3, x4, x5, mod_intra_x5, x5_inter, (mod_pred[mod] for mod in mod_pred)
            # return mod_intra_x5, x5_inter
        
        if self.decoder_fuse:
            fuse_pred, preds = self.decoder_fuse(x1, x2, x3, x4, x5_inter)
        ########### InterFormer
        
        if is_training:
            return fuse_pred, (mod_pred[mod] for mod in mod_pred), preds
        return fuse_pred
    


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.img_size = cfg.img_size
        self.patch_size = cfg.patch_size
        self.complete_modalities = ["T1", "T2", "FLAIR"] if 'complete' in cfg.arch else None
        self.modal = cfg.modal
        init_modal = self.modal if self.complete_modalities is None else self.complete_modalities
        self.num_modals = len(init_modal)

        self.flair_encoder = Encoder(cfg)
        self.t1ce_encoder = Encoder(cfg)
        self.t1_encoder = Encoder(cfg)
        self.t2_encoder = Encoder(cfg)

        ########### IntraFormer
        self.flair_encode_conv = nn.Conv3d(cfg.conv_dims*16, cfg.transformer_dims, kernel_size=1, stride=1, padding=0)
        self.t1ce_encode_conv = nn.Conv3d(cfg.conv_dims*16, cfg.transformer_dims, kernel_size=1, stride=1, padding=0)
        self.t1_encode_conv = nn.Conv3d(cfg.conv_dims*16, cfg.transformer_dims, kernel_size=1, stride=1, padding=0)
        self.t2_encode_conv = nn.Conv3d(cfg.conv_dims*16, cfg.transformer_dims, kernel_size=1, stride=1, padding=0)
        # self.flair_decode_conv = nn.Conv3d(cfg.transformer_dims, basic_dims*16, kernel_size=1, stride=1, padding=0)
        # self.t1ce_decode_conv = nn.Conv3d(cfg.transformer_dims, basic_dims*16, kernel_size=1, stride=1, padding=0)
        # self.t1_decode_conv = nn.Conv3d(cfg.transformer_dims, basic_dims*16, kernel_size=1, stride=1, padding=0)
        # self.t2_decode_conv = nn.Conv3d(cfg.transformer_dims, basic_dims*16, kernel_size=1, stride=1, padding=0)

        self.flair_pos = nn.Parameter(torch.zeros(1, self.patch_size**3, cfg.transformer_dims))
        self.t1ce_pos = nn.Parameter(torch.zeros(1, self.patch_size**3, cfg.transformer_dims))
        self.t1_pos = nn.Parameter(torch.zeros(1, self.patch_size**3, cfg.transformer_dims))
        self.t2_pos = nn.Parameter(torch.zeros(1, self.patch_size**3, cfg.transformer_dims))

        self.flair_transformer = Transformer(embedding_dim=cfg.transformer_dims, depth=cfg.depth, heads=cfg.num_heads, mlp_dim=cfg.mlp_dim)
        self.t1ce_transformer = Transformer(embedding_dim=cfg.transformer_dims, depth=cfg.depth, heads=cfg.num_heads, mlp_dim=cfg.mlp_dim)
        self.t1_transformer = Transformer(embedding_dim=cfg.transformer_dims, depth=cfg.depth, heads=cfg.num_heads, mlp_dim=cfg.mlp_dim)
        self.t2_transformer = Transformer(embedding_dim=cfg.transformer_dims, depth=cfg.depth, heads=cfg.num_heads, mlp_dim=cfg.mlp_dim)
        ########### IntraFormer

        ########### InterFormer
        self.multimodal_transformer = Transformer(embedding_dim=cfg.transformer_dims, depth=cfg.depth, heads=cfg.num_heads, mlp_dim=cfg.mlp_dim, n_levels=self.num_modals)
        self.multimodal_decode_conv = nn.Conv3d(cfg.transformer_dims*self.num_modals, cfg.conv_dims*16*self.num_modals, kernel_size=1, padding=0)
        ########### InterFormer

        self.masker = MaskModal()

        self.decoder_fuse = Decoder_fuse(cfg)
        self.decoder_sep = Decoder_sep(cfg)

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #

        dummy_input = torch.randn(1, self.num_modals, self.img_size, self.img_size, self.img_size)
        mask = torch.randn(1, self.num_modals)
        feats = self.forward(dummy_input, mask, features_only=True, is_training=True)
        x5_inter = feats[-2]
        self.in_features = list(torch.squeeze(x5_inter).size())

    def forward(self, x, mask, features_only=False, is_training=False):
        #extract feature from different layers
        ic(x.size(), mask.size())
        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4, :, :, :])

        ########### IntraFormer
        flair_token_x5 = self.flair_encode_conv(flair_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.cfg.transformer_dims)
        t1ce_token_x5 = self.t1ce_encode_conv(t1ce_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.cfg.transformer_dims)
        t1_token_x5 = self.t1_encode_conv(t1_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.cfg.transformer_dims)
        t2_token_x5 = self.t2_encode_conv(t2_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.cfg.transformer_dims)

        flair_intra_token_x5 = self.flair_transformer(flair_token_x5, self.flair_pos)
        t1ce_intra_token_x5 = self.t1ce_transformer(t1ce_token_x5, self.t1ce_pos)
        t1_intra_token_x5 = self.t1_transformer(t1_token_x5, self.t1_pos)
        t2_intra_token_x5 = self.t2_transformer(t2_token_x5, self.t2_pos)

        flair_intra_x5 = flair_intra_token_x5.view(x.size(0), self.patch_size, self.patch_size, self.patch_size, self.cfg.transformer_dims).permute(0, 4, 1, 2, 3).contiguous()
        t1ce_intra_x5 = t1ce_intra_token_x5.view(x.size(0), self.patch_size, self.patch_size, self.patch_size, self.cfg.transformer_dims).permute(0, 4, 1, 2, 3).contiguous()
        t1_intra_x5 = t1_intra_token_x5.view(x.size(0), self.patch_size, self.patch_size, self.patch_size, self.cfg.transformer_dims).permute(0, 4, 1, 2, 3).contiguous()
        t2_intra_x5 = t2_intra_token_x5.view(x.size(0), self.patch_size, self.patch_size, self.patch_size, self.cfg.transformer_dims).permute(0, 4, 1, 2, 3).contiguous()

        if is_training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)
        ########### IntraFormer

        x1 = self.masker(torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1), mask) #Bx4xCxHWZ
        x2 = self.masker(torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1), mask)
        x3 = self.masker(torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1), mask)
        x4 = self.masker(torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1), mask)
        x5 = self.masker(torch.stack((flair_x5, t1ce_x5, t1_x5, t2_x5), dim=1), mask)
        x5_intra = self.masker(torch.stack((flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5), dim=1), mask)
        mod_intra_x5 = (flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5)
        ########### InterFormer
        flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5 = torch.chunk(x5_intra, self.num_modals, dim=1)
        multimodal_token_x5 = torch.cat((flair_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.cfg.transformer_dims),
                                         t1ce_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.cfg.transformer_dims),
                                         t1_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.cfg.transformer_dims),
                                         t2_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.cfg.transformer_dims),
                                        ), dim=1)
        multimodal_pos = torch.cat((self.flair_pos, self.t1ce_pos, self.t1_pos, self.t2_pos), dim=1)
        multimodal_inter_token_x5 = self.multimodal_transformer(multimodal_token_x5, multimodal_pos)
        multimodal_inter_x5 = self.multimodal_decode_conv(multimodal_inter_token_x5.view(multimodal_inter_token_x5.size(0), self.patch_size, self.patch_size, self.patch_size, self.cfg.transformer_dims*self.num_modals).permute(0, 4, 1, 2, 3).contiguous())
        x5_inter = multimodal_inter_x5

        if features_only:
            return x1, x2, x3, x4, x5, mod_intra_x5, x5_inter, (flair_pred, t1ce_pred, t1_pred, t2_pred)

        fuse_pred, preds = self.decoder_fuse(x1, x2, x3, x4, x5_inter)
        ########### InterFormer
        
        if is_training:
            return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), preds
        return fuse_pred