import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import (general_conv3d, normalization, prm_generator,
                    prm_generator_laststage, region_aware_modal_fusion)

class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()

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

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        return x1, x2, x3, x4

class Decoder_sep(nn.Module):
    def __init__(self, cfg):
        super(Decoder_sep, self).__init__()
        
        num_cls = cfg.num_cls
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

    def forward(self, x1, x2, x3, x4):
        de_x4 = self.d3_c1(self.d3(x4))

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

        self.RFM4 = region_aware_modal_fusion(in_channel=cfg.conv_dims*8, num_cls=num_cls, num_modal=len(cfg.modal))
        self.RFM3 = region_aware_modal_fusion(in_channel=cfg.conv_dims*4, num_cls=num_cls, num_modal=len(cfg.modal))
        self.RFM2 = region_aware_modal_fusion(in_channel=cfg.conv_dims*2, num_cls=num_cls, num_modal=len(cfg.modal))
        self.RFM1 = region_aware_modal_fusion(in_channel=cfg.conv_dims*1, num_cls=num_cls, num_modal=len(cfg.modal))

        self.prm_generator4 = prm_generator_laststage(in_channel=cfg.conv_dims*8, num_cls=num_cls, num_modal=len(cfg.modal))
        self.prm_generator3 = prm_generator(in_channel=cfg.conv_dims*4, num_cls=num_cls, num_modal=len(cfg.modal))
        self.prm_generator2 = prm_generator(in_channel=cfg.conv_dims*2, num_cls=num_cls, num_modal=len(cfg.modal))
        self.prm_generator1 = prm_generator(in_channel=cfg.conv_dims*1, num_cls=num_cls, num_modal=len(cfg.modal))


    def forward(self, x1, x2, x3, x4, mask):
        ic(x4.size(), mask.size())
        prm_pred4 = self.prm_generator4(x4, mask)
        ic(prm_pred4.size(), self.up8(prm_pred4).size())
        de_x4 = self.RFM4(x4, prm_pred4.detach(), mask)
        de_x4 = self.d3_c1(self.up2(de_x4))

        prm_pred3 = self.prm_generator3(de_x4, x3, mask)
        de_x3 = self.RFM3(x3, prm_pred3.detach(), mask)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        prm_pred2 = self.prm_generator2(de_x3, x2, mask)
        de_x2 = self.RFM2(x2, prm_pred2.detach(), mask)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        prm_pred1 = self.prm_generator1(de_x2, x1, mask)
        de_x1 = self.RFM1(x1, prm_pred1.detach(), mask)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (prm_pred1, self.up2(prm_pred2), self.up4(prm_pred3), self.up8(prm_pred4))

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        num_cls = cfg.num_cls
        self.img_size = cfg.img_size
        self.flair_encoder = Encoder(cfg)
        self.t1ce_encoder = Encoder(cfg)
        self.t1_encoder = Encoder(cfg)
        self.t2_encoder = Encoder(cfg)

        self.decoder_fuse = Decoder_fuse(cfg)
        self.decoder_sep = Decoder_sep(cfg)

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #

    def forward(self, x, mask, is_training=False):
        ic(x.size())
        
        if len(x.size()) < 5:
            x = x.unsqueeze(2)
        
        #extract feature from different layers
        flair_x1, flair_x2, flair_x3, flair_x4 = self.flair_encoder(x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4 = self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4 = self.t2_encoder(x[:, 3:4, :, :, :])

        x1 = torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1) #Bx4xCxHWZ
        x2 = torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1)
        x3 = torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1)
        x4 = torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1)

        ic(x1.size(), x2.size(), x3.size(), mask.size())
        fuse_pred, prm_preds = self.decoder_fuse(x1, x2, x3, x4, mask)

        if is_training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4)
            return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), prm_preds
        return fuse_pred
    
class Model2(nn.Module):
    def __init__(self, cfg):
        super(Model2, self).__init__()
        num_cls = cfg.num_cls
        self.img_size = cfg.img_size
        self.modal = cfg.modal
        self.modal_encoders = nn.ModuleDict({mod: Encoder(cfg) for mod in self.modal})

        self.decoder_fuse = Decoder_fuse(cfg)
        self.decoder_sep = Decoder_sep(cfg)

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #

    def forward(self, x, mask, is_training=False):
        #extract feature from different layers
        if len(x.size()) < 6:
            x = x.unsqueeze(2)
        
        mask = mask.bool()

        mod_x1, mod_x2, mod_x3, mod_x4 = {}, {}, {}, {}

        for idx, mod in enumerate(self.modal): 
            mod_x1[mod], mod_x2[mod], mod_x3[mod], mod_x4[mod] = self.modal_encoders[mod](x[:, idx, :, :, :, :])
        
        

        x1 = torch.stack(list(mod_x1.values()), dim=1) #BxMxCxHWZ
        x2 = torch.stack(list(mod_x2.values()), dim=1)
        x3 = torch.stack(list(mod_x3.values()), dim=1)
        x4 = torch.stack(list(mod_x4.values()), dim=1)

        fuse_pred, prm_preds = self.decoder_fuse(x1, x2, x3, x4, mask)

        mod_preds = {}
        if is_training:
            for mod in self.modal:
                mod_preds[mod] = self.decoder_sep(mod_x1[mod], mod_x2[mod], mod_x3[mod], mod_x4[mod])
            
            return fuse_pred, (mod_preds[mod] for mod in self.modal), prm_preds
        
        return fuse_pred
