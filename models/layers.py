import torch
import torch.nn as nn

def normalization(planes, norm='bn', eps=1e-3):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes, eps=eps)
    # elif norm == 'sync_bn':
    #     m = SynchronizedBatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

class general_conv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        assert not torch.isnan(x).any(), f"NaN detected at {self.conv}"
        x = self.activation(x)
        assert not torch.isnan(x).any(), f"NaN detected at {self.activation}"
        return x

class general_conv3d_prenorm(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv3d_prenorm, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)


    def forward(self, x, last=False):
        # ic(x.min(), x.max(), x.std(), torch.isnan(x).any())
        assert not torch.isnan(x).any(), f"NaN detected before {self.norm}"
        if not last:
            x = self.norm(x)
            assert not torch.isnan(x).any(), f"NaN detected at {self.norm}"
        x = self.activation(x)
        assert not torch.isnan(x).any(), f"NaN detected at {self.activation}"
        x = self.conv(x)
        assert not torch.isnan(x).any(), f"NaN detected at {self.conv}"
        return x

class general_conv3d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)


    def forward(self, x, last=False):
        x = self.conv(x)
        # ic(x.min(), x.max())
        x = torch.clamp(x, min=-1e4, max=1e4)  # Clamp to avoid extreme values
        assert not torch.isnan(x).any(), "NaN detected after convolution & clamping"
        if not last:
            x = self.norm(x)
            assert not torch.isnan(x).any(), "NaN detected after normalization"
        x = self.activation(x)
        assert not torch.isnan(x).any(), "NaN detected after activation"
        return x
    
class EfficientConv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm='in'):
        super(EfficientConv3DBlock, self).__init__()
        ic(in_channels, out_channels)
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels)  # Depthwise Conv
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False) # Pointwise conv

        # **Two-stage Pooling for Aggressive Downsampling**
        self.pool = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),  # 64 → 32
            nn.AvgPool3d(kernel_size=2, stride=2),  # 32 → 16
            # nn.AvgPool3d(kernel_size=2, stride=2),  # 16 → 8
            nn.AvgPool3d(kernel_size=2, stride=2))  # 8 → 4

        self.norm = norm
        if norm == 'bn':
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.norm = normalization(out_channels, norm=norm)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        ic(x.shape)
        x = self.depthwise(x)
        ic(x.shape)
        x = self.pointwise(x)
        ic(x.shape)
        x = self.pool(x)
        ic(x.shape)
        if self.norm == 'bn':
            x = self.bn(x)
        else:
            x = self.norm(x)
        x = self.act(x)
        return x

class fusion_prenorm(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(fusion_prenorm, self).__init__()
        self.fusion_layer = nn.Sequential(
                        general_conv3d_prenorm(in_channel*num_cls, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d_prenorm(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d_prenorm(in_channel, in_channel, k_size=1, padding=0, stride=1))

    def forward(self, x):
        return self.fusion_layer(x)

class fusion_postnorm(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(fusion_postnorm, self).__init__()
        self.fusion_layer = nn.Sequential(
                        general_conv3d(in_channel*num_cls, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d(in_channel, in_channel, k_size=1, padding=0, stride=1))

    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        y = y.view(B, -1, H, W, Z)
        return self.fusion_layer(y)

    
###### Used by RFNet #######

class prm_generator_laststage(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4, num_modal=4):
        super(prm_generator_laststage, self).__init__()

        self.embedding_layer = nn.Sequential(
                            general_conv3d(in_channel*num_modal, int(in_channel//4), k_size=1, padding=0, stride=1),
                            general_conv3d(int(in_channel//4), int(in_channel//4), k_size=3, padding=1, stride=1),
                            general_conv3d(int(in_channel//4), in_channel, k_size=1, padding=0, stride=1))

        self.prm_layer = nn.Sequential(
                            general_conv3d(in_channel, 16, k_size=1, stride=1, padding=0),
                            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
                            nn.Softmax(dim=1))

    def forward(self, x, mask):
        ic(x.size(), mask.size())
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        y = y.view(B, -1, H, W, Z)
        emb = self.embedding_layer(y)
        ic(emb.size())
        seg = self.prm_layer(emb)
        return seg

class prm_generator(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4, num_modal=4):
        super(prm_generator, self).__init__()

        self.embedding_layer = nn.Sequential(
                            general_conv3d(in_channel*num_modal, int(in_channel//4), k_size=1, padding=0, stride=1),
                            general_conv3d(int(in_channel//4), int(in_channel//4), k_size=3, padding=1, stride=1),
                            general_conv3d(int(in_channel//4), in_channel, k_size=1, padding=0, stride=1))


        self.prm_layer = nn.Sequential(
                            general_conv3d(in_channel*2, 16, k_size=1, stride=1, padding=0),
                            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
                            nn.Softmax(dim=1))

    def forward(self, x1, x2, mask):
        B, K, C, H, W, Z = x2.size()
        y = torch.zeros_like(x2)
        y[mask, ...] = x2[mask, ...]
        y = y.view(B, -1, H, W, Z)

        seg = self.prm_layer(torch.cat((x1, self.embedding_layer(y)), dim=1))
        return seg

class prm_fusion(nn.Module):
    def __init__(self, in_channel=64, basic_dim=16, norm='in', num_cls=4):
        super(prm_fusion, self).__init__()

        self.prm_layer = nn.Sequential(
                            general_conv3d(in_channel, basic_dim, k_size=1, stride=1, padding=0),
                            nn.Conv3d(basic_dim, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
                            nn.Softmax(dim=1))

    def forward(self, x1):

        seg = self.prm_layer(x1)
        return seg

####modal fusion in each region
class modal_fusion(nn.Module):
    def __init__(self, in_channel=64, num_modal=4):
        super(modal_fusion, self).__init__()
        self.weight_layer = nn.Sequential(
                            nn.Conv3d(num_modal*in_channel+1, 128, 1, padding=0, bias=True),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Conv3d(128, num_modal, 1, padding=0, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, prm):
        B, K, C, H, W, Z = x.size()
        ic(x.size())

        prm_avg = torch.mean(prm, dim=(3,4,5), keepdim=False) + 1e-7
        feat_avg = torch.mean(x, dim=(3,4,5), keepdim=False) / prm_avg

        feat_avg = feat_avg.view(B, K*C, 1, 1, 1)
        ic(feat_avg.size())
        feat_avg = torch.cat((feat_avg, prm_avg[:, 0, 0, ...].view(B, 1, 1, 1, 1)), dim=1)
        ic(feat_avg.size())
        ic(self.weight_layer)
        weight = self.weight_layer(feat_avg)
        ic(weight.size())
        weight = weight.view(B, K, 1)
        weight = self.sigmoid(weight).view(B, K, 1, 1, 1, 1)

        ###we find directly using weighted sum still achieve competing performance
        region_feat = torch.sum(x * weight, dim=1)
        return region_feat

###fuse region feature
class region_fusion(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(region_fusion, self).__init__()
        self.fusion_layer = nn.Sequential(
                        general_conv3d(in_channel*num_cls, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d(in_channel, in_channel//2, k_size=1, padding=0, stride=1))

    def forward(self, x):
        B, _, _, H, W, Z = x.size()
        x = torch.reshape(x, (B, -1, H, W, Z))
        return self.fusion_layer(x)

class region_aware_modal_fusion(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4, num_modal=4):
        super(region_aware_modal_fusion, self).__init__()
        # self.num_cls = 2 if num_cls == 1 else num_cls
        self.num_cls = num_cls
        self.num_modal = num_modal

        self.modal_fusion = nn.ModuleList([modal_fusion(in_channel=in_channel, num_modal=num_modal) for i in range(self.num_cls)])
        self.region_fusion = region_fusion(in_channel=in_channel, num_cls=self.num_cls)
        self.short_cut = nn.Sequential(
                        general_conv3d(in_channel*self.num_modal, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d(in_channel, in_channel//2, k_size=1, padding=0, stride=1))

        # self.clsname_list = ['BG', 'NCR/NET', 'ED', 'ET'] ##BRATS2020 and BRATS2018
        # # self.clsname_list = ['BG', 'NCR', 'ED', 'NET', 'ET'] ##BRATS2015
        # if num_cls == 2: # ISLES dataset
        #     self.clsname_list = ['BG', 'lesion']

    def forward(self, x, prm, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]

        prm = torch.unsqueeze(prm, 2).repeat(1, 1, C, 1, 1, 1)
        ###divide modal features into different regions
        # modal_feat = y * prm.unsqueeze(1)

        # region_feat = [modal_feat[:, :, i, :, :, :, :] for i in range(K)]

        region_feat = []
        for m in range(K):
            region_feat.append(y[:,m:m+1,...] * prm)
        modal_feat = torch.stack(region_feat, dim=1)
        modal_feat = modal_feat.permute(0,2,1,3,4,5,6)
        ic(modal_feat.shape)
        ###modal fusion in each region
        region_fused_feat = []
        for i in range(self.num_cls):
            ic(modal_feat[:,i,...].shape, prm[:,i:i+1,...].shape)
            modal_fused = self.modal_fusion[i](modal_feat[:,i, ...], prm[:, i:i+1, ...])
            ic(modal_fused.shape)
            region_fused_feat.append(modal_fused)
        region_fused_feat = torch.stack(region_fused_feat, dim=1)
        ic(region_fused_feat.size())
        '''
        region_fused_feat = torch.stack((self.modal_fusion[0](region_feat[0], prm[:, 0:1, ...], 'BG'),
                                         self.modal_fusion[1](region_feat[1], prm[:, 1:2, ...], 'NCR/NET'),
                                         self.modal_fusion[2](region_feat[2], prm[:, 2:3, ...], 'ED'),
                                         self.modal_fusion[3](region_feat[3], prm[:, 3:4, ...], 'ET')), dim=1)
        '''

        ###gain final feat with a short cut
        region_fused_feat = self.region_fusion(region_fused_feat)
        ic(region_fused_feat.shape)
        y = self.short_cut(y.view(B, -1, H, W, Z))
        ic(y.shape)
        final_feat = torch.cat((region_fused_feat, y), dim=1)
        return final_feat


######### LoRA ###########
class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module, r: int, alpha: int):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b
        self.r = r
        self.alpha = alpha

    def forward(self, x):
        x = self.w(x) + (self.alpha // self.r) * self.w_b(self.w_a(x))
        return x


class _LoRA_qkv_timm(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        r: int,
        alpha: int
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)
        self.r = r
        self.alpha = alpha

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += (self.alpha // self.r) * new_q
        qkv[:, :, -self.dim :] += (self.alpha // self.r) * new_v
        return qkv