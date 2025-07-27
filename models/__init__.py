from .vision_transformer import *
from .embed_layer_3d_modality import VoxelEmbed, VoxelEmbed_Hybrid_no_average, \
                                        VoxelEmbed_Hybrid, VoxelNaiveProjection, \
                                        VoxelEmbed_Hybrid_no_average, VoxelEmbed_no_average
from monai.networks.nets.swin_unetr import SwinUNETR
from .mmFormer import Model as mmFormer
from .ims2trans import Model as IMS2Trans

VALID_EMBED_LAYER={
    'VoxelEmbed': VoxelEmbed,
    'VoxelEmbed_no_zdim': VoxelNaiveProjection,
    'VoxelEmbed_no_average': VoxelEmbed_no_average,
    'VoxelEmbed': VoxelEmbed,
}

BACKBONE_EMBED_DIM={
    'deit_base_patch16_224': 768,
    'deit_small_patch16_224': 384,
    'deit_tiny_patch16_224': 192
}

weight_Info = {
    "base_dino": "vit_base_patch16_224.dino",  # 21k -> 1k
    "base_sam": "vit_base_patch16_224.sam",  # 1k
    "base_mill": "vit_base_patch16_224_miil.in21k_ft_in1k",  # 1k
    "base_beit": "beitv2_base_patch16_224.in1k_ft_in22k_in1k",
    "base_clip": "vit_base_patch16_clip_224.laion2b_ft_in1k",  # 1k
    "base_deit": "deit_base_distilled_patch16_224",  # 1k
    "large_clip": "vit_large_patch14_clip_224.laion2b_ft_in1k",  # laion-> 1k
    "large_beit": "beitv2_large_patch16_224.in1k_ft_in22k_in1k",
    "huge_clip": "vit_huge_patch14_clip_224.laion2b_ft_in1k",  # laion-> 1k
    "giant_eva": "eva_giant_patch14_224.clip_ft_in1k",  # laion-> 1k
    "giant_clip": "vit_giant_patch14_clip_224.laion2b",
    "giga_clip": "vit_gigantic_patch14_clip_224.laion2b",
}

