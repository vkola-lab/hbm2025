from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional

from omegaconf import MISSING

from helper_classes.channel_initialization import ChannelInitialization
from helper_classes.feature_pooling import FeaturePooling
from helper_classes.first_layer_init import FirstLayerInit, NewChannelLeaveOneOut
from helper_classes.norm_type import NormType
from helper_classes.channel_pooling_type import ChannelPoolingType

# fmt: off

@dataclass
class OptimizerParams(Dict):
    pass


@dataclass
class Optimizer:
    name: str
    params: OptimizerParams
    lr: float
    momentum: float
    weight_decay: float


@dataclass
class SchedulerParams(Dict):
    pass


@dataclass
class Scheduler:
    name: str
    convert_to_batch: bool
    params: SchedulerParams
    warmup_steps: int
    num_epochs: int
    optimizer: Optimizer


@dataclass
class Train:
    batch_strategy: None
    resume_train: bool
    resume_model: Optional[str]
    checkpoint_key: str
    noamp: bool
    grad_clip: bool
    max_grad_norm: float
    batch_size: int
    accum_iter: int
    num_epochs: int
    iter_per_epoch: int
    start_epoch: Optional[int]
    seed: int
    debug: bool
    logdir: Optional[str]
    deterministic: Optional[bool]

    ## dataloaders, sampling,
    sampling: str
    smote_strategy: Optional[str]
    augment: str = "none"
    # adaptive_interface_epochs: int = 0
    # adaptive_interface_lr: Optional[float] = None
    swa: Optional[bool] = False
    swad: Optional[bool] = False
    swa_lr: Optional[float] = 0.05
    swa_start: Optional[int] = 5

    ## MIRO
    miro: Optional[bool] = False
    miro_lr_mult: Optional[float] = 10.0
    miro_ld: Optional[float] = 0.01  
    
    ## TPS Transform (Augmentation)
    tps_prob: Optional[float] = 0.0

    ## Supervised Loss
    criterion: str = "CE"
    label_smoothing: float = 0.0
    
    ## Self-Supervised Learning (SSL) 
    ssl: Optional[bool] = False
    cont_lambda: Optional[float] = 0.0
    cont_alpha: Optional[float] = 0.0
    cont_beta: Optional[float] = 0.0

    ## Training chunks, for leave one out
    training_chunks: Optional[str] = None

    ## extra loss: channel proxy loss
    extra_loss_lambda: Optional[float] = 0.0
    
    ## for SEGMENTATION loss fusion.
    region_fusion_start_epoch: Optional[int] = 0

    # plot_attn: Optional[bool] = False


@dataclass
class Eval:
    batch_size: int
    write_raw_score: bool
    suffix: Optional[str]
    combination: Optional[str]
    apply_mask: Optional[str]
    save_path: Optional[str] = None
    wandb_artifact: Optional[bool] = False
    # dest_dir: str = ""  ## where to save results
    # feature_dir: str = ""  ## where to save features for evaluation
    # root_dir: str = ""  ## folder that contains images and metadata
    # classifiers: List[str] = field(default_factory=list)  ## classifier to use
    # classifier: str = ""  ## placeholder for classifier
    # feature_file: str = ""  ## feature file to use
    # use_gpu: bool = True  ## use gpu for evaluation
    # knn_metrics: List[str] = field(default_factory=list)  ## "l2" or "cosine"
    # knn_metric: str = ""  ## should be "l2" or "cosine", placeholder
    # meta_csv_file: str = ""  ## metadata csv file
    # clean_up: bool = True  ## whether to delete the feature file after evaluation
    # only_eval_first_and_last: bool = False  ## whether to only evaluate first (off the shelf) and last (final fune-tuned) epochs
    # every_n_epochs: int = 1  ## evaluate every n epochs
    # skip_eval_first_epoch: Optional[bool] = False  ## whether to skip evaluation on the first epoch
    # eval_subset_channels: Optional[bool] = False  ## whether to evaluate on a subset of channels

@dataclass
class AttentionPoolingParams:
    """
    param for ChannelAttentionPoolingLayer class.
    initialize all arguments in the class.
    """

    max_num_channels: int
    dim: int
    depth: int
    dim_head: int
    heads: int
    mlp_dim: int
    dropout: float
    use_cls_token: bool
    use_channel_tokens: bool
    init_channel_tokens: ChannelInitialization


@dataclass
class Model:
    arch: str
    conv_dims: Optional[int]
    transformer_dims: Optional[int]
    num_heads: Optional[int]
    mlp_dim: Optional[int]
    mlp_ratio: Optional[int]
    depth: Optional[int]
    # init_weights: bool
    # in_dim: int = MISSING
    labels: List[str]  ## list of labels
    num_cls: Optional[int]
    freeze_other: Optional[bool] = None  ## used in Shared Models
    modal: List[str] = MISSING  ## also used to compute total number of channels
    separate_norm: Optional[
        bool
    ] = None  ## use a separate norm layer for each data chunk
    img_size: int = 64 
    patch_size: int = 4 
    classifier: Optional[str] = None
    fusion_type: Optional[str] = None
    norm_type: Optional[
        NormType
    ] = None  # one of ["batch_norm", "norm_type", "instance_norm"]
    duplicate: Optional[
        bool
    ] = None  # whether to only use the first param bank and duplicate for all the channels
    pooling_channel_type: Optional[ChannelPoolingType] = None
    kernels_per_channel: Optional[int] = None
    num_templates: Optional[int] = None  # number of templates to use in template mixing
    separate_coef: Optional[bool] = None  # whether to use a separate set of coefficients for each chunk
    coefs_init: Optional[bool] = None # whether to initialize the coefficients, used in templ mixing ver2
    freeze_coefs_epochs: Optional[int] = None # TODO: add this. Whether to freeze the coefficients for some first epoch, used in templ mixing ver2
    separate_emb: Optional[bool] = None  # whether to use a separate embedding (hypernetwork) for each chunk
    z_dim: Optional[int] = None  # dimension of the latent space, hypernetwork
    hidden_dim: Optional[int] = None  # dimension of the hidden layer, hypernetwork

    ## temperature in the loss
    learnable_temp: bool = False

    ## leave one out
    new_channel_inits: Optional[List[NewChannelLeaveOneOut]] = None

    ## use_hcs
    enable_sample: Optional[bool] = False

    use_channelvit_channels: Optional[bool] = True

    ## hypernet
    orthogonal_init: Optional[bool] = False  ## whether to use orthogonal initialization embedding (`conv1_emb`)
    use_conv1x1: Optional[bool] = False ## reduce the number of parameters in the hypernetwork
    
    z_emb_init: Optional[str] = None ## random, orthogonal, or a path storing pytorch tensor
    freeze_z_emb: Optional[bool] = False ## whether to freeze the z_emb

    attn_type: Optional[str] = None ## used in hyper channel Vit

    is_conv_small: Optional[bool] = False ## norm to 22M parameters

    z_dim_0: Optional[int] = 0 ## used in hyper hyper net
    
    reduce_size: Optional[bool] = True ## used in depthwise conv

    sample_by_weights: Optional[bool] = False ## used in depthwise

    sample_by_weights_warmup: Optional[int] = 0 ## used in depthwise
    sample_by_weights_scale : Optional[float] = 0.3 ## used in depthwise
    generate_first_layer: Optional[bool] = False ## used in depthwise
    channel_extractor_dim: Optional[int] = 64 ## used in depthwise
    channel_extractor_patch_size: Optional[int] = 0 ## used in depthwise

    ## used in DiChaViT
    orth_loss_v1_lambda: Optional[float] = 0.0 
    proxy_loss_lambda: Optional[float] = 0.0 

    ## used in RegionModalMix


    dropout_rate: Optional[float] = 0.0 ## used in channel vit models
    proj_drop: Optional[float] = 0.0 ## used in self attention

@dataclass
class Dataset:
    name: str = MISSING
    test_name: Optional[str] = None
    img_size: int = 128
    labels: List[str] = MISSING
    # root_dir: str = ""
    train_path: str = MISSING
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    inflate: bool = False
    stripped: str = "_stripped_MNI"
    modal: Optional[List[str]] = None ## used to compute total number of channels
    load_segmentations: bool = False
    apply_mask: Optional[str] = None
    random_masking: Optional[str] = "none"
    split_ratio: float = 0.6

@dataclass
class Wandb:
    use_wandb: bool
    log_freq: int
    project_name: str
    run_name: Optional[str]
    entity: str
    log_code: str
    sweep_id: Optional[str] = None

@dataclass
class Logging:
    wandb: Wandb
    use_py_log: bool
    scc_jobid: Optional[str] = None


@dataclass
class Hardware:
    num_workers: int
    device: str
    multi_gpus: str
    rank: Optional[int]
    gpu: Optional[int]
    world_size: Optional[int]
    master_port: int
    dist_url: Optional[str] = "env://"


@dataclass
class MyConfig:
    train: Train
    eval: Eval
    test: bool
    optimizer: Optimizer
    scheduler: Scheduler
    model: Model
    dataset: Dataset
    logging: Logging
    hardware: Hardware
    tag: str
    attn_pooling: Optional[AttentionPoolingParams] = None