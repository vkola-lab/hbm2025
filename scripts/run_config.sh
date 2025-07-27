#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml

# Request 4 CPUs
#$ -pe omp 3

#$ -m bea

# Request 1 GPU 
#$ -l gpus=1
#$ -l gpu_memory=48G

#$ -l h_rt=48:00:00

# nvidia-smi
# echo $CUDA_VISIBLE_DEVICES

# module load miniconda
conda activate mmMRI
export WANDB_CACHE_DIR="/projectnb/ivc-ml/dlteif/.cache/"
echo $WANDB_CACHE_DIR
export WANDB_DIR=$WANDB_CACHE_DIR
echo $WANDB_DIR
export TMPDIR="/projectnb/ivc-ml/dlteif/tmp"
mkdir -p $TMPDIR

export MASTER_PORT=$((29435 + RANDOM % 20000))
echo $MASTER_PORT


## -----------------------
## RUN W/ WANDB SWEEP ID
## -----------------------
# python main.py -cn main_cfg logging=wandb ++logging.wandb.sweep_id="dlteif/MultiModalMRI/9i10ll5l" dataset=naccIE ++dataset.load_segmentations=True ++train.debug=False model=mmformer
## -----------------------

roi=64
ps=4
augment="RegionModalMix1.1" #"3DMMCutMixorig"
fusion_type="null" #"interprod"
criterion="GeneralizedDSC"
dset="BRATS2020"
model="IMS2Transdim24"
ckpt="/projectnb/ivc-ml/dlteif/multimodalMRI/checkpoints/${model}_roi${roi}ps${ps}_batchsize2_iter200_epoch1000_lr2e-4_rfse0_augment${augment}"
ddp=null
python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 --master_port=$1 main.py -cn cfg_SEG ++hardware.multi_gpus=$ddp ++train.debug=False +optimizer=adamw logging=wandb model=rfnet ++train.batch_size=2 ++model.img_size=${roi} ++model.patch_size=${ps} ++dataset.img_size=${roi} ++train.criterion=${criterion} ++train.cont_alpha=0.2 ++train.cont_beta=0.2 ++train.augment=${augment} ++dataset.load_segmentations=True ++dataset.random_masking=True ++train.logdir=${ckpt} ++train.iter_per_epoch=200



## -----------------------
## EVALUATE
## -----------------------
# fusion_type=null #"attn" #"interprod"
# ckpt="/projectnb/ivc-ml/dlteif/M2FTrans/M2FTrans_v1/output/mmformer_roi64ps4_batchsize4_iter200_epoch1000_lr2e-4_rfse0"
# suffix="_last"
# export TORCH_USE_CUDA_DSA=1
# CUDA_LAUNCH_BLOCKING=1 python main.py -cn cfg_SEG model=m2ftrans ++train.debug=False logging=no ++model.img_size=${roi} ++model.patch_size=${ps} ++model.fusion_type=$fusion_type ++dataset.img_size=${roi} ++train.batch_size=1 +optimizer=adamw ++train.logdir=${ckpt}/ ++train.resume_train=True ++train.resume_model=${ckpt}/model${suffix}.pth test=True eval=test ++eval.write_raw_score=False ++eval.batch_size=4 ++eval.suffix=${suffix} ++train.checkpoint_key='state_dict' ++eval.wandb_artifact=False ++train.augment=${augment}
