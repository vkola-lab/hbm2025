#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml

# Request n CPUs
#$ -pe omp 3

#$ -m ea

# Request 1 GPU 
#$ -l gpus=1
#$ -l gpu_memory=48G

#$ -l h_rt=1:00:00

module load miniconda/23.11.0
conda activate py3.11

roi=64
ps=4
augment="none" #"RegionModalMix1.1" #"none" #
fusion_type=null #"concat"
# ckpt="checkpoints/IMS2Transdim24_roi${roi}ps${ps}_NoneFusion_clsNone_BRATS2020_augment${augment}"
suffix="_last"
# ckpt="checkpoints/RFNetdimNone_roi64ps4_clsNone_GeneralizedDSC_BRATS2020_augmentRegionModalMix1.1_a0.2b0.2_noamp_noclip" 
# ckpt="checkpoints/RFNetdimNone_roi64ps4_clsNone_BRATS2020_augmentnone_noamp_noclip"
# ckpt="/projectnb/ivc-ml/dlteif/M2FTrans/M2FTrans_v1/output/rfnet_roi64_batchsize4_iter150_epoch1000_lr2e-4_rfse0"
# ckpt="/projectnb/ivc-ml/dlteif/M2FTrans/M2FTrans_v1/output/m2ftrans_roi64ps4_batchsize2_iter200_epoch1000_lr2e-4_rfse0_augmentRegionModalMix1.1"
# ckpt="/projectnb/ivc-ml/dlteif/M2FTrans/M2FTrans_v1/output/m2ftrans_roi64ps4_batchsize4_iter200_epoch1000_lr2e-4_rfse0"
ckpt="checkpoints/IMS2Transdim24_roi64ps4_clsNone_BRATS2020_augmentRegionModalMix1.1_a0.2b0.2_noamp_noclip"

python -m evaluate.visualizeBraTS -cn cfg_SEG model=ims2trans ++model.transformer_dims=24 ++train.debug=False logging=no ++model.img_size=${roi} ++model.patch_size=${ps} ++model.fusion_type=${fusion_type} ++dataset.img_size=${roi} ++train.batch_size=1 +optimizer=adamw ++train.logdir=${ckpt}/ ++train.resume_train=True ++train.resume_model=${ckpt}/model${suffix}.pt test=True eval=test ++eval.write_raw_score=False ++eval.suffix=${suffix} ++train.checkpoint_key='model' ++eval.wandb_artifact=False
