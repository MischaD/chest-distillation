#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --job-name=ComputeMetrics
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --export=NONE

cd $WORK/pycharm/chest-distillation

unset SLURM_EXPORT_ENV
export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80

#
#load required modules (compiler, ...)
module load python/3.9-anaconda
module load cudnn/8.2.4.15-11.4
module load cuda/11.4
#
# anaconda
source activate chest

python scripts/compute_bbox_iou.py experiments/chestxray/train_baseline_reontgen_hpc.py compute-metrics-sdv2 --ckpt=/home/atuin/b143dc/b143dc11/diffusionmodels/latentdiffusion/512-base-ema.ckpt
python scripts/compute_bbox_iou.py experiments/chestxray/train_baseline_reontgen_hpc.py compute-metrics-finetune-sd-128 --ckpt=/home/atuin/b143dc/b143dc11/diffusionmodels/chest/finetune-sd-128/global_step=10000.ckpt
python scripts/compute_bbox_iou.py experiments/chestxray/train_baseline_reontgen_hpc.py compute-metrics-finetune-sd-128 --ckpt=/home/atuin/b143dc/b143dc11/diffusionmodels/chest/finetune-sd-128/global_step=20000.ckpt
python scripts/compute_bbox_iou.py experiments/chestxray/train_baseline_reontgen_hpc.py compute-metrics-finetune-sd-128 --ckpt=/home/atuin/b143dc/b143dc11/diffusionmodels/chest/finetune-sd-128/global_step=30000.ckpt
python scripts/compute_bbox_iou.py experiments/chestxray/train_baseline_reontgen_hpc.py compute-metrics-finetune-sd-128 --ckpt=/home/atuin/b143dc/b143dc11/diffusionmodels/chest/finetune-sd-128/global_step=40000.ckpt
python scripts/compute_bbox_iou.py experiments/chestxray/train_baseline_reontgen_hpc.py compute-metrics-finetune-sd-128 --ckpt=/home/atuin/b143dc/b143dc11/diffusionmodels/chest/finetune-sd-128/global_step=50000.ckpt
python scripts/compute_bbox_iou.py experiments/chestxray/train_baseline_reontgen_hpc.py compute-metrics-finetune-sd-128 --ckpt=/home/atuin/b143dc/b143dc11/diffusionmodels/chest/finetune-sd-128/global_step=60000.ckpt
