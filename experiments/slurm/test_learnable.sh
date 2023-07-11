#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --job-name=TestLearnablePGMOff
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH -C a100_80
#SBATCH --export=NONE

EXPERIMENT_NAME=pgm_off
EXPERIMENT_FILE_PATH=src/experiments/default_cfg_hpc_learnable.py
CKPT_PATH=/home/atuin/b180dc/b180dc10/diffusionmodels/payattention/learnable_60k.ckpt

cd $WORK/pycharm/chest-distillation

unset SLURM_EXPORT_ENV
export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80

module load python/3.9-anaconda
module load cuda

source activate chest

python scripts/compute_bbox_iou.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME --ckpt=$CKPT_PATH #--phrase_grounding_mode
