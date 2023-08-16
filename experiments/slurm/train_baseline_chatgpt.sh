#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --job-name=ChatGPTAsImpressionLearnableCont
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=2
#SBATCH --partition=a100
#SBATCH --qos=a100multi
#SBATCH --gres=gpu:a100:8
#SBATCH -C a100_80
#SBATCH --export=NONE

cd $WORK/pycharm/chest-distillation
CKPT_PATH=/home/atuin/b143dc/b143dc11/diffusionmodels/chest/chatgpt_as_impression/learnable_60k.ckpt
LOG_DIR=/home/atuin/b143dc/b143dc11/pycharm/chest-distillation/log/chatgpt_as_impression_learnable
EXPERIMENT_FILE_PATH=src/experiments/chatgpt_cfg_learnable_hpc.py
EXPERIMENT_NAME=chatgpt_as_impression_learnable

unset SLURM_EXPORT_ENV
export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80

module load python/3.9-anaconda
module load cuda

source activate chest

srun python scripts/train_baseline.py  $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME --save_to=$CKPT_PATH --cond_stage_trainable

#python scripts/compute_bbox_iou.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME --ckpt=$CKPT_PATH --mask_dir=$LOG_DIR/preliminary_masks --phrase_grounding_mode
