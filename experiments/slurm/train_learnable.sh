#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --job-name=Baseline-Learnable-2-cont
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=2
#SBATCH --partition=a100
#SBATCH --qos=a100multi
#SBATCH --gres=gpu:a100:8
#SBATCH -C a100_80
#SBATCH --export=NONE

MODEL_NUM=2
EXPERIMENT_NAME=statistical_learnable_$MODEL_NUM
EXPERIMENT_FILE_PATH=src/experiments/default_cfg_hpc_learnable.py
CKPT_PATH=/home/atuin/b143dc/b143dc11/diffusionmodels/chest/statistical/$EXPERIMENT_NAME.ckpt

cd $WORK/pycharm/chest-distillation

unset SLURM_EXPORT_ENV
export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80

module load python/3.9-anaconda
module load cuda

source activate chest

srun python scripts/train_baseline.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME --save_to=$CKPT_PATH --cond_stage_trainable