#!/bin/bash -l
#SBATCH --time=04:00:00
#SBATCH --job-name=Train
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=a100
#SBATCH -C a100_80
#SBATCH --export=NONE

EXPERIMENT_NAME=language-encoder-oom-test-run
EXPERIMENT_FILE_PATH=experiments/chestxray/train_baseline_reontgen_hpc.py

cd $WORK/pycharm/chest-distillation
unset SLURM_EXPORT_ENV
export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80

module load python/3.9-anaconda
moduel load cuda

source activate chest

srun python scripts/train_baseline.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME
