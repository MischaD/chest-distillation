#!/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH --job-name=SDv2_Baseline_batchsize16
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH -C a100_80
#SBATCH --export=NONE

cd $WORK/pycharm/chest-distillation

unset SLURM_EXPORT_ENV
export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80

#
#load required modules (compiler, ...)
module load python/3.9-anaconda
moduel load cuda

# anaconda
source activate chest

python scripts/train_baseline.py experiments/chestxray/train_baseline_reontgen_hpc.py finetune-sd
