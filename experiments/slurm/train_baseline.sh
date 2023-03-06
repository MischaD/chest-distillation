#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --job-name=TrainReduced
#SBATCH --ntasks-per-node=8
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:8
#SBATCH -C a100_80
#SBATCH --export=NONE


cd $WORK/pycharm/chest-distillation

unset SLURM_EXPORT_ENV
export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80

module load python/3.9-anaconda
moduel load cuda

source activate chest

srun python scripts/train_baseline.py experiments/chestxray/train_baseline_reontgen_hpc.py pa_only
