#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --job-name=Baseline-Learnable-Train-bs256
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=2
#SBATCH --partition=a100
#SBATCH --qos=a100multi
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

srun python scripts/train_baseline.py experiments/chestxray/train_baseline_reontgen_hpc_multinode.py baseline-bs256
