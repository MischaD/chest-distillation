#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --job-name=TrainTrueTrue77
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=a100
#SBATCH -C a100_80
#SBATCH --export=NONE

EXPERIMENT_NAME=mlf-invariancetrue-singlehealthytrue-77
EXPERIMENT_FILE_PATH=experiments/chestxray/train_multilabel_hpc_tt_77.py

cd $WORK/pycharm/chest-distillation

unset SLURM_EXPORT_ENV
export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80

module load python/3.9-anaconda
module load cuda

source activate chest

srun python scripts/train_baseline.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME
