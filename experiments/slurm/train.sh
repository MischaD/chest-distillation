#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --job-name=TrainStatistical1
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:8
#SBATCH -C a100_80
#SBATCH --export=NONE

MODEL_NUM=1
EXPERIMENT_NAME=statistical$MODEL_NUM
EXPERIMENT_FILE_PATH=src/experiments/default_cfg_hpc.py
CKPT_PATH=/home/atuin/b143dc/b143dc11/diffusionmodels/chest/statistical/$EXPERIMENT_NAME.ckpt

cd $WORK/pycharm/chest-distillation

unset SLURM_EXPORT_ENV
export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80

module load python/3.9-anaconda
module load cuda

source activate chest

python scripts/train_baseline.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME --save_to=$CKPT_PATH
#python scripts/train_baseline.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME --save_to=$CKPT_PATH --cond_stage_trainable

python scripts/compute_bbox_iou.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME --ckpt=$CKPT_PATH --phrase_grounding_mode
python scripts/sample_model.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME --ckpt=$CKPT_PATH --label_list_path=$WORK/data/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/p19_5k_preprocessed_evenly.csv
