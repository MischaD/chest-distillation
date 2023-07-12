#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --job-name=TrainFindingLabelsAsImpression
#SBATCH --ntasks-per-node=8
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:8
#SBATCH -C a100_80
#SBATCH --export=NONE


cd $WORK/pycharm/chest-distillation
CKPT_PATH=/home/atuin/b143dc/b143dc11/diffusionmodels/chest/label_as_impression/30k.ckpt
LOG_DIR=/home/atuin/b143dc/b143dc11/pycharm/chest-distillation/log/finding_labels_as_impression
EXPERIMENT_FILE_PATH=src/experiments/label_only_training_cfg_hpc.py
EXPERIMENT_NAME=finding_labels_as_impression

unset SLURM_EXPORT_ENV
export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80

module load python/3.9-anaconda
module load cuda

source activate chest

srun python scripts/train_baseline.py  $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME --save_to=$CKPT_PATH

python scripts/compute_bbox_iou.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME --ckpt=$CKPT_PATH --mask_dir=$LOG_DIR/preliminary_masks --phrase_grounding_mode
