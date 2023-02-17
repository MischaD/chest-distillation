#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --job-name=Test
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH -C a100_80
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80

module load python/3.9-anaconda
moduel load cuda
source activate chest

cd $WORK/pycharm/chest-distillation

EXPERIMENT_NAME='finetune-sd-128'
LOG_DIR_TIMESTAMP='2023-02-16T00-37-59'
CHECKPOINT_FILENAME='global_step=10000.ckpt' # rest determined automatically for your own safety
EXPERIMENT_FILE_PATH='experiments/chestxray/train_baseline_reontgen_hpc.py'

# ======================================================================================================================

FID_REFERENCE_DATASET=/home/atuin/b143dc/b143dc11/data/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic_p19_preprocessed.csv
LOG_DIR=/home/atuin/b143dc/b143dc11/pycharm/chest-distillation/log/$EXPERIMENT_NAME/$LOG_DIR_TIMESTAMP
CHECKPOINT_PATH=$LOG_DIR/checkpoints/$CHECKPOINT_FILENAME

srun python scripts/train_baseline.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME

#discriminative
python scripts/compute_bbox_iou.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME --ckpt=$CHECKPOINT_PATH --mask_dir=$LOG_DIR/preliminary_masks

#genrative
python scripts/sample_model.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME --ckpt=$CHECKPOINT_PATH --use_mscxrlabels
python scripts/calc_fid.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME $LOG_DIR/generated $FID_REFERENCE_DATASET

