#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --job-name=TestArm1
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH -C a100_80
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80

module load python/3.9-anaconda
module load cuda
source activate chest

cd $WORK/pycharm/chest-distillation

EXPERIMENT_NAME='mlf-invariancefalse-singlehealthytrue-arm1'
LOG_DIR_TIMESTAMP='2023-02-27T10-00-29'
CHECKPOINT_FILENAME='global_step=30000.ckpt' # rest determined automatically for your own safety
EXPERIMENT_FILE_PATH='experiments/chestxray/train_multilabel_hpc_ft_arm1.py'
# TODO DOUBLE CHECK If this is a baseline run - if so --> do not use mscxr-labels

# ======================================================================================================================

#FID_REFERENCE_DATASET=/home/atuin/b143dc/b143dc11/data/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/p19_5k_preprocessed.csv
FID_REFERENCE_DATASET_EVENLY=/home/atuin/b143dc/b143dc11/data/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/p19_5k_preprocessed_evenly.csv
LOG_DIR=/home/atuin/b143dc/b143dc11/pycharm/chest-distillation/log/$EXPERIMENT_NAME/$LOG_DIR_TIMESTAMP
CHECKPOINT_PATH=$LOG_DIR/checkpoints/$CHECKPOINT_FILENAME

#discriminative
#python scripts/compute_bbox_iou.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME --ckpt=$CHECKPOINT_PATH --mask_dir=$LOG_DIR/preliminary_masks

python scripts/compute_bbox_iou.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME --ckpt=$CHECKPOINT_PATH --mask_dir=$LOG_DIR/preliminary_masks_filtered --filter_bad_impressions
python scripts/compute_bbox_iou.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME --ckpt=$CHECKPOINT_PATH --mask_dir=$LOG_DIR/preliminary_masks_phraseground --phrase_grounding_mode

python scripts/sample_model.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME --ckpt=$CHECKPOINT_PATH --img_dir=$LOG_DIR/generatedevenly --N=5000 --label_list_path=$FID_REFERENCE_DATASET_EVENLY

python scripts/calc_fid.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME $LOG_DIR/generatedevenly $FID_REFERENCE_DATASET_EVENLY --result_dir=$LOG_DIR/generatedevenly
python scripts/calc_fid.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME $LOG_DIR/generatedevenly $FID_REFERENCE_DATASET_EVENLY --result_dir=$LOG_DIR/generatedevenlymscxr

python scripts/calc_ms_ssim.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME --ckpt=$CHECKPOINT_PATH --n_sample_sets=100 --trial_size=4 --img_dir=$LOG_DIR/ms_ssim_mscxr
python scripts/classify_chest_xray.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME $FID_REFERENCE_DATASET_EVENLY
