#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH --job-name=TestVariableFID
#SBATCH --ntasks-per-node=4
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH -C a100_80
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80

FID_REFERENCE_DATASET_EVENLY=/home/atuin/b143dc/b143dc11/data/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/p19_5k_preprocessed_evenly.csv

cd $WORK/pycharm/chest-distillation

model_name="chatgpt_f" # chatgpt_l, label_f, label_l
if [ "$model_name" = "label_l" ]; then
  # LABEL LEARNABLE
  CKPT_PATH=/home/atuin/b143dc/b143dc11/diffusionmodels/chest/label_as_impression/learnable_60k.ckpt
  LOG_DIR=/home/atuin/b143dc/b143dc11/pycharm/chest-distillation/log/finding_labels_as_impression_learnable
  EXPERIMENT_FILE_PATH=src/experiments/label_only_training_cfg_learnable_hpc.py
  EXPERIMENT_NAME=finding_labels_as_impression_learnable
elif [ "$model_name" = "chatgpt_l" ]; then
  # CHATGPT LEARNABLE
  CKPT_PATH=/home/atuin/b143dc/b143dc11/diffusionmodels/chest/chatgpt_as_impression/learnable_60k.ckpt
  LOG_DIR=/home/atuin/b143dc/b143dc11/pycharm/chest-distillation/log/chatgpt_as_impression_learnable
  EXPERIMENT_FILE_PATH=src/experiments/chatgpt_cfg_learnable_hpc.py
  EXPERIMENT_NAME=chatgpt_as_impression_learnable
elif [ "$model_name" = "label_f" ]; then
  # LABEL FROZEN
   CKPT_PATH=/home/atuin/b143dc/b143dc11/diffusionmodels/chest/label_as_impression/frozen_30k.ckpt
   LOG_DIR=/home/atuin/b143dc/b143dc11/pycharm/chest-distillation/log/finding_labels_as_impression
   EXPERIMENT_FILE_PATH=src/experiments/label_only_training_cfg_hpc.py
   EXPERIMENT_NAME=finding_labels_as_impression
elif [ "$model_name" = "chatgpt_f" ]; then
  # CHATGPT FROZEN
    CKPT_PATH=/home/atuin/b143dc/b143dc11/diffusionmodels/chest/chatgpt_as_impression/frozen_30k.ckpt
    LOG_DIR=/home/atuin/b143dc/b143dc11/pycharm/chest-distillation/log/chatgpt_impression
    EXPERIMENT_FILE_PATH=src/experiments/chatgpt_cfg_hpc.py
    EXPERIMENT_NAME=chatgpt_as_impression
fi

module load python/3.9-anaconda
module load cuda

source activate chest

python scripts/sample_model.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME --ckpt=$CKPT_PATH --img_dir=$LOG_DIR/generatedevenly --N=5000 --label_list_path=$FID_REFERENCE_DATASET_EVENLY
python scripts/calc_fid.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME $LOG_DIR/generatedevenly $FID_REFERENCE_DATASET_EVENLY --result_dir=$LOG_DIR/generatedevenly

