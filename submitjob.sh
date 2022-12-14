#!/bin/bash -l
#SBATCH --partition=a100 --gres=gpu:a100:2 --time=24:00:00
#SBATCH --job-name=UnetRefinedBirds

unset SLURM_EXPORT_ENV
export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80

cd $WORK/pycharm/foba
module load python/3.9-anaconda
conda activate fobaunet
wandb online

python ./scripts/train_segmentation_refined.py experiments/birds/compute_preliminary_bird_masks_train_hpc.py

#  tinyx.nhr.fau.de
#SBATCH --export=NONE


#CUDA_VISIBLE_DEVICES=1 python scripts/compute_attention_masks_raw.py experiments/human36/compute_masks_human.py

# Finetuning
# Create config file
# Change paths in config file
#python main.py finetune-stable-diffusion/main.py -t --base /vol/ideadata/ed52egek/pycharm/foba/experiments/configs/human36_inpainting.yaml --gpus 0,1 --scale_lr False --num_nodes 1 --check_val_every_n_epoch 1 --finetune_from /vol/ideadata/ed52egek/pycharm/foba/stable-diffusion/sd-v1-4-full-ema.ckpt data.params.batch_size=4 lightning.trainer.accumulate_grad_batches=1 data.params.validation.params.n_gpus=2
#/home/saturn/iwai/iwai003h/sd-v1-4-full-ema.ckpt