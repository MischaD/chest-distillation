#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --job-name=ComputeFID
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
#
# anaconda
source activate chest

#python scripts/sample_model.py experiments/chestxray/train_baseline_reontgen_hpc.py finetuned-sd-synthesis --ckpt=/home/atuin/b143dc/b143dc11/diffusionmodels/chest/finetune-sd-128/global_step=60000.ckpt --use_mscxrlabels
#python scripts/sample_model.py experiments/chestxray/train_baseline_reontgen_hpc.py finetuned-sd-synthesis --ckpt=/home/atuin/b143dc/b143dc11/diffusionmodels/chest/finetune-sd-128/global_step=60000.ckpt
python scripts/calc_xrv_fid.py experiments/chestxray/train_baseline_reontgen_hpc.py compute_fid /home/atuin/b143dc/b143dc11/pycharm/chest-distillation/log/finetuned-sd-synthesis/2023-02-16T00-37-59/generated /home/atuin/b143dc/b143dc11/data/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic_p19_preprocessed.csv
python scripts/calc_xrv_fid.py experiments/chestxray/train_baseline_reontgen_hpc.py compute_fid /home/atuin/b143dc/b143dc11/pycharm/chest-distillation/log/finetuned-sd-synthesis/2023-02-16T08-17-08/generated /home/atuin/b143dc/b143dc11/data/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic_p19_preprocessed.csv
