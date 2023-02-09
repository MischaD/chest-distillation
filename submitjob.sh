#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --job-name=SDv2_Baseline
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --export=NONE
#SBATCH -C a100_80

cd $WORK/pycharm/chest-distillation

unset SLURM_EXPORT_ENV
export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80

#
#load required modules (compiler, ...)
module load python/3.9-anaconda
module load cudnn/8.2.4.15-11.4
module load cuda/11.4.2
#
# anaconda
source activate chest

#timeout 23h python scripts/txt2img.py experiments/chestxray/generate_sdv2_baseline_hpc.py --from_file="./experiments/chestxray/prompts/chestxraytest.txt" --out_dir=output/sd_unfinetuned_baseline_4p0 --n_samples=5000 --scale=4

#python -m pytorch_fid output/sd_unfinetuned_baseline_4p0/samplesa_photo_of_a_chest_xray/ /home/atuin/b143dc/b143dc11/data/fobadiffusion/chestxray14/test_images/
#python scripts/calc_ms_ssim.py output/sd_unfinetuned_baseline_4p0/samplesa_photo_of_a_chest_xray/ /home/atuin/b143dc/b143dc11/data/fobadiffusion/chestxray14/test_images/
python scripts/calc_xrv_fid.py /home/atuin/b143dc/b143dc11/data/fobadiffusion/chestxray14/test_images/ output/sd_unfinetuned_baseline_4p0/samplesa_photo_of_a_chest_xray/

# restart slurm script after 24h
if [[ $? -eq 124 ]]; then
  sbatch submitjob.sh
fi

#rsync -r . alexp:/home/atuin/b143dc/b143dc11/pycharm/chest-distillation