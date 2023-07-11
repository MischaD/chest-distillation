#!/bin/bash -l

conda activate $CONDA/chest
cd /vol/ideadata/ed52egek/pycharm/chest-distillation

export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=0,1,2,3
export COCO_TEST_PATH=/vol/ideadata/ed52egek/data/mscoco/val2017_meta.csv

#python scripts/compute_bbox_iou_multi_caption.py src/experiments/default_cfg_mscoco.py mscoco_singlegpu --ckpt=/vol/ideadata/ed52egek/diffusionmodels/latentdiffusion/512-base-ema.ckpt --phrase_grounding_mode --mask_dir=/vol/ideadata/ed52egek/pycharm/chest-distillation/log/mscoco_singlegpu/bbox-"sdv2"

#export SAMPLE_PATH=/vol/ideadata/ed52egek/pycharm/chest-distillation/log/mscoco_singlegpu/samples-sdv2
#python scripts/sample_model.py src/experiments/default_cfg_mscoco.py mscoco_singlegpu --ckpt=/vol/ideadata/ed52egek/diffusionmodels/latentdiffusion/512-base-ema.ckpt --img_dir=$SAMPLE_PATH
#python scripts/calc_fid.py     src/experiments/default_cfg_mscoco.py mscoco_singlegpu $SAMPLE_PATH $COCO_TEST_PATH --result_dir=$SAMPLE_PATH


#MODES=("learnable" "frozen")
MODES=("frozen")
for NUM in 15 #2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
  for MODE in "${MODES[@]}"
  do
    if [ "$MODE" = "frozen" ]
    then
      export CKPT_PATH=/vol/ideadata/ed52egek/pycharm/chest-distillation/log/mscoco_singlegpu/2023-06-20T16-35-03/checkpoints/global_step="$NUM"0000.ckpt
    else
      export CKPT_PATH=/vol/ideadata/ed52egek/pycharm/chest-distillation/log/mscoco_singlegpu/2023-06-23T08-24-06/checkpoints/global_step="$NUM"0000.ckpt
    fi

    # compute_bbox
    python scripts/compute_bbox_iou_multi_caption.py src/experiments/default_cfg_mscoco.py mscoco_singlegpu --ckpt=$CKPT_PATH --phrase_grounding_mode --mask_dir=/vol/ideadata/ed52egek/pycharm/chest-distillation/log/mscoco_singlegpu/bbox-"$MODE"-"$NUM"

    # 4568 images will be generated due to caption constraints
    export SAMPLE_PATH=/vol/ideadata/ed52egek/pycharm/chest-distillation/log/mscoco_singlegpu/samples-"$MODE"-"$NUM"

    # sample model
    python scripts/sample_model.py src/experiments/default_cfg_mscoco.py mscoco_singlegpu --ckpt=$CKPT_PATH --img_dir=$SAMPLE_PATH
    # calculate fid  -results will be in SAMPLE_PATH
    python scripts/calc_fid.py     src/experiments/default_cfg_mscoco.py mscoco_singlegpu $SAMPLE_PATH $COCO_TEST_PATH --result_dir=$SAMPLE_PATH

  done
done




