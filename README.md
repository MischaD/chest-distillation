# Source Code for Trade-offs in Fine-tuned Diffusion Models Between Accuracy and Interpretability 

**UPDATE**: The paper got accepted as **oral** to AAAI as part of the Main Conference Safe, Robust, and Responsible AI (SRRAI) track 2024!

# Training

Go to main directory (with ./src) and add the package to the python directory:

    export PYTHONPATH=$PWD
    pip install -e . 

Train baseline model. Requieres a pre-trained [Stable diffusion v2 checkpoint](https://github.com/Stability-AI/stablediffusion)
We use the 512x512 model for our experiments. 

As a preliminary task you have to prepare a directory with the trainings dataset. The folder with the data has to contain a .csv list with the relative path to all files name train2017_meta.csv or mimic_metadata_preprocessed.csv
Examples can be found in *./experiments/train2017_meta.csv*

You need to set your own paths in src/experiments/default_cfg_mscoco.py: 

    config.data_dir # path to .csv containing paths to images and their correspoingng text label
    config.work_dir # path to this repo (where ./src is located)
    config.ckpt # path to Stable diffusion v2 512x512 img geneation ema model


Start the finetuning with: 

    python scripts/train_baseline.py src/experiments/default_cfg_mscoco.py mscoco

# Generative Results 

To sample the model prepare csv file sample.csv 

    python scripts/sample_model.py src/experiments/default_cfg.py sample_baseline --ckpt=path/to/finetuned/model.ckpt --N=10 --label_list_path=experiments/p19_test.csv 

where experiments/p19_test.csv has the same structure as mimic_metadata_preprocessed.csv. 


# Localization 

To reproduce localization results from Table 1 and Table 2 and Table 3 (requires MS_CXR_Local_Alignment_v1.0.0.csv from MS-CXR in data_dir):

## Mimic 

    python scripts/compute_bbox_iou.py src/experiments/default_cfg.py mimic --ckpt=path/to/finetuned/ckpt.ckpt --filter_bad_impressions

add "--filter_bad_impressions" to reproduce results from Table 7.

## MS-COCO 

    python scripts/compute_bbox_iou_multi_caption.py src/experiments/default_cfg_mscoco.py mscoco_singlegpu --ckpt=path/to/ckpt/512-base-ema.ckpt --phrase_grounding_mode --mask_dir=output/save/dir
