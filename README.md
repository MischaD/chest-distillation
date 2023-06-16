# Source Code for Pay Attention: Accuracy 


## Execution

Define path where you want to save the checkpoint: 

    CKPT_PATH=$WORK/mimic_frozen_finetuned.ckpt

Specify the path to the training dataset in `src/experiments/default_cfg.py`.

Start training of the frozen model (change to learnable by setting the *--cond_stage_trainable* flag):

    python scripts/train_baseline.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME --save_to=$CKPT_PATH

Sample model: 

    python scripts/sample_model.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME --ckpt=$CKPT_PATH --label_list_path=$WORK/data/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/p19_5k_preprocessed_evenly.csv

Compute zero shot segmentation bbox iou:

    python scripts/compute_bbox_iou.py $EXPERIMENT_FILE_PATH $EXPERIMENT_NAME --ckpt=$CKPT_PATH --phrase_grounding_mode



## If you want to use our code or cite this project please use: 

    @article{dombrowski2023pay,
      title={Pay Attention: Accuracy Versus Interpretability Trade-off in Fine-tuned Diffusion Models},
      author={Dombrowski, Mischa and Reynaud, Hadrien and M{\"u}ller, Johanna P and Baugh, Matthew and Kainz, Bernhard},
      journal={arXiv preprint arXiv:2303.17908},
      year={2023}
    }