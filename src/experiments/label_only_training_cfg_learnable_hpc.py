from src.experiments.default_cfg_hpc_learnable import config

config.datasets.train.text_label_key = "finding_labels"

config.ckpt = "/home/atuin/b143dc/b143dc11/diffusionmodels/chest/label_as_impression/learnable_30k.ckpt"