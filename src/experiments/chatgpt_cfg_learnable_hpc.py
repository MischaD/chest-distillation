from src.experiments.default_cfg_hpc_learnable import config

config.datasets.train.text_label_key = "chatgpt"

config.ckpt = "/home/atuin/b143dc/b143dc11/diffusionmodels/chest/chatgpt_as_impression/learnable_30k.ckpt"