from src.experiments.default_cfg_hpc import config


config.cond_stage_trainable = True

config.trainer.max_steps=30001#just to make sure 60k is saved
config.trainer.checkpoint_save_frequency=10000
config.trainer.num_nodes=2
config.ckpt = "/home/atuin/b143dc/b143dc11/diffusionmodels/chest/statistical/statistical_learnable_2_3Ok.ckpt" #continued run