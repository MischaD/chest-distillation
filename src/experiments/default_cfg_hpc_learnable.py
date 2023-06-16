from src.experiments.default_cfg_hpc import config


config.cond_stage_trainable = True

config.trainer.max_steps=60001#just to make sure 60k is saved
config.trainer.checkpoint_save_frequency=60000
config.trainer.num_nodes=2