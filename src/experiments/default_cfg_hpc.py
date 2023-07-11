from .default_cfg import config


#config.root = "/home/atuin/b143dc/b143dc11"
config.root = "/home/atuin/b180dc/b180dc10"

config.datasets.train.limit_dataset = None

config.datasets.test.limit_dataset = None

config.dataloading.batch_size = 16

config.trainer.max_steps=30001#just to make sure 60k is saved
config.trainer.checkpoint_save_frequency=30000
