import ml_collections
import torch
import os
from utils import DatasetSplit


config = ml_collections.ConfigDict()
config.root = "/home/atuin/b143dc/b143dc11"
config.data_dir = os.path.join(config.root, "data/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/") # data
config.work_dir = os.path.join(config.root, "pycharm/chest-distillation") # code, config
config.ckpt = os.path.join(config.root, "diffusionmodels/latentdiffusion/512-base-ema.ckpt")
config.config_path = os.path.join(config.work_dir, "experiments/chestxray/configs/v2-chest-training.yaml")
config.config_path_inference = os.path.join(config.work_dir, "experiments/chestxray/configs/v2-inference.yaml")
config.latent_attention_masks = False
config.cond_stage_trainable = False
config.seed=4200

# datasets
config.datasets = datasets = ml_collections.ConfigDict()
config.datasets.C = 4
config.datasets.H = 512
config.datasets.W = 512
config.datasets.f = 8

config.datasets.train = ml_collections.ConfigDict()
config.datasets.train.dataset = "chestxraymimic"
config.datasets.train.base_dir = config.data_dir
config.datasets.train.split = DatasetSplit("train")
#config.datasets.train.limit_dataset = [0, 64]  # 6d79a86d53fe64e8ea8dca6e81be75b0edfd98c4
config.datasets.train.preload = True

config.datasets.val = ml_collections.ConfigDict()
config.datasets.val.dataset = "chestxraymimic"
config.datasets.val.base_dir = config.data_dir
config.datasets.val.split = DatasetSplit("mscxr")
config.datasets.val.limit_dataset = [0, 64]  # 6d79a86d53fe64e8ea8dca6e81be75b0edfd98c4
config.datasets.val.preload = True

config.datasets.test = ml_collections.ConfigDict()
config.datasets.test.dataset = "chestxraymimicbbox"
config.datasets.test.base_dir = config.data_dir
config.datasets.test.split = DatasetSplit("mscxr")
#config.datasets.test.limit_dataset=[0, 32]
config.datasets.test.preload = True
config.datasets.test.save_original_images = True

config.datasets.testp19 = ml_collections.ConfigDict()
config.datasets.testp19.dataset = "chestxraymimic"
config.datasets.testp19.base_dir = config.data_dir
config.datasets.testp19.split = DatasetSplit("p19")
config.datasets.testp19.preload = True

# stable diffusion args
config.stable_diffusion = ml_collections.ConfigDict()
config.stable_diffusion.ddim_eta = 0.0 # 0 corresponds to deterministic sampling
config.stable_diffusion.scale = 4
config.stable_diffusion.optimizer_type="adam" # adam or lion
config.stable_diffusion.learning_rate=5e-5
config.stable_diffusion.ucg_probability=0.3

# dataloading
config.dataloading = ml_collections.ConfigDict()
config.dataloading.batch_size = 16
config.dataloading.num_workers = 1

# trainer
config.trainer = ml_collections.ConfigDict()
config.trainer.max_steps=30001#just to make sure 60k is saved
config.trainer.checkpoint_save_frequency=30000
config.trainer.num_nodes=1
config.trainer.precompute_latent_training_data=True

# sample
config.sample = ml_collections.ConfigDict()
config.sample.n_synth_samples_per_class=625
config.sample.ddim_steps=75
config.sample.plms=True
config.sample.iou_batch_size=8