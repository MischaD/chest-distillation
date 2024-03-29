from utils import DatasetSplit, get_tok_idx
import os
from src.preliminary_masks import AttentionExtractor
debug = True

root = "/vol/ideadata/ed52egek"
data_dir = os.path.join(root, "data/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/") # data
work_dir = os.path.join(root, "pycharm/chest-distillation") # code, config
ckpt = os.path.join(root, "diffusionmodels/latentdiffusion/512-base-ema.ckpt")

config_path = os.path.join(work_dir, "experiments/chestxray/configs/v2-chest-training.yaml")
config_path_inference = os.path.join(work_dir, "experiments/chestxray/configs/v2-inference.yaml")

latent_attention_masks = False
dataset_args_train = dict(
    dataset="chestxraymimic",
    base_dir=data_dir,
    split=DatasetSplit("train"),
    preload=True,
)
dataset_args_val = dict(
    dataset="chestxraymimicbbox",
    base_dir=data_dir,
    split=DatasetSplit("mscxr"),
    limit_dataset=[0, 64],  # 6d79a86d53fe64e8ea8dca6e81be75b0edfd98c4
    preload=True,
)
dataset_args_testp19 = dict(
    dataset="chestxraymimic",
    base_dir=data_dir,
    split=DatasetSplit("p19"),
    preload=True,
)

dataset_args_test = dict(
    dataset="chestxraymimicbbox",
    base_dir=data_dir,
    split=DatasetSplit("mscxr"),
    preload=True,
    save_original_images=True,
)

# dataset
C=4 # latent channels
H=512
W=512
f=8

# stable diffusion args
seed=4200
ddim_eta = 0.0 # 0 corresponds to deterministic sampling
scale = 4
cond_stage_trainable=False
optimizer_type="adam" # adam or lion
learning_rate=5e-5
ucg_probability=0.0

# dataloading
batch_size=8
num_workers=1

#trainer
max_steps=30001#just to make sure 60k is saved
checkpoint_save_frequency=10000
num_nodes=1
precompute_latent_training_data=True

#sample
n_synth_samples_per_class=625
ddim_steps=75
plms=True