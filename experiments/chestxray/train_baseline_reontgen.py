from utils import DatasetSplit, get_tok_idx
import os
from src.preliminary_masks import AttentionExtractor

root = "/vol/ideadata/ed52egek"
data_dir = os.path.join(root, "data/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/") # data
work_dir = os.path.join(root, "pycharm/chest-distillation") # code, config
ckpt = os.path.join(root, "diffusionmodels/latentdiffusion/v2-1_512-ema-pruned.ckpt")
ckpt_ft = os.path.join(root, "diffusionmodels/models_finetuned/chest/chest_finetuned.ckpt")

config_path = os.path.join(work_dir, "experiments/chestxray/configs/v2-inference.yaml")
out_dir = os.path.join(data_dir, "preliminary_masks/", "chestxrayofpleuraleffusion")

latent_attention_masks = False
dataset = "chestxraymimic"
dataset_args = dict() # will be overwritten during execution to contain our split
dataset_args_train = dict(
    base_dir=data_dir,
    split=DatasetSplit("train"),
    limit_dataset=[0, 10],
)
dataset_args_val = dict(
    base_dir=data_dir,
    split=DatasetSplit("val"),
    limit_dataset=[0, 10],
)
dataset_args_test = dict(
    base_dir=data_dir,
    split=DatasetSplit("test"),
    limit_dataset=[0, 10],
)

# dataset
C=4 # latent channels
H=1024
W=1024
f=16

# stable diffusion args
seed=4200
ddim_steps=50
ddim_eta = 0.0 # 0 corresponds to deterministic sampling
fixed_code = True
scale = 4
synthesis_steps = 75

# dataloading
batch_size=4
num_workers=1
