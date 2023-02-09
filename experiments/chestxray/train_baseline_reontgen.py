from utils import DatasetSplit, get_tok_idx
import os
from src.preliminary_masks import AttentionExtractor

debug = True

root = "/vol/ideadata/ed52egek"
data_dir = os.path.join(root, "data/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/") # data
work_dir = os.path.join(root, "pycharm/chest-distillation") # code, config
ckpt = os.path.join(root, "diffusionmodels/latentdiffusion/512-base-ema.ckpt")
ckpt_ft = os.path.join(root, "diffusionmodels/models_finetuned/chest/chest_finetuned.ckpt")

config_path = os.path.join(work_dir, "experiments/chestxray/configs/v2-chest-training.yaml")
out_dir = os.path.join(data_dir, "preliminary_masks/", "chestxrayofpleuraleffusion")

latent_attention_masks = False
dataset_args_train = dict(
    dataset="chestxraymimic",
    base_dir=data_dir,
    split=DatasetSplit("train"),
    #all 8b308d1ff146fc994156bb7f50775f99891bdd33
    #limit_dataset=[0, 10],#c0a08655ac43528158bef787cbfa549c447665dfb
    preload=True,
)
dataset_args_val = dict(
    dataset="chestxraymimicbbox",
    base_dir=data_dir,
    split=DatasetSplit("mscxr"),
    limit_dataset=[0, 4], #213851912adf554689226fff69183d41d96f6d44
    #limit_dataset=[0, 10], #c0a08655ac43528158bef787cbfa549c447665df
    preload=True,
)

# dataset
C=4 # latent channels
H=512
W=512
f=8

# stable diffusion args
seed=4200
ddim_steps=75
ddim_eta = 0.0 # 0 corresponds to deterministic sampling
fixed_code = True
scale = 4
synthesis_steps = 75

# dataloading
batch_size=4
num_workers=1

#trainer
max_steps=60001#just to make sure 60k is saved
checkpoint_save_frequency=10000

precompute_latent_training_data=True
