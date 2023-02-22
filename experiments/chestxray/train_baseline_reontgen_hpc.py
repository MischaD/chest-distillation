from utils import DatasetSplit, get_tok_idx
import os
from src.preliminary_masks import AttentionExtractor

debug = True

root = "/home/atuin/b143dc/b143dc11"
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
    #all 8b308d1ff146fc994156bb7f50775f99891bdd33
    #limit_dataset=[0, 10],#c0a08655ac43528158bef787cbfa549c447665dfb
    limit_dataset=[0, 100],
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
    #0-1133 10d6f749d36ca86d83cdd19bca06a7e9d52a08b5
    #limit_dataset=[0, 12],
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
cond_stage_trainable=True
optimizer_type="adam" # adam or lion
learning_rate=5e-5

# dataloading
batch_size=16
num_workers=1

#trainer
max_steps=60001#just to make sure 60k is saved
checkpoint_save_frequency=10000
num_nodes=1
precompute_latent_training_data=True

#sample
n_synth_samples_per_class=625
ddim_steps=50
plms=False