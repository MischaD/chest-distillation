from utils import DatasetSplit, get_tok_idx
import os
from src.preliminary_masks import AttentionExtractor

root = "/vol/ideadata/ed52egek"
data_dir = os.path.join(root, "data/chestxray14") # data
work_dir = os.path.join(root, "pycharm/chest-distillation") # code, config
ckpt = os.path.join(root, "diffusionmodels/latentdiffusion/v2-1_512-ema-pruned.ckpt")
ckpt_ft = os.path.join(root, "diffusionmodels/models_finetuned/chest/chest_finetuned.ckpt")

config_path = os.path.join(work_dir, "experiments/chestxray/configs/v2-inference.yaml")
out_dir = os.path.join(data_dir, "preliminary_masks/", "chestxrayofpleuraleffusion")

#prompt = "final report examination chest mass"
prompt = "a photo of a bird"
foreground_prompt = "final report examination chest mass"
background_prompt = "final report examination no finding"
attention_extractor = AttentionExtractor("multi_relevant_token_step_mean", tok_idx=[5], steps=40)
latent_attention_masks = True

dataset = "chestxray14bbox"
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
    limit_dataset=[0, 10000],
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

synthesis_caption_mask = "full"
synthesis_steps=50
exp_name="train_bird_final"
