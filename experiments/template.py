from utils import DatasetSplit, get_tok_idx
import os
from src.preliminary_masks import AttentionExtractor

base_dir = "/vol/ideadata/ed52egek/data/fobadiffusion/CUB_200_2011"
out_dir = os.path.join(base_dir, "preliminary_masks/", "compute_preliminary_bird_masks")

rev_diff_steps = 40
num_repeat_each_diffusion_step = 1
prompt = "a photo of a bird"
goal_prompt = "bird" # object we want to detect
foreground_prompt = "a photo of a bird"
background_prompt = "a photo of a background"
attention_extractor = AttentionExtractor("relevant_token_step_mean", tok_idx=5, steps=40)
latent_attention_masks = True

dataset = "bird"
dataset_args = dict(
    base_dir=base_dir,
    split=DatasetSplit("train"),
    limit_dataset=[0, 10],
)

# dataset
C=4 # latent channels
H=512
W=512
f=8

# stable diffusion args
seed=4200
ddim_steps=50
ddim_eta = 0.0 # 0 corresponds to deterministic sampling
fixed_code = True
scale = 1
ckpt = "/vol/ideadata/ed52egek/pycharm/foba/stable-diffusion/sd-v1-4.ckpt"
config = None #TODO
ckpt_ft = None #TODO

#gmm_unet_path = "/vol/ideadata/ed52egek/pycharm/foba/log/ggm-unet-segmentation/checkpoints/last.ckpt"
synthesis_caption_mask = "full"
synthesis_steps=50
exp_name="train_bird_final"

# dataloading
batch_size=1 #unused
num_workers=1#unused

# logging
log_dir = "./log/"