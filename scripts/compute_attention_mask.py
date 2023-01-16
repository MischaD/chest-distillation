import argparse
import shutil
import pprint
import time
import os
import pickle
import numpy as np
import pytorch_lightning as pl
import datetime
import torchvision
from tqdm import tqdm
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
import torch
from torch import autocast
from contextlib import contextmanager, nullcontext
from src.datasets import get_dataset
from src.visualization.utils import model_to_viz
from log import logger, log_experiment, file_handler
from utils import get_compute_mask_args, make_exp_config, load_model_from_config, collate_batch, img_to_viz
from einops import reduce, rearrange, repeat
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from src.ldm.util import instantiate_from_config
from src.ldm.util import AttentionSaveMode
from src.ldm.models.diffusion.plms import PLMSSampler
from src.preliminary_masks import reorder_attention_maps, normalize_attention_map_size
from src.ldm.models.diffusion.ddim import DDIMSampler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def get_latent_slice(batch, opt):
    ds_slice = []
    for slice_ in batch["slice"]:
        if slice_.start is None:
            ds_slice.append(slice(None, None, None))
        else:
            ds_slice.append(slice(slice_.start // opt.f, slice_.stop // opt.f, None))
    return tuple(ds_slice)

from torch.utils.data import DataLoader

def add_viz_of_data_and_pred(images, batch, x_samples_ddim, opt):
    # append input
    x0_norm = torch.clamp((batch["x"] + 1.0) / 2.0, min=0.0, max=1.0).cpu()
    x0_norm = reduce(x0_norm, 'b c (h h2) (w w2) -> b c h w', 'mean', h2=opt.f, w2=opt.f)
    images.append(x0_norm)

    # append model output

    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()
    images.append(
        reduce(x_samples_ddim, 'b c (h h2) (w w2) -> b c h w', 'mean', h2=opt.f, w2=opt.f))

    # append gt mask
    images.append(
        reduce(batch["segmentation_x"], 'b c (h h2) (w w2) -> b c h w', 'max', h2=opt.f, w2=opt.f))


def main(opt):
    logger.info(f"=" * 50 + f"Running with prompt: {opt.prompt}" + "="*50)

    dataset = get_dataset(opt, split="train")
    logger.info(f"Length of dataset: {len(dataset)}")

    config = OmegaConf.load(f"{opt.config_path}")
    config["model"]["params"]["use_ema"] = False
    attention_save_mode = config["model"]["params"]["unet_config"]["params"]["attention_save_mode"]
    logger.info(f"enable attention save mode: {attention_save_mode}")

    model = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.out_dir, exist_ok=True)

    start_code = None
    seed_everything(opt.seed)
    if opt.fixed_code:
        start_code = torch.randn([opt.batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast

    # visualization args
    rev_diff_steps = 40
    num_repeat_each_diffusion_step = 1

    start_reverse_diffusion_from_t = int((rev_diff_steps - 1) * (sampler.ddpm_num_timesteps // opt.ddim_steps) + 1)

    logger.info(f"Relative path to first sample: {dataset[0]['rel_path']}")

    print(start_reverse_diffusion_from_t)
    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=0,#opt.num_workers,
                            collate_fn=collate_batch,
                            )


    # for visualization of intermediate results
    log_initial_run = True
    topil = torchvision.transforms.ToPILImage()
    resize_to_imag_size = torchvision.transforms.Resize(512)
    for samples in tqdm(dataloader, "generating masks"):
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():

                    img = resize_to_imag_size(samples["x"].to(torch.float32).to(device))
                    x0_posterior = model.encode_first_stage(img)
                    x0 = model.get_first_stage_encoding(x0_posterior)
                    mask = torch.ones_like(x0)

                    c = model.get_learned_conditioning(opt.batch_size * [opt.prompt, ])
                    logger.info(f"Start reverse diffusion from {start_reverse_diffusion_from_t}")

                    b = len(x0)
                    samples_ddim, intermediates = sampler.sample(
                                                                model=model,
                                                                t=start_reverse_diffusion_from_t,
                                                                repeat_steps=num_repeat_each_diffusion_step,
                                                                S=opt.ddim_steps,
                                                                conditioning=c[:b],
                                                                batch_size=b,
                                                                shape=x0.size()[1:],
                                                                verbose=False,
                                                                unconditional_guidance_scale=opt.scale,
                                                                eta=opt.ddim_eta,
                                                                x_T=start_code[:b],
                                                                mask=mask,
                                                                x0=x0,
                                                                save_attention=True,
                                                            )


                    attention_masks = intermediates["attention"]
                    attention_masks = reorder_attention_maps(attention_masks)
                    attention_masks = normalize_attention_map_size(attention_masks)

                    for i in range(len(attention_masks)):
                        attention_mask = opt.attention_extractor(attention_masks[i])
                        attention_mask = attention_mask.cpu()

                        path = os.path.join(opt.out_dir, samples["rel_path"][i])
                        os.makedirs(os.path.dirname(path), exist_ok=True)

                        # save intermediate attention maps
                        path = os.path.join(opt.out_dir, samples["rel_path"][i]) + ".pt"
                        logger.info(f"Saving attention mask to {path}")
                        torch.save(attention_mask, path)

                    if log_initial_run:
                        log_path = os.path.join(opt.log_dir, os.path.dirname(samples["rel_path"][i]))
                        os.makedirs(log_path, exist_ok=True)
                        log_path = os.path.join(opt.log_dir, samples["rel_path"][i])

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        attention_mask = (attention_mask - attention_mask.min()) / (attention_mask.max() - attention_mask.min())
                        attention_mask = repeat(resize_to_imag_size(rearrange(attention_mask, "1 1 1 h w -> 1 h w")), "1 h w -> c h w", c=3)

                        attention_masks = torch.stack([opt.attention_extractor(attention_masks[i]).squeeze() for i in range(len(attention_masks))]).unsqueeze(dim=1)
                        attention_masks = (attention_masks - attention_masks.min()) / (attention_masks.max() - attention_masks.min())
                        attention_masks = repeat(resize_to_imag_size(attention_masks), "b 1 h w -> b 3 h w")

                        logger.info(f"Saving preliminary results to {log_path}")
                        topil(make_grid(torch.cat([x_samples_ddim.cpu(), attention_masks.cpu()], dim=0), nrow=2)).save(log_path)
                        log_initial_run = False


if __name__ == '__main__':
    args = get_compute_mask_args()
    opt = make_exp_config(args.EXP_PATH)
    for key, value in vars(args).items():
        if value is not None:
            setattr(opt, key, value)
            logger.info(f"Overwriting exp file key {key} with: {value}")

    # make log dir (same as the one for the console log)
    log_dir = os.path.join(os.path.dirname(file_handler.baseFilename))
    setattr(opt, "log_dir", log_dir)
    logger.debug(f"Current file: {__file__}")
    log_experiment(logger, args, opt.config_path)
    main(opt)