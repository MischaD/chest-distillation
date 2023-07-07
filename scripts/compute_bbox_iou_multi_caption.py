import argparse
import random
import shutil
import pprint
import time
import os
import cv2
import json
import pickle
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import datetime
import torchvision
from tqdm import tqdm
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
import torch
import torch.multiprocessing as mp
from torch import autocast
from contextlib import contextmanager, nullcontext
from src.datasets import get_dataset
from src.foreground_masks import GMMMaskSuggestor
from src.preliminary_masks import preprocess_attention_maps
from src.visualization.utils import word_to_slice
from src.visualization.utils import MIMIC_STRING_TO_ATTENTION
from src.datasets.mscoco import get_classes_for_caption
from src.visualization.utils import model_to_viz
from log import logger, log_experiment
from sklearn.metrics import jaccard_score
from log import logger, log_experiment
from log import formatter as log_formatter
from tqdm import tqdm
import skimage.io as io
import logging
from utils import get_compute_mask_args, make_exp_config, load_model_from_config, collate_batch, img_to_viz, main_setup
from einops import reduce, rearrange, repeat
from pytorch_lightning import seed_everything
from mpl_toolkits.axes_grid1 import ImageGrid
from omegaconf import OmegaConf
from src.ldm.util import instantiate_from_config
from src.ldm.util import AttentionSaveMode
from src.ldm.models.diffusion.plms import PLMSSampler
from src.preliminary_masks import reorder_attention_maps, normalize_attention_map_size
from src.ldm.models.diffusion.ddim import DDIMSampler
from src.evaluation.utils import compute_metrics, compute_prediction_from_binary_mask
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib.patches as patches
from src.datasets.utils import path_to_tensor
from src.ldm.encoders.modules import OpenClipDummyTokenizer
from src.evaluation.utils import samples_to_path_multiquery, contrast_to_noise_ratio, check_mask_exists_multiquery
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from src.datasets.mscoco import MSCOCO_CLASSES, MultiCaptionDataset
from torchvision.transforms import Resize, CenterCrop, Compose


def compute_masks(rank, config, world_size):
    logger.info(f"Current rank: {rank}")
    if config.phrase_grounding_mode:
        config.datasets.test["phrase_grounding"] = True
    else:
        config.datasets.test["phrase_grounding"] = False

    dataset = get_dataset(config, "test")

    model_config = OmegaConf.load(f"{config.config_path}")
    model_config["model"]["params"]["use_ema"] = False
    model_config["model"]["params"]["unet_config"]["params"]["attention_save_mode"] = "cross"
    logger.info(f"Enabling attention save mode")

    model = load_model_from_config(model_config, f"{config.ckpt}")
    device = torch.device(rank) if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    if config.sample.plms:
        logger.info("Always using DDIM sampler due to empirically better results")

    dataset.load_precomputed(model)

    len_raw_dataset = len(dataset)
    dataset = MultiCaptionDataset(dataset)
    logger.info(f"Length of Multi-caption dataset: {len(dataset)}, Length of raw dataset: {len_raw_dataset}")

    seed_everything(time.time())
    precision_scope = autocast

    # visualization args
    rev_diff_steps = 40

    cond_key = config.cond_stage_key if hasattr(config, "cond_stage_key") else "label_text"

    data_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    logger.info(f"Relative path to first sample: {dataset[0]['rel_path']}")

    dataloader = DataLoader(dataset,
                            batch_size=config.sample.iou_batch_size,
                            shuffle=False,
                            num_workers=0,  #opt.num_workers,
                            collate_fn=collate_batch,
                            drop_last=False,
                            sampler=data_sampler,
                            )

    model = model.to(rank)
    STRING_TO_ATTENTION={l:[l.lower(),] for l in MSCOCO_CLASSES} # in case we want to define multiple words as query words we can add them here

    if hasattr(config, "mask_dir"):
        mask_dir = config.mask_dir
    else:
        mask_dir = os.path.join(config.log_dir, "preliminary_masks")
    logger.info(f"Mask dir: {mask_dir}")

    for samples in tqdm(dataloader, "generating masks"):
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    if check_mask_exists_multiquery(mask_dir, samples):
                        logger.info(f"Masks already exists for {samples['rel_path']}")
                        continue
                    model.cond_stage_model = model.cond_stage_model.to(model.device)
                    images = model.log_images(samples, N=len(samples[cond_key]), split="test", sample=False, inpaint=True,
                                                  plot_progressive_rows=False, plot_diffusion_rows=False,
                                                  use_ema_scope=False, cond_key=cond_key, mask=1.,
                                                  save_attention=True)
                    attention_maps = images.pop("attention")
                    attention_images = preprocess_attention_maps(attention_maps, on_cpu=False)

                    for j, attention in enumerate(attention_images):
                        tok_attentions = []
                        txt_label = samples[cond_key][j]
                        token_lens = model.cond_stage_model.compute_word_len(txt_label.split(" "))
                        token_positions = list(np.cumsum(token_lens) + 1)
                        token_positions = [1,] + token_positions

                        # multiple query classes
                        query_classes = samples["query_classes"][j]
                        for query_class in query_classes:
                            query_words = STRING_TO_ATTENTION[query_class]
                            locations = word_to_slice(txt_label.split(" "), query_words)
                            if len(locations) == 0:
                                # use all
                                tok_attention = attention[-1*rev_diff_steps:,:,token_positions[0]:token_positions[-1]]
                                tok_attentions.append(tok_attention.mean(dim=(0,1,2)))
                            else:
                                for location in locations:
                                    tok_attention = attention[-1*rev_diff_steps:,:,token_positions[location]:token_positions[location+1]]
                                    tok_attentions.append(tok_attention.mean(dim=(0,1,2)))

                            preliminary_attention_mask = torch.stack(tok_attentions).mean(dim=(0))
                            path = samples_to_path_multiquery(mask_dir, samples, j, query=query_class)
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                            logger.info(f"(rank({rank}): Saving attention mask to {path}")
                            torch.save(preliminary_attention_mask.to("cpu"), path)


def compute_iou_score(config):
    if config.phrase_grounding_mode:
        config.datasets.test["phrase_grounding"] = True
    else:
        config.datasets.test["phrase_grounding"] = False

    dataset = get_dataset(config, "test")

    model_config = OmegaConf.load(f"{config.config_path}")
    model_config["model"]["params"]["use_ema"] = False
    model_config["model"]["params"]["unet_config"]["params"]["attention_save_mode"] = "cross"
    logger.info(f"Enabling attention save mode")

    model = load_model_from_config(model_config, f"{config.ckpt}")
    dataset.load_precomputed(model)
    dataset = MultiCaptionDataset(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=config.sample.iou_batch_size,
                            shuffle=False,
                            num_workers=0,  #opt.num_workers,
                            collate_fn=collate_batch,
                            drop_last=False,
                            )

    seed_everything(config.seed)
    cond_key = config.cond_stage_key if hasattr(config, "cond_stage_key") else "label_text"

    if hasattr(config, "mask_dir"):
        mask_dir = config.mask_dir
    else:
        mask_dir = os.path.join(config.log_dir, "preliminary_masks")
    logger.info(f"Mask dir: {mask_dir}")

    dataset.add_preliminary_masks(mask_dir, sanity_check=False)
    log_some = 10
    results = {"rel_path":[], "caption":[], "query_word":[], "top1":[], "aucroc": [], "cnr":[]}

    resize_to_imag_size = torchvision.transforms.Resize(512)
    for samples in tqdm(dataloader, "computing metrics"):
        samples["impression"] = samples[cond_key]

        for i in range(len(samples["img"])):
            sample = {k: v[i] for k, v in samples.items()}

            query_classes = samples["query_classes"][i]
            for j in range(len(query_classes)):
                query_word = query_classes[j]
                path_to_pred = samples_to_path_multiquery(mask_dir, samples, i, query=query_word)
                pred = torch.load(path_to_pred) #pred.size() == (64,64)
                gt_segmentation = dataset.get_gt_segmentation(sample, query_word)
                if gt_segmentation is None:
                    results["rel_path"].append(sample["rel_path"])
                    results["query_word"].append(query_word)
                    results["caption"].append(sample["captions"])
                    results["cnr"].append(torch.nan)
                    results["aucroc"].append(torch.nan)
                    results["top1"].append(torch.nan)
                    continue
                gt_segmentation = gt_segmentation.squeeze(dim=0)

                #binary_mask = repeat(mask_suggestor(sample, key="preliminary_mask"), "h w -> 3 h w")
                #binary_mask_large = resize_to_imag_size(binary_mask.float()).round()
                prelim_mask = (pred - pred.min())/(pred.max() - pred.min())
                prelim_mask_large = resize_to_imag_size(prelim_mask.unsqueeze(dim=0)).squeeze(dim=0)

                results["rel_path"].append(sample["rel_path"])
                results["query_word"].append(query_word)
                results["caption"].append(sample["captions"])
                results["cnr"].append(float(contrast_to_noise_ratio(gt_segmentation, prelim_mask_large)))

                argmax_idx = np.unravel_index(prelim_mask_large.argmax(), prelim_mask_large.size())
                mode_is_outlier = gt_segmentation[argmax_idx]
                results["top1"].append(float(mode_is_outlier))

                auc = roc_auc_score(gt_segmentation.flatten(), prelim_mask_large.flatten())
                results["aucroc"].append(auc)
                if log_some > 0:
                    logger.info(f"Logging example bboxes and attention maps to {config.log_dir}")
                    img_name = sample["img_name"]  # 000000212226
                    imgIds = dataset.coco_meta.getImgIds(imgIds=[int(img_name)])
                    img_draw = dataset.coco_meta.loadImgs(imgIds)[0]

                    img_raw = io.imread(img_draw['coco_url'])
                    img_raw = rearrange(torch.tensor(img_raw), "h w c -> c h w")
                    img_raw = Compose([Resize(dataset.dataset.W), CenterCrop(dataset.dataset.W)])(img_raw)


                    ground_truth_img = repeat(gt_segmentation, "h w -> 3 h w").to(torch.float32)
                    prelim_mask_large = repeat(prelim_mask_large, "h w -> 3 h w")

                    fig = plt.figure(figsize=(6, 20))
                    grid = ImageGrid(fig, 111,
                                     nrows_ncols=(4, 1),
                                     axes_pad=0.1)
                    for t, ax, im in zip(np.arange(4), grid, [img_raw, img_raw, prelim_mask_large, ground_truth_img]): # 2nd img is below prelim mask
                        ax.imshow(rearrange(im, "c h w -> h w c"))
                        if t == 1:
                            ax.imshow(prelim_mask_large.mean(axis=0), cmap="jet", alpha=0.25)
                            ax.scatter(argmax_idx[1], argmax_idx[0], s=100, c='red', marker='o')
                        ax.axis('off')

                    path = os.path.join(config.log_dir, "localization_examples", os.path.basename(sample["rel_path"]).rstrip(".png") + f"__{results['caption'][-1]}_"+ f"query_{results['query_word'][-1]}")
                    os.makedirs(path, exist_ok=True)
                    logger.info(f"Logging example images to {path}")
                    plt.title(f"Caption: {sample['captions']} ---- query: {query_word}")
                    plt.savefig(path + "_raw.png", bbox_inches="tight")
                    log_some -= 1

    # drop nan
    df = pd.DataFrame(results)
    logger.info(f"Saving file with results to {mask_dir}")
    df.to_csv(os.path.join(mask_dir, f"mscoco_localization_all.csv"))

    df = df.dropna()
    df = df.groupby(["query_word", "rel_path"]).mean(["top1", "aucroc", "cnr"])  # if one object has multiple descriptions --> aggregate
    df = df.groupby("query_word").mean(numeric_only=True)
    df.to_csv(os.path.join(mask_dir,  f"mscoco_localization_means.csv"))
    with open(os.path.join(mask_dir, f"mscoco_localization_means.json"), "w") as file:
        json_results = {}
        json_results["all"] = dict(df.mean(numeric_only=True))
        for x in df.index:
            json_results[x] = dict(df.loc[x])
        json.dump(json_results, file, indent=4)


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("EXP_NAME", type=str, help="Path to Experiment results")
    parser.add_argument("--ckpt", type=str, default="train")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="dir to save masks in. Default will be inside log dir and should be used!")
    parser.add_argument("--filter_bad_impressions", action="store_true", default=False,
                        help="If set, then we use shortned impressions from mscxr")
    parser.add_argument("--phrase_grounding_mode", action="store_true", default=False,
                        help="If set, then we use shortned impressions from mscxr")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    config = main_setup(args)
    world_size = torch.cuda.device_count()

    # mask computation
    mp.spawn(
        compute_masks,
        args=(config, world_size),
        nprocs=world_size
    )

    compute_iou_score(config)