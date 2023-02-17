import os
import random

import torch
import torchvision.transforms
import numpy as np
from tqdm import tqdm
from log import logger
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode
from src.datasets.utils import path_to_tensor
import pandas as pd
import pathlib
from src.evaluation.xrv_fid import IMAGE_EXTENSIONS



def calc_ms_ssim(imgs):
    msssim = MultiScaleStructuralSimilarityIndexMeasure(gaussian_kernel=True,
                                                        kernel_size=11,
                                                        sigma=1.5)
    msssim.to("cuda")
    scores = []
    for i in range(len(imgs)):
        for j in range(i+1, len(imgs)):
            score = msssim(imgs[i], imgs[j])
            scores.append(score)
    scores = torch.tensor(scores)
    return scores


def calc_ms_ssim_for_path(path, n=4, trials=1, limit_dataset=100):
    logger.info(f"Computing Mean and SDV of MSSSIM with n={n} for path: {path}")
    logger.info(f"Repeating {trials} times.")

    if path.endswith(".csv"):
        df = pd.read_csv(path)
        files = df["path"].to_list()
    else:
        path = pathlib.Path(path)
        files = [os.path.join(path, file) for ext in IMAGE_EXTENSIONS
                 for file in path.rglob('*.{}'.format(ext))]

    files = files[:limit_dataset]
    logger.info(f"Dataset size: {len(files)}")
    imgs = torch.stack([path_to_tensor(x, normalize=False) for x in files])

    scores = []
    for _ in tqdm(range(trials)):
        torch.randperm(imgs.shape[0])
        subset = imgs[:n]
        score = calc_ms_ssim(subset)
        scores.append(score)

    scores = torch.cat(scores)
    filtered_scores = scores[~scores.isnan()]
    if len(filtered_scores)/len(scores) < .90:
        #this happens if the boundary is just black (sometimes happens for real images)
        logger.error("Too many NaN in calculation of MSSSIM - carefully interpret results!")

    mean, sd = filtered_scores.mean(), torch.sqrt(filtered_scores.var())
    logger.info(f"Mean/std: {mean} +- {sd} --> $.{round(float(mean)*100):02d} \pm .{round(float(sd)*100):02d}$")

    return mean, sd


def calc_ms_ssim_for_path_ordered(path, trial_size=4):
    logger.info(f"Computing Mean and SDV of MSSSIM with for path: {path}")

    scores = []
    prompt_dirs = os.listdir(path)
    for prompt_dir in tqdm(prompt_dirs, "Calculating MSSIM"):
        files = [os.path.join(path, prompt_dir, file) for ext in IMAGE_EXTENSIONS
                 for file in pathlib.Path(path).rglob('*.{}'.format(ext))]
        random.shuffle(files)
        files = files[:trial_size]

        imgs = torch.stack([path_to_tensor(x, normalize=False) for x in files])

        score = calc_ms_ssim(imgs.to("cuda"))
        scores.append(score)

    scores = torch.cat(scores)
    filtered_scores = scores[~scores.isnan()]
    if len(filtered_scores)/len(scores) < .90:
        #this happens if the boundary is just black (sometimes happens for real images)
        logger.error("Too many NaN in calculation of MSSSIM - carefully interpret results!")

    mean, sd = filtered_scores.mean(), filtered_scores.std()
    logger.info(f"Mean/std: {mean} +- {sd} --> $.{round(float(mean)*100):02d} \pm .{round(float(sd)*100):02d}$")

    return mean, sd
