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


def calc_ms_ssim(imgs):
    msssim = MultiScaleStructuralSimilarityIndexMeasure(gaussian_kernel=True,
                                                        kernel_size=11,
                                                        sigma=1.5)
    scores = []
    for i in range(len(imgs)):
        for j in range(i+1, len(imgs)):
            score = msssim(imgs[i], imgs[j])
            scores.append(score)
    scores = torch.tensor(scores)
    return scores


def calc_ms_ssim_for_path(path, n=4, trials=1):
    logger.info(f"Computing Mean and SDV of MSSSIM with n={n} for path: {path}")
    logger.info(f"Repeating {trials} times.")

    scores = []
    for _ in tqdm(range(trials)):
        img_list_real = os.listdir(path)
        random.shuffle(img_list_real)
        img_list_real = img_list_real[:n]

        imgs_real = torch.stack([path_to_tensor(os.path.join(path, x), normalize=False).to("cuda") for x in img_list_real])

        score = calc_ms_ssim(imgs_real)
        scores.append(score)
        print(len(scores))

    scores = torch.cat(scores)
    filtered_scores = scores[~scores.isnan()]
    if len(filtered_scores)/len(scores) < .90:
        #this happens if the boundary is just black (sometimes happens for real images)
        logger.error("Too many NaN in calculation of MSSSIM - carefully interpret results!")

    mean, sd = filtered_scores.mean(), torch.sqrt(filtered_scores.var())
    logger.info(f"Mean/std: {mean} +- {sd} --> $.{round(float(mean)*100):02d} \pm .{round(float(sd)*100):02d}$")

    return mean, sd
