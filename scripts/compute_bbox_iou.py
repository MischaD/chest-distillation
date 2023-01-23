import argparse
import shutil
import pprint
import time
import os
import cv2
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
from torch import autocast
from contextlib import contextmanager, nullcontext
from src.datasets import get_dataset
from src.foreground_masks import GMMMaskSuggestor
from src.visualization.utils import model_to_viz
from log import logger, log_experiment, file_handler
from sklearn.metrics import jaccard_score
from utils import get_compute_mask_args, make_exp_config, load_model_from_config, collate_batch, img_to_viz
from einops import reduce, rearrange, repeat
from pytorch_lightning import seed_everything
from mpl_toolkits.axes_grid1 import ImageGrid
from omegaconf import OmegaConf
from src.ldm.util import instantiate_from_config
from src.ldm.util import AttentionSaveMode
from src.ldm.models.diffusion.plms import PLMSSampler
from src.preliminary_masks import reorder_attention_maps, normalize_attention_map_size
from src.ldm.models.diffusion.ddim import DDIMSampler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib.patches as patches


def get_latent_slice(batch, opt):
    ds_slice = []
    for slice_ in batch["slice"]:
        if slice_.start is None:
            ds_slice.append(slice(None, None, None))
        else:
            ds_slice.append(slice(slice_.start // opt.f, slice_.stop // opt.f, None))
    return tuple(ds_slice)

def apply_rect(img, x, y, h, w, color="red"):
    img = (img * 255).to(torch.uint8).numpy()
    img = rearrange(img, "c h w -> h w c")
    if color == "red":
        color = (255, 0, 0)
    elif color == "blue":
        color = (0, 0, 255)

    img = cv2.rectangle(img.copy(), [x, y], [x + h, y + w], color, 3)
    img = rearrange(img, "h w c -> c h w") / 255.
    return torch.tensor(img)


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


def check_mask_exists(opt, samples):
    for i in range(len(samples["rel_path"])):
        path = os.path.join(opt.out_dir, samples["rel_path"][i] + ".pt")
        if not os.path.exists(path):
            return False
    return True


def main(opt):
    logger.info(f"=" * 50 + f"Running with prompt: {opt.prompt}" + "="*50)

    dataset = get_dataset(opt, opt.split)
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
                    if check_mask_exists(opt, samples):
                        logger.info(f"Masks already exists for {samples['rel_path']}")
                        continue

                    else:
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

    dataset.add_preliminary_masks(opt.out_dir)
    mask_suggestor = GMMMaskSuggestor(opt)
    log_some = 10
    results = {"rel_path":[], "Finding Label":[], "iou":[], "miou":[], "distance":[]}

    for samples in tqdm(dataloader, "computing metrics"):
        for i in range(len(samples["x"])):
            sample = {k: v[i] for k, v in samples.items()}
            resize_to_input_size = torchvision.transforms.Resize(1024)
            bbox = int(sample["bbox"]["bbox_x"]), int(sample["bbox"]["bbox_y"]), int(sample["bbox"]["bbox_w"]), int(sample["bbox"]["bbox_h"])
            x, y, h, w = tuple(bbox)

            binary_mask = repeat(mask_suggestor(sample, key="preliminary_mask"), "h w -> 3 h w")
            binary_mask_large = resize_to_input_size(binary_mask)

            bbox_img = torch.zeros(img.size()[1:])
            bbox_img[x:(x + w), y: (y + h)] = 1
            iou, miou, distance, bboxpred = compute_metrics(bbox, binary_mask_large.float().mean(axis=0))
            results["rel_path"].append(sample["rel_path"])
            results["iou"].append(float(iou))
            results["miou"].append(float(miou))
            results["distance"].append(float(distance))
            results["Finding Label"].append(sample["bboxlabel"])

            if log_some > 0:
                logger.info(f"Logging example bboxes and attention maps to {opt.log_dir}")
                img = (sample["img"] + 1) / 2

                prelim_mask = sample["preliminary_mask"]
                prelim_mask_large = resize_to_input_size(repeat(prelim_mask.squeeze(), "h w -> 3 h w"))
                prelim_mask_large = (prelim_mask_large - prelim_mask_large.min())/(prelim_mask_large.max() - prelim_mask_large.min())
                prelim_mask_large = resize_to_input_size(prelim_mask_large)

                binary_mask_large = apply_rect(binary_mask_large, x, y, h, w)
                binary_mask_large = apply_rect(binary_mask_large, bboxpred[0], bboxpred[2], (bboxpred[1] - bboxpred[0]), (bboxpred[3] - bboxpred[2]), color="blue")

                fig = plt.figure(figsize=(6, 20))
                grid = ImageGrid(fig, 111,
                                 nrows_ncols=(3, 1),
                                 axes_pad=0.1)
                for j, ax, im in zip(np.arange(3), grid, [img, prelim_mask_large, binary_mask_large]):
                    ax.imshow(rearrange(im, "c h w -> h w c"))
                    if j == 1:
                        ax.imshow(prelim_mask_large.mean(axis=0), cmap="jet", alpha=0.3)
                    ax.axis('off')

                path = os.path.join(opt.log_dir, os.path.basename(sample["rel_path"]).rstrip(".png") + f"_{sample['bboxlabel']}")
                logger.info(f"Logging to {path}")
                plt.savefig(path + "_raw.png", bbox_inches="tight")
                fig.suptitle(f"IoU: {float(iou):.3}, mIoU:{miou:.3}, distance (pixel): {distance:.3f}\n Red bbox is gt, blue is prediction")
                plt.savefig(path + "_detailed.png", bbox_inches="tight")
                log_some -= 1



    df = pd.DataFrame(results)
    logger.info(f"Saving file with results to {opt.log_dir}")
    df.to_csv(os.path.join(opt.log_dir, "bbox_results.csv"))


def compute_prediction_from_binary_mask(binary_prediction):
    binary_prediction = binary_prediction.to(torch.bool).numpy()
    horizontal_indicies = np.where(np.any(binary_prediction, axis=0))[0]
    vertical_indicies = np.where(np.any(binary_prediction, axis=1))[0]
    x1, x2 = horizontal_indicies[[0, -1]]
    y1, y2 = vertical_indicies[[0, -1]]
    prediction = np.zeros_like(binary_prediction)
    prediction[y1:y2, x1:x2] = 1
    center_of_mass = [x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2]
    return prediction, center_of_mass, (x1, x2, y1, y2)


def compute_metrics(bbox, binary_prediction):
    x, y, h, w = bbox
    ground_truth_bbox_img = torch.zeros_like(binary_prediction)
    ground_truth_bbox_img[x:(x + w), y: (y + h)] = 1

    prediction, center_of_mass_prediction, bbox_pred = compute_prediction_from_binary_mask(binary_prediction)

    iou = torch.tensor(jaccard_score(ground_truth_bbox_img.flatten(), prediction.flatten()))
    iou_rev = torch.tensor(jaccard_score(1 - ground_truth_bbox_img.flatten(), 1 - prediction.flatten()))

    center_of_mass = [x + w / 2,
                      y + h / 2]
    miou = (iou + iou_rev)/2

    distance = np.sqrt((center_of_mass[0] - center_of_mass_prediction[0])**2 +
                       (center_of_mass[1] - center_of_mass_prediction[1])**2
                      )
    return iou, miou, distance, bbox_pred


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