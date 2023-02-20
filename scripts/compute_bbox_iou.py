import argparse
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
from torch import autocast
from contextlib import contextmanager, nullcontext
from src.datasets import get_dataset
from src.foreground_masks import GMMMaskSuggestor
from src.preliminary_masks import preprocess_attention_maps
from src.visualization.utils import word_to_slice
from src.visualization.utils import MIMIC_STRING_TO_ATTENTION
from src.visualization.utils import model_to_viz
from log import logger, log_experiment
from sklearn.metrics import jaccard_score
from log import logger, log_experiment
from log import formatter as log_formatter
from tqdm import tqdm
import logging
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
from src.evaluation.utils import compute_metrics, compute_prediction_from_binary_mask
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib.patches as patches
from src.datasets.utils import path_to_tensor


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


def contrast_to_noise_ratio(ground_truth_img, prelim_mask_large):
    gt_mask = ground_truth_img.flatten()
    pr_mask = prelim_mask_large.flatten()

    roi_values = pr_mask[gt_mask == 1.0]
    not_roi_values = pr_mask[gt_mask != 1.0]

    contrast = roi_values.mean() - not_roi_values.mean()
    noise = torch.sqrt(
        roi_values.var() / 2 + not_roi_values.var() / 2
    )
    cnr = contrast / noise
    return cnr


def check_mask_exists(mask_dir, samples):
    for i in range(len(samples["rel_path"])):
        path = os.path.join(mask_dir, samples["rel_path"][i] + ".pt")
        if not os.path.exists(path):
            return False
    return True


def main(opt):
    dataset = get_dataset(opt, "test")
    logger.info(f"Length of dataset: {len(dataset)}")

    config = OmegaConf.load(f"{opt.config_path}")
    config["model"]["params"]["use_ema"] = False
    config["model"]["params"]["unet_config"]["params"]["attention_save_mode"] = "cross"
    logger.info(f"Enabling attention save mode")

    model = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    dataset.apply_filter_for_disease_in_txt()
    dataset.load_precomputed(model)

    seed_everything(opt.seed)
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


    if hasattr(opt, "mask_dir"):
        mask_dir = opt.mask_dir
    else:
        mask_dir = os.path.join(opt.log_dir, "preliminary_masks")
    logger.info(f"Mask dir: {mask_dir}")

    for samples in tqdm(dataloader, "generating masks"):
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    if check_mask_exists(mask_dir, samples):
                        logger.info(f"Masks already exists for {samples['rel_path']}")
                        continue
                    #img = model.log_images(samples, cond_key="label_text", unconditional_guidance_scale=1.0, inpaint=False)
                    images = model.log_images(samples, split="test", sample=False, inpaint=True,
                                                  plot_progressive_rows=False, plot_diffusion_rows=False,
                                                  use_ema_scope=False, cond_key="label_text", mask=1.,
                                                  save_attention=True)
                    attention_maps = images.pop("attention")
                    attention_images = preprocess_attention_maps(attention_maps, on_cpu=False)

                    for j, attention in enumerate(attention_images):
                        txt_label = samples["label_text"][j]
                        # determine tokenization
                        txt_label = txt_label.split("|")[0]  # we filter cases with different text labels, all are the same thanks for filtering
                        token_lens = model.cond_stage_model.compute_word_len(txt_label.split(" "))
                        token_positions = list(np.cumsum(token_lens) + 1)
                        token_positions = [1,] + token_positions
                        label = samples["finding_labels"][j]
                        query_words = MIMIC_STRING_TO_ATTENTION[label]

                        locations = word_to_slice(txt_label.split(" "), query_words)
                        assert len(locations) >= 1, f"{samples['dicom_id'][j]}"

                        tok_attentions = []
                        for location in locations:
                            tok_attention = attention[-1*rev_diff_steps:,:,token_positions[location]:token_positions[location+1]]
                            tok_attentions.append(tok_attention.mean(dim=(0,1,2)))
                        preliminary_attention_mask = torch.stack(tok_attentions).mean(dim=(0))

                        path = os.path.join(mask_dir, samples["rel_path"][j])+ ".pt"
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        logger.info(f"Saving attention mask to {path}")
                        torch.save(preliminary_attention_mask.to("cpu"), path)


    topil = torchvision.transforms.ToPILImage()
    resize_to_imag_size = torchvision.transforms.Resize(512)
    resize_to_latent_size = torchvision.transforms.Resize(64)

    dataset.add_preliminary_masks(mask_dir)
    mask_suggestor = GMMMaskSuggestor(opt)
    log_some = 1e5
    results = {"rel_path":[], "finding_labels":[], "iou":[], "miou":[], "bboxiou":[], "bboxmiou":[], "distance":[], "top1":[], "aucroc": [], "cnr":[]}

    for samples in tqdm(dataloader, "computing metrics"):
        #z, c, x, xrec = model.get_input(samples, "img", cond_key="label_text", bs=len(samples["rel_path"]),
        #                                    return_first_stage_outputs=True)

        for i in range(len(samples["img"])):
            sample = {k: v[i] for k, v in samples.items()}
            dataset.add_preliminary_to_sample(sample)
            bboxes = sample["bboxxywh"].split("|")
            bbox_meta = dataset.bbox_meta_data.loc[sample["dicom_id"]]
            img_size = [bbox_meta["image_width"], bbox_meta["image_height"]]
            #bbox_img = torch.zeros(img_size)
            for i in range(len(bboxes)):
                bbox = [int(box) for box in bboxes[i].split("-")]
                bboxes[i] = bbox
            #    x, y, w, h = tuple(bbox)
            #    bbox_img[x:(x + w), y: (y + h)] = 1

            ground_truth_img = sample["bbox_img"].float()

            #reconstructed_large = xrec[i].clamp(-1, 1)
            #reconstructed_large = (reconstructed_large + 1)/2
            #reconstructed = resize_to_latent_size(reconstructed_large)

            binary_mask = repeat(mask_suggestor(sample, key="preliminary_mask"), "h w -> 3 h w")
            binary_mask_large = resize_to_imag_size(binary_mask.float()).round()

            prelim_mask = (sample["preliminary_mask"] - sample["preliminary_mask"].min())/(sample["preliminary_mask"].max() - sample["preliminary_mask"].min())
            prelim_mask_large = resize_to_imag_size(prelim_mask.unsqueeze(dim=0)).squeeze(dim=0)



            results["rel_path"].append(sample["rel_path"])
            results["finding_labels"].append(sample["finding_labels"])
            results["cnr"].append(float(contrast_to_noise_ratio(ground_truth_img, prelim_mask_large)))
            prediction, center_of_mass_prediction, bbox_gmm_pred = compute_prediction_from_binary_mask(binary_mask_large[0])
            iou = torch.tensor(jaccard_score(ground_truth_img.flatten(), binary_mask_large[0].flatten()))
            iou_rev = torch.tensor(jaccard_score(1 - ground_truth_img.flatten(), 1 - binary_mask_large[0].flatten()))
            results["iou"].append(float(iou))
            results["miou"].append(float((iou + iou_rev)/2))

            bboxiou = torch.tensor(jaccard_score(ground_truth_img.flatten(), prediction.flatten()))
            bboxiou_rev = torch.tensor(jaccard_score(1 - ground_truth_img.flatten(), 1 - prediction.flatten()))
            results["bboxiou"].append(float(bboxiou))
            results["bboxmiou"].append(float((bboxiou + bboxiou_rev)/2))

            if len(bboxes) > 1:
                results["distance"].append(np.nan)
            else:
                _, center_of_mass, _ = compute_prediction_from_binary_mask(ground_truth_img)
                distance = np.sqrt((center_of_mass[0] - center_of_mass_prediction[0]) ** 2 +
                                   (center_of_mass[1] - center_of_mass_prediction[1]) ** 2
                                   )
                results["distance"].append(float(distance))


            argmax_idx = np.unravel_index(prelim_mask_large.argmax(), prelim_mask_large.size())
            mode_is_outlier = ground_truth_img[argmax_idx]
            results["top1"].append(float(mode_is_outlier))

            auc = roc_auc_score(ground_truth_img.flatten(), prelim_mask_large.flatten())
            results["aucroc"].append(auc)

            if log_some > 0:
                logger.info(f"Logging example bboxes and attention maps to {opt.log_dir}")
                img = (sample["img_raw"] + 1) / 2

                ground_truth_img = repeat(ground_truth_img, "h w -> 3 h w")
                prelim_mask_large = repeat(prelim_mask_large, "h w -> 3 h w")

                #for bbox in bboxes:
                #    x, y, w, h = tuple(bbox)
                #    img = apply_rect(img, x, y, h, w)
                #    prelim_mask_large = apply_rect(prelim_mask_large, x, y, h, w)
                #    binary_mask_large = apply_rect(binary_mask_large, x, y, h, w)

                #binary_mask_large = apply_rect(binary_mask_large, bbox_gmm_pred[0], bbox_gmm_pred[2], (bbox_gmm_pred[1] - bbox_gmm_pred[0]), (bbox_gmm_pred[3] - bbox_gmm_pred[2]), color="blue")

                fig = plt.figure(figsize=(6, 20))
                grid = ImageGrid(fig, 111,
                                 nrows_ncols=(4, 1),
                                 axes_pad=0.1)
                for j, ax, im in zip(np.arange(4), grid, [img, img, binary_mask_large, ground_truth_img]): # 2nd img is below prelim mask
                    ax.imshow(rearrange(im, "c h w -> h w c"))
                    if j == 1:
                        ax.imshow(prelim_mask_large.mean(axis=0), cmap="jet", alpha=0.25)
                        ax.scatter(argmax_idx[1], argmax_idx[0], s=100, c='red', marker='o')
                    ax.axis('off')

                path = os.path.join(opt.log_dir, "localization_examples", os.path.basename(sample["rel_path"]).rstrip(".png") + f"_{sample['finding_labels']}")
                os.makedirs(path)
                logger.info(f"Logging to {path}")
                plt.savefig(path + "_raw.png", bbox_inches="tight")
                #fig.suptitle(f"IoU: {float(iou):.3}, mIoU:{miou:.3}, distance (pixel): {distance:.3f}\n Red bbox is gt, blue is prediction")
                #plt.savefig(path + "_detailed.png", bbox_inches="tight")
                log_some -= 1

    df = pd.DataFrame(results)
    logger.info(f"Saving file with results to {opt.log_dir}")
    df.to_csv(os.path.join(opt.log_dir, "bbox_results.csv"))
    mean_results = df.groupby("finding_labels").mean(numeric_only=True)
    mean_results.to_csv(os.path.join(opt.log_dir, "bbox_results_means.csv"))
    logger.info(df.mean())
    logger.info(df.groupby("finding_labels").mean(numeric_only=True))

    with open(os.path.join(opt.log_dir, "bbox_results.json"), "w") as file:
        json_results = {}
        json_results["all"] = dict(df.mean(numeric_only=True))
        for x in mean_results.index:
            json_results[x] = dict(mean_results.loc[x])

        json.dump(json_results, file, indent=4)



if __name__ == '__main__':
    args = get_compute_mask_args()
    log_dir = os.path.join(os.path.abspath("."), "log", args.EXP_NAME, datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'console.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.debug("="*30 + f"Running {os.path.basename(__file__)}" + "="*30)
    logger.debug(f"Logging to {log_dir}")

    opt = make_exp_config(args.EXP_PATH)
    for key, value in vars(args).items():
        if value is not None:
            setattr(opt, key, value)
            logger.info(f"Overwriting exp file key {key} with: {value}")

    # make log dir (same as the one for the console log)
    log_dir = os.path.join(os.path.dirname(file_handler.baseFilename))
    setattr(opt, "log_dir", log_dir)
    logger.info(f"Log dir: {log_dir}")
    logger.debug(f"Current file: {__file__}")
    log_experiment(logger, args, opt.config_path)
    main(opt)
