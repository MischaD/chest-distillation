import argparse
import os
import pandas as pd
if os.path.basename(os.path.abspath(".")) == "notebooks":
    os.chdir("..")
from utils import main_setup, AttributeDict
from omegaconf import OmegaConf
from torch import autocast
from src.datasets import get_dataset
from utils import load_model_from_config, collate_batch
from src.visualization.utils import word_to_slice
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from src.preliminary_masks import preprocess_attention_maps
from src.visualization.utils import MIMIC_STRING_TO_ATTENTION
import numpy as np
import torchvision
from src.foreground_masks import GMMMaskSuggestor
from einops import repeat


def main(args):
    phrase_grounding_mode = args.phrase_grounding_mode
    model_name = args.model_name
    print(f"Model Name: {model_name} -- PGM: {phrase_grounding_mode}")

    args = AttributeDict(EXP_PATH="src/experiments/default_cfg.py", EXP_NAME="confounder_notebook_learnable")
    config = main_setup(args)
    config.datasets.test["phrase_grounding"] = phrase_grounding_mode

    MODEL_MAP = {"sdv2":"/vol/ideadata/ed52egek/diffusionmodels/latentdiffusion/512-base-ema.ckpt",
                 "frozen":"/vol/ideadata/ed52egek/diffusionmodels/chest/miccai_models/frozen_30k.ckpt",
                 "learnable":"/vol/ideadata/ed52egek/diffusionmodels/chest/miccai_models/learnable_60k.ckpt"}

    config.ckpt = MODEL_MAP[model_name]
    config.log_dir = os.path.dirname(config.log_dir)
    print(f"Saving to {config.log_dir}")
    delattr(config.datasets.test, "limit_dataset")
    dataset = get_dataset(config, "test")
    model_config = OmegaConf.load(f"{config.config_path}")
    model_config["model"]["params"]["use_ema"] = False
    model_config["model"]["params"]["unet_config"]["params"]["attention_save_mode"] = "cross"

    model = load_model_from_config(model_config, f"{config.ckpt}")
    dataset.load_precomputed(model)
    dataset[0].keys()
    cond_key = "label_text"


    def samples_to_path(mask_dir, samples, j):
        sample_path = samples["rel_path"][j]
        label = samples["finding_labels"][j]
        impr = samples["impression"][j].replace(" ", "_")
        path = os.path.join(mask_dir, sample_path + label + impr) + ".pt"
        return path


    idx = 0
    precision_scope = autocast

    # visualization args
    rev_diff_steps = 40

    model = model.to("cuda")
    mask_dir =  os.path.join(config.log_dir, "precomputed_masks")


    def contrast_to_noise_ratio(ground_truth_img, prelim_mask_large):
        gt_mask = ground_truth_img.flatten()
        pr_mask = prelim_mask_large.flatten()

        roi_values = pr_mask[gt_mask == 1.0]
        not_roi_values = pr_mask[gt_mask != 1.0]

        contrast = roi_values.mean() - not_roi_values.mean()
        noise = torch.sqrt(
            roi_values.var() + not_roi_values.var()
        )
        cnr = contrast / noise
        return cnr

    resize_to_imag_size = torchvision.transforms.Resize(512)
    mask_suggestor = GMMMaskSuggestor(config)

    results = {"rel_path":[], "word":[], "finding_labels":[], "cnr":[]}
    results_positional = {"rel_path":[], "position":[], "finding_labels":[], "cnr":[]}

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for i, sample in enumerate(dataset):
                    print(sample["label_text"])
                    label_text = [sample["label_text"].split("|")[0],]
                    sample["impression"] = label_text
                    sample["label_text"] = label_text
                    model.cond_stage_model = model.cond_stage_model.to(model.device)
                    images = model.log_images(sample, N=1, split="test", sample=False, inpaint=True,
                                                   plot_progressive_rows=False, plot_diffusion_rows=False,
                                                   use_ema_scope=False, cond_key=cond_key, mask=1.,
                                                   save_attention=True)
                    attention_maps = images.pop("attention")
                    attention_images = preprocess_attention_maps(attention_maps, on_cpu=False)
                    attention = attention_images[0]

                    bboxes = sample["bboxxywh"].split("|")
                    bbox_meta = dataset.bbox_meta_data.loc[sample["dicom_id"]]
                    img_size = [bbox_meta["image_width"], bbox_meta["image_height"]]
                    for j in range(len(bboxes)):
                        bbox = [int(box) for box in bboxes[j].split("-")]
                        bboxes[j] = bbox
                    ground_truth_img = sample["bbox_img"].float()

                    token_lens = model.cond_stage_model.compute_word_len(label_text[0].split(" "))
                    token_positions = list(np.cumsum(token_lens) + 1)
                    token_positions = [1,] + token_positions
                    words = ["<SOS>",] + label_text[0].split(" ") + ["<EOS>",]
                    attention = attention[-1 * rev_diff_steps:].mean(dim=(0,1))
                    for j, word in enumerate(words):
                        if j == 0:
                            attention_map = attention[0:1]
                        elif j == (len(words) - 1):
                            attention_map = attention[token_positions[-1]:token_positions[-1]+1]
                        else:
                            attention_map = attention[token_positions[j-1]:token_positions[j-1]+1]
                        attention_map = attention_map.mean(dim=0)

                        prelim_mask = (attention_map - attention_map.min())/(attention_map.max() - attention_map.min())

                        prelim_mask_large = resize_to_imag_size(prelim_mask.unsqueeze(dim=0)).squeeze(dim=0)
                        cnr = contrast_to_noise_ratio(ground_truth_img, prelim_mask_large)

                        results["rel_path"].append(sample["rel_path"])
                        results["word"].append(word)
                        results["finding_labels"].append(sample["finding_labels"])
                        results["cnr"].append(cnr.cpu())

                    for j in range(len(attention)):
                        attention_map = attention[j]
                        prelim_mask = (attention_map - attention_map.min())/(attention_map.max() - attention_map.min())

                        prelim_mask_large = resize_to_imag_size(prelim_mask.unsqueeze(dim=0)).squeeze(dim=0)
                        cnr = contrast_to_noise_ratio(ground_truth_img, prelim_mask_large)

                        results_positional["rel_path"].append(sample["rel_path"])
                        results_positional["position"].append(j)
                        results_positional["finding_labels"].append(sample["finding_labels"])
                        results_positional["cnr"].append(cnr.cpu())

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(config.log_dir, f"{model_name}_{phrase_grounding_mode}_results.csv"))
    res_positional = pd.DataFrame(results_positional)
    res_positional.to_csv(os.path.join(config.log_dir, f"{model_name}_{phrase_grounding_mode}_results_positional.csv"))


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--phrase_grounding_mode", action="store_true", default=False,
                        help="If set, then we use shortned impressions from mscxr")
    parser.add_argument("--model_name", type=str, default="sdv2")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)