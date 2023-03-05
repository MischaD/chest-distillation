import random
from log import logger, log_experiment
import os
import json
import logging
import pandas as pd
import torch
from utils import get_sample_model_args, make_exp_config, load_model_from_config, collate_batch, img_to_viz, get_classification_args
from log import formatter as log_formatter
from src.datasets import get_dataset
import datetime
from src.ldm.util import instantiate_from_config
from src.ldm.models.diffusion.ddim import DDIMSampler
from src.ldm.models.diffusion.plms import PLMSSampler
from omegaconf import OmegaConf
from torch import autocast
from tqdm import tqdm
from PIL import Image
from pytorch_lightning import seed_everything
import time
import random
import numpy as np
from einops import rearrange
from src.datasets.sample_dataset import get_mscxr_synth_dataset, get_mscxr_synth_dataset_size_n
from torchvision.transforms import Compose, CenterCrop, Resize
import torchxrayvision as xrv
from pathlib import Path
import skimage
import torchvision
from sklearn.metrics import roc_auc_score


def main(opt):
    if opt.IMG_PATH.endswith(".csv"):
        imagdf = pd.read_csv(opt.IMG_PATH)
        images = list(imagdf.path)
        labels = list(imagdf["Finding Labels"])
    else:
        path = Path(opt.IMG_PATH)
        images = [os.path.join(opt.IMG_PATH, file) for ext in ["jpg", "png"]
                 for file in path.rglob('*.{}'.format(ext))]
        labels = [os.path.basename(os.path.dirname(image)) for image in images]

    #random.shuffle(images)
    #random.shuffle(labels)
    #images = images[:100]
    #labels = labels[:100]

    model = xrv.models.DenseNet(weights="densenet121-res224-all").to("cuda")
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])

    results = []
    for image_path, label in tqdm(zip(images, labels), "classify images with xrv", total=len(labels)):
        img = skimage.io.imread(image_path)
        img = xrv.datasets.normalize(img, 255)  # convert 8-bit image to [-1024, 1024] range
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = img.mean(2)[None, ...]  # Make single color channel
        else:
            img = img[None, ...]

        img = transform(img)
        img = torch.from_numpy(img).to("cuda")

        outputs = model(img[None, ...])

        preds = dict(zip(model.pathologies,outputs[0].detach().cpu().numpy()))
        preds["image"] = image_path
        preds["label"] = label
        results.append(preds)

    final_table_cols = ["label", "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Lung Opacity", "Effusion", "Pneumonia", "Pneumothorax", "No Finding"]
    results = pd.DataFrame(results).set_index("image")
    results = results.drop(columns=[col for col in results if col not in final_table_cols])
    val_cols = results.drop(columns="label")
    log_path = opt.IMG_PATH if not opt.IMG_PATH.endswith(".csv") else os.path.dirname(opt.IMG_PATH)
    results.to_csv(os.path.join(log_path, "classfication_results.csv"))

    results["idxmax"] = val_cols.idxmax(axis="columns")
    #results["idxmaxvalue"] = val_cols.max(axis="columns")
    results["idxmax"] = results["idxmax"].map(lambda x: x.replace("Effusion", "Plural Effusion"))
    results["top1"] = results["idxmax"] == results["label"]

    top1score = dict(results.groupby("label")["top1"].mean())
    top1score["all"] = results["top1"].sum()
    top1score = {k: (float(v)) for k,v in top1score.items()}

    with open(os.path.join(log_path, "classifier_acc.json"), "w") as file:
        json.dump(top1score, file, indent=4)
    # aucroc
    # avg confidence if highest
    roc_auc_scores = {}
    results["label"] = results["label"].map(lambda x: x.replace("Pleural Effusion", "Effusion"))
    for disease in final_table_cols:
        if disease == "label" or disease == "No Finding":
            continue
        healthy_scores = results[results["label"] == "No Finding"][disease]
        unhealty_scores = results[results["label"] == disease][disease]
        clf_labels = np.zeros(len(healthy_scores) + len(unhealty_scores))
        clf_labels[len(healthy_scores):] = 1
        scores = np.concatenate([np.array(healthy_scores), np.array(unhealty_scores)])
        roc_auc = roc_auc_score(clf_labels, scores)
        roc_auc_scores[disease] = roc_auc

    with open(os.path.join(log_path, "classifier_roc_auc.json"), "w") as file:
        json.dump(roc_auc_scores, file, indent=4)




if __name__ == '__main__':
    args = get_classification_args()
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