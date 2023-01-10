import os
import random
import torch
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
from utils import DatasetSplit
from random import shuffle
from src.datasets.utils import file_to_list, resize, path_to_tensor
from src.datasets.dataset import FOBADataset
from einops import rearrange, repeat
import scipy.ndimage as ndimage
from utils import DatasetSplit, SPLIT_TO_DATASETSPLIT, DATASETSPLIT_TO_SPLIT


class ChestXrayDataset(FOBADataset):
    def __init__(self, opt, H, W, mask_dir=None):
        super().__init__(opt, H, W, mask_dir)
        self.load_bboxes = opt.dataset_args.get("load_bboxes", False)
        self._build_dataset()
        self.opt = opt

    def _build_dataset(self):
        image_paths, splits = [], []
        for entry in file_to_list(os.path.join(self.base_dir, "images.txt")):
            entry = entry.strip().split(" ")
            image_paths.append(entry[0])
            splits.append(entry[1])

        self.meta_data = pd.read_csv(os.path.join(self.base_dir, "data_paonly_joined.csv"))
        self.meta_data = self.meta_data[self.meta_data["split"] == DATASETSPLIT_TO_SPLIT[self.split.name]]

        data = [dict(rel_path=img_path, img_path=os.path.join(self.base_dir, img_path), label=label) for img_path, label in zip(image_paths, self.meta_data["Finding Labels"])]
        #data = [x for x in data if "".join(x["label"]).find("Mass") != -1] #[x for x in data if "".join(x["label"]).find("Mass")]

        self._get_split(data, splits)

        if self.shuffle:
            np.random.seed(42)
            np.random.shuffle(self.data)

        if self.limit_dataset is not None:
            self.data = self.data[self.limit_dataset[0]:min(self.limit_dataset[1], len(self.data))]

        if self.preload:
            self._load_images(np.arange(len(self)))


    def _load_images(self, index):
        assert len(index)
        entry = self.data[index[0]].copy()
        img = path_to_tensor(entry["img_path"])

        # images too large are resized to self.W^2
        if max(img.size()) > self.W:
            img = resize(img, tosize=self.W)
        entry["img"] = img

        x = torch.full((1, 3, self.H, self.W), -1.)
        x[0, :, :img.size()[2], :img.size()[3]] = img

        entry["x"] = x
        entry["slice"] = (slice(None), slice(None), slice(0, img.size()[2]), slice(0, img.size()[3]))

        entry["prompt"] = "final report examination chest " + entry["label"].replace("|", " ")

        if self._preliminary_masks_path is not None:
            entry["preliminary_mask"] = torch.load(os.path.join(self._preliminary_masks_path, entry["rel_path"] + ".pt"))
            if not self.latent_attention_mask:
                entry["preliminary_mask"] = repeat(entry["preliminary_mask"], "1 1 c h w -> 1 1 c (h h2) (w w2)", h2=self.opt.f, w2=self.opt.f)

        if self._inpainted_images_path is not None:
            entry["inpainted_image"] = torch.load(os.path.join(self._inpainted_images_path, entry["rel_path"] + ".pt"))

        tmp_mask_path = os.path.join(self.opt.base_dir, "refined_mask_tmp", entry["rel_path"] + ".pt")
        if os.path.isfile(tmp_mask_path):
            entry["refined_mask"] = torch.load(tmp_mask_path)
            if max(entry["refined_mask"].size()) > self.W:
                assert False, "reimplement this"

        return entry


