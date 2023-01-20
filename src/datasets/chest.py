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
from log import logger
import scipy.ndimage as ndimage
from utils import DatasetSplit, SPLIT_TO_DATASETSPLIT, DATASETSPLIT_TO_SPLIT


class ChestXray14Dataset(FOBADataset):
    def __init__(self, opt, H, W, mask_dir=None):
        super().__init__(opt, H, W, mask_dir)
        self.load_bboxes = opt.dataset_args.get("load_bboxes", False)
        self._meta_data = None
        self._build_dataset()
        self.opt = opt

    @property
    def meta_data_path(self):
        return os.path.join(self.base_dir, "data_paonly_joined.csv")

    @property
    def meta_data(self):
        if self._meta_data is None:
            logger.info(f"Loading image list from {self.meta_data_path}")
            self._meta_data = pd.read_csv(self.meta_data_path, index_col="idx")
            return self._meta_data
        else:
            return self._meta_data

    def _build_dataset(self):
        data = [dict(rel_path=os.path.join("images", img_path), img_path=os.path.join(self.base_dir, "images", img_path), label=label) for img_path, label in zip(list(self.meta_data.index), list(self.meta_data["Finding Labels"]))]
        splits = self.meta_data["split"].astype(int)
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

        tmp_mask_path = os.path.join(self.base_dir, "refined_mask_tmp", entry["rel_path"] + ".pt")
        if os.path.isfile(tmp_mask_path):
            entry["refined_mask"] = torch.load(tmp_mask_path)
            if max(entry["refined_mask"].size()) > self.W:
                assert False, "reimplement this"

        entry["bbox"] = self.get_bbox(entry["rel_path"], bboxlabel=entry["bboxlabel"])
        return entry

    def get_bbox(self, sample_path):
        return self.meta_data.loc[os.path.basename(sample_path)]


class ChestXray14BboxDataset(ChestXray14Dataset):
    def __init__(self, opt, H, W, mask_dir=None):
        super().__init__(opt, H, W, mask_dir)

    @property
    def meta_data_path(self):
        return os.path.join(self.base_dir, "data_paonly_bboxonly_joined.csv")

    def _build_dataset(self):
        data = [dict(rel_path=os.path.join("images", img_path), img_path=os.path.join(self.base_dir, "images", img_path), label=label, bboxlabel=chest_label) for img_path, label, chest_label in zip(list(self.meta_data.index), list(self.meta_data["Finding Labels"]), list(self.meta_data["Finding Label"]))]
        splits = self.meta_data["split"].astype(int)
        self._get_split(data, splits)

        if self.shuffle:
            np.random.seed(42)
            np.random.shuffle(self.data)

        if self.limit_dataset is not None:
            self.data = self.data[self.limit_dataset[0]:min(self.limit_dataset[1], len(self.data))]

        if self.preload:
            self._load_images(np.arange(len(self)))

    def get_bbox(self, index, bboxlabel):
        meta_data = self.get_meta_data(index, bboxlabel)
        meta_data = meta_data[["bbox_h", "bbox_w", "bbox_x", "bbox_y"]]
        return meta_data

    def get_meta_data(self, index, bboxlabel):
        meta_data = self.meta_data.loc[os.path.basename(index)]
        if isinstance(meta_data, pd.DataFrame):
            meta_data = meta_data[meta_data["Finding Label"] == bboxlabel]
            assert len(meta_data) == 1
            meta_data = meta_data.iloc[0]
        # series
        return meta_data

