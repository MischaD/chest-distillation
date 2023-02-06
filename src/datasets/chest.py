import os
import random
import torch
import cv2
import hashlib
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
from utils import DatasetSplit
from random import shuffle
from src.datasets.utils import file_to_list, resize, path_to_tensor
from torchvision.transforms import Resize, CenterCrop, Compose
from src.datasets.dataset import FOBADataset
from einops import rearrange, repeat
from log import logger
import scipy.ndimage as ndimage
from utils import DatasetSplit, SPLIT_TO_DATASETSPLIT, DATASETSPLIT_TO_SPLIT
from tqdm import tqdm
from time import time
import pickle


class ChestXray14Dataset(FOBADataset):
    def __init__(self, dataset_args, opt):
        super().__init__(dataset_args, opt)
        self.load_bboxes = dataset_args.get("load_bboxes", False)
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
    def __init__(self, dataset_args, opt):
        super().__init__(dataset_args, opt)

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

        self.data = [self._load_images(i) for i in np.arange(len(self))]

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


class MimicCXRDataset(FOBADataset):
    def __init__(self, dataset_args, opt):
        super().__init__(dataset_args, opt)
        self._meta_data = None
        self._build_dataset()
        self.opt = opt
        self._precomputed_path = None

    @property
    def precomputed_path(self):
        if self._precomputed_path is None:
            name = "".join([x["rel_path"] for x in self.data])
            name = hashlib.sha1(name.encode("utf-8")).hexdigest()
            precompute_path = os.path.join(self.base_dir, str(name))
            self._precomputed_path = precompute_path
        return self._precomputed_path

    @property
    def is_precomputed(self):
        return os.path.isdir(self.precomputed_path)

    def load_precomputed(self, model):
        if not self.is_precomputed:
            self.precompute(model)

        logger.info(f"Using precomputed dataset with name: {self.precomputed_path}")
        entries = pickle.load(open(os.path.join(self.precomputed_path, "entries.pkl"), "rb"))
        dir_list = os.listdir(self.precomputed_path)
        for file in dir_list:
            if not file.endswith(".pt"):
                continue
            tensor_key = os.path.basename(file.rstrip(".pt"))
            entries[tensor_key] = torch.load(os.path.join(self.precomputed_path, file))

        self._data = []
        for i in range(len(entries["rel_path"])):
            self._data.append({k: entries[k][i] for k in entries.keys()})

    def compute_latent(self, img, model):
        """
        Preprocoessing. Img is already 512x512 tensor 1xCx512x512 --> compute latent using vqvae - saves Gaussian parameters
        """
        img = img.to("cuda")
        encoder_posterior = model.encode_first_stage(img)
        encoder_posterior = encoder_posterior.parameters.detach().cpu()
        return encoder_posterior

    def sample_latent(self, encoder_posterior, scale_factor):
        from src.ldm.modules.distributions.distributions import DiagonalGaussianDistribution
        z = DiagonalGaussianDistribution(encoder_posterior).sample()
        return z * scale_factor

    def decode_from_latent(self, encoder_posterior, model):
        """
        Helper function to decode latent space of vqvae
        """
        n, c, h, w = encoder_posterior.size()
        assert encoder_posterior.ndim == 4 and n == 1
        old_device = encoder_posterior.device
        encoder_posterior = encoder_posterior.to("cuda")

        if c == 8:
            # params for latent gaussian
            z = self.sample_latent(encoder_posterior, model.scale_factor)
        elif c == 4:
            # sampled values
            z = encoder_posterior
        else:
            raise ValueError(f"Unable to interpret encoder_posterior of shape: {encoder_posterior.size()}")
        img = model.decode_first_stage(z).detach()
        img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
        return img.to(old_device)

    def precompute(self, model):
        #load entries
        entry = self._load_images([0])
        keys = entry.keys()
        entries = {key: [] for key in keys}
        for i in tqdm(range(len(self)), "Precomputing Dataset"):
            entry = self._load_images([i])
            for k in keys:
                entries[k].append(entry[k])

            # preprocess --> 1 x 8 x 64 x 64 diag gaussian latent
            entries["img"][i] = self.compute_latent(entry["img"], model)

        # save entries
        entry_keys = list(entries.keys())
        data_tensors = {}
        for key in entry_keys:
            if isinstance(entries[key][0], torch.Tensor):
                data_tensors[key] = torch.stack(entries.pop(key))

        path = self.precomputed_path
        logger.info(f"Saving precomputed dataset to: {path}")
        os.makedirs(path)
        pickle.dump(entries, open(os.path.join(path, "entries.pkl"), "wb"))
        for key in data_tensors.keys():
            torch.save(data_tensors[key], os.path.join(path, f"{key}.pt"))

    @property
    def meta_data_path(self):
        return os.path.join(self.base_dir, "mimic_metadata_preprocessed.csv")

    @property
    def meta_data(self):
        if self._meta_data is None:
            logger.info(f"Loading image list from {self.meta_data_path}")
            self._meta_data = pd.read_csv(self.meta_data_path, index_col="dicom_id")
            return self._meta_data
        else:
            return self._meta_data

    def _build_dataset(self):
        data = [dict(rel_path=os.path.join(img_path.replace(".dcm", ".jpg")), finding_labels=labels) for img_path, labels in zip(list(self.meta_data.path), list(self.meta_data["Finding Labels"]))]
        splits = self.meta_data["split"].astype(int)
        self._get_split(data, splits)

        if self.shuffle:
            np.random.shuffle(self.data)

        if self.limit_dataset is not None:
            self.data = self.data[self.limit_dataset[0]:min(self.limit_dataset[1], len(self.data))]

    def _load_image(self, img_path):
        img = path_to_tensor(img_path)
        # images too large are resized to self.W^2 using center cropping
        if max(img.size()) > self.W:
            transforms = Compose([Resize(self.W), CenterCrop(self.W)])
            img = transforms(img)
        return img

    def _load_images(self, index):
        assert len(index)
        entry = self.data[index[0]].copy()
        entry["dicom_id"] = os.path.basename(entry["rel_path"]).rstrip(".jpg")
        img_path = os.path.join(self.base_dir, entry["rel_path"].replace(".dcm", ".jpg"))
        entry["img"] = self._load_image(img_path)
        entry["impression"] = self.meta_data.loc[entry["dicom_id"]]["impression"]
        return entry


class MimicCXRDatasetMSBBOX(MimicCXRDataset):
    def __init__(self, dataset_args, opt):
        self._bbox_meta_data = None
        assert dataset_args["split"] == DatasetSplit("mscxr")
        super().__init__(dataset_args, opt)

    @property
    def bbox_meta_data(self):
        return pd.read_csv(os.path.join(self.base_dir, "mimic_sccxr_preprocessed.csv"), index_col="dicom_id")

    def _build_dataset(self):
        data = [dict(rel_path=os.path.join(img_path.replace(".dcm", ".jpg")), finding_labels=labels) for img_path, labels in zip(list(self.bbox_meta_data.paths), list(self.bbox_meta_data["category_name"]))]
        self.data = data
        if self.shuffle:
            np.random.shuffle(self.data)

        if self.limit_dataset is not None:
            self.data = self.data[self.limit_dataset[0]:min(self.limit_dataset[1], len(self.data))]

    def _load_images(self, index):
        assert len(index)
        entry = self.data[index[0]].copy()
        entry["dicom_id"] = os.path.basename(entry["rel_path"]).rstrip(".jpg")
        entry["img"] = self._load_image(os.path.join(self.base_dir, entry["rel_path"].replace(".dcm", ".jpg")))

        meta_data_entry = self.bbox_meta_data.loc[entry["dicom_id"]]
        image_width, image_height = meta_data_entry[["image_width", "image_height"]]
        bboxes = meta_data_entry["bboxxywh"].split("|")
        bbox_img = torch.zeros((image_height, image_width), dtype=bool)

        for bbox in bboxes:
            bbox = bbox.split("-")
            bbox = tuple(map(lambda y: int(y), bbox))
            x, y, w, h = bbox
            bbox_img[y: (y + h), x:(x + w)] = True

        if max(bbox_img.size()) > self.W:
            transforms = Compose([Resize(self.W), CenterCrop(self.W)])
            bbox_img = transforms(bbox_img.unsqueeze(dim=0)).squeeze()

        entry["bbox_img"] = bbox_img
        entry["bboxxywh"] = meta_data_entry["bboxxywh"]
        entry["label_text"] = meta_data_entry["label_text"]
        return entry
