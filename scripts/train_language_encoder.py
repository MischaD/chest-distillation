import argparse, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
import os

from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor

from src.datasets import get_dataset
from src.callbacks import CUDACallback, SetupCallback
from utils import get_train_args, make_exp_config, load_model_from_config, collate_batch, img_to_viz, instantiate_from_config
from log import logger, log_experiment
from log import formatter as log_formatter
from tqdm import tqdm
import logging


MULTINODE_HACKS = False


def get_trainer_logger(log_dir, **kwargs):
    def_kwargs = {
        "target": "pytorch_lightning.loggers.WandbLogger",
        "params": {
            "name": os.path.dirname(log_dir),
            "save_dir": log_dir,
            "offline": False,
            "id": "__".join(log_dir.split("/")[-2:]),  # used for resuming
        }
    }

    for k, v in kwargs.items():
        def_kwargs["params"][k] = v
    return OmegaConf.create(def_kwargs)


def get_model_checkpoint_config(ckptdir, **kwargs):
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
        }
    }

    return default_modelckpt_cfg


class MimicDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False, num_val_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = 0 #num_workers if num_workers is not None else batch_size * 2
        if num_val_workers is None:
            self.num_val_workers = self.num_workers
        else:
            self.num_val_workers = num_val_workers
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None: # train 'target' hf_dataset
            self.datasets["train"] = train
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.datasets["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.datasets["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.datasets["predict"] = predict
            self.predict_dataloader = self._predict_dataloader


    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          )

    def _val_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_val_workers,
                          shuffle=shuffle)


def main(opt):
    sys.path.append(os.getcwd())

    ckptdir = os.path.join(opt.log_dir, "checkpoints")
    cfgdir = os.path.join(opt.log_dir, "configs")
    seed_everything(opt.seed)

    ckpt = opt.ckpt

    resume = False
    if os.path.isfile(os.path.join(ckptdir, "last.ckpt")):
        resume = True
        ckpt = os.path.join(ckptdir, "last.ckpt")

    config = OmegaConf.load(f"{opt.config_path}")
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config["accelerator"] = "gpu"
    trainer_config["strategy"] = "ddp"
    trainer_config["precision"] = 16
    trainer_config["devices"] = torch.cuda.device_count()
    lightning_config.trainer = trainer_config


    model = load_model_from_config(config, f"{opt.ckpt}")

    if not ckpt == "":
        print(f"Attempting to load state from {ckpt}")
        old_state = torch.load(ckpt, map_location="cpu")
        if "state_dict" in old_state:
            print(f"Found nested key 'state_dict' in checkpoint, loading this instead")
            old_state = old_state["state_dict"]
        m, u = model.load_state_dict(old_state, strict=False)
        if len(m) > 0:
            print("missing keys:")
            print(m)
        if len(u) > 0:
            print("unexpected keys:")
            print(u)

    trainer_kwargs = {}
    logger_cfg = get_trainer_logger(log_dir=opt.log_dir,
                                    name=opt.EXP_NAME+os.path.basename(opt.log_dir),
                                    offline=opt.debug,
                                    group="train_language_encoder"
                                    )
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
    modelckpt_cfg = get_model_checkpoint_config(ckptdir)
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        modelckpt_cfg["params"]["monitor"] = model.monitor
        modelckpt_cfg["params"]["save_top_k"] = 3

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg_tmp = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(modelckpt_cfg, modelckpt_cfg_tmp)
    trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

    setup_cb = SetupCallback(resume=resume,
                  logdir=log_dir,
                  ckptdir=ckptdir,
                  cfgdir=cfgdir,
                  config=config,
                  lightning_config=lightning_config,
                  debug= opt.debug,
                  enable_multinode_hacks=MULTINODE_HACKS,
    )
    from src.preliminary_masks import AttentionExtractor
    image_logger = instantiate_from_config(lightning_config["callbacks"]["image_logger"])
    image_logger.set_attention_extractor(AttentionExtractor("all_token_mean", steps=int(opt.ddim_steps * 4/5), max_token=10))
    learning_rate_logger = LearningRateMonitor(logging_interval="step")
    cuda_callback = CUDACallback()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(ckptdir, 'trainstep_checkpoints'),
        filename="{epoch:06}-{step:09}",
        verbose=True,
        save_top_k=-1,
        every_n_train_steps=10000,
        save_weights_only=True,

    )

    trainer_kwargs["precision"] = trainer_config["precision"]
    trainer_kwargs["accelerator"] = trainer_config["accelerator"]
    trainer_kwargs["strategy"] = trainer_config["strategy"]
    trainer_kwargs["devices"] = trainer_config["devices"]
    trainer_kwargs["callbacks"] = [setup_cb, image_logger, learning_rate_logger, cuda_callback, checkpoint_callback]
    trainer = Trainer(**trainer_kwargs)
    trainer.logdir = log_dir

    # configure learning rate
    bs, base_lr = opt.batch_size, config.model.base_learning_rate
    model.learning_rate = base_lr
    logger.info("++++ NOT USING LR SCALING ++++")
    logger.info(f"Setting learning rate to {model.learning_rate:.2e}")

    train_dataset = get_dataset(opt, "train")
    val_dataset = get_dataset(opt, "val")

    train_dataset.load_precomputed(model)
    val_dataset.load_precomputed(model)

    dataset = DataModuleFromConfig(
        batch_size=opt.batch_size,
        train=train_dataset,
        validation=val_dataset,
        num_workers=opt.num_workers,
        num_val_workers=0,
    )

    logger.info(f"Length of train dataset: {len(train_dataset)}")
    trainer.fit(model, dataset)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False,
                 shuffle_val_dataloader=False, num_val_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 0 #num_workers if num_workers is not None else batch_size * 2
        self.datasets = {}
        if num_val_workers is None:
            self.num_val_workers = self.num_workers
        else:
            self.num_val_workers = num_val_workers
        if train is not None: # train 'target' hf_dataset
            self.datasets["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.datasets["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.datasets["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.datasets["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def _val_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_val_workers,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          shuffle=shuffle,
                          num_workers=self.num_workers)


if __name__ == '__main__':
    args = get_train_args()
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
    logger.debug(f"Current file: {__file__}")
    log_experiment(logger, args, opt.config_path)

    main(opt)
