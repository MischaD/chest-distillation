import argparse, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
import os

from src.ldm.encoders.modules import OpenClipDummyTokenizer
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor

from src.preliminary_masks import AttentionExtractor
from src.datasets import get_dataset
from src.callbacks import CUDACallback, SetupCallback, CheckpointEveryNSteps
from utils import get_train_args, make_exp_config, load_model_from_config, collate_batch, img_to_viz, instantiate_from_config, main_setup
from log import logger, log_experiment
from log import formatter as log_formatter
from tqdm import tqdm
import logging


MULTINODE_HACKS = False


def get_trainer_logger(log_dir, **kwargs):
    def_kwargs = {
        "target": "pytorch_lightning.loggers.WandbLogger",
        "params": {
            "project": "chest-distillation",
            "name": os.path.dirname(log_dir),
            "save_dir": log_dir,
            "offline": False,
            "id": "__".join(log_dir.split("/")[-2:]),  # used for resuming
            "tags": kwargs.pop("tags")
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
            "verbose": True,
            "monitor": None,
            "save_last": False,
            "save_weights_only": True,
            "save_top_k": 0,
            "every_n_train_steps": 0,
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


def train(config):
    sys.path.append(os.getcwd())
    config = config
    ckptdir = os.path.join(config.log_dir, "checkpoints")
    cfgdir = os.path.join(config.log_dir, "configs")
    seed_everything(config.seed)

    ckpt = config.ckpt

    resume = False
    if os.path.isfile(os.path.join(ckptdir, "last.ckpt")):
        resume = True
        ckpt = os.path.join(ckptdir, "last.ckpt")

    model_config = OmegaConf.load(f"{config.config_path}")
    lightning_config = model_config.pop("lightning", OmegaConf.create())

    if hasattr(config, "cond_stage_trainable"):
        model_config["model"]["params"]["cond_stage_trainable"] = config.cond_stage_trainable

    attention_regularzation = False
    if hasattr(config, "mlf_args"):
        logger.info(f"Overwriting default arguments of config with {config.mlf_args}")
        attention_regularzation = config.mlf_args.get("attention_regularization", False)
        model_config["model"]["params"]["attention_regularization"] = attention_regularzation
        model_config["model"]["params"]["cond_stage_key"] = config.mlf_args.get("cond_stage_key")
        model_config["model"]["params"]["cond_stage_config"]["params"]["multi_label_finetuning"] = config.mlf_args.get("multi_label_finetuning")

    if attention_regularzation:
        logger.info("Applying Attention-Regularization!")
        model_config["model"]["params"]["unet_config"]["params"]["attention_save_mode"] = "arm"

    if hasattr(config.stable_diffusion, "ucg_probability"):
        logger.info(f"Overwriting default arguments of ucg probability with {config.stable_diffusion.ucg_probability}")
        model_config["model"]["params"]["ucg_probability"] = config.stable_diffusion.ucg_probability

    if hasattr(config, "mlf_args") and config.mlf_args["rali"] is not None:
        rali_mode = config.mlf_args["rali"]# rali_mode used in tokenizer later
        logger.info(f"Activating Rali Mode: {rali_mode} - attention_regularization (bool): {attention_regularzation}")
        model_config["model"]["params"]["rali"] = True
        model_config["model"]["params"]["cond_stage_config"]["params"]["rali"] = rali_mode
    else:
        model_config["model"]["params"]["rali"] = False

    model_config["model"]["base_learning_rate"] = config.stable_diffusion.learning_rate
    model_config["model"]["params"]["optimizer_type"] = config.stable_diffusion.optimizer_type
    logger.info(f"Setting learning rate to {config.stable_diffusion.learning_rate} and optimizer to {config.stable_diffusion.optimizer_type}")

    model = load_model_from_config(model_config, f"{config.ckpt}")

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
    logger_cfg = get_trainer_logger(log_dir=config.log_dir,
                                    name=config.EXP_NAME + os.path.basename(config.log_dir),
                                    group="train_language_encoder",
                                    tags=["repeat_exp", f"trainable_{config.cond_stage_trainable}"],
                                    )
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
    modelckpt_cfg_default = get_model_checkpoint_config(ckptdir)
    checkpoint_callback = instantiate_from_config(modelckpt_cfg_default)

    setup_cb = SetupCallback(resume=resume,
                             logdir=config.log_dir,
                             ckptdir=ckptdir,
                             cfgdir=cfgdir,
                             config=model_config,
                             debug=False,
                             lightning_config=lightning_config,
                             enable_multinode_hacks=MULTINODE_HACKS,
                             )
    image_logger = instantiate_from_config(lightning_config["callbacks"]["image_logger"])
    image_logger.set_attention_extractor(AttentionExtractor("all_token_mean", steps=int(config.sample.ddim_steps * 4 / 5), max_token=10))
    learning_rate_logger = LearningRateMonitor(logging_interval="step")

    cuda_callback = CUDACallback()
    step_checkpoint_callback = CheckpointEveryNSteps(save_step_frequency=config.trainer.checkpoint_save_frequency)

    trainer_kwargs["callbacks"] = [setup_cb, image_logger, learning_rate_logger, cuda_callback, checkpoint_callback, step_checkpoint_callback]
    trainer_kwargs["precision"] = 16
    trainer_kwargs["accelerator"] = "gpu"
    trainer_kwargs["strategy"] = "ddp" # trainer_config["strategy"]
    trainer_kwargs["num_nodes"] = config.trainer.num_nodes # trainer_config["strategy"]
    trainer_kwargs["devices"] = torch.cuda.device_count()
    trainer_kwargs["max_steps"] = config.trainer.max_steps if hasattr(config.trainer, "max_steps") else 60001
    trainer_kwargs["num_sanity_val_steps"] = 0
    trainer = Trainer(**trainer_kwargs)
    trainer.logdir = config.log_dir

    # configure learning rate
    bs, base_lr = config.dataloading.batch_size, model_config.model.base_learning_rate
    model.learning_rate = base_lr
    logger.info("++++ NOT USING LR SCALING ++++")
    logger.info(f"Setting learning rate to {model.learning_rate:.2e}")

    train_dataset = get_dataset(config, "train")
    val_dataset = get_dataset(config, "val")

    train_dataset.load_precomputed(model)
    val_dataset.load_precomputed(model)

    dataset = DataModuleFromConfig(
        batch_size=config.dataloading.batch_size,
        train=train_dataset,
        validation=val_dataset,
        num_workers=config.dataloading.num_workers,
        num_val_workers=0,
    )

    if hasattr(config, "mlf_args"):
        tokenizer = OpenClipDummyTokenizer(config.seed,
                                           config.mlf_args.get("append_invariance_tokens", False),
                                           config.mlf_args.get("single_healthy_class_token", False),
                                           rali=config.mlf_args.get("rali"),
                                           )
        model.cond_stage_model.set_multi_label_tokenizer(tokenizer)

    seed_everything(time.time())

    logger.info(f"Length of train dataset: {len(train_dataset)}")
    trainer.fit(model, dataset)
    if hasattr(config, "save_to") and config.save_to is not None:
        from shutil import copy
        copy(step_checkpoint_callback.last_ckpt, config.save_to)



class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False,
                 shuffle_val_dataloader=False, num_val_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 0  # num_workers if num_workers is not None else batch_size * 2
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
                          num_workers=self.num_workers, shuffle=True, collate_fn=collate_batch)

    def _val_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_val_workers,
                          shuffle=shuffle, collate_fn=collate_batch)

    def _test_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=shuffle, collate_fn=collate_batch)

    def _predict_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          shuffle=shuffle,
                          num_workers=self.num_workers, collate_fn=collate_batch)


def get_args():
    parser = argparse.ArgumentParser(description="Compute Masks")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("EXP_NAME", type=str, help="Path to Experiment results")
    parser.add_argument("--cond_stage_trainable", action="store_true", default=False, help="Trainable or frozen language encoder")
    parser.add_argument("--save_to", type=str, default=None, help="Path to save final model to")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    config = main_setup(args)
    train(config)