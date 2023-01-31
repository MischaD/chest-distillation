import argparse, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
import datetime
import os

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image
from einops import repeat

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from src.datasets import get_dataset
from utils import get_train_args, make_exp_config, load_model_from_config, collate_batch, img_to_viz, instantiate_from_config
from log import logger, log_experiment
from log import formatter as log_formatter
import logging


MULTINODE_HACKS = False


def get_trainer_logger(**kwargs):
    log_dir = os.path.join("./logger", datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    def_kwargs = {
        "target": "pytorch_lightning.loggers.WandbLogger",
        "params": {
            "name": os.path.dirname(log_dir),
            "save_dir": log_dir,
            "offline": False,
            "id": os.path.dirname(log_dir),
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


class DataModuleFromConfig(pl.LightningDataModule):
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

    def preload(self):
        pass

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          )

    def _val_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_val_workers,
                          shuffle=shuffle)

class SetupCallback(Callback):
    def __init__(self, resume, logdir, ckptdir, cfgdir, config,
                 lightning_config, debug):
        super().__init__()
        self.resume = resume
        self.now = os.path.dirname(logdir)
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.debug = debug

    def on_keyboard_interrupt(self, trainer, pl_module):
        if not self.debug and trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            if MULTINODE_HACKS:
                import time
                time.sleep(5)
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not MULTINODE_HACKS and not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, log_all_val=False):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_all_val = log_all_val

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if self.log_all_val and split == "val":
            should_log = True
        else:
            should_log = self.check_frequency(check_idx)
        if (should_log and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                self.log_images_kwargs["unconditional_guidance_scale"] = 1 # classifier free guidance
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            images["masks"] = (repeat(batch["mask"][:, :3].cpu(), "b c h w -> b c (h h4) (w w4)", h4=8, w4=8).float() - 0.5) * 2

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")
            #print(f"Skip logging image")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


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
    trainer_config["accelerator"] = "ddp"
    lightning_config.trainer = trainer_config

    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda")
    model = model.to(device)

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
    logger_cfg = get_trainer_logger(name=os.path.basename(log_dir),
                                    save_dir=log_dir,
                                    offline=opt.debug,
                                    id=os.path.dirname(log_dir),
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
    )
    image_logger = ImageLogger(batch_frequency=750, max_images=4, clamp=True)
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

    dataset = DataModuleFromConfig(
        batch_size=opt.batch_size,
        train=train_dataset,
        validation=val_dataset,
        num_workers=4,
        num_val_workers=0,
    )

    dataset.preload()

    logger.info(f"Length of train dataset: {len(train_dataset)}")
    trainer.fit(model, dataset)



class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
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
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_val_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)

#class LatentDataset():


if __name__ == '__main__':
    args = get_train_args()
    log_dir = os.path.join(os.path.abspath("."), "log", args.EXP_NAME)
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'console.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.debug("="*30 + "Running train_language_encoder.py" + "="*30)

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
