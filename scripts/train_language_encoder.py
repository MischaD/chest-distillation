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

from src.datasets.chest import ChestXray14Dataset
from utils import get_train_args, make_exp_config, load_model_from_config, collate_batch, img_to_viz, instantiate_from_config
from log import logger, log_experiment, file_handler


MULTINODE_HACKS = False

def main(opt):
    sys.path.append(os.getcwd())

    ckpt = opt.ckpt
    base_configs = sorted(glob.glob(os.path.join(opt.log_dir, "configs/*.yaml")))
    opt.base = base_configs + opt.base

    ckptdir = os.path.join(opt.log_dir, "checkpoints")
    cfgdir = os.path.join(opt.log_dir, "configs")
    seed_everything(opt.seed)

    """
    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["accelerator"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        if not opt.finetune_from == "":
            print(f"Attempting to load state from {opt.finetune_from}")
            old_state = torch.load(opt.finetune_from, map_location="cpu")
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

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                    "debug": opt.debug,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': -1,
                         'every_n_train_steps': 10000,
                         'save_weights_only': True
                     }
                     }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        if not "plugins" in trainer_kwargs:
            trainer_kwargs["plugins"] = list()
        if not lightning_config.get("find_unused_parameters", True):
            from pytorch_lightning.plugins import DDPPlugin

            trainer_kwargs["plugins"].append(DDPPlugin(find_unused_parameters=False))
        if MULTINODE_HACKS:
            # disable resume from hpc ckpts
            # NOTE below only works in later versions
            # from pytorch_lightning.plugins.environments import SLURMEnvironment
            # trainer_kwargs["plugins"].append(SLURMEnvironment(auto_requeue=False))
            # hence we monkey patch things
            from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector

            setattr(CheckpointConnector, "hpc_resume_path", None)

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir  ###

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        try:
            for k in data.datasets:
                print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
        except:
            print("datasets not yet initialized.")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1

        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")


        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)


        import signal

        signal.signal(signal.SIGUSR1, melk)


        # signal.signal(signal.SIGUSR2, divein)

        # run
        def check_validity():
            from torchvision.transforms import ToPILImage
            from einops import rearrange
            txts = ""
            masks = []
            images = []
            for i in range(10):
                sample = data.datasets["train"][i]
                txts += sample["txt"].replace(" ", "") + "___"
                masks.append(sample["mask"].to(torch.float32))
                images.append(sample["image"])
            images = rearrange(torch.cat(images, dim=1), "h w c -> c h w")
            masks = torch.cat(masks, dim=2)[:3]
            images = (images + 1) / 2

            to_pil = ToPILImage()
            abs_path_to_log = os.path.join(os.path.abspath("."), logdir)
            os.makedirs(abs_path_to_log, exist_ok=True)
            txts = txts[:100]
            to_pil(images).save(os.path.join(os.path.abspath("."), logdir, f"imag_{txts}.png"))
            to_pil(masks).save(os.path.join(os.path.abspath("."), logdir, f"mask_{txts}.png"))
            print(f"Saving samples of training objective to {abs_path_to_log}")

        check_validity()

        if opt.train:
            trainer.fit(model, data)
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except RuntimeError as err:
        if MULTINODE_HACKS:
            import requests
            import datetime
            import os
            import socket

            device = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
            hostname = socket.gethostname()
            ts = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            # resp = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
            # print(f'ERROR at {ts} on {hostname}/{resp.text} (CUDA_VISIBLE_DEVICES={device}): {type(err).__name__}: {err}', flush=True)
        raise err
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())
    """


if __name__ == '__main__':
    args = get_train_args()
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




"""
if __name__ == '__main__':
    args = get_train_args()
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
"""