import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
import os

from omegaconf import OmegaConf
from PIL import Image
from einops import repeat
from torch import autocast
from log import logger

from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage, ToTensor
from einops import rearrange, repeat
from PIL import ImageDraw

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from src.preliminary_masks import preprocess_attention_maps
from src.visualization.utils import log_images_helper
from pytorch_lightning.loggers import WandbLogger




class SetupCallback(Callback):
    def __init__(self, resume, logdir, ckptdir, cfgdir, config,
                 lightning_config, debug, enable_multinode_hacks=False):
        super().__init__()
        self.resume = resume
        self.now = os.path.dirname(logdir)
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.debug = debug
        self.enable_multinode_hacks = enable_multinode_hacks

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
            if self.enable_multinode_hacks:
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
            if not self.enable_multinode_hacks and not self.resume and os.path.exists(self.logdir):
                pass
                #dst, name = os.path.split(self.logdir)
                #dst = os.path.join(dst, "child_runs", name)
                #os.makedirs(os.path.split(dst)[0], exist_ok=True)
                #try:
                #    os.rename(self.logdir, dst)
                #except FileNotFoundError:
                #    pass


class ImageLogger(Callback):
    def __init__(self, epoch_frequency, max_images, clamp=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, log_all_val=False):
        super().__init__()
        self.rescale = rescale
        self.max_images = max_images
        self.epoch_frequency = epoch_frequency
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_all_val = log_all_val
        self.attention_extractor = None

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

    def set_attention_extractor(self, attention_extractor):
        self.attention_extractor = attention_extractor

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (hasattr(pl_module, "log_images") and
                callable(pl_module.log_images)):

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                with autocast("cuda"):
                    images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
                if images.get("attention") is not None:
                    attention = images.pop("attention")
                    attention_maps = []
                    attention_images = preprocess_attention_maps(attention, on_cpu=True)
                    #attention_images = torch.cat(attention_images, dim=1)
                    for i in range(len(batch["img"])):  # batch
                        attention_maps.append(self.attention_extractor(attention_images[i]))
                    images["attention"] = attention_maps

                    labeled_attention = []
                    for i, attention in enumerate(images["attention"]):
                        # 1 x 1 x tok_idx x 64 x 64
                        assert len(attention.size()) == 5
                        attention = attention.squeeze(dim=(0))

                        attention = rearrange(repeat(attention, "1 b h w -> 3 b h w"), "c b h w -> b c h w")
                        grid_img = make_grid(attention, nrow=len(attention), normalize=True)
                        grid_img = ToPILImage()(grid_img)
                        ImageDraw.Draw(grid_img).text(
                            (0, 0),
                            batch["label_text"][i],
                            (255, 0, 0)
                        )
                        labeled_attention.append(ToTensor()(grid_img))
                    labeled_attention = torch.stack(labeled_attention, dim=0)
                    labeled_attention = (labeled_attention - 0.5) * 2
                    images["attention"] = labeled_attention

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k].to(float), -1., 1.)

            if images.get("inputs") is not None and images["inputs"].size()[-3] == 8:
                # Gaussian - just ignore as input
                images.pop("inputs")
            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if isinstance(pl_module.logger, WandbLogger):
                log_images_helper(pl_module.logger, images, prefix="", drop_samples=False)

            if is_train:
                pl_module.train()

    def log_attention(self, pl_module, batch, batch_idx, split="train"):
        loc_logger = type(pl_module.logger)

        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            with autocast("cuda"):
                images = pl_module.log_images(batch, split=split, sample=False, inpaint=True, plot_progressive_rows=False, plot_diffusion_rows=False, use_ema_scope=False, cond_key="label_text", mask=1., save_attention=True)
                images.pop("inputs")
                attention = images.pop("attention")
                attention_maps = []
                attention_images = preprocess_attention_maps(attention, on_cpu=True)
                for i in range(len(batch["img"])):  # batch
                    attention_maps.append(self.attention_extractor(attention_images[i]))
                images["attention"] = attention_maps

                labeled_attention = []
                for i, attention in enumerate(images["attention"]):
                    # 1 x 1 x tok_idx x 64 x 64
                    assert len(attention.size()) == 5
                    attention = attention.squeeze(dim=(0))
                    attention = rearrange(repeat(attention, "1 b h w -> 3 b h w"), "c b h w -> b c h w")
                    grid_img = make_grid(attention, nrow=len(attention), normalize=True, scale_each=True)
                    grid_img = ToPILImage()(grid_img)

                    txt_label = batch["label_text"][i]
                    token_lens = pl_module.cond_stage_model.compute_word_len(txt_label.split(" "))
                    token_positions = np.cumsum(token_lens)
                    token_positions = token_positions - token_positions[0]
                    for word, token_pos in zip(txt_label.split(" "), token_positions):
                        ImageDraw.Draw(grid_img).text(
                            ((1 + token_pos)*66 + 5, 0), # +5 just offset to look nicer, 66 because grid add pixels
                            word,
                            (255, 0, 0)
                        )


                    labeled_attention.append(ToTensor()(grid_img))
                labeled_attention = torch.stack(labeled_attention, dim=0)
                labeled_attention = (labeled_attention - 0.5) * 2
                images["attention"] = labeled_attention

        for k in images:
            N = min(images[k].shape[0], self.max_images)
            images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k].to(float), -1., 1.)

        if images.get("inputs") is not None and images["inputs"].size()[-3] == 8:
            # Gaussian - just ignore as input
            images.pop("inputs")
        self.log_local(pl_module.logger.save_dir, "attention", images,
                       pl_module.global_step, pl_module.current_epoch, batch_idx)

        if isinstance(pl_module.logger, WandbLogger):
            log_images_helper(pl_module.logger, images, prefix="imgin-", drop_samples=False)

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

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if (trainer.current_epoch == 0 and self.log_first_step) or (trainer.current_epoch > 0 and trainer.current_epoch % self.epoch_frequency == 0):
            if not self.disabled:
                logger.info("Start sampling of image.")
                self.log_img(pl_module, batch, batch_idx, split="val") # logs image trained from scratch
                logger.info("Start sampling with attention.")
                self.log_attention(pl_module, batch, batch_idx, split="val") # logs attention maps if we condtion on data from validation set


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
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



class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency

    def on_train_batch_end(self, trainer: pl.Trainer, *args, **kwargs):
        """ Check if we should save a checkpoint after every train batch """
        if trainer.global_step % self.save_step_frequency == 0 and trainer.global_step != 0:
            logger.info("Start saving model")
            global_step=trainer.global_step
            filename = f"{global_step=}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

