from enum import Enum
import argparse
from importlib.machinery import SourceFileLoader
import importlib
import torch
import numpy as np
from log import logger, log_experiment
from log import formatter as log_formatter
import os
import datetime
import logging
from log import logger
from einops import rearrange
from scipy import ndimage
import torchvision


def get_compute_mask_args():
    parser = argparse.ArgumentParser(description="Compute Masks for Localization Metrics")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("EXP_NAME", type=str, help="Path to Experiment results")
    parser.add_argument("--ckpt", type=str, default="train")
    parser.add_argument("--mask_dir", type=str, default=None, help="dir to save masks in. Default will be inside log dir and should be used!")
    parser.add_argument("--filter_bad_impressions", action="store_true", default=False, help="If set, then we use shortned impressions from mscxr")
    parser.add_argument("--phrase_grounding_mode", action="store_true", default=False, help="If set, then we use shortned impressions from mscxr")
    return parser.parse_args()


def get_compute_mssim():
    parser = argparse.ArgumentParser(description="Compute MS-SSIM of dataset")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("EXP_NAME", type=str, help="Path to Experiment results")
    parser.add_argument("--ckpt", type=str, default="to generate_propmts with")
    parser.add_argument("--n_sample_sets", type=int, default=100)
    parser.add_argument("--trial_size", type=int, default=4)
    parser.add_argument("--use_mscxrlabels", action="store_true", default=False, help="If set, then we use shortned impressions from mscxr")
    parser.add_argument("--img_dir", type=str, default=None,
                        help="dir to save images in. Default will be inside log dir and should be used!")
    return parser.parse_args()

def get_classification_args():
    parser = argparse.ArgumentParser(description="Classify Generated Samples")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("EXP_NAME", type=str, help="Path to Experiment results")
    parser.add_argument("IMG_PATH", type=str, default=None,
                        help="Either path to directory containing images with the folder names being the label, or path to csv")
    return parser.parse_args()

def get_sample_model_args():
    parser = argparse.ArgumentParser(description="Compute Masks")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("EXP_NAME", type=str, help="Path to Experiment results")
    parser.add_argument("--ckpt", type=str, default="train")
    parser.add_argument("--img_dir", type=str, default=None, help="dir to save images in. Default will be inside log dir and should be used!")
    parser.add_argument("--use_mscxrlabels", action="store_true", default=False, help="")
    parser.add_argument("--N", type=int, default=None, help="")
    parser.add_argument("--label_list_path", type=str, default=None, help="")
    return parser.parse_args()

def get_comput_fid_args():
    parser = argparse.ArgumentParser(description="Compute FID of dataset")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("EXP_NAME", type=str, help="Path to Experiment results")
    parser.add_argument("path_src", type=str, help="Path to first dataset")
    parser.add_argument("path_tgt", type=str, help="Path to second dataset")
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size to use')
    parser.add_argument('--num-workers', type=int,
                        help=('Number of processes to use for data loading. '
                              'Defaults to `min(8, num_cpus)`'))
    parser.add_argument("--result_dir", type=str, default=None, help="dir to save results in.")
    return parser.parse_args()

def get_train_args():
    parser = argparse.ArgumentParser(description="Compute Masks")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("EXP_NAME", type=str, help="Path to Experiment results")
    parser.add_argument("--save_to", type=str, default=None, help="Path to save final model to")
    return parser.parse_args()

def get_train_segmentation_refined():
    parser = argparse.ArgumentParser(description="Compute Unet Refined")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("--exp_name", type=str, default=None, help="Path to experiment files")
    parser.add_argument("--postprocess", action="store_true", default=False)
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--bbox_mode", action="store_true", default=False)
    parser.add_argument("--ckpt_path", default=None)
    return parser.parse_args()

def get_compute_background_args():
    parser = argparse.ArgumentParser(description="Compute Background")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("NAME", type=str, help="Name of experiment - will be used to save masks")
    parser.add_argument("--log_all", action="store_true", default=False, help="logs all information s.a. mask, input, reconstruction ")
    parser.add_argument("--save_output_dir", default=None, help="Save all output to single outdir")
    parser.add_argument("--use_plms", action="store_true", default=False, help="Use plms sampling")
    parser.add_argument("--start", type=int, default=None, help="first sample to generate, inclusive")
    parser.add_argument("--stop", type=int, default=None, help="last sample to generate, exclusive")
    parser.add_argument("--synthesis_caption_mask", type=str, default=None, choices=["fg", "bg", "full"], help="last sample to generate, exclusive")
    parser.add_argument("--caption", type=str, default=None, choices=["fg", "bg"], help="Diffusion Model Prompt - either foreground prompt or background prompt")
    parser.add_argument("--scale", type=float, default=1., help="Classifier free guidance scale")
    return parser.parse_args()

def get_inpaint_baseline_args():
    parser = argparse.ArgumentParser(description="Compute Background")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("NAME", type=str, help="Name of experiment - will be used to save masks")
    parser.add_argument("--log_all", action="store_true", default=False, help="logs all information s.a. mask, input, reconstruction ")
    parser.add_argument("--use_plms", action="store_true", default=False, help="Use plms sampling")
    parser.add_argument("--start", type=int, default=None, help="first sample to generate, inclusive")
    parser.add_argument("--stop", type=int, default=None, help="last sample to generate, exclusive")
    return parser.parse_args()

def make_exp_config(exp_file):
    # get path to experiment
    exp_name = exp_file.split('/')[-1].rstrip('.py')

    # import experiment configuration
    exp_config = SourceFileLoader(exp_name, exp_file).load_module()
    exp_config.name = exp_name
    return exp_config


def resize_to(img, tosize):
    assert img.ndim == 4
    b, c, h, w = img.size()
    max_size = max(h, w)

    zoom_factor = tosize / max_size

    return torch.tensor(ndimage.zoom(img, (1, 1, zoom_factor,zoom_factor)))

class DatasetSplit(Enum):
    train="train"
    test="test"
    val="val"
    mscxr="mscxr"
    p19="p19"
    all="all"

def resize_long_edge(img, size_long_edge):
    # torchvision resizes so shorter edge has length - I want longer edge to have spec. length
    assert img.size()[-3] == 3, "Channel dimension expected at third position"
    img_longer_edge = max(img.size()[-2:])
    img_shorter_edge = min(img.size()[-2:])
    resize_factor = size_long_edge / img_longer_edge

    # resized_img = torchvision.transforms.functional.resize(img_longer_edge/img_shorter_edge)
    resize_to = img_shorter_edge * resize_factor
    resizer = torchvision.transforms.Resize(size=round(resize_to))
    return resizer(img)[..., :size_long_edge, :size_long_edge]


SPLIT_TO_DATASETSPLIT = {0:DatasetSplit("test"), 1:DatasetSplit("train"), 2:DatasetSplit("val"), 3:DatasetSplit("p19"), 4:DatasetSplit("mscxr")} #p19 - 3
DATASETSPLIT_TO_SPLIT = {"test":0, "train":1, "val":2, "p19":3, "mscxr":4}#p19 - 3


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def get_tok_idx(prompt, obj):
    object_categories = prompt.split(" ")
    tok_idx = [i for i in range(len(object_categories)) if object_categories[i] == obj][0]
    return tok_idx + 1

def img_to_viz(img):
    img = rearrange(img, "1 c h w -> h w c")
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    img = np.array(((img + 1) * 127.5), np.uint8)
    return img


def collate_batch(batch):
    # make list of dirs to dirs of lists with batchlen
    batched_data = {}
    for data in batch:
        # label could be img, label, path, etc
        for key, value in data.items():
            if batched_data.get(key) is None:
                batched_data[key] = []
            batched_data[key].append(value)

    # cast to torch.tensor
    for key, value in batched_data.items():
        if isinstance(value[0],torch.Tensor):
            if value[0].size()[0] != 1:
                for i in range(len(value)):
                    value[i] = value[i][None,...]
            # check if concatenatable
            if all([value[0].size() == value[i].size() for i in range(len(value))]):
                batched_data[key] = torch.concat(batched_data[key])
    return batched_data

def prompts_from_file():
    pass

def viz_array(x):
    # 1 x c x h x w
    # c x h x w
    # h x w x c
    from einops import rearrange
    import matplotlib.pyplot as plt
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    x = x.float()
    x = x.squeeze()
    x = x.detach().cpu()
    x = (x - x.min()) / (x.max() - x.min())
    if x.ndim == 3:
        if x.size()[-1] != 3:
            x = rearrange(x, "c h w -> h w c")
        plt.imshow(x)
    else:
        #ndim == 2
        plt.imshow(x, cmap="Greys_r")
    plt.show()

def main_setup(args, name=__file__):
    config = make_exp_config(args.EXP_PATH).config
    for key, value in vars(args).items():
        if value is not None:
            keys = key.split(".")
            if len(keys) == 1:
                key = keys
                setattr(config, keys[0], value)
            else:
                # keys with more depth
                cfg_key = config
                for i in range(len(keys) - 1):
                    cfg_key = getattr(cfg_key, keys[i])
                setattr(cfg_key, keys[-1], value)
            logger.info(f"Overwriting exp file key {key} with: {value}")

    if not hasattr(config, "log_dir"):
        setattr(config, "log_dir", os.path.join(os.path.abspath("."), "log", args.EXP_NAME, datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")))
    else:
        # /vol/ideadata/ed52egek/pycharm/privacy/log/score_sde/2023-04-13T21-35-52
        config.EXP_NAME = config.log_dir.split("/")[-2] # overwrite exp name if log dir is defined

    log_dir = config.log_dir
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'console.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.debug("="*30 + f"Running {os.path.basename(name)}" + "="*30)
    logger.debug(f"Logging to {log_dir}")

    # make log dir (same as the one for the console log)
    log_dir = os.path.join(os.path.dirname(file_handler.baseFilename))
    setattr(config, "log_dir", log_dir)
    logger.info(f"Log dir: {log_dir}")
    logger.debug(f"Current file: {__file__}")
    logger.debug(f"config")
    log_experiment(logger, args)
    return config


def save_copy_checkpoint(src_path, tgt_path, log_logdir=None, log_wandb=None):
    os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
    if not os.path.exists(tgt_path):
        logger.info(f"Save best checkpoint to:{tgt_path}")
        shutil.copy(src_path, tgt_path)
    else:
        out_dir = os.path.dirname(tgt_path)
        extension = os.path.basename(tgt_path)
        i = 1
        while os.path.exists(os.path.join(out_dir, '{}_{}'.format(i, extension))):
            i += 1
        new_tgt_path = os.path.join(out_dir, '{}_{}'.format(i, extension))
        logger.info(f"Best path {tgt_path} already exists")
        logger.info(f"Copying old checkpoint to {new_tgt_path} as backup")
        shutil.copy(tgt_path, new_tgt_path)
        logger.info(f"Saving new checkpoint to {tgt_path}")
        shutil.copy(src_path, tgt_path)

    if log_logdir is not None:
        # some debug information
        base_path = os.path.dirname(tgt_path)
        extension = os.path.basename(tgt_path)
        with open(os.path.join(base_path, "." + extension + ".log"), "w", encoding="utf-8") as fp:
            fp.write(f"{tgt_path} comes from {log_logdir}\n")
            fp.write(f"wandb:{log_wandb}\n")


def update_matplotlib_font(fontsize=11, fontsize_ticks=8, tex=True):
    import matplotlib.pyplot as plt
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": tex,
        "font.family": "serif",
        # Use 11pt font in plots, to match 11pt font in document
        "axes.labelsize": fontsize,
        "font.size": fontsize,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": fontsize_ticks,
        "xtick.labelsize": fontsize_ticks,
        "ytick.labelsize": fontsize_ticks
    }
    plt.rcParams.update(tex_fonts)


def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "MICCAI":
        width_pt = 347.12354
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)




class AttributeDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'AttributeDict' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

