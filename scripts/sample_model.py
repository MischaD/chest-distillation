import random

from log import logger, log_experiment
import os
import logging
import torch
from utils import get_sample_model_args, make_exp_config, load_model_from_config, collate_batch, img_to_viz
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
from src.datasets.sample_dataset import get_mscxr_synth_dataset




def main(opt):
    if not hasattr(opt, "img_dir") or opt.img_dir is None:
        img_dir = os.path.join(opt.log_dir, "generated")
    else:
        img_dir = opt.img_dir

    logger.info(f"Saving Images to {img_dir}")
    config = OmegaConf.load(f"{opt.config_path_inference}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda")
    model = model.to(device)

    if opt.use_mscxrlabels:
        dataset = get_dataset(opt, "test")
        dataset.load_precomputed(model)
        synth_dataset, labels = get_mscxr_synth_dataset(opt, dataset)
    else:
        dataset = get_dataset(opt, "testp19")
        dataset.load_precomputed(model)
        synth_dataset, labels = get_mscxr_synth_dataset(opt, dataset, finding_key="impression", label_key="finding_labels")


    os.makedirs(img_dir, exist_ok=True)
    for label in labels:
        os.makedirs(os.path.join(img_dir, label), exist_ok=True)

    batch_size = opt.batch_size

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    seed_everything(time.time())

    batched_dataset = [synth_dataset[i:i+batch_size] for i in range(0, len(synth_dataset), batch_size)]

    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():
                for samples in tqdm(batched_dataset, desc="data"):
                    uc = None
                    prompts = [list(x.values())[0] for x in samples]
                    classes = [list(x.keys())[0] for x in samples]
                    c = model.get_learned_conditioning(prompts)
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(len(c) * [""])

                    start_code = torch.randn([len(c), opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    output, _ = sampler.sample(S=opt.ddim_steps,
                                                conditioning=c,
                                                batch_size=len(samples),
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=opt.scale,
                                                unconditional_conditioning=uc,
                                                x_T=start_code)

                    output = model.decode_first_stage(output)
                    output = torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0)

                    output = output.cpu()
                    for i in range(len(output)):
                        sample_path = os.path.join(os.path.join(img_dir, classes[i]))
                        base_count = len(os.listdir(sample_path))
                        sample = 255. * rearrange(output[i].numpy(), 'c h w -> h w c')
                        Image.fromarray(sample.astype(np.uint8)).save(
                            os.path.join(sample_path, f"{base_count:05}.png"))


if __name__ == '__main__':
    args = get_sample_model_args()
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
