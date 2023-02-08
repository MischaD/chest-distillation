import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from utils import get_compute_mask_args, make_exp_config, load_model_from_config, collate_batch, img_to_viz
from log import logger
import pprint

from src.ldm.util import instantiate_from_config
from src.ldm.models.diffusion.ddim import DDIMSampler
from src.ldm.models.diffusion.plms import PLMSSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument(
        "--out_dir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        help="How many samples to create for each prompt",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )

    parser.add_argument(
        "--scale",
        type=float,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--from_file",
        type=str,
        help="if specified, load prompts from this file",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="the seed (for reproducible sampling)",
    )

    args = parser.parse_args()
    opt = make_exp_config(args.EXP_PATH)
    for key, value in vars(args).items():
        if value is not None:
            setattr(opt, key, value)
            logger.info(f"Overwriting exp file key {key} with: {value}")
    logger.debug(pprint.PrettyPrinter(depth=4).pformat({k:v for k, v in vars(opt).items() if not k.startswith("__")}))

    print(os.path.abspath("."))
    config = OmegaConf.load(f"{opt.config_path}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.out_dir, exist_ok=True)
    outpath = opt.out_dir

    batch_size = opt.batch_size
    logger.info(f"reading prompts from {opt.from_file}")
    with open(opt.from_file, "r") as f:
        data = f.read().splitlines()


    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():
                for prompt in tqdm(data, desc="data"):
                    sample_path = os.path.join(outpath, "samples" + f"{prompt.replace(' ', '_')}")
                    os.makedirs(sample_path, exist_ok=True)
                    base_count = len(os.listdir(sample_path))
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])

                    if isinstance(prompt, tuple) or isinstance(prompt, str):
                        prompts = list((prompt,))
                    c = model.get_learned_conditioning(prompts)

                    for n_samples in range(opt.n_samples):
                        if not opt.fixed_code:
                            start_code = torch.randn([batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        #samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                        #                                 conditioning=c,
                        #                                 batch_size=opt.n_samples,
                        #                                 shape=shape,
                        #                                 verbose=False,
                        #                                 unconditional_guidance_scale=opt.scale,
                        #                                 unconditional_conditioning=uc,
                        #                                 eta=opt.ddim_eta,
                        #                                 dynamic_threshold=opt.dyn,
                        #                                 x_T=start_code)
                        samples, _ = sampler.sample(S=opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=len(prompts),
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    x_T=start_code)

                        samples = model.decode_first_stage(samples)
                        samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(sample_path, f"{base_count:05}.png"))
                            base_count += 1


if __name__ == "__main__":
    main()
