import random

from log import logger, log_experiment
import os
import logging
import torch
from utils import get_sample_model_args, make_exp_config, load_model_from_config, collate_batch, img_to_viz, get_comput_fid_args, get_compute_mssim
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
import json
import random
import numpy as np
from einops import rearrange
from src.datasets.sample_dataset import get_mscxr_synth_dataset
from src.evaluation.mssim import calc_ms_ssim_for_path_ordered, calc_ms_ssim_for_path
from src.ldm.encoders.modules import OpenClipDummyTokenizer


def main(opt):
    #mean, sd = calc_ms_ssim_for_path(opt.path, n=opt.n_samples, trials=opt.trials)
    if not hasattr(opt, "img_dir") or opt.img_dir is None:
        img_dir = os.path.join(opt.log_dir, "ms_ssim")
    else:
        img_dir = opt.img_dir

    logger.info(f"Saving Images to {img_dir}")
    config = OmegaConf.load(f"{opt.config_path_inference}")

    is_mlf = False
    if hasattr(opt, "mlf_args"):
        is_mlf = opt.mlf_args.get("multi_label_finetuning", False)
        logger.info(f"Overwriting default arguments of config with {opt.mlf_args}")
        config["model"]["params"]["attention_regularization"] = opt.mlf_args.get("attention_regularization")
        config["model"]["params"]["cond_stage_key"] = opt.mlf_args.get("cond_stage_key")
        config["model"]["params"]["cond_stage_config"]["params"]["multi_label_finetuning"] = opt.mlf_args.get("multi_label_finetuning")

    if not os.path.exists(img_dir):
        os.makedirs(img_dir, exist_ok=True)
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
            synth_dataset, labels = get_mscxr_synth_dataset(opt, dataset, finding_key="impression",
                                                            label_key="finding_labels")

        actual_batch_size = opt.trial_size
        batch_size = 1

        if opt.plms:
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)

        if is_mlf:
            tokenizer = OpenClipDummyTokenizer(opt.seed, opt.mlf_args.get("append_invariance_tokens", False),
                                               opt.mlf_args.get("single_healthy_class_token", False))
            if opt.seed == 4200:
                tokenization = tokenizer("Consolidation|Cardiomegaly|Pleural Effusion".split("|"))
                if len(tokenization) != 9:
                    tokenization = tokenization[1:-1]
                # assert tokenization[1] == 15598 and tokenization[3] == 22073
            model.cond_stage_model.set_multi_label_tokenizer(tokenizer)

        seed_everything(time.time())

        batched_dataset = [synth_dataset[i:i+batch_size] for i in range(0, len(synth_dataset), batch_size)]
        prompt_list = set()
        with torch.no_grad():
            with autocast("cuda"):
                with model.ema_scope():
                    for sample_num in range(len(batched_dataset)):
                        samples = batched_dataset[sample_num]
                        prompts = [list(x.values())[0] for x in samples]
                        classes = [list(x.keys())[0] for x in samples]
                        prompt = prompts[0]
                        if prompt in prompt_list:
                            continue
                        else:
                            prompt_list.add(prompt)

                        prompts = prompts * actual_batch_size if not is_mlf else classes * actual_batch_size

                        c = model.get_learned_conditioning(prompts)
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(actual_batch_size * [""])

                        start_code = torch.randn([actual_batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        output , _ = sampler.sample(S=opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=len(prompts),
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    x_T=start_code)

                        output = model.decode_first_stage(output)
                        output = torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0)
                        output = output.cpu()
                        base_count_dir = len(os.listdir(img_dir))
                        dir_name = f"{base_count_dir:05}"
                        dir_path = os.path.join(img_dir, dir_name)
                        os.makedirs(dir_path,exist_ok=True)

                        for i in range(len(output)):
                            base_count = len(os.listdir(dir_path))
                            sample = 255. * rearrange(output[i].numpy(), 'c h w -> h w c')
                            Image.fromarray(sample.astype(np.uint8)).save(
                                os.path.join(dir_path, f"{base_count:05}.png"))

                        with open(os.path.join(dir_path, "prompt.txt"), 'w') as file:
                            # Write a string to the file
                            file.write("\n".join(prompts))
                            if is_mlf:
                                mlfclasses = "\n".join(classes)
                                file.write(f"MLF- Input:\n {mlfclasses}")

                        logger.info(f"Computed trial set {base_count_dir+1} out of {opt.n_sample_sets} for prompt {prompt}")
                        if base_count_dir+1 == opt.n_sample_sets:
                            break

    if len(os.listdir(img_dir)) < opt.n_sample_sets:
        logger.warning(f"Found fewer samples than specified. Fallback to using fewer samples. Given: {os.listdir(img_dir)}, Needed: {opt.n_sample_sets}")

    mean, sd = calc_ms_ssim_for_path_ordered(img_dir, trial_size=opt.trial_size)
    with open(os.path.join(img_dir, "ms_ssim_results.json"), "w") as file:
        as_str =  f"$.{round(float(mean)*100):02d} \pm .{round(float(sd)*100):02d}$"
        json.dump({"mean":float(mean), "sdv":float(mean), "as_string":as_str}, file)


if __name__ == '__main__':
    args = get_compute_mssim()
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
