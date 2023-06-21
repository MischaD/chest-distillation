import random
import argparse
import torch.multiprocessing as mp
from log import logger, log_experiment
import os
import logging
import pandas as pd
import torch
from torch.utils.data.distributed import DistributedSampler
from utils import get_sample_model_args, make_exp_config, load_model_from_config, collate_batch, img_to_viz
from log import formatter as log_formatter
from src.datasets import get_dataset
import datetime
from src.ldm.util import instantiate_from_config
from src.ldm.models.diffusion.ddim import DDIMSampler
from src.ldm.models.diffusion.plms import PLMSSampler
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from torch import autocast
from tqdm import tqdm
from PIL import Image
from pytorch_lightning import seed_everything
import time
import random
import numpy as np
from einops import rearrange
from src.datasets.sample_dataset import get_mscxr_synth_dataset, get_mscxr_synth_dataset_size_n
from src.ldm.encoders.modules import OpenClipDummyTokenizer
from utils import main_setup
from src.datasets.mscoco import get_classes_for_caption


def load_datalist(config, model):
    if config.data_dir.endswith("mscoco"):
        dataset = get_dataset(config, "test_sample")
        dataset.load_precomputed(model)
        ds = []
        labels = []
        for i in range(len(dataset)):
            caption = dataset[i]["captions"].split("|")[0]
            classes = get_classes_for_caption(caption)

            ds.append(caption)
            labels.append("_".join(classes))

        synth_dataset = [{"labels": {k: v}} for k, v in zip(labels, ds)]
        if hasattr(config, "N") and config.N is not None:
            synth_dataset = synth_dataset[:config.N]
            labels = labels[:config.N]
    else:
        if hasattr(config, "label_list_path"):
            df = pd.read_csv(config.label_list_path)
            synth_dataset = [{"labels":{k: v}} for k, v in zip(list(df["Finding Labels"]), list(df["impression"]))]
            labels = set(df["Finding Labels"].unique())
            if hasattr(config, "N") and config.N is not None:
                synth_dataset = synth_dataset[:config.N]
        else:
            if config.use_mscxrlabels:
                dataset = get_dataset(config, "test")
                dataset.load_precomputed(model)
                if hasattr(config, "N") and config.N is not None:
                    logger.info(f"Sampling {config.N} from dataset mscxr")
                    synth_dataset, labels = get_mscxr_synth_dataset_size_n(config.N, dataset)
                else:
                    synth_dataset, labels = get_mscxr_synth_dataset(config, dataset)
            else:
                dataset = get_dataset(config, "testp19")
                dataset.load_precomputed(model)
                if hasattr(config, "N") and config.N is not None:
                    logger.info(f"Sampling {config.N} from dataset p19")
                    synth_dataset, labels = get_mscxr_synth_dataset_size_n(config.N, dataset, finding_key="impression",
                                                                           label_key="finding_labels")
                else:
                    synth_dataset, labels = get_mscxr_synth_dataset(config, dataset, finding_key="impression", label_key="finding_labels")
    return synth_dataset, labels


def sample_model(rank, config, world_size):
    logger.info(f"Rank {rank} - Total {world_size}")
    if not hasattr(config, "img_dir") or config.img_dir is None:
        img_dir = os.path.join(config.log_dir, "generated")
    else:
        img_dir = config.img_dir

    logger.info(f"Saving Images to {img_dir}")
    model_config = OmegaConf.load(f"{config.config_path_inference}")

    is_mlf = False
    if hasattr(config, "mlf_args"):
        is_mlf = config.mlf_args.get("multi_label_finetuning", False)
        logger.info(f"Overwriting default arguments of config with {config.mlf_args}")
        model_config["model"]["params"]["attention_regularization"] = config.mlf_args.get("attention_regularization")
        model_config["model"]["params"]["cond_stage_key"] = config.mlf_args.get("cond_stage_key")
        model_config["model"]["params"]["cond_stage_config"]["params"]["multi_label_finetuning"] = config.mlf_args.get("multi_label_finetuning")

    is_rali = False
    if hasattr(config, "mlf_args") and config.mlf_args["rali"] is not None:
        model_config["model"]["params"]["rali"] = True
        is_rali = True

    model = load_model_from_config(model_config, f"{config.ckpt}")

    device = torch.device(rank)
    model = model.to(device)
    seed_everything(config.seed)

    synth_dataset, labels = load_datalist(config, model)

    if is_mlf:
        tokenizer = OpenClipDummyTokenizer(config.seed, config.mlf_args.get("append_invariance_tokens", False), config.mlf_args.get("single_healthy_class_token", False))
        if config.seed == 4200:
            tokenization = tokenizer("Consolidation|Cardiomegaly|Pleural Effusion".split("|"))
            if len(tokenization) != 9:
                tokenization = tokenization[1:-1]
            #assert tokenization[1] == 15598 and tokenization[3] == 22073
        model.cond_stage_model.set_multi_label_tokenizer(tokenizer)
    elif is_rali:
        tokenizer = OpenClipDummyTokenizer(config.seed, False, True, rali=True)
        model.cond_stage_model.set_multi_label_tokenizer(tokenizer)

    os.makedirs(img_dir, exist_ok=True)
    for label in labels:
        os.makedirs(os.path.join(img_dir, label), exist_ok=True)

    batch_size = config.sample.iou_batch_size

    if config.sample.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    seed_everything(time.time())

    #batched_dataset = [synth_dataset[i:i+batch_size] for i in range(0, len(synth_dataset), batch_size)]

    data_sampler = DistributedSampler(synth_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(synth_dataset,
                            batch_size=config.sample.iou_batch_size,
                            shuffle=False,
                            num_workers=0,  #opt.num_workers,
                            collate_fn=collate_batch,
                            drop_last=False,
                            sampler=data_sampler,
                            )
    log_first = True
    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():
                for samples in tqdm(dataloader, desc="Sampling", total=len(dataloader)):
                    uc = None
                    prompts = [list(x.values())[0] for x in samples["labels"]]
                    classes = [list(x.keys())[0] for x in samples["labels"]]
                    if is_mlf:
                        c = model.get_learned_conditioning(classes)
                    elif is_rali:
                        c = model.get_learned_conditioning([classes, prompts])
                    else:
                        c = model.get_learned_conditioning(prompts)

                    if config.stable_diffusion.scale != 1.0:
                        if is_rali:
                            uc = model.get_learned_conditioning([len(c) * ["No Finding",], len(c) * [""]])
                        else:
                            uc = model.get_learned_conditioning(len(c) * [""])

                    start_code = torch.randn([len(c), config.datasets.C, config.datasets.H // config.datasets.f, config.datasets.W // config.datasets.f], device=device)

                    shape = [config.datasets.C, config.datasets.H // config.datasets.f, config.datasets.W // config.datasets.f]
                    output, _ = sampler.sample(S=config.sample.ddim_steps,
                                                conditioning=c.to(device),
                                                batch_size=len(samples["labels"]),
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=config.stable_diffusion.scale,
                                                unconditional_conditioning=uc.to(device),
                                                x_T=start_code.to(device))

                    output = model.decode_first_stage(output)
                    output = torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0)

                    output = output.cpu()
                    for i in range(len(output)):
                        sample_path = os.path.join(os.path.join(img_dir, classes[i]))
                        base_count = len(os.listdir(sample_path))
                        sample = 255. * rearrange(output[i].numpy(), 'c h w -> h w c')
                        img_path = os.path.join(sample_path, f"{base_count:05}_{rank}.png")
                        logger.info(f"Saving sample to {img_path}")
                        Image.fromarray(sample.astype(np.uint8)).save(img_path)

                    if log_first:
                        log_first = False
                        import matplotlib.pyplot as plt
                        fig, axes = plt.subplots(nrows=2, ncols=int(len(output) // 2), figsize=(int(4 * len(output) // 2), 8))

                        images = (255. * rearrange(output.to(torch.float32).clip(0,1).numpy(), 'b c h w -> b h w c')).astype(np.uint8)
                        for i, (image, caption) in enumerate(zip(images, prompts)):
                            axes.flat[i].imshow(image)
                            axes.flat[i].axis('off')
                            axes.flat[i].set_title(caption)

                        # Adjust the spacing between subplots
                        plt.tight_layout()
                        plt.savefig(os.path.join(config.log_dir, f"{os.path.basename(config.ckpt)}_samples.pdf"))


def get_args():
    parser = argparse.ArgumentParser(description="Sample Model")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("EXP_NAME", type=str, help="Path to Experiment results")
    parser.add_argument("--ckpt", type=str, default="train")
    parser.add_argument("--img_dir", type=str, default=None, help="dir to save images in. Default will be inside log dir and should be used!")
    parser.add_argument("--use_mscxrlabels", action="store_true", default=False, help="")
    parser.add_argument("--N", type=int, default=None, help="")
    parser.add_argument("--label_list_path", type=str, default=None, help="")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    config = main_setup(args)
    world_size = torch.cuda.device_count()

    # mask computation
    mp.spawn(
        sample_model,
        args=(config, world_size),
        nprocs=world_size
    )
