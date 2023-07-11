import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torch
import argparse
import json
from src.evaluation.inception import InceptionV3
from src.evaluation.xrv_fid import calculate_fid_given_paths
from utils import get_comput_fid_args
import random
from log import logger, log_experiment
import os
import logging
import torch
from utils import make_exp_config
from log import formatter as log_formatter
import datetime
import torchxrayvision as xrv
from utils import get_compute_mask_args, make_exp_config, load_model_from_config, collate_batch, img_to_viz, main_setup


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


def main(opt):
    device = torch.device('cuda')
    if opt.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = opt.num_workers

    results = {}
    for fid_model in ["inception", "xrv"]:
        if fid_model == "xrv":
            dims = 1024
            model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)
        elif fid_model == "inception":
            dims = 2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            model = InceptionV3([block_idx]).to(device)

        fid_value = calculate_fid_given_paths([opt.path_src, opt.path_tgt],
                                              opt.batch_size,
                                              device,
                                              fid_model,
                                              model=model,
                                              dims=dims,
                                              num_workers=num_workers)
        logger.info(f"FID of the following paths: {opt.path_src} -- {opt.path_tgt}")
        logger.info(f'{fid_model} FID: {fid_value} --> ${fid_value:.1f}$')
        results[fid_model] = fid_value

    if hasattr(opt, "result_dir") and opt.result_dir is not None:
        with open(os.path.join(opt.result_dir, "fid_results.json"), "w") as file:
            results_file = {}
            results_file["dataset_src"] = opt.path_src
            results_file["dataset_tgt"] = opt.path_tgt
            for fid_model, fid_value in results.items():
                results_file[fid_model] = {"FID": fid_value,
                                          "as_string": f"{fid_value:.1f}"
                                          }
            json.dump(results_file, file)


def get_args():
    parser = argparse.ArgumentParser(description="Compute FID of dataset")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("EXP_NAME", type=str, help="Path to Experiment results")
    parser.add_argument("path_src", type=str, help="Path to first dataset")
    parser.add_argument("path_tgt", type=str, help="Path to second dataset")
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size to use')
    parser.add_argument('--num-workers', type=int, default=16,
                        help=('Number of processes to use for data loading. '
                              'Defaults to `min(8, num_cpus)`'))
    parser.add_argument("--result_dir", type=str, default=None, help="dir to save results in.")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    config = main_setup(args)
    main(config)
