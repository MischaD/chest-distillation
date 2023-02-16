import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torch
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
        logger.info(f'{fid_model} FID: {fid_value} --> ${fid_value:2.01f}$', fid_value)
        results[fid_model] = fid_value

    for fid_model, fid_value in results.items():
        print(f'{fid_model} FID: {fid_value} --> ${fid_value:2.01f}$', fid_value)


if __name__ == '__main__':
    args = get_comput_fid_args()
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
