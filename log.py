import logging
from logging.handlers import RotatingFileHandler
import sys
import os
import yaml
import pprint
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(filename)s-%(funcName)s-%(lineno)04d | %(levelname)s | %(message)s')

log_dir = "./log/" + datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
os.makedirs(log_dir, exist_ok=True)

file_handler = logging.FileHandler(os.path.join(log_dir,'console.log'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

def log_experiment(_logger, args, config_path):
    _logger.debug("="*40 + "Command line arguments: " + "="*40)
    _logger.info("Args: " + str(args))
    _logger.debug("="*40 + " Experiment settings " + "="*40)
    _logger.info("Exp Path: " + args.EXP_PATH)
    _logger.debug(open(args.EXP_PATH).read())
    _logger.debug("="*40 + " Config file " + "="*40)
    _logger.debug("Path: " + config_path)
    config_file = yaml.load(open(config_path), Loader=yaml.SafeLoader)
    _logger.debug(pprint.PrettyPrinter(depth=10).pformat(config_file))