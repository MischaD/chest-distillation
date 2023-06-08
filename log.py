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

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)

logger.addHandler(stdout_handler)

def log_experiment(_logger, args):
    _logger.debug("="*40 + "Command line arguments: " + "="*40)
    _logger.info("Args: " + str(args))
    _logger.debug("="*40 + " Experiment settings " + "="*40)
    _logger.info("Exp Path: " + args.EXP_PATH)
    _logger.debug(open(args.EXP_PATH).read())
    _logger.debug("="*40 + " Config file " + "="*40)