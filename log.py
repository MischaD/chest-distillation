import logging
from logging.handlers import RotatingFileHandler
import sys
import os
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(filename)s-%(funcName)s-%(lineno)04d | %(levelname)s | %(message)s')

os.makedirs('./log/console_logs/', exist_ok=True)

file_handler = logging.FileHandler('./log/console_logs/' + datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + '.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)