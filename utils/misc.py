import yaml
import random
import time
import os
import logging

from easydict import EasyDict
import numpy as np
import torch


def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))
    

def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_new_log_dir(root: str = 'logs', prefix='', tag=''):

    # Create directory name based on current time, prefix, and tag
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix:
        fn = f'{prefix}_{fn}'
    if tag:
        fn = f'{fn}_{tag}'

    # Join the directory name with the root path and create the directory
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir, exist_ok=True)

    return log_dir


def get_logger(name, log_dir=None):

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    # Stream handler (console)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler (log file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger



