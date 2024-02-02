import yaml
import random
import time
import os
from typing import Optional
import logging

from easydict import EasyDict
import numpy as np
import torch


def load_config(path: str) -> EasyDict:
    """
    Loads a configuration file from the specified path and returns it as an EasyDict object.

    Args:
        path (str): The file path to the YAML configuration file.

    Returns:
        EasyDict: An EasyDict object containing the configuration data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If the YAML file is not properly formatted.
    """
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))
    

def seed_all(seed: int):
    """
    Sets the seed for generating random numbers in torch, numpy, and the built-in random module.

    This function is useful for ensuring reproducibility of results in experiments involving
    random number generation.

    Args:
        seed (int): An integer value to be used as the seed for random number generators.

    Example:
        seed_all(42)  # Sets the same seed for torch, numpy, and random

    Note:
        This function does not guarantee reproducibility across different versions of these libraries
        or across different Python implementations.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_new_log_dir(root: str = 'logs', prefix: Optional[str] = '', tag: Optional[str] = '') -> str:
    """
    Generates a new directory for logging, with an optional prefix and tag, and creates it in the file system.

    Args:
        root (str, optional): The root directory where the log directory will be created. Defaults to 'logs'.
        prefix (Optional[str], optional): An optional prefix for the log directory name. Defaults to ''.
        tag (Optional[str], optional): An optional tag to append to the log directory name. Defaults to ''.

    Returns:
        str: The path to the newly created log directory.

    Example:
        log_dir = get_new_log_dir(root='logs', prefix='training', tag='experiment1')
        # This might create and return a directory like 'logs/training_2021_08_15__12_30_45_experiment1'
    """

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


def get_logger(name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """
    Configures and returns a logger with both stream and file handlers.

    This function sets up a logger to output to both the console and a file (if a log directory is provided).
    It configures the logger to use a specific format for its messages.

    Args:
        name (str): Name of the logger. This is typically the name of the module calling the logger.
        log_dir (Optional[str], optional): Directory where the log file will be saved. If None, no file logging is set up. Defaults to None.

    Returns:
        logging.Logger: Configured logger object.

    Example:
        logger = get_logger(__name__, log_dir='./logs')
        logger.info("This is an info message")
    """

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



