import argparse
import os
from typing import Tuple
import shutil
from easydict import EasyDict
from logging import Logger

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.transforms import Compose

import utils.misc as utils_misc
import utils.transforms_prop as utils_trans
from datasets import get_dataset


def load_configs(config_path: str) -> Tuple[EasyDict, str]:
    """
    Loads a configuration file from the specified path and returns the configuration along with its name.

    This function is designed to load a configuration file, typically in JSON or YAML format, and
    extract its name (without the file extension) for further use.

    Args:
        config_path (str): The path to the configuration file. This should be a relative or absolute path.

    Returns:
        Tuple[dict, str]: A tuple containing two elements:
            - The first element is a dictionary with the contents of the configuration file.
            - The second element is a string representing the name of the configuration file, without its extension.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        ValueError: If the file extension cannot be determined.

    Example:
        config, config_name = load_configs('configs/my_config.yaml')
        # `config` is now a dictionary with the contents of 'my_config.yaml'
        # `config_name` is the string 'my_config'

    Note:
        The function assumes that the configuration file can be loaded into a dictionary. It relies on the
        `utils_misc.load_config` function, which should be capable of handling the file format of the configuration.
    """
    config = utils_misc.load_config(config_path)
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]

    return config, config_name


def setup_logging(
    config_name: str, 
    input_args: argparse.Namespace, 
    config: EasyDict
) -> Tuple[Logger, SummaryWriter]:
    """
    Sets up logging and TensorBoard writer for a training session, including creating 
    directories for logs and checkpoints, initializing a logger, and saving configuration files.

    Args:
        config_name (str): The name of the configuration, used as a prefix for the log directory.
        input_args (argparse.Namespace): Parsed command-line arguments.
        config (EasyDict): Configuration parameters, typically loaded from a config file.

    Returns:
        Tuple[Logger, SummaryWriter]: A tuple containing a logger instance set up for logging 
        training information and a TensorBoard SummaryWriter instance.

    Side Effects:
        - Creates a new logging directory with a unique name based on `config_name` and `tag`.
        - Creates a subdirectory 'checkpoints' within the logging directory.
        - Initializes a TensorBoard SummaryWriter.
        - Logs the command-line arguments and configuration parameters.
        - Copies the configuration file to the logging directory.
        - Copies the 'models' directory to the logging directory.
    """
    log_dir = utils_misc.get_new_log_dir(input_args.logdir, prefix=config_name, tag=input_args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = utils_misc.get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(input_args)
    logger.info(config)
    shutil.copyfile(input_args.config, os.path.join(log_dir, os.path.basename(input_args.config)))
    shutil.copytree('models', os.path.join(log_dir, 'models'))

    return logger, writer


    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    # Load the config file 
    config, config_name = load_configs(args.config)

    # Set the seed for generating random numbers
    utils_misc.seed_all(config.train.seed)

    # Set up the logging and TensorBoard Writer
    logger, writer = setup_logging(config_name, args, config)

    # Data transformers 
    portein_featurizer = utils_trans.FeaturizeProteinAtom()
    ligand_featurizer = utils_trans.FeaturizeLigandAtom()
    transform = Compose([
        portein_featurizer,
        ligand_featurizer,
    ])

    # Datasets and loaders
    logger.info('Loading dataset ...')
    dataset, subsets = get_dataset(
        config = config.dataset,
        transform = transform,
        emb_path = config.dataset.emb_path if 'emb_path' in config.dataset else None,
        heavy_only = config.dataset.heavy_only
    )

    train_set, val_set, test_set = subsets['train'], subsets['val'], subsets['test']
        logger.info(f'Train set: {len(train_set)} Val set: {len(val_set)} Test set: {len(test_set)}')
