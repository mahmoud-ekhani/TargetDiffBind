import argparse
import os
import shutil
from easydict import EasyDict
from logging import Logger

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.transforms import Compose
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit.Chem.rdchem import HybridizationType

import utils.misc as utils_misc
from datasets import get_dataset


def load_configs(config_path):
    config = utils_misc.load_config(config_path)
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    return config, config_name


def setup_logging(config_name, input_args, config):
    log_dir = utils_misc.get_new_log_dir(input_args.logdir, prefix=config_name, tag=input_args.tag)
    # ckpt_dir = os.path.join(log_dir, 'checkpoints')
    # os.makedirs(ckpt_dir, exist_ok=True)
    logger = utils_misc.get_logger('data', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(input_args)
    logger.info(config)
    shutil.copyfile(input_args.config, os.path.join(log_dir, os.path.basename(input_args.config)))
    # shutil.copytree('models', os.path.join(log_dir, 'models'))

    return logger, writer


class ProteinLigandData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance['protein_' + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance['ligand_' + key] = item

        instance['ligand_nbh_list'] = {i.item(): [j.item() for k, j in enumerate(instance.ligand_bond_index[1])
                                                  if instance.ligand_bond_index[0, k].item() == i]
                                       for i in instance.ligand_bond_index[0]}
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ligand_bond_index':
            return self['ligand_element'].size(0)
        else:
            return super().__inc__(key, value)


class FeaturizeProteinAtom(object):

    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])    # H, C, N, O, S, Se
        self.max_num_aa = 20

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1

    def __call__(self, data: ProteinLigandData):
        element = data.protein_element.view(-1, 1) == self.atomic_numbers.view(1, -1)   # (N_atoms, N_elements)
        amino_acid = F.one_hot(data.protein_atom_to_aa_type, num_classes=self.max_num_aa)
        is_backbone = data.protein_is_backbone.view(-1, 1).long()
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        data.protein_atom_feature = x
        return data


class FeaturizeLigandAtom(object):
    ATOM_FEATS = {'AtomicNumber': 1, 'Aromatic': 1, 'Degree': 6, 'NumHs': 6, 'Hybridization': len(HybridizationType.values)}
    
    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17])  # H, C, N, O, F, P, S, Cl

    @property
    def num_properties(self):
        return sum(self.ATOM_FEATS.values())

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.num_properties

    def __call__(self, data: ProteinLigandData):
        element = data.ligand_element.view(-1, 1) == self.atomic_numbers.view(1, -1)   # (N_atoms, N_elements)
        # convert some features to one-hot vectors
        atom_feature = []
        for i, (k, v) in enumerate(self.ATOM_FEATS.items()):
            feat = data.ligand_atom_feature[:, i:i+1]
            if v > 1:
                feat = (feat == torch.LongTensor(list(range(v))).view(1, -1))
            else:
                if k == 'AtomicNumber':
                    feat = feat / 100.
            atom_feature.append(feat)

        atom_feature = torch.cat(atom_feature, dim=-1)
        data.ligand_atom_feature_full = torch.cat([element, atom_feature], dim=-1)
        return data
    

def get_dataloader(train_set, val_set, test_set, config):
    follow_batch = ['protein_element', 'ligand_element']
    collate_exclude_keys = ['ligand_nbh_list']
    train_loader = DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=follow_batch,
        exclude_keys=collate_exclude_keys
    )
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=follow_batch, exclude_keys=collate_exclude_keys)
    test_loader = DataLoader(test_set, config.train.batch_size, shuffle=False,
                             follow_batch=follow_batch, exclude_keys=collate_exclude_keys)
    return train_loader, val_loader, test_loader


def get_model(config, protein_atom_feat_dim, ligand_atom_feat_dim):
    if config.model.encoder.name == 'egnn_enc':
        model = PropPredNetEnc(
            config.model,
            protein_atom_feature_dim=protein_atom_feat_dim,
            ligand_atom_feature_dim=ligand_atom_feat_dim,
            enc_ligand_dim=config.model.enc_ligand_dim,
            enc_node_dim=config.model.enc_node_dim,
            enc_graph_dim=config.model.enc_graph_dim,
            enc_feature_type=config.model.enc_feature_type,
            output_dim=1
        )
    else:
        model = PropPredNet(
            config.model,
            protein_atom_feature_dim=protein_atom_feat_dim,
            ligand_atom_feature_dim=ligand_atom_feat_dim,
            output_dim=3
        )
    return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/pdbbind_general_egnn.yml')
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
    portein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
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
    train_loader, val_loader, test_loader = get_dataloader(train_set, val_set, test_set, config)

    # Model
    logger.info('Building model ...')
    model = get_model(config, portein_featurizer.feature_dim, ligand_featurizer.feature_dim)
