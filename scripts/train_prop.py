import argparse
import os
import shutil
import warnings

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.transforms import Compose
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit.Chem.rdchem import HybridizationType
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

import utils.misc as utils_misc
from datasets import get_dataset
from models.prop_model import PropPredNet, PropPredNetEnc

KMAP = {'Ki': 1, 'Kd': 2, 'IC50': 3}


def load_configs(config_path):
    config = utils_misc.load_config(config_path)
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    return config, config_name


def setup_logging(config_name, input_args, config):
    log_dir = utils_misc.get_new_log_dir(input_args.logdir, prefix=config_name, tag=input_args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = utils_misc.get_logger('data', log_dir)
    writer = SummaryWriter(log_dir)
    logger.info(input_args)
    logger.info(config)
    shutil.copyfile(input_args.config, os.path.join(log_dir, os.path.basename(input_args.config)))
    shutil.copytree('models', os.path.join(log_dir, 'models'))

    return logger, writer, log_dir, ckpt_dir


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
        return self.atomic_numbers.size(0) + self.max_num_aa + 1 # number of possible elements + number of possible amino acids + 1

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


def get_optimizer(cfg, model):
    if cfg.type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2,)
        )
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)
    

def get_scheduler(cfg, optimizer):
    if cfg.type == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr
        )
    elif cfg.type == 'warmup_plateau':
        return GradualWarmupScheduler(
            optimizer,
            multiplier=cfg.multiplier,
            total_epoch=cfg.total_epoch,
            after_scheduler=ReduceLROnPlateau(
                optimizer,
                factor=cfg.factor,
                patience=cfg.patience,
                min_lr=cfg.min_lr
            )
        )
    elif cfg.type == 'expmin':
        return ExponentialLR_with_minLr(
            optimizer,
            gamma=cfg.factor,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == 'expmin_milestone':
        gamma = np.exp(np.log(cfg.factor) / cfg.milestone)
        return ExponentialLR_with_minLr(
            optimizer,
            gamma=gamma,
            min_lr=cfg.min_lr,
        )
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)
    

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


# customize exp lr scheduler with min lr
class ExponentialLR_with_minLr(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, min_lr=1e-4, last_epoch=-1, verbose=False):
        self.gamma = gamma
        self.min_lr = min_lr
        super(ExponentialLR_with_minLr, self).__init__(optimizer, gamma, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return self.base_lrs
        return [max(group['lr'] * self.gamma, self.min_lr)
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
                for base_lr in self.base_lrs]
    

def get_eval_scores(ypred_arr, ytrue_arr, logger, prefix='All'):
    if len(ypred_arr) == 0:
        return None
    rmse = np.sqrt(mean_squared_error(ytrue_arr, ypred_arr))
    mae = mean_absolute_error(ytrue_arr, ypred_arr)
    r2 = r2_score(ytrue_arr, ypred_arr)
    pearson, ppval = pearsonr(ytrue_arr, ypred_arr)
    spearman, spval = spearmanr(ytrue_arr, ypred_arr)
    mean = np.mean(ypred_arr)
    std = np.std(ypred_arr)
    logger.info("Evaluation Summary:")
    logger.info(
        "[%4s] num: %3d, RMSE: %.3f, MAE: %.3f, "
        "R^2 score: %.3f, Pearson: %.3f, Spearman: %.3f, mean/std: %.3f/%.3f" % (
            prefix, len(ypred_arr), rmse, mae, r2, pearson, spearman, mean, std))
    return rmse


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
    logger, writer, log_dir, ckpt_dir = setup_logging(config_name, args, config)

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
    model = model.to(args.device)
    logger.info(f"# trainable parameters: {utils_misc.count_parameters(model) / 1e6:.4f} M")

    # Optimizer and scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)


    def train(epoch):
        model.train()
        optimizer.zero_grad()
        it = 0
        num_it = len(train_loader)
        for batch in tqdm(train_loader, dynamic_ncols=True, desc=f'Epoch {epoch}', position=1):
            it += 1
            batch = batch.to(args.device)
            # compute loss
            loss = model.get_loss(batch, pos_noise_std=config.train.pos_noise_std)
            loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            if it % config.train.report_iter == 0:
                logger.info('[Train] Epoch %03d Iter %04d | Loss %.6f | Lr %.4f * 1e-3' % (
                    epoch, it, loss.item(), optimizer.param_groups[0]['lr'] * 1000
                ))

            writer.add_scalar('train/loss', loss, it + epoch * num_it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it + epoch * num_it)
            writer.add_scalar('train/grad', orig_grad_norm, it + epoch * num_it)
            writer.flush()

    def validate(epoch, data_loader, scheduler, writer, prefix='Validate'):
        sum_loss, sum_n = 0, 0
        ytrue_arr, ypred_arr = [], []
        y_kind = []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(data_loader, desc=prefix):
                batch = batch.to(args.device)
                loss, pred = model.get_loss(batch, pos_noise_std=0., return_pred=True)
                sum_loss += loss.item() * len(batch.y)
                sum_n += len(batch.y)
                ypred_arr.append(pred.view(-1))
                ytrue_arr.append(batch.y)
                y_kind.append(batch.kind)
        avg_loss = sum_loss / sum_n
        logger.info('[%s] Epoch %03d | Loss %.6f' % (
            prefix, epoch, avg_loss,
        ))
        ypred_arr = torch.cat(ypred_arr).cpu().numpy().astype(np.float64)
        ytrue_arr = torch.cat(ytrue_arr).cpu().numpy().astype(np.float64)
        y_kind = torch.cat(y_kind).cpu().numpy()
        rmse = get_eval_scores(ypred_arr, ytrue_arr, logger)
        for k, v in KMAP.items():
            get_eval_scores(ypred_arr[y_kind == v], ytrue_arr[y_kind == v], logger, prefix=k)

        if scheduler:
            if config.train.scheduler.type == 'plateau':
                scheduler.step(avg_loss)
            elif config.train.scheduler.type == 'warmup_plateau':
                scheduler.step_ReduceLROnPlateau(avg_loss)
            else:
                scheduler.step()

        if writer:
            writer.add_scalar('val/loss', avg_loss, epoch)
            writer.add_scalar('val/rmse', rmse, epoch)
            writer.flush()

        return avg_loss
    
    try:
        best_val_loss = float('inf')
        best_val_epoch = 0
        patience = 0
        for epoch in range(1, config.train.max_epochs + 1):
            # with torch.autograd.detect_anomaly():
            train(epoch)
            if epoch % config.train.val_freq == 0 or epoch == config.train.max_epochs:
                val_loss = validate(epoch, val_loader, scheduler, writer)
                validate(epoch, test_loader, scheduler=None, writer=None, prefix='Test')

                if val_loss < best_val_loss:
                    patience = 0
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    logger.info(f'Best val achieved at epoch {epoch}, val loss: {best_val_loss:.3f}')
                    logger.info(f'Eval on Test set:')
                    validate(epoch, test_loader, scheduler=None, writer=None, prefix='Test')
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % epoch)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                    }, ckpt_path)
                    logger.info(f'Model {log_dir}/{epoch}.pt saved!')
                else:
                    patience += 1
                    logger.info(f'Val loss does not improve, patience: {patience} '
                                f'(Best val loss: {best_val_loss:.3f} at epoch {best_val_epoch})')

    except KeyboardInterrupt:
        logger.info('Terminating...')

    