import argparse
import os
import multiprocessing as mp
from functools import partial
import sys
from io import StringIO
import pickle

from tqdm.auto import tqdm
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType, HybridizationType
import torch
from torch_scatter import scatter

RDLogger.DisableLog('rdApp.*')

KMAP = {'Ki': 1, 'Kd': 2, 'IC50': 3}


def parse_pdbbind_index_file(raw_path, subset='refined'):
    all_index = []
    version = int(raw_path[-4:])
    assert version >= 2016
    if subset == 'refined':
        data_path = os.path.join(raw_path, f'refined-set')
        index_path = os.path.join(data_path, 'index', f'INDEX_refined_data.{version}')
    elif subset == 'general':
        data_path = os.path.join(raw_path, f'general-set-except-refined')
        index_path = os.path.join(data_path, 'index', f'INDEX_general_PL_data.{version}')
    else:
        raise ValueError(subset)

    all_files = os.listdir(data_path)
    with open(index_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#'): continue
        index, res, year, pka, kv = line.split('//')[0].strip().split()
        kind = [v for k, v in KMAP.items() if k in kv]
        assert len(kind) == 1
        if index in all_files:
            all_index.append([index, res, year, pka, kind[0]])
    return all_index


def process_item(item, args):
    pdb_idx, res, year, pka, kind = item
    try:
        if args.subset == 'refined':
            pdb_path = os.path.join(args.source, 'refined-set', pdb_idx)
        elif args.subset == 'general':
            pdb_path = os.path.join(args.source, 'general-set-except-refined', pdb_idx)
        else:
            raise ValueError(args.subset)

        protein_path = os.path.join(pdb_path, f'{pdb_idx}_protein.pdb')
        ligand_sdf_path = os.path.join(pdb_path, f'{pdb_idx}_ligand.sdf')
        ligand_mol2_path = os.path.join(pdb_path, f'{pdb_idx}_ligand.mol2')
        mol, problem, ligand_path = read_mol(ligand_sdf_path, ligand_mol2_path)
        if problem:
            print('Read mol error.', item)
            return None, ligand_path, res, pka, kind

        protein = PDBProtein(protein_path)
        ligand = parse_sdf_file_mol(ligand_path, heavy_only=False)
        pocket_path = os.path.join(pdb_path, f'{pdb_idx}_pocket{args.radius}.pdb')
        if not os.path.exists(pocket_path):
            pdb_block_pocket = protein.residues_to_pdb_block(
                protein.query_residues_ligand(ligand, args.radius)
            )
            with open(pocket_path, 'w') as f:
                f.write(pdb_block_pocket)
        return pocket_path, ligand_path, res, pka, kind

    except Exception as e:
        print('Exception occured.', item)
        print(e)
        return None, ligand_path, res, pka, kind
    

def read_mol(sdf_fileName, mol2_fileName, verbose=False):
    Chem.WrapLogs()
    stderr = sys.stderr
    sio = sys.stderr = StringIO()
    mol = Chem.MolFromMolFile(sdf_fileName, sanitize=False)
    problem = False
    ligand_path = None
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        sm = Chem.MolToSmiles(mol)
        ligand_path = sdf_fileName
    except Exception as e:
        sm = str(e)
        problem = True
    if problem:
        mol = Chem.MolFromMol2File(mol2_fileName, sanitize=False)
        problem = False
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
            sm = Chem.MolToSmiles(mol)
            problem = False
            ligand_path = mol2_fileName
        except Exception as e:
            sm = str(e)
            problem = True

    if verbose:
        print(sio.getvalue())
    sys.stderr = stderr
    return mol, problem, ligand_path
    
class PDBProtein(object):
    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        }

    AA_NAME_NUMBER = {k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())}

    BACKBONE_NAMES = ["CA", "C", "N", "O"]

    def __init__(self, data, mode='auto'):
        super().__init__()
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line[0:6].strip() == 'ATOM':
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': 'ATOM',
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break  # Some PDBs have more than 1 model.

    def _parse(self):
        # Process atoms
        residues_tmp = {}
        for atom in self._enum_formatted_atom_lines():
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            self.atoms.append(atom)
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            next_ptr = len(self.element)
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
            self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])

            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in residues_tmp:
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                }
            else:
                assert residues_tmp[chain_res_id]['name'] == atom['res_name']
                assert residues_tmp[chain_res_id]['chain'] == atom['chain']
                residues_tmp[chain_res_id]['atoms'].append(next_ptr)

        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in self.BACKBONE_NAMES:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            residue['center_of_mass'] = sum_pos / sum_mass

        # Process backbone atoms of residues
        for residue in self.residues:
            self.amino_acid.append(self.AA_NAME_NUMBER[residue['name']])
            self.center_of_mass.append(residue['center_of_mass'])
            for name in self.BACKBONE_NAMES:
                pos_key = 'pos_%s' % name  # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])

    def to_dict_atom(self):
        return {
            'element': np.array(self.element, dtype=np.int64),
            'molecule_name': self.title,
            'pos': np.array(self.pos, dtype=np.float32),
            'is_backbone': np.array(self.is_backbone, dtype=np.bool),
            'atom_name': self.atom_name,
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)
        }

    def to_dict_residue(self):
        return {
            'amino_acid': np.array(self.amino_acid, dtype=np.int64),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'pos_C': np.array(self.pos_C, dtype=np.float32),
            'pos_N': np.array(self.pos_N, dtype=np.float32),
            'pos_O': np.array(self.pos_O, dtype=np.float32),
        }

    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            print(residue[criterion], distance)
            if distance < radius:
                selected.append(residue)
        return selected

    def query_residues_ligand(self, ligand, radius, criterion='center_of_mass'):
        selected = []
        sel_idx = set()
        # The time-complexity is O(mn).
        for center in ligand['pos']:
            for i, residue in enumerate(self.residues):
                distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        return selected

    def residues_to_pdb_block(self, residues, name='POCKET'):
        block = "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        for residue in residues:
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        return block


def parse_sdf_file_mol(path, heavy_only=True, mol=None):
    if mol is None:
        if path.endswith('.sdf'):
            mol = Chem.MolFromMolFile(path, sanitize=False)
        elif path.endswith('.mol2'):
            mol = Chem.MolFromMol2File(path, sanitize=False)
        else:
            raise ValueError
        Chem.SanitizeMol(mol)
        if heavy_only:
            mol = Chem.RemoveHs(mol)
            
    feat_mat = get_ligand_atom_features(mol)

    ptable = Chem.GetPeriodicTable()

    num_atoms = mol.GetNumAtoms()
    pos = mol.GetConformer().GetPositions()

    element = []
    accum_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    accum_mass = 0.0
    for atom_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        atomic_number = atom.GetAtomicNum()
        element.append(atomic_number)
        x, y, z = pos[atom_idx]
        atomic_weight = ptable.GetAtomicWeight(atomic_number)
        accum_pos += np.array([x, y, z]) * atomic_weight
        accum_mass += atomic_weight
    center_of_mass = np.array(accum_pos / accum_mass, dtype=np.float32)
    element = np.array(element, dtype=np.int64)
    pos = np.array(pos, dtype=np.float32)

    row, col, edge_type = [], [], []
    BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]
    edge_index = np.array([row, col], dtype=np.int64)
    edge_type = np.array(edge_type, dtype=np.int64)
    perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    data = {
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat
    }
    return data


def get_ligand_atom_features(rdmol):
    num_atoms = rdmol.GetNumAtoms()
    atomic_number = []
    aromatic = []
    hybrid = []
    degree = []
    for atom_idx in range(num_atoms):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        HYBRID_TYPES = {t: i for i, t in enumerate(HybridizationType.names.values())}
        hybrid.append(HYBRID_TYPES[hybridization])
        degree.append(atom.GetDegree())
    node_type = torch.tensor(atomic_number, dtype=torch.long)

    row, col = [], []
    for bond in rdmol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
    row = torch.tensor(row, dtype=torch.long)
    col = torch.tensor(col, dtype=torch.long)
    hs = (node_type == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=num_atoms).numpy()
    feat_mat = np.array([atomic_number, aromatic, degree, num_hs, hybrid], dtype=np.int64).transpose()
    return feat_mat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='ml-data/mahmoud/pdbbind/pdbbind_v2020')
    parser.add_argument('--subset', type=str, default='refined')
    parser.add_argument('--refined_index_pkl', type=str, default=None)
    parser.add_argument('--radius', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())
    args = parser.parse_args()
    
    index = parse_pdbbind_index_file(args.source, args.subset)

    pool = mp.Pool(args.num_workers)
    index_pocket = []
    for item_pocket in tqdm(pool.imap_unordered(partial(process_item, args=args), index), total=len(index)):
        index_pocket.append(item_pocket)
    pool.close()

    valid_index_pocket = []
    for index in index_pocket:
        if index[0] is not None:
            valid_index_pocket.append(index)

    save_path = os.path.join(args.source, f'pocket_{args.radius}_{args.subset}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    index_path = os.path.join(save_path, 'index.pkl')

    if args.subset == 'general' and args.refined_index_pkl is not None:
        with open(args.refined_index_pkl, 'rb') as f:
            refined_index = pickle.load(f)
        valid_index_pocket += refined_index
    with open(index_path, 'wb') as f:
        pickle.dump(valid_index_pocket, f)
    print('Done. %d protein-ligand pairs in total.' % len(valid_index_pocket))


