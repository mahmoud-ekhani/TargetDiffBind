# TargetDiffBind

## Installation

### Dependencies

### Installation via Conda and Pip

To install the required packages, follow these steps. The script will automatically detect and use the appropriate CUDA version for PyTorch installation:

```bash
# Create and activate a new Conda environment
conda create -n targetdiffbind python=3.8
conda activate targetdiffbind

# Install PyTorch with CUDA support that matches the system's CUDA version
conda install pytorch -c pytorch -c nvidia

# Install PyTorch Geometric
conda install pyg -c pyg

# Install other dependencies
conda install rdkit openbabel tensorboard pyyaml easydict python-lmdb -c conda-forge
```

## Data Preparation

### Using the PDBBind Dataset

#### Downloading the Dataset
- The PDBBind dataset can be downloaded from [PDBBind website](http://www.pdbbind.org.cn).
- Save the downloaded `refined` and `general` sets in the `data/pdbbind_v{YEAR}` directory, where `{YEAR}` corresponds to the version of the dataset (e.g., `2020`).

#### Preparing the Data

1. **Unzipping the Data**: 
   Create a directory for the dataset and extract the contents. Replace `{YEAR}` with the appropriate year.
   ```bash
   mkdir -p data/pdbbind_v{YEAR} && tar -xzvf data/pdbbind_v{YEAR}_refined.tar.gz -C data/pdbbind_v{YEAR}
   ```
   For PDBBind v2020, the command would be:
   ```bash
   mkdir -p data/pdbbind_v2020 && tar -xzvf data/pdbbind_v2020_refined.tar.gz -C data/pdbbind_v2020
   ```

2. **Extracting Pockets**:
   Use the following command to extract 10A pockets. Replace `{YEAR}` with the dataset year.
   ```bash
   python utils/extract_pdbbind_pockets.py --source data/pdbbind_v{YEAR} --subset refined --refined_index_pkl data/pdbbind_v{YEAR}/pocket_10_refined/index.pkl
   ```
   For PDBBind v2020, it becomes:
   ```bash
   python utils/extract_pdbbind_pockets.py --source data/pdbbind_v2020 --subset refined --refined_index_pkl data/pdbbind_v2020/pocket_10_refined/index.pkl
   ```
3. **Dataset Splitting**:
   We use the PDBbind core set (CASF-2016) as the test set. This file can be download from [PDBbind core set](https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/CASF-2016.tar.gz).

   Use the following command to split the dataset between training and validation sets. Replace `{YEAR}` with the dataset year.
   ```bash
   python utils/pdbbind_split.py --index_path data/pdbbind_v{YEAR}/pocket_10_refined/index.pkl --split_mode coreset --test_path data/pdbbind_v{YEAR}/CASF-2016/coreset
   ```
   For PDBBind v2020, it becomes:
   ```bash
    python utils/pdbbind_split.py --index_path data/pdbbind_v2020/pocket_10_refined/index.pkl --split_mode coreset --test_path data/pdbbind_v2020/CASF-2016/coreset
   ```

