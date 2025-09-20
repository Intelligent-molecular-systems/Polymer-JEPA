# Polymer-JEPA


### Setup Environment

#### Option 1: Conda Environment
```bash
conda create -n polymer-jepa python=3.10
conda activate polymer-jepa
pip install -r requirements.txt
```

#### Option 2: PyTorch Geometric Installation
```bash
conda create -n polymer-jepa python=3.10
conda activate polymer-jepa
pip install torch-geometric==2.3.0 torch-sparse torch-scatter torch-cluster -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
pip install torch==2.1 wandb networkx yacs metis kaleido tensorboard einops pillow tqdm pandas rdkit matplotlib plotly seaborn umap
```

#### Option 3: Containerization
Alternatively setup the environment using containerization (see `environment_apptainer.yml` and `apptainer_env.def` files).

### Project Structure

See [STRUCTURE.md](STRUCTURE.md) for detailed project organization.

### Codebase Overview

#### Core Files
- **`main.py`** - Main entry point that orchestrates the complete training pipeline
- **`src/config.py`** - Configuration management with all hyperparameters and settings
- **`src/data.py`** - Dataset creation, loading, and data selection strategies
- **`src/pretrain.py`** - JEPA pretraining implementation with self-supervised learning
- **`src/finetune.py`** - Supervised finetuning for downstream property prediction
- **`src/training.py`** - Core training utilities and optimization functions

#### Model Architecture
- **`src/JEPA_models/`**
  - `PolymerJEPAv1.py` - JEPA model V1 (using transformers as encoders)
  - `PolymerJEPAv2.py` - JEPA model V2, no transformers, GNNs as encoders. This is the best-performing version utilized in the paper as default.
  - `WDNodeMPNN.py` - Weighted-directed (multi-layer) node message passing neural network
  - `WDNodeMPNNLayer.py` - Single layer weighted-directed node message passing, designed to be used as a building block
  - `model_utils/` - Utility functions for model components

#### Analysis & Visualization
- **`src/visualize.py`** - Embedding space visualization and result plotting
- **`src/plots.py`** - Additional plotting utilities
- **`src/logger.py`** - Weights & Biases logging configuration
- **`src/extract_results.py`** - Result extraction and analysis tools

#### Baseline Methods
- **`src/train_test_rf.py`** - Random Forest baseline for Aldeghi dataset
- **`src/train_test_rf_diblock.py`** - Random Forest baseline for diblock dataset
- **`src/aggregate_results_rf.py`** - Random Forest result aggregation
- **`src/linearFinetune.py`** - Linear probe evaluation

#### Data Processing
- **`src/featurization_utils/`**
  - `featurization.py` - Core molecular graph featurization from SMILES
  - `featurization_helper.py` - Helper functions for atom/bond features and molecular processing
- **`src/transform.py`** - Graph transformations for positional encoding and JEPA partitioning
- **`src/subgraphing_utils/`** - Subgraph extraction methods (motif, METIS, random walk)

The data preprocessing pipeline automatically handles polymer SMILES string conversion to graph representations:

1. **Raw Data**: CSV files containing polymer SMILES strings and target properties
   - `Data/aldeghi_coley_ea_ip_dataset.csv` - Aldeghi dataset with EA/IP properties
   - `Data/diblock_copolymer_dataset.csv` - Diblock copolymer dataset

2. **Graph Conversion**: Polymer SMILES are converted to PyTorch Geometric graphs via `src/data.py` and `src/featurization_utils/featurization.py`:
   - **`get_graphs()`** (in `data.py`): Main function that processes CSV files row by row
   - **`poly_smiles_to_graph()`** (in `featurization.py`): Core featurization function that:
     - **SMILES Parsing**: Splits polymer strings into monomers, weights, and bond rules
     - **Molecule Construction**: Creates RDKit molecule objects using `make_polymer_mol()`
     - **Atom Tagging**: Tags attachment points and core atoms via `tag_atoms_in_repeating_unit()`
     - **Feature Generation**: Extracts 133-dim node features and 14-dim edge features
     - **Bond Processing**: Handles both intra-monomer (weight=1.0) and inter-monomer bonds
     - **Motif Detection**: Identifies molecular substructures using `get_motifs()`
     - **Graph Assembly**: Constructs PyTorch Geometric `Data` objects with all metadata
   - **Helper functions** (in `featurization_helper.py`):
     - `atom_features()`: Extracts atomic properties (hybridization, formal charge, etc.)
     - `bond_features()`: Extracts bond properties (type, aromaticity, ring membership)
     - `parse_polymer_rules()`: Parses inter-monomer connection rules
     - `make_polymer_mol()`: Constructs polymer molecules from SMILES
     - `check_missing_bonds()`: Validates bond coverage in motif detection

3. **Data Cleaning & Validation**: During graph conversion:
   - **SMILES Validation**: RDKit parsing with error handling for invalid structures
   - **Molecular Sanitization**: Standardization of bond orders and formal charges
   - **Bond Connectivity**: Validation of inter-monomer connections based on attachment points
   - **Missing Bond Detection**: Automated detection and reporting of bonds not covered by motifs
   - **Atomic Consistency**: Verification that all atoms have valid feature representations
   - **Stoichiometry Validation**: Ensures polymer composition ratios are correctly parsed

4. **Processed Data**: Graphs are cached as `.pt` files for faster loading:
   - `Data/aldeghi_graphs_list.pt` - conjugated copolymer (Aldeghi) dataset graphs
   - `Data/diblock_graphs_list.pt` - Diblock copolymer dataset graphs  
   - **Caching Logic**: Files are automatically generated on first run and reused unless deleted
   - **PyTorch Geometric Format**: Graphs stored as `Data` objects with node/edge features and metadata

#### Dataset Creation Pipeline
The `create_data()` function orchestrates the complete data processing workflow:
1. **Pre-transforms**: Applies positional encoding (`PositionalEncodingTransform`)
2. **Training Transforms**: Configures graph partitioning for JEPA (`GraphJEPAPartitionTransform`)
3. **Dataset Wrapping**: Creates PyTorch Geometric `InMemoryDataset` objects
5. **Transform Application**: Applies different transforms for training vs. validation

**Manual Data Processing** (if needed):
```bash
# Force regeneration of graph files
rm Data/*_graphs_list.pt
python main.py  # Will regenerate graphs automatically

# Or run data processing directly
python -c "from src.data import get_graphs; get_graphs('aldeghi')"
```

**Note**: Graph conversion runs automatically on first execution. Subsequent runs load cached graphs unless files are deleted.

#### Data Selection Strategies
- Random Sampling** (`getRandomData()`): Standard random selection from available data (method used in the research paper)

#### Data Format and Structure
**Input Format**: Polymer SMILES strings follow the pattern:
```
[*:1]monomer1.[*:2]monomer2|weight1|weight2|bond_rules
```
Example: `[*:1]c1cc(F)c([*:2])cc1F.[*:3]c1c(O)cc(O)c([*:4])c1O|0.5|0.5|<1-2:0.375:0.375<...`

**Graph Structure**: Each processed graph contains:
- `x`: Node features (133-dim: atom type, hybridization, formal charge, etc.)
- `edge_index`: Graph connectivity in COO format
- `edge_attr`: Edge features (14-dim: bond type, aromaticity, ring membership, etc.)
- `node_weight`: Stoichiometry-based weights for polymer composition
- `edge_weight`: Bond weights (1.0 for intra-monomer, variable for inter-monomer)
- `motifs`: Detected molecular motifs and subgraph partitions
- `monomer_mask`: Identifies which monomer each atom belongs to

#### Configuration
Key data processing parameters in `src/config.py`:
- `cfg.finetuneDataset`: Dataset selection ('aldeghi', 'diblock')
- `cfg.finetune.aldeghiFTPercentage`: Training data percentage for Aldeghi
- `cfg.finetune.diblockFTPercentage`: Training data percentage for diblock

### Usage

#### Basic Usage
```bash
# Default configuration (pretraining + finetuning)
python main.py

# Skip pretraining, only finetune
python main.py shouldPretrain False

# Use GNN-based model (recommended)
python main.py modelVersion v2

# Train on diblock dataset
python main.py finetuneDataset diblock modelVersion v2
```

#### Configuration Options
```bash
# Override specific parameters
python main.py pretrain.epochs 50 model.hidden_size 300

# Use configuration file
python main.py --config config_example.yaml

# See runs.sh and run.sbatch for more examples
```

#### Key Parameters
- `modelVersion`: 'v1' (transformer-based) or 'v2' (GNN-based, recommended)
- `finetuneDataset`: 'aldeghi' or 'diblock' (diblock requires v2)
- `shouldPretrain`: Enable/disable pretraining phase
- `shouldFinetune`: Enable/disable finetuning phase
- `experimentName`: Name for wandb experiment tracking

### Experiments
Besides running experiments with JEPA pretraining, we also run the experiments using a baseline random forest model, as implemented in [polymer-chemprop](https://github.com/coleygroup/polymer-chemprop/tree/master/chemprop/features), and using an input-space SSL architecture developed for the same architecture [[1]](#1).
We provide the results in the folder Results/experiments_paper/. The code for running the random forest model is located in the scripts `src.train_test_rf.py`
 

### Credits 
The JEPA models code and `transform.py` are largely based on [Graph-JEPA](https://github.com/geriskenderi/graph-jepa) code, itself based on the [Graph-ViT-MLPMixer](https://github.com/XiaoxinHe/Graph-ViT-MLPMixer) code.
The featurization process code is taken from [polymer-chemprop](https://github.com/coleygroup/polymer-chemprop/tree/master/chemprop/features) code. 



### References
<a id="1">[1]</a> 
Gao, Q., Dukker, T., Schweidtmann, A. M., & Weber, J. M. (2024). 
Self-supervised graph neural networks for polymer property prediction. Molecular Systems Design & Engineering, 9(11), 1130-1143.