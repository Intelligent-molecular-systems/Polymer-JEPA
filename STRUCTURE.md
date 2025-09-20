# Polymer-JEPA Project Structure

This document describes the organization and structure of the Polymer-JEPA codebase.

## Directory Structure

```
Polymer-JEPA/
├── main.py                     # Main entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── STRUCTURE.md               # This file
├── environment_apptainer.yml   # Containerization config
├── apptainer_env.def          # Apptainer definition
├── run.sbatch                 # SLURM batch script
├── runs.sh                    # Shell script for experiments
│
├── Data/                      # Datasets and processed graphs
│   ├── aldeghi_coley_ea_ip_dataset.csv
│   ├── diblock_copolymer_dataset.csv
│   ├── aldeghi_graphs_list.pt
│   └── diblock_graphs_list.pt
│
├── Models/                    # Saved model checkpoints
│   └── Pretrain/             # Pretrained model weights
│
├── Results/                   # Experimental results
│   └── experiments_paper/     # Published results
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── config.py             # Configuration management
│   ├── data.py               # Data loading and preprocessing
│   ├── pretrain.py           # Self-supervised pretraining
│   ├── finetune.py           # Supervised finetuning
│   ├── linearFinetune.py     # Linear probe evaluation
│   ├── training.py           # Core training utilities
│   ├── logger.py             # Weights & Biases logging
│   ├── transform.py          # Graph transformations
│   ├── visualize.py          # Embedding visualization
│   ├── plots.py              # Additional plotting utilities
│   ├── extract_results.py    # Result extraction tools
│   │
│   ├── JEPA_models/          # Model architectures
│   │   ├── __init__.py
│   │   ├── PolymerJEPAv1.py  # Transformer-based JEPA
│   │   ├── PolymerJEPAv2.py  # GNN-based JEPA (recommended)
│   │   ├── WDNodeMPNN.py     # Weighted-directed MPNN
│   │   ├── WDNodeMPNNLayer.py # MPNN layer component
│   │   └── model_utils/      # Model utility functions
│   │
│   ├── featurization_utils/  # Molecular graph featurization
│   │   ├── __init__.py
│   │   ├── featurization.py  # SMILES to graph conversion
│   │   └── featurization_helper.py # Helper functions
│   │
│   ├── subgraphing_utils/    # Graph partitioning methods
│   │   ├── __init__.py
│   │   ├── motif_subgraphing.py
│   │   ├── context_subgraph_extractor.py
│   │   ├── target_subgraph_extractor.py
│   │   └── small_molecules_extractor.py
│   │
│   └── baseline_methods/     # Baseline implementations
│       ├── train_test_rf.py  # Random Forest (Aldeghi)
│       ├── train_test_rf_diblock.py # Random Forest (Diblock)
│       └── aggregate_results_rf.py  # RF result aggregation
│
└── wandb/                    # Weights & Biases logs (offline)
```

## Key Components

### Core Training Pipeline
- **main.py**: Orchestrates the complete training workflow
- **src/config.py**: Centralized configuration management with YACS
- **src/data.py**: Dataset creation and data loading utilities
- **src/pretrain.py**: Self-supervised JEPA pretraining implementation
- **src/finetune.py**: Supervised finetuning for property prediction

### Model Architecture
- **PolymerJEPAv1**: Transformer-based JEPA model
- **PolymerJEPAv2**: GNN-based JEPA model (best performance, default)
- **WDNodeMPNN**: Weighted-directed message passing neural network
- **model_utils/**: Attention mechanisms, encoders, and utilities

### Data Processing Pipeline
1. **Raw Data**: CSV files with polymer SMILES and properties
2. **Graph Conversion**: SMILES → PyTorch Geometric graphs
3. **Featurization**: 133-dim node features, 14-dim edge features
4. **Caching**: Processed graphs saved as .pt files
5. **Transforms**: Positional encoding and JEPA partitioning

### Configuration System
The project uses YACS for hierarchical configuration:
- **General**: Experiment settings, model selection
- **Pretraining**: Hyperparameters, regularization, VICReg weights
- **Finetuning**: Task configuration, dataset percentages
- **Model**: Architecture parameters, attention types
- **Subgraphing**: Partitioning methods and sizes
- **Visualization**: Plotting options

### Supported Datasets
- **Aldeghi**: Conjugated copolymers with EA/IP properties (~16k samples)
- **Diblock**: Diblock copolymers with phase behavior (~4.8k samples)

### Baseline Methods
- Random Forest implementations for comparison
- Linear probe evaluation for representation quality assessment

## Usage Patterns

### Basic Training
```bash
python main.py                          # Default configuration
python main.py shouldPretrain False     # Skip pretraining
python main.py modelVersion v2          # Use GNN-based model
```

### Configuration Override
```bash
python main.py pretrain.epochs 20 model.hidden_size 512
python main.py --config experiments/config.yaml
```

### Dataset Selection
```bash
python main.py finetuneDataset aldeghi   # Aldeghi dataset
python main.py finetuneDataset diblock   # Diblock dataset (v2 only)
```

## Development Guidelines

### Adding New Models
1. Create model class in `src/JEPA_models/`
2. Update `__init__.py` imports
3. Add model selection logic in `main.py`
4. Update configuration options in `config.py`

### Adding New Datasets
1. Add CSV processing logic in `src/data.py`
2. Update featurization if needed
3. Add dataset option to configuration
4. Update documentation