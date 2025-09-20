"""Configuration management for Polymer-JEPA.

This module defines all hyperparameters and settings for the training pipeline.
Configuration is managed using YACS (Yet Another Configuration System) which
allows for hierarchical configuration with command-line overrides.

Usage:
    # Use default config
    from src.config import cfg
    
    # Override from command line
    python main.py modelVersion v2 pretrain.epochs 20
    
    # Override from config file
    python main.py --config path/to/config.yaml
"""

import argparse
import os
from yacs.config import CfgNode as CN


def set_cfg(cfg):
    # ========================================================================
    # GENERAL CONFIGURATION
    # ========================================================================
    
    # Experiment settings
    cfg.experimentName = 'default'  # Experiment name for wandb tracking
    cfg.seeds = 0  # Seed set selection (0, 1, or 2)
    cfg.runs = 5  # Number of cross-validation runs
    cfg.num_workers = 0  # Number of workers for data loading
    
    # Training pipeline control
    cfg.shouldPretrain = True  # Enable pretraining phase
    cfg.shouldFinetune = True  # Enable finetuning phase
    cfg.shouldFinetuneOnPretrainedModel = True  # Use pretrained weights for finetuning
    cfg.frozenWeights = False  # Freeze pretrained weights during finetuning
    
    # Model and dataset selection
    cfg.modelVersion = 'v2'  # Model version: 'v1' (transformer-based) or 'v2' (GNN-based)
    cfg.finetuneDataset = 'aldeghi'  # Dataset: 'aldeghi' or 'diblock' (v2 only)
    cfg.pretrainedModelName = ''  # Pretrained model name (if not pretraining)

    # Pseudo-labeling configuration (experimental)
    cfg.pseudolabel = CN()
    cfg.pseudolabel.shouldUsePseudoLabel = False  # Enable pseudo-labeling
    cfg.pseudolabel.jepa_weight = 1.0  # JEPA loss weight
    cfg.pseudolabel.m_w_weight = 1.0  # Molecular weight loss weight
    # ========================================================================
    # PRETRAINING CONFIGURATION
    # ========================================================================
    cfg.pretrain = CN()
    
    # Training hyperparameters
    cfg.pretrain.batch_size = 128  # Mini-batch size
    cfg.pretrain.epochs = 10  # Maximum number of epochs
    cfg.pretrain.lr = 0.0005  # Base learning rate
    cfg.pretrain.wd = 0.0  # L2 regularization (weight decay)
    cfg.pretrain.optimizer = 'Adam'  # Optimizer type
    cfg.pretrain.min_lr = 1e-5  # Minimum learning rate
    
    # Learning rate scheduling
    cfg.pretrain.lr_patience = 20  # Steps before LR reduction
    cfg.pretrain.lr_decay = 0.5  # LR decay factor
    
    # Early stopping
    cfg.pretrain.early_stopping = 0  # Enable early stopping (0=False, 1=True)
    cfg.pretrain.early_stopping_patience = 2  # Early stopping patience
    
    # Regularization
    cfg.pretrain.dropout = 0.1  # Standard dropout rate
    cfg.pretrain.mlpmixer_dropout = 0.35  # MLPMixer dropout rate
    cfg.pretrain.regularization = False  # Enable VICReg regularization
    cfg.pretrain.shouldShareWeights = False  # Share encoder weights (for VICReg)
    cfg.pretrain.layer_norm = 1  # Layer normalization after encoders
    
    # VICReg loss weights (used when regularization=True)
    # Recommended: λ=μ=25, ν=1 for stable training
    cfg.pretrain.inv_weight = 25  # Invariance loss weight (λ)
    cfg.pretrain.var_weight = 25  # Variance loss weight (μ) 
    cfg.pretrain.cov_weight = 1  # Covariance loss weight (ν)

    # ========================================================================
    # FINETUNING CONFIGURATION
    # ========================================================================
    
    cfg.finetune = CN()
    
    # Training hyperparameters
    cfg.finetune.batch_size = 64  # Mini-batch size
    cfg.finetune.epochs = 100  # Maximum number of epochs
    cfg.finetune.lr = 0.001  # Base learning rate
    cfg.finetune.wd = 0.0  # L2 regularization (weight decay)
    
    # Early stopping
    cfg.finetune.early_stopping = 0  # Enable early stopping (0=False, 1=True)
    cfg.finetune.early_stopping_patience = 5  # Early stopping patience
    
    # Task configuration
    cfg.finetune.property = 'ea'  # Target property: 'ea' (electron affinity) or 'ip' (ionization potential)
    cfg.finetune.isLinear = False  # Use linear probe instead of full finetuning
    
    # Dataset size configuration
    # Aldeghi dataset: percentage relative to 40% of full dataset
    # Values: 0.01, 0.02, 0.04, 0.1, 0.2 correspond to 0.4%, 0.8%, 1.6%, 4%, 8% of total
    cfg.finetune.aldeghiFTPercentage = 0.01
    
    # Diblock dataset: percentage of ~4800 total graphs
    # Max 0.8 to match Aldeghi paper dataset size
    cfg.finetune.diblockFTPercentage = 0.06

    # ========================================================================
    # MODEL ARCHITECTURE
    # ========================================================================
    cfg.model = CN()
    
    # Architecture parameters
    cfg.model.hidden_size = 256  # Hidden dimension (use power of 2 for v1, 300 recommended for v2)
    cfg.model.nlayer_gnn = 2  # Number of GNN layers
    cfg.model.nlayer_mlpmixer = 2  # Number of MLP-Mixer layers (v1 only)
    cfg.model.residual = True  # Enable residual connections
    
    # Attention mechanism (v1 only)
    # Options: MLPMixer, Hadamard, Standard, Graph, Addictive, Kernel
    cfg.model.gMHA_type = 'Hadamard'  # Graph multihead attention type
    
    # Pooling and weighting
    cfg.model.pool = 'mean'  # Pooling method for graph/subgraph embeddings
    cfg.model.shouldUseNodeWeights = True  # Use stoichiometry-based node weights

    # ========================================================================
    # POSITIONAL ENCODING
    # ========================================================================
    cfg.pos_enc = CN()
    
    # Random walk positional encoding dimensions
    cfg.pos_enc.rw_dim = 20  # Node-level random walk encoding
    cfg.pos_enc.patch_rw_dim = 20  # Patch-level random walk encoding
    cfg.pos_enc.patch_num_diff = 0  # Diffusion steps for patch PE

    # ========================================================================
    # SUBGRAPH PARTITIONING
    # ========================================================================
    cfg.subgraphing = CN()
    
    # Partitioning parameters
    cfg.subgraphing.n_patches = 32  # Maximum number of patches (some may be empty)
    cfg.subgraphing.type = 2  # Partitioning method: 0=motif, 1=metis, 2=random_walk
    cfg.subgraphing.drop_rate = 0.3  # Edge dropout rate before partitioning
    
    # JEPA subgraph sizes (as fraction of original graph)
    cfg.subgraphing.context_size = 0.6  # Context subgraph size
    cfg.subgraphing.target_size = 0.10  # Target subgraph size


    # ========================================================================
    # JEPA CONFIGURATION
    # ========================================================================
    cfg.jepa = CN()
    
    # JEPA architecture
    cfg.jepa.num_context = 1  # Number of context patches
    cfg.jepa.num_targets = 1  # Number of target patches
    
    # Loss function
    # 0=2D Hyperbolic, 1=Euclidean, 2=Hyperbolic
    cfg.jepa.dist = 1

    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    cfg.visualize = CN()
    
    # Plotting options
    cfg.visualize.should3DPlot = False  # Generate 3D embedding plots
    cfg.visualize.shouldEmbeddingSpace = True  # Generate 2D embedding plots
    cfg.visualize.shouldLoss = False  # Plot training/validation loss
    cfg.visualize.shouldPlotMetrics = False  # Plot evaluation metrics

    return cfg



def update_cfg(cfg, args_str=None):
    """Update configuration from command line arguments or config file.
    
    Args:
        cfg: Base configuration object
        args_str: Optional string of arguments (for programmatic use)
        
    Returns:
        Updated configuration object
        
    Examples:
        # From command line
        python main.py modelVersion v2 pretrain.epochs 20
        
        # From config file
        python main.py --config experiments/config.yaml
        
        # Programmatic use
        cfg = update_cfg(cfg, "modelVersion v2 pretrain.epochs 20")
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="",
                        metavar="FILE", help="Path to config file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg
    cfg = cfg.clone()

    # Update from config file
    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    # Update from command line
    cfg.merge_from_list(args.opts)

    return cfg

# ============================================================================
# GLOBAL CONFIGURATION INSTANCE
# ============================================================================

cfg = set_cfg(CN())
