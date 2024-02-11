import os
import argparse
from yacs.config import CfgNode as CN


def set_cfg(cfg):

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    # Additional num of worker for data loading
    cfg.num_workers = 1
    # Cuda device number, used for machine with multiple gpus
    cfg.device = 'cpu'
    # Whether fix the running seed to remove randomness
    cfg.seed = None
    # Custom log file name
    cfg.logfile = None

    cfg.shouldPretrain = False
    cfg.shouldFinetune = True

    # v1 for PolymerJEPA, v2 for PolymerJEPAv2
    cfg.modelVersion = 'v1'

    # ------------------------------------------------------------------------ #
    # Training options
    # ------------------------------------------------------------------------ #
    cfg.pretrain = CN()
    # Total graph mini-batch size
    cfg.pretrain.batch_size = 128
    # Maximal number of epochs
    cfg.pretrain.epochs = 30
    # Number of runs with random init
    cfg.pretrain.runs = 4
    # Base learning rate
    cfg.pretrain.lr = 0.001
    # number of steps before reduce learning rate
    cfg.pretrain.lr_patience = 20
    # learning rate decay factor
    cfg.pretrain.lr_decay = 0.5
    # L2 regularization, weight decay
    cfg.pretrain.wd = 0.
    # Dropout rate
    cfg.pretrain.dropout = 0.
    # Dropout rate for MLPMixer
    cfg.pretrain.mlpmixer_dropout = 0.
    # A lower bound on the learning rate.
    cfg.pretrain.min_lr = 1e-5
    # optimizer
    cfg.pretrain.optimizer = 'Adam'
    # Multiscale training
    cfg.pretrain.multiscale = False    
    # Regularization (vcReg), between 0 and 1, tell the weight of the regularization loss, if 0 then no regularization
    cfg.pretrain.regularization = 0.

    
    cfg.finetune = CN()
    # Property to train (finetune) on: 'ea' or 'ip'
    cfg.finetune.property = 'ea'
    cfg.finetune.epochs = 50
    # Base learning rate
    cfg.finetune.lr = 0.001
    # L2 regularization, weight decay
    cfg.finetune.wd = 0.
    # Total graph mini-batch size
    cfg.finetune.batch_size = 64

    # ------------------------------------------------------------------------ #
    # Model options
    # ------------------------------------------------------------------------ #
    cfg.model = CN()
    # GraphMLPMixer or graph-based multihead attention: [MLPMixer, Hadamard, Standard, Graph, Addictive, Kernel]
    cfg.model.gMHA_type = 'Hadamard' # Hadamard is the default one for all datsets (yaml files) in original code
    # Hidden size of the model
    cfg.model.hidden_size = 256
    # Number of mlp mixer layers
    cfg.model.nlayer_mlpmixer = 4
    # Pooling type for generaating graph/subgraph embedding from node embeddings
    cfg.model.pool = 'mean'
    # Use residual connection
    cfg.model.residual = True

    # ------------------------------------------------------------------------ #
    # Positional encoding options
    # ------------------------------------------------------------------------ #
    cfg.pos_enc = CN()
    # Random walk structural encoding
    # if set to 0 non pretransform? in graph-jepa it is set to 0
    cfg.pos_enc.rw_dim = 20 # [TODO]: idk what is the best val for this
    # Patch random walk structural encoding
    cfg.pos_enc.patch_rw_dim = 20
    # Patch PE diffusion steps
    cfg.pos_enc.patch_num_diff = 0

    # ------------------------------------------------------------------------ #
    # Metis patch extraction options
    # ------------------------------------------------------------------------ #
    cfg.subgraphing = CN()
    # The number of partitions (upper bound) RISK in case of more subgraphs this would break
    cfg.subgraphing.n_patches = 20
    # 0 = motif, 1 = metis, 2 = random walk
    cfg.subgraphing.type = 1

    # ------------------------------------------------------------------------ #
    # JEPA options
    # ------------------------------------------------------------------------ #
    cfg.jepa = CN()
    # Use Graph-JEPA
    # Number of patches to use as context
    cfg.jepa.num_context = 1
    # Number of patches to use as targets
    cfg.jepa.num_targets = 4
    # Distance function: 0 = 2d Hyper, 1 = Euclidean, 2 = Hyperbolic
    cfg.jepa.dist = 0

    return cfg


# Principle means that if an option is defined in a YACS config object,
# then your program should set that configuration option using cfg.merge_from_list(opts) and not by defining,
# for example, --train-scales as a command line argument that is then used to set cfg.pretrain.SCALES.


def update_cfg(cfg, args_str=None):
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


"""
    Global variable
"""
cfg = set_cfg(CN())
