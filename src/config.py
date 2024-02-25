import os
import argparse
from yacs.config import CfgNode as CN

# This file was copied from graph-jepa, some of the defined options are not used in the code at the moment

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

    cfg.shouldPretrain = True
    cfg.shouldFinetune = True
    # in case we want to finetune on a model that was pretrained
    cfg.shouldFinetuneOnPretrainedModel = True
    cfg.frozenWeights = False

    # v1 for PolymerJEPA, v2 for PolymerJEPAv2
    cfg.modelVersion = 'v2'

    # finetuning dataset, values: 'aldeghi' or 'diblock', 'diblock' can only be finetuned on a v2 model, not v1.
    cfg.finetuneDataset = 'aldeghi'

    # ------------------------------------------------------------------------ #
    # Training options
    # ------------------------------------------------------------------------ #
    cfg.pretrain = CN()
    # Total graph mini-batch size
    cfg.pretrain.batch_size = 128
    # Maximal number of epochs
    cfg.pretrain.epochs = 20
    # Number of runs with random init
    cfg.pretrain.runs = 4
    # Base learning rate
    cfg.pretrain.lr = 0.0005 # RISK before 0.001, i should try even a lower one 0.0001
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
    cfg.pretrain.regularization = True
    # this should be used only when using the vicReg objective, where sharing weights is beneficial
    cfg.pretrain.shouldShareWeights = False
    # From vicReg: we have empirically found that using very different values for λ (inv) and μ (var), or taking λ = μ with ν (cov) > μ leads to unstable training. On the other hand taking λ = μ and picking ν < μ leads to stable convergence, setting lambda = mu = 25 and nu = 1 works best
    cfg.pretrain.inv_weight = 25
    cfg.pretrain.var_weight = 25
    cfg.pretrain.cov_weight = 1
    # which percentage of the full dataset should be used to pretrain
    # (0.1%, 0.2%, 0.4%, 0.6%, 0.8%, 1%, 10%, 20%, 40%, 60%, 80% and 100%) of 40%, which are equivalent to 0.04%, 0.08%, 1.6%, 2.4%, 3.2%, 4%, 8%, 16%, 24%, 32%, 40% of the total dataset
    # this value is relative to 40%: 0.01 -> 1 % 
    cfg.pretrain.aldeghiFTPercentage = 0.01
    cfg.pretrain.diblockFTPercentage = 0.2

    
    cfg.finetune = CN()
    # Property to train (finetune) on: 'ea' or 'ip'
    cfg.finetune.property = 'ip'
    cfg.finetune.epochs = 100
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
    cfg.model.gMHA_type = 'Standard' # Hadamard is the default one for all datsets (yaml files) in original code
    # Hidden size of the model
    cfg.model.hidden_size = 64 # make it a power of 2 if using the default model with transformer attention heads
    # The hidden dimension of the wdmpnn 
    cfg.model.wdmpnn_hid_dim = 128
    # Number of mlp mixer layers
    cfg.model.nlayer_mlpmixer = 2
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
    cfg.subgraphing.type = 2

    # ------------------------------------------------------------------------ #
    # JEPA options
    # ------------------------------------------------------------------------ #
    cfg.jepa = CN()
    # Use Graph-JEPA
    # Number of patches to use as context
    cfg.jepa.num_context = 1
    # Number of patches to use as targets
    cfg.jepa.num_targets = 4
    # loss/criterion/Distance function: 0 = 2d Hyper, 1 = Euclidean, 2 = Hyperbolic
    cfg.jepa.dist = 0


    cfg.metis = CN()
    cfg.metis.drop_rate = 0.3

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
