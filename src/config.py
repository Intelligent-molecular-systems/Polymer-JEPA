import os
import argparse
from yacs.config import CfgNode as CN

def set_cfg(cfg):
    # ------------------------------------------------------------------------ #
    # General options
    # ------------------------------------------------------------------------ #
    # Additional num of worker for data loading
    cfg.num_workers = 0
    # Number of runs 
    cfg.runs = 5
    # Whether to pretrain the model
    cfg.shouldPretrain = True
    # Whether to finetune the model
    cfg.shouldFinetune = True
    # in case we want to finetune on a model that was pretrained
    cfg.shouldFinetuneOnPretrainedModel = True
    # name of the pretrained model, used only if shouldPretrain is False
    cfg.pretrainedModelName = 'name'
    # whether to freeze the weights of the pretrained model when finetuning
    cfg.frozenWeights = False
    # v1 for PolymerJEPA, v2 for PolymerJEPAv2
    cfg.modelVersion = 'v2'
    # finetuning dataset, values: 'aldeghi' or 'diblock' or 'zinc', 'diblock' can only be finetuned on a v2 model, not v1.
    cfg.finetuneDataset = 'aldeghi'
    # name fo the experiment to track on wandb
    cfg.experimentName = 'default'

    # ------------------------------------------------------------------------ #
    # Training options
    # ------------------------------------------------------------------------ #
    cfg.pretrain = CN()
    # Total graph mini-batch size
    cfg.pretrain.batch_size = 128
    # Maximal number of epochs
    cfg.pretrain.epochs = 10
    # Base learning rate
    cfg.pretrain.lr = 0.0005
    # number of steps before reduce learning rate
    cfg.pretrain.lr_patience = 20
    # learning rate decay factor
    cfg.pretrain.lr_decay = 0.5
    # L2 regularization, weight decay
    cfg.pretrain.wd = 0.0
    # Dropout rate
    cfg.pretrain.dropout = 0.1
    # Dropout rate for MLPMixer
    cfg.pretrain.mlpmixer_dropout = 0.35
    # A lower bound on the learning rate.
    cfg.pretrain.min_lr = 1e-5
    # optimizer
    cfg.pretrain.optimizer = 'Adam'   
    # Regularization (vcReg), between 0 and 1, tell the weight of the regularization loss, if 0 then no regularization
    cfg.pretrain.regularization = False
    # this should be used only when using the vicReg objective, where sharing weights is beneficial
    cfg.pretrain.shouldShareWeights = False
    # From vicReg: we have empirically found that using very different values for λ (inv) and μ (var), or taking λ = μ with ν (cov) > μ leads to unstable training. On the other hand taking λ = μ and picking ν < μ leads to stable convergence, setting lambda = mu = 25 and nu = 1 works best
    # invariance loss weight
    cfg.pretrain.inv_weight = 25
    # variance loss weight
    cfg.pretrain.var_weight = 25
    # covariance loss weight
    cfg.pretrain.cov_weight = 1

    
    cfg.finetune = CN()
    # Property to train (finetune) on: 'ea' or 'ip'
    cfg.finetune.property = 'ea'
    # Number of finetuning epochs
    cfg.finetune.epochs = 100
    # Base learning rate
    cfg.finetune.lr = 0.001
    # L2 regularization, weight decay
    cfg.finetune.wd = 0.
    # Total graph mini-batch size
    cfg.finetune.batch_size = 64
    cfg.finetune.isLinear = False
    # which percentage of the full dataset should be used to pretrain
    # (1%, 2%, 4%, 6%, 8%, 10%, 20%, 40%, 60%, 80% and 100%) of 40%, which are equivalent to 0.04%, 0.08%, 1.6%, 2.4%, 3.2%, 4%, 8%, 16%, 24%, 32%, 40% of the total dataset
    # this value is relative to 40%: 0.01 -> 1 % = 160 graphs 0.2 * 40
    cfg.finetune.aldeghiFTPercentage = 0.01
    # diblock has around 4800 graphs in total, 0.8 should be used as max size to match same full dataset size used by Aldeghi paper, 
    cfg.finetune.diblockFTPercentage = 0.06
    # 0 for random, 1 for lab data, 2 for maximizied variation, works only for aldeghi
    cfg.finetune.dataScenario = 0

    # ------------------------------------------------------------------------ #
    # Model options
    # ------------------------------------------------------------------------ #
    cfg.model = CN()
    # whether to use node weights for stoichiometry
    cfg.model.shouldUseNodeWeights = True
    # GraphMLPMixer or graph-based multihead attention: [MLPMixer, Hadamard, Standard, Graph, Addictive, Kernel]
    cfg.model.gMHA_type = 'Hadamard' # Hadamard is the default one for all datasets (yaml files) in original code
    # Hidden size of the model, I should use 300 for v2, 128 for v1
    cfg.model.hidden_size = 300 # make it a power of 2 if using the default model with transformer attention heads
    # Number of GNN layers
    cfg.model.nlayer_gnn = 3
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
    cfg.pos_enc.rw_dim = 20 
    # Patch random walk structural encoding
    cfg.pos_enc.patch_rw_dim = 20
    # Patch PE diffusion steps
    cfg.pos_enc.patch_num_diff = 0

    # ------------------------------------------------------------------------ #
    # Metis patch extraction options
    # ------------------------------------------------------------------------ #
    cfg.subgraphing = CN()
    # The number of partitions we want (upper bound) some subgraphs could be empty
    cfg.subgraphing.n_patches = 32
    # 0 = motif, 1 = metis, 2 = random walk
    cfg.subgraphing.type = 2
    # Whether to randomly drop a set of edges before each metis partition
    cfg.subgraphing.drop_rate = 0.3
    # The size of the context subgraph in percentage of the original graph
    cfg.subgraphing.context_size=0.6
    # The size of the target subgraph in percentage of the original graph
    cfg.subgraphing.target_size=0.10


    # ------------------------------------------------------------------------ #
    # JEPA options
    # ------------------------------------------------------------------------ #
    cfg.jepa = CN()
    # Use Graph-JEPA
    # Number of patches to use as context
    cfg.jepa.num_context = 1
    # Number of patches to use as targets
    cfg.jepa.num_targets = 3
    # loss/criterion/Distance function: 0 = 2d Hyper, 1 = Euclidean, 2 = Hyperbolic
    cfg.jepa.dist = 1

    # ------------------------------------------------------------------------ #
    # Plotting options
    # ------------------------------------------------------------------------ #
    cfg.visualize = CN()
    # Whether to plot the 3D embedding space
    cfg.visualize.should3DPlot = False
    # Whether to plot the embedding space (2D)
    cfg.visualize.shouldEmbeddingSpace = True
    # Whether to plot the loss space
    cfg.visualize.shouldLoss = False
    # Whether to plot the metrics (i.e RMSE, R2, PCR, ROC)
    cfg.visualize.shouldPlotMetrics = False

    return cfg



# Update cfg from command line
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
