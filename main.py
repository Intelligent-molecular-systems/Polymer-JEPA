"""Main entry point for Polymer-JEPA training pipeline.

This module orchestrates the complete training workflow including:
- Data loading and preprocessing
- Model pretraining (optional)
- Model finetuning
- Cross-validation and evaluation
- Results logging and saving

Usage:
    python main.py [config_options]
    
Example:
    python main.py shouldPretrain False modelVersion v2
"""

# Standard library imports
import collections
import math
import os
import random
import string
import time

# Third-party imports
import pandas as pd
import torch
import wandb
from sklearn.model_selection import KFold, StratifiedShuffleSplit, train_test_split

# Local imports
from src.config import cfg, update_cfg
from src.data import create_data, get_random_data
from src.finetune import finetune
from src.linearFinetune import finetune as linearFinetune
from src.logger import start_WB_log_hyperparameters
from src.JEPA_models.PolymerJEPAv1 import PolymerJEPAv1
from src.JEPA_models.PolymerJEPAv2 import PolymerJEPAv2
from src.pretrain import pretrain
from src.training import reset_parameters

# Configure wandb to run offline
os.environ["WANDB_MODE"] = "offline"

# Constants
NODE_FEATURES = 133  # Number of node features in molecular graphs
EDGE_FEATURES = 14   # Number of edge features in molecular graphs
RANDOM_STATE = 12345 # Fixed random state for reproducibility

# Seed sets for cross-validation runs
SEED_SETS = {
    0: [42, 123, 777, 888, 999],
    1: [421, 1231, 7771, 8881, 9991],
    2: [422, 1232, 7772, 8882, 9992]
}

def run(pretrn_trn_dataset, pretrn_val_dataset, pretrn_test_dataset, 
        ft_trn_dataset, ft_val_dataset, ft_test_dataset):
    """Execute training pipeline for one fold.
    
    Args:
        pretrn_trn_dataset: Pretraining training dataset
        pretrn_val_dataset: Pretraining validation dataset
        pretrn_test_dataset: Pretraining test dataset
        ft_trn_dataset: Finetuning training dataset
        ft_val_dataset: Finetuning validation dataset
        ft_test_dataset: Finetuning test dataset
        
    Returns:
        tuple: (train_loss, val_loss, test_loss, val_metrics, test_metrics)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model_name = None

    if cfg.shouldPretrain:
        # pretraining only needs, validation set not test set (testing is done after finetuning)
        model, model_name = pretrain(pretrn_trn_dataset, pretrn_val_dataset, cfg, device)

    ft_trn_loss = 0.0
    ft_val_loss = 0.0
    if cfg.shouldFinetune:
        print(f'Finetuning on {cfg.finetuneDataset} dataset...')
        if cfg.finetuneDataset == 'aldeghi' or cfg.finetuneDataset == 'diblock':
            if cfg.modelVersion == 'v1':
                model = PolymerJEPAv1(
                    nfeat_node=NODE_FEATURES,
                    nfeat_edge=EDGE_FEATURES,
                    nhid=cfg.model.hidden_size,
                    nlayer_gnn=cfg.model.nlayer_gnn,
                    nlayer_mlpmixer=cfg.model.nlayer_mlpmixer,
                    gMHA_type=cfg.model.gMHA_type,
                    rw_dim=cfg.pos_enc.rw_dim,
                    patch_rw_dim=cfg.pos_enc.patch_rw_dim,
                    pooling=cfg.model.pool,
                    n_patches=cfg.subgraphing.n_patches,
                    mlpmixer_dropout=cfg.pretrain.mlpmixer_dropout,
                    num_target_patches=cfg.jepa.num_targets,
                    should_share_weights=cfg.pretrain.shouldShareWeights,
                    regularization=cfg.pretrain.regularization,
                    shouldUse2dHyperbola=cfg.jepa.dist == 0,
                    shouldUseNodeWeights=True
                ).to(device)

            elif cfg.modelVersion == 'v2':
                model = PolymerJEPAv2(
                    nfeat_node=NODE_FEATURES,
                    nfeat_edge=EDGE_FEATURES,
                    nhid=cfg.model.hidden_size,
                    nlayer_gnn=cfg.model.nlayer_gnn,
                    rw_dim=cfg.pos_enc.rw_dim,
                    patch_rw_dim=cfg.pos_enc.patch_rw_dim,
                    pooling=cfg.model.pool,
                    num_target_patches=cfg.jepa.num_targets,
                    should_share_weights=cfg.pretrain.shouldShareWeights,
                    regularization=cfg.pretrain.regularization,
                    layer_norm=cfg.pretrain.layer_norm,
                    shouldUse2dHyperbola=cfg.jepa.dist == 0,
                    shouldUseNodeWeights=True,
                    shouldUsePseudoLabel=cfg.pseudolabel.shouldUsePseudoLabel
                ).to(device)

            else:
                raise ValueError('Invalid model version')


            
        reset_parameters(model)

        if cfg.shouldFinetuneOnPretrainedModel:
            if not model_name: # it means we have not pretrained in the current run, so we need to load a pretrained model to finetune
                model_name = cfg.pretrainedModelName
            wandb.config.update({'local_model_name': model_name})

            model.load_state_dict(torch.load(f'Models/Pretrain/{model_name}/model.pt', map_location=device))

            if cfg.finetune.isLinear:
                metrics = linearFinetune(ft_trn_dataset, ft_val_dataset, model, model_name, cfg, device)
            else:
                ft_trn_loss, ft_val_loss, ft_test_loss, metrics, metrics_test = finetune(ft_trn_dataset, ft_val_dataset, ft_test_dataset, model, model_name, cfg, device)
        
        else:
            # in case we are not finetuning on a pretrained model
            random.seed(time.time())
            model_name = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))
            model_name += '_NotPretrained'
            wandb.config.update({'local_model_name': model_name})
            if cfg.finetune.isLinear:
                metrics = linearFinetune(ft_trn_dataset, ft_val_dataset, model, model_name, cfg, device)
            else:
                ft_trn_loss, ft_val_loss, ft_test_loss, metrics, metrics_test = finetune(ft_trn_dataset, ft_val_dataset, ft_test_dataset, model, model_name, cfg, device)
    
    # check if folder Results/{model_name} exists, if so, delete it to save space
    # delete this code if you want to keep the plots of each run saved in the Results folder locally
    if os.path.exists(f'Results/{model_name}'):
        os.system(f'rm -r Results/{model_name}')

    return ft_trn_loss, ft_val_loss, ft_test_loss, metrics, metrics_test

    

if __name__ == '__main__':
    cfg = update_cfg(cfg) # update cfg with command line arguments
    trn_losses = []
    val_losses = []
    test_losses = []
    metrics = collections.defaultdict(list)
    metrics_test = collections.defaultdict(list)
    # Get seed set for reproducible cross-validation
    seeds = SEED_SETS.get(cfg.seeds, SEED_SETS[0])
    
    
    print("Used seeds:")
    print(seeds)

    if cfg.finetuneDataset == 'aldeghi' or cfg.finetuneDataset == 'diblock':
        full_aldeghi_dataset, train_transform, val_transform = create_data(cfg)
        
        # !! setting folds = runs is risky, they shouldn't be used as done here !!
        kf = KFold(n_splits=cfg.runs, shuffle=True, random_state=RANDOM_STATE)
        train_indices, test_indices = [], []
        for train_index, test_index in kf.split(torch.zeros(len(full_aldeghi_dataset))):
            train_indices.append(torch.from_numpy(train_index).to(torch.long))
            test_indices.append(torch.from_numpy(test_index).to(torch.long))

        pretrn_trn_dataset = []
        pretrn_val_dataset = []

        for run_idx, (train_index, test_index) in enumerate(zip(train_indices, test_indices)):
            start_WB_log_hyperparameters(cfg)                
            print("----------------------------------------")
            print(f'Run {run_idx}/{cfg.runs-1}')
            if cfg.finetuneDataset == 'aldeghi': # pretrain and finetune on same dataset (aldeghi), pretrain and finetune val dataset are the same.
                train_dataset = full_aldeghi_dataset[train_index].copy()
                if cfg.shouldPretrain:
                    # keep 50% of the train dataset for finetuning, corresponding to 40% of the full dataset
                    pretrn_trn_dataset = train_dataset[:int((len(train_dataset)/100)*50)] # half of the train dataset for pretraining
                    
                    # pretrn_trn_dataset = train_dataset[:len(train_dataset)//2] # half of the train dataset for pretraining
                    pretrn_trn_dataset.transform = train_transform
                
                # split test set in val and test set, so we can do early stopping
                val_idx, test_idx = train_test_split(test_index, test_size=0.5, random_state=RANDOM_STATE)  # Split 50/50

                pretrn_val_dataset = full_aldeghi_dataset[val_idx].copy()
                pretrn_test_dataset = full_aldeghi_dataset[test_idx].copy()
                
                pretrn_val_dataset.transform = val_transform
                pretrn_val_dataset = [x for x in pretrn_val_dataset] # apply transform only once
                ft_val_dataset = pretrn_val_dataset # use same val dataset for pretraining and finetuning

                pretrn_test_dataset.transform = val_transform
                pretrn_test_dataset = [x for x in pretrn_test_dataset] # apply transform only once
                ft_test_dataset = pretrn_test_dataset # use same test dataset for pretraining and finetuning


                ft_trn_dataset = train_dataset[int((len(train_dataset)/100)*50):] # half of the train dataset for finetuning
                # ft_trn_dataset = train_dataset[len(train_dataset)//2:] # half of the train dataset for finetuning
                ft_trn_dataset.transform = train_transform
                # use math.ceil in order to get the same exact amount of data used in Gao, Qinghe, et al. "Self-supervised graph neural networks for polymer property prediction." Molecular Systems Design & Engineering paper.
                if cfg.finetune.aldeghiFTPercentage == 1:
                    dataset_size = len(ft_trn_dataset)
                else:
                    dataset_size = int(math.ceil(cfg.finetune.aldeghiFTPercentage*len(ft_trn_dataset)/64)*64)
                # dataset_size = int(cfg.finetune.aldeghiFTPercentage*len(ft_trn_dataset))

                
                ft_trn_dataset = get_random_data(ft_trn_dataset, dataset_size, seeds[run_idx])
                                
            elif cfg.finetuneDataset == 'diblock':
                if cfg.shouldPretrain: # only compute pretrain datasets if we are pretraining, it's an expensive operation
                    pretrn_trn_dataset = full_aldeghi_dataset[train_index].copy()
                    pretrn_trn_dataset.transform = train_transform
                    pretrn_val_dataset = full_aldeghi_dataset[test_index].copy()
                    pretrn_val_dataset.transform = val_transform
                    pretrn_val_dataset = [x for x in pretrn_val_dataset]

                if run_idx == 0: # load the dataset only once
                    diblock_dataset = torch.load('Data/diblock_graphs_list.pt') 
                random.seed(seeds[run_idx])

                phase1_labels = [graph.phase1 for graph in diblock_dataset]

                sss = StratifiedShuffleSplit(n_splits=1, test_size=1-cfg.finetune.diblockFTPercentage, random_state=seeds[run_idx])

                for train_index, test_index in sss.split(diblock_dataset, phase1_labels):
                    ft_trn_dataset = [diblock_dataset[i] for i in train_index]
                    ft_val_dataset = [diblock_dataset[i] for i in test_index]

            ft_trn_loss, ft_val_loss, ft_test_loss, metric, metric_test = run(pretrn_trn_dataset, pretrn_val_dataset, pretrn_test_dataset, ft_trn_dataset, ft_val_dataset, ft_test_dataset)
            if not cfg.finetune.isLinear:
                print(f"losses_{run_idx}:", ft_trn_loss.item(), ft_val_loss.item())
            trn_losses.append(ft_trn_loss)
            val_losses.append(ft_val_loss)
            test_losses.append(ft_test_loss)
            wandb_dict = {'final_ft_test_loss': ft_test_loss}
            print(f"metrics_{run_idx}:", end=' ')
            for k, v in metric.items():
                metrics[k].append(v)
                print(f"{k}={v}:", end=' ')
            for k, v in metric_test.items():
                metrics_test[k].append(v)
                print(f"{k}={v}:", end=' ')
            wandb_dict.update(metric)
            wandb_dict.update(metric_test)
            wandb.log(wandb_dict)
            wandb.finish()

            # if we are not pretraining and we are finetuning on a pretrained model, we only need to run once
            # if not cfg.shouldPretrain and cfg.shouldFinetuneOnPretrainedModel:
            #     break

        # Save the metrics 
        # Save results as excel
        df = pd.DataFrame(dict(metrics))  # Convert defaultdict to DataFrame
        df_test = pd.DataFrame(dict(metrics_test))
        variables = {
            "PL": cfg.pseudolabel.shouldUsePseudoLabel,
            "layer_norm": cfg.pretrain.layer_norm,
            "seeds": seeds[0],
            "finetune_percentage": cfg.finetune.aldeghiFTPercentage,
            "pretraining": cfg.shouldPretrain

        }
        csv_filename = "metrics_train_" + "_".join(f"{k}_{v}" for k, v in variables.items()) + ".csv"
        csv_filename_test = "metrics_test_" + "_".join(f"{k}_{v}" for k, v in variables.items()) + ".csv"
        if cfg.finetuneDataset == 'diblock':
            variables = {
            "PL": cfg.pseudolabel.shouldUsePseudoLabel,
            "layer_norm": cfg.pretrain.layer_norm,
            "seeds": seeds[0],
            "finetune_percentage": cfg.finetune.diblockFTPercentage,
            "pretraining": cfg.shouldPretrain

        }
        csv_filename = "metrics_diblock_train_" + "_".join(f"{k}_{v}" for k, v in variables.items()) + ".csv"
        csv_filename_test = "metrics_diblock_test_" + "_".join(f"{k}_{v}" for k, v in variables.items()) + ".csv"
        df.to_csv(csv_filename, index=False)  # Save as csv
        df_test.to_csv(csv_filename_test, index=False)  # Save as csv

    


    else:
        raise ValueError('Invalid dataset name')
    
    print("----------------------------------------")
    print(f'N of total runs {cfg.runs}')
    print(f'Avg train loss {sum(trn_losses)/len(trn_losses)}')
    print(f'Avg val loss {sum(val_losses)/len(val_losses)}')
    for k, v in metrics.items():
        print(f'Avg {k} {sum(v)/len(v)}')
    print("----------------------------------------")
    print("config used:")
    print(cfg)
    print("Seeds used")
    print(seeds)