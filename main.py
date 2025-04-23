import collections
import os
import math
import random
from sklearn.model_selection import KFold, StratifiedShuffleSplit, train_test_split
from src.config import cfg, update_cfg
from src.data import create_data, getMaximizedVariedData, getLabData, getRandomData, getTammoData
from src.finetune import finetune
from src.linearFinetune import finetune as linearFinetune
from src.logger import start_WB_log_hyperparameters
# from PolymerJEPA_old import PolymerJEPA
from src.JEPA_models.PolymerJEPAv2 import PolymerJEPAv2
from src.JEPA_models.PolymerJEPA import PolymerJEPA
from src.JEPA_models.GeneralJEPA import GeneralJEPAv1
from src.JEPA_models.GeneralJEPAv2 import GeneralJEPAv2
from src.pretrain import pretrain
from src.training import reset_parameters
import string
import time
import torch
import wandb
import pandas as pd

os.environ["WANDB_MODE"]="offline"

def run(pretrn_trn_dataset, pretrn_val_dataset, pretrn_test_dataset, ft_trn_dataset, ft_val_dataset, ft_test_dataset):
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
                model = PolymerJEPA(
                    nfeat_node=133,
                    nfeat_edge=14,
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
                    nfeat_node=133,
                    nfeat_edge=14,
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

        if cfg.finetuneDataset == 'zinc':
            if cfg.modelVersion == 'v1':
                model = GeneralJEPAv1(
                    nfeat_node=28,
                    nfeat_edge=4,
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
                    shouldUse2dHyperbola=cfg.jepa.dist == 0
                ).to(device)

            elif cfg.modelVersion == 'v2':
                model = GeneralJEPAv2(
                    nfeat_node=28,
                    nfeat_edge=4,
                    nhid=cfg.model.hidden_size,
                    nlayer_gnn=cfg.model.nlayer_gnn,
                    rw_dim=cfg.pos_enc.rw_dim,
                    patch_rw_dim=cfg.pos_enc.patch_rw_dim,
                    pooling=cfg.model.pool,
                    num_target_patches=cfg.jepa.num_targets,
                    should_share_weights=cfg.pretrain.shouldShareWeights,
                    regularization=cfg.pretrain.regularization,
                    shouldUse2dHyperbola=cfg.jepa.dist == 0
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
                # TODO: Implement early stopping for linearFinetune
                metrics = linearFinetune(ft_trn_dataset, ft_val_dataset, model, model_name, cfg, device)
            else:
                #train_loss, val_loss, test_loss, metrics, metrics_test
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
    # generate a random seed for each run, always the same for reproducibility
    if cfg.seeds == 0:
        seeds = [42, 123, 777, 888, 999]
    elif cfg.seeds == 1:
        seeds = [421, 1231, 7771, 8881, 9991]
    elif cfg.seeds ==2:
        seeds = [422, 1232, 7772, 8882, 9992]
    
    
    print("Used seeds:")
    print(seeds)

    if cfg.finetuneDataset == 'aldeghi' or cfg.finetuneDataset == 'diblock':
        full_aldeghi_dataset, augmented_dataset, train_transform, val_transform = create_data(cfg)
        
        # !! setting folds = runs is risky, they shouldn't be used as done here !!
        kf = KFold(n_splits=cfg.runs, shuffle=True, random_state=12345)
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

                    # Optionally there can be more augmented data added to the pretraining data
                    if cfg.use_augmented_data and cfg.augmented_data_fraction:  # Check if augmented data should be used
                        # Shuffle augmented data
                        num_aug_samples = int(cfg.augmented_data_fraction * len(augmented_dataset))
                        shuffled_indices = torch.randperm(len(augmented_dataset))  # Random permutation
                        aug_subset = augmented_dataset[shuffled_indices][:num_aug_samples]  # Select fraction

                        # Extract data objects
                        data_list = pretrn_trn_dataset[:].copy() + aug_subset[:].copy()

                        # Instead of modifying a sliced dataset, create a new dataset instance, root=None -> in memory only 
                        pretrn_trn_dataset = train_dataset.__class__(root=None, data_list=data_list)
                        print(f"Using augmented data with {num_aug_samples} samples.")
                    else:
                        print("Augmented data is not used for pretraining.")
                    # pretrn_trn_dataset = train_dataset[:len(train_dataset)//2] # half of the train dataset for pretraining
                    pretrn_trn_dataset.transform = train_transform
                
                # split test set in val and test set, so we can do early stopping
                val_idx, test_idx = train_test_split(test_index, test_size=0.5, random_state=12345)  # Split 50/50

                pretrn_val_dataset = full_aldeghi_dataset[val_idx].copy()
                pretrn_test_dataset = full_aldeghi_dataset[test_idx].copy()
                
                #pretrn_val_dataset = full_aldeghi_dataset[test_index].copy()
                pretrn_val_dataset.transform = val_transform
                pretrn_val_dataset = [x for x in pretrn_val_dataset] # apply transform only once
                ft_val_dataset = pretrn_val_dataset # use same val dataset for pretraining and finetuning

                pretrn_test_dataset.transform = val_transform
                pretrn_test_dataset = [x for x in pretrn_test_dataset] # apply transform only once
                ft_test_dataset = pretrn_test_dataset # use same test dataset for pretraining and finetuning


                ft_trn_dataset = train_dataset[int((len(train_dataset)/100)*50):] # half of the train dataset for finetuning
                # ft_trn_dataset = train_dataset[len(train_dataset)//2:] # half of the train dataset for finetuning
                ft_trn_dataset.transform = train_transform
                # use math.ceil in order to get the same exact amount of data used by Tammo in his code
                if cfg.finetune.aldeghiFTPercentage == 1:
                    dataset_size = len(ft_trn_dataset)
                else:
                    dataset_size = int(math.ceil(cfg.finetune.aldeghiFTPercentage*len(ft_trn_dataset)/64)*64)
                # dataset_size = int(cfg.finetune.aldeghiFTPercentage*len(ft_trn_dataset))

                if cfg.finetune.dataScenario == 0:
                    ft_trn_dataset = getRandomData(ft_trn_dataset, dataset_size, seeds[run_idx])
                elif cfg.finetune.dataScenario == 1:
                    ft_trn_dataset = getLabData(ft_trn_dataset, dataset_size, seeds[run_idx])
                elif cfg.finetune.dataScenario == 2:
                    ft_trn_dataset = getMaximizedVariedData(ft_trn_dataset, dataset_size, seeds[run_idx])
                                
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
        df.to_csv(csv_filename, index=False)  # Save as csv
        df_test.to_csv(csv_filename_test, index=False)  # Save as csv

    
    elif cfg.finetuneDataset == 'zinc':
        # for zinc, create_data returns directly the datasets, not the trasforms
        pretrn_trn_dataset, ft_dataset, val_dataset = create_data(cfg) 

        for i in range(cfg.runs):
            print("----------------------------------------")
            print(f'Run {i}/{cfg.runs-1}')
            start_WB_log_hyperparameters(cfg)
            ft_trn_loss, ft_val_loss, metric = run(pretrn_trn_dataset, val_dataset, ft_dataset, val_dataset)
            print(f"losses_{i}:", ft_trn_loss.item(), ft_val_loss.item())
            trn_losses.append(ft_trn_loss)
            val_losses.append(ft_val_loss)
            wandb_dict = {'final_ft_val_loss': ft_val_loss}
            print(f"metrics_{i}:", end=' ')
            for k, v in metric.items():
                metrics[k].append(v)
                print(f"{k}={v}:", end=' ')
            print()
            wandb_dict.update(metric)
            wandb.log(wandb_dict)
            wandb.finish()

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