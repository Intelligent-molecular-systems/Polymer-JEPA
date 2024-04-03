import collections
import os
import random
from sklearn.model_selection import KFold
from src.config import cfg, update_cfg
from src.data import create_data, getMaximizedVariedData, getLabData, getRandomData, getTammoData
from src.finetune import finetune
from src.linearFinetune import finetune as linearFinetune
from src.logger import start_WB_log_hyperparameters
# from PolymerJEPA_old import PolymerJEPA
from src.PolymerJEPAv2 import PolymerJEPAv2
from src.PolymerJEPA import PolymerJEPA
from src.GeneralJEPA import GeneralJEPAv1
from src.GeneralJEPAv2 import GeneralJEPAv2
from src.pretrain import pretrain
from src.training import reset_parameters
import string
import time
import torch
import wandb

# os.environ["WANDB_MODE"]="offline"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def run(pretrn_trn_dataset, pretrn_val_dataset, ft_trn_dataset, ft_val_dataset):

    model_name = None

    if cfg.shouldPretrain:
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
                    shouldUse2dHyperbola=cfg.jepa.dist == 0,
                    shouldUseNodeWeights=True
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
                    shouldUse2dHyperbola=cfg.jepa.dist == 0,
                    shouldUseNodeWeights=cfg.model.shouldUseNodeWeights
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
                    shouldUse2dHyperbola=cfg.jepa.dist == 0,
                    shouldUseNodeWeights=True
                ).to(device)
            else:
                raise ValueError('Invalid model version')

        if cfg.shouldFinetuneOnPretrainedModel:
            if not model_name: # it means we are not pretraining in the current run
                model_name = '232lmXan'
            wandb.config.update({'local_model_name': model_name})

            model.load_state_dict(torch.load(f'Models/Pretrain/{model_name}/model.pt', map_location=device))

            if cfg.finetune.isLinear:
                metrics = linearFinetune(ft_trn_dataset, ft_val_dataset, model, model_name, cfg, device)
            else:
                ft_trn_loss, ft_val_loss, metrics = finetune(ft_trn_dataset, ft_val_dataset, model, model_name, cfg, device)
        
        else:
            reset_parameters(model)
            # in case we are not finetuning on a pretrained model
            random.seed(time.time())
            model_name = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))
            model_name += '_NotPretrained'
            wandb.config.update({'local_model_name': model_name})
            if cfg.finetune.isLinear:
                metrics = linearFinetune(ft_trn_dataset, ft_val_dataset, model, model_name, cfg, device)
            else:
                ft_trn_loss, ft_val_loss, metrics = finetune(ft_trn_dataset, ft_val_dataset, model, model_name, cfg, device)
    
    return ft_trn_loss, ft_val_loss, metrics

    

if __name__ == '__main__':
    cfg = update_cfg(cfg) # update cfg with command line arguments
    trn_losses = []
    val_losses = []
    metrics = collections.defaultdict(list)

    if cfg.finetuneDataset == 'aldeghi' or cfg.finetuneDataset == 'diblock':
        full_aldeghi_dataset, train_transform, val_transform = create_data(cfg)
        
        # !! setting folds = runs is risky !!
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
            if cfg.finetuneDataset == 'aldeghi':
                train_dataset = full_aldeghi_dataset[train_index].copy()

                if cfg.shouldPretrain:
                    pretrn_trn_dataset = train_dataset[:len(train_dataset)//2]
                    pretrn_trn_dataset.transform = train_transform

                pretrn_val_dataset = full_aldeghi_dataset[test_index].copy()
                pretrn_val_dataset.transform = val_transform
                pretrn_val_dataset = [x for x in pretrn_val_dataset]
                ft_val_dataset = pretrn_val_dataset # use same val dataset for pretraining and finetuning

                ft_trn_dataset = train_dataset[len(train_dataset)//2:]
                ft_trn_dataset.transform = train_transform
                #ft_data = getMaximizedVariedData(ft_dataset.copy(), int(cfg.finetune.aldeghiFTPercentage*len(ft_dataset))) #ft_dataset[:int(cfg.finetune.aldeghiFTPercentage*len(ft_dataset))]
                #ft_data = getLabData(ft_dataset.copy(), int(cfg.finetune.aldeghiFTPercentage*len(ft_dataset)))
                ft_trn_dataset = getRandomData(ft_trn_dataset, int(cfg.finetune.aldeghiFTPercentage*len(ft_trn_dataset)))
                #ft_data = getTammoData(pretrn_dataset + ft_dataset)
                
                

            elif cfg.finetuneDataset == 'diblock':
                if cfg.shouldPretrain:
                    pretrn_trn_dataset = full_aldeghi_dataset[train_index].copy()
                    pretrn_trn_dataset.transform = train_transform
                    pretrn_val_dataset = full_aldeghi_dataset[test_index].copy()
                    pretrn_val_dataset.transform = val_transform
                    pretrn_val_dataset = [x for x in pretrn_val_dataset]

                if run_idx == 0: # load the dataset only once
                    diblock_dataset = torch.load('Data/diblock_graphs_list.pt') 
                random.seed(time.time())
                diblock_dataset = random.sample(diblock_dataset, len(diblock_dataset))
                ft_trn_dataset = diblock_dataset[:int(cfg.finetune.diblockFTPercentage*len(diblock_dataset))]
                ft_val_dataset = diblock_dataset[int(cfg.finetune.diblockFTPercentage*len(diblock_dataset)):]

            
            # pretrn_dataset.shuffle()
            # ft_dataset.shuffle()

            ft_trn_loss, ft_val_loss, metric = run(pretrn_trn_dataset, pretrn_val_dataset, ft_trn_dataset, ft_val_dataset)
            print(f"losses_{run_idx}:", ft_trn_loss, ft_val_loss)
            wandb_dict = {'final_ft_val_loss': ft_val_loss}
            trn_losses.append(ft_trn_loss)
            val_losses.append(ft_val_loss)
            print(f"metrics_{run_idx}:", end=' ')
            for k, v in metric.items():
                metrics[k].append(v)
                print(f"{k}={v}:", end=' ')
            print()
            wandb_dict.update(metric)
            wandb.log(wandb_dict)
            wandb.finish()
            


    elif cfg.finetuneDataset == 'zinc':
        pretrn_trn_dataset, ft_dataset, val_dataset = create_data(cfg)
        # pretrn_dataset.shuffle()
        # ft_dataset.shuffle()

        for i in range(cfg.runs):
            print("----------------------------------------")
            print(f'Run {i}/{cfg.runs-1}')
            start_WB_log_hyperparameters(cfg)
            ft_trn_loss, ft_val_loss, metric = run(pretrn_trn_dataset, val_dataset, ft_dataset, val_dataset)
            print(f"losses_{i}:", ft_trn_loss, ft_val_loss)
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
    





# check features dimensions for batches
# for i in range(0, len(dataset), 20):
#     try:
#         Batch.from_data_list(dataset[i:i+20])
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         for data in dataset[i:i+20]:
#             print(data)
#             quit()


# params_before = {name: param.clone() for name, param in model.named_parameters()}
# params_after = {name: param.clone() for name, param in model.named_parameters()}

            # Compare parameters
            # for name, param_before in params_before.items():
            #     param_after = params_after[name]
            #     # Check if the same (using torch.equal to compare tensors)
            #     if not torch.equal(param_before, param_after):
            #         print(f"Parameter {name} has changed.")
            #     else:
            #         print(f"Parameter {name} remains the same.")