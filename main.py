import collections
import os
import random
from sklearn.model_selection import KFold
from src.config import cfg
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

def run(pretrn_dataset, ft_dataset, val_dataset):

    ft_trn_loss = 0.0
    ft_val_loss = 0.0
    # ft_dataset.shuffle()
    # pretraning always done on the aldeghi dataset since its bigger dataset and no issues with homopolymer or tri, penta...blocks polymers
    # which would require different subgraphing techniques
    
    if cfg.finetuneDataset == 'aldeghi':
        print('Finetuning will be on aldeghi dataset...')
        #ft_data = getMaximizedVariedData(ft_dataset.copy(), int(cfg.finetune.aldeghiFTPercentage*len(ft_dataset))) #ft_dataset[:int(cfg.finetune.aldeghiFTPercentage*len(ft_dataset))]
        #ft_data = getLabData(ft_dataset.copy(), int(cfg.finetune.aldeghiFTPercentage*len(ft_dataset)))
        ft_data = getRandomData(ft_dataset, int(cfg.finetune.aldeghiFTPercentage*len(ft_dataset)))
        #ft_data = getTammoData(pretrn_dataset + ft_dataset)

        # for official result use the full val_dataset, but to run experiments fast i can use 0.95
        # print(ft_data)
        # get a list of all .smiles        
       
        pre_test_data = val_dataset # .copy()
        ft_test_data = val_dataset #.copy()


    elif cfg.finetuneDataset == 'diblock':
        # for official result use the full val_dataset, but to run experiments fast i can use 0.95
        pre_test_data = val_dataset # .copy()
        ft_data = ft_dataset[:int(cfg.finetune.diblockFTPercentage*len(ft_dataset))]
        ft_test_data = ft_dataset[int(cfg.finetune.diblockFTPercentage*len(ft_dataset)):]

    elif cfg.finetuneDataset == 'zinc':
        print('Loading zinc dataset for finetuning...')
        ft_data = ft_dataset
        pre_test_data = val_dataset
        ft_test_data = val_dataset

    else:
        raise ValueError('Invalid dataset name')

    model_name = None

    if cfg.shouldPretrain:
        model, model_name = pretrain(pretrn_dataset, pre_test_data, cfg)

    if cfg.shouldFinetune:
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
                ).to(cfg.device)

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
                ).to(cfg.device)

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
                ).to(cfg.device)

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
                ).to(cfg.device)
            else:
                raise ValueError('Invalid model version')

        if cfg.shouldFinetuneOnPretrainedModel:
            if not model_name: # it means we are not pretraining in the current run
                model_name = '232lmXan'
            wandb.config.update({'local_model_name': model_name})

            model.load_state_dict(torch.load(f'Models/Pretrain/{model_name}/model.pt', map_location=cfg.device))

            if cfg.finetune.isLinear:
                metrics = linearFinetune(ft_data, ft_test_data, model, model_name, cfg)
            else:
                ft_trn_loss, ft_val_loss, metrics = finetune(ft_data, ft_test_data, model, model_name, cfg)
        
        else:
            reset_parameters(model)
            # in case we are not finetuning on a pretrained model
            random.seed(time.time())
            model_name = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))
            model_name += '_NotPretrained'
            wandb.config.update({'local_model_name': model_name})
            if cfg.finetune.isLinear:
                metrics = linearFinetune(ft_data, ft_test_data, model, model_name, cfg)
            else:
                ft_trn_loss, ft_val_loss, metrics = finetune(ft_data, ft_test_data, model, model_name, cfg)
    
    return ft_trn_loss, ft_val_loss, metrics

    

if __name__ == '__main__':
    trn_losses = []
    val_losses = []
    metrics = collections.defaultdict(list)

    if cfg.finetuneDataset == 'aldeghi' or cfg.finetuneDataset == 'diblock':
        full_dataset, train_transform, val_transform = create_data(cfg)
        
        # !! setting folds = runs is risky !!
        kf = KFold(n_splits=cfg.runs, shuffle=True, random_state=12345)
        train_indices, test_indices = [], []
        for train_index, test_index in kf.split(torch.zeros(len(full_dataset))):
            train_indices.append(torch.from_numpy(train_index).to(torch.long))
            test_indices.append(torch.from_numpy(test_index).to(torch.long))

        for (train_index, test_index) in zip(train_indices, test_indices):
            start_WB_log_hyperparameters(cfg)
            train_dataset = full_dataset[train_index].copy()
            if cfg.finetuneDataset == 'aldeghi':
                pretrn_dataset = train_dataset[:len(train_dataset)//2]
                pretrn_dataset.transform = train_transform
                ft_dataset = train_dataset[len(train_dataset)//2:]
                ft_dataset.transform = train_transform
                if cfg.modelVersion == 'v2':
                    ft_dataset = [x for x in ft_dataset] # no need for transform at each iteration for model v2

            elif cfg.finetuneDataset == 'diblock':
                pretrn_dataset = full_dataset
                pretrn_dataset.transform = train_transform
                diblock_dataset = torch.load('Data/diblock_graphs_list.pt')
                random.seed(time.time())
                diblock_dataset = random.sample(diblock_dataset, len(diblock_dataset))
                ft_dataset = diblock_dataset

            
            val_dataset = full_dataset[test_index].copy()
            val_dataset.transform = val_transform
            val_dataset = [x for x in val_dataset]

            # pretrn_dataset.shuffle()
            # ft_dataset.shuffle()

            ft_trn_loss, ft_val_loss, metric = run(pretrn_dataset, ft_dataset, val_dataset)
            wandb_dict = {'final_ft_val_loss': ft_val_loss}
            trn_losses.append(ft_trn_loss)
            val_losses.append(ft_val_loss)
            for k, v in metric.items():
                metrics[k].append(v)
            wandb_dict.update(metric)
            wandb.log(wandb_dict)
            wandb.finish()
            


    elif cfg.finetuneDataset == 'zinc':
        pretrn_dataset, ft_dataset, val_dataset = create_data(cfg)
        # pretrn_dataset.shuffle()
        # ft_dataset.shuffle()

        for i in range(cfg.runs):
            start_WB_log_hyperparameters(cfg)
            ft_trn_loss, ft_val_loss, metric = run(pretrn_dataset, ft_dataset, val_dataset)
            trn_losses.append(ft_trn_loss)
            val_losses.append(ft_val_loss)
            wandb_dict = {'final_ft_val_loss': ft_val_loss}
            for k, v in metric.items():
                metrics[k].append(v)
            wandb_dict.update(metric)
            wandb.log(wandb_dict)
            wandb.finish()

    else:
        raise ValueError('Invalid dataset name')
    
    print(f'N of runs {cfg.runs}')
    print(f'Avg train loss {sum(trn_losses)/len(trn_losses)}')
    print(f'Avg val loss {sum(val_losses)/len(val_losses)}')
    for k, v in metrics.items():
        print(f'Avg {k} {sum(v)/len(v)}')

    





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