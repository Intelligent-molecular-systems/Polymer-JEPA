from contextlib import redirect_stdout
import os
import numpy as np
import random
from src.config import cfg
from src.visualize import visualize_aldeghi_results, visualize_diblock_results
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import wandb


def finetune(ft_trn_data, ft_val_data, model, model_name, cfg, device):
    print(f'Finetuning training on: {len(ft_trn_data)} graphs')
    print(f'Finetuning validating on: {len(ft_val_data)} graphs')
    
    if cfg.modelVersion == 'v2':
        # no need to use transform at every data access
        ft_trn_data = [x for x in ft_trn_data]
        

    ft_trn_loader = DataLoader(dataset=ft_trn_data, batch_size=cfg.finetune.batch_size, shuffle=True, num_workers=cfg.num_workers)
    ft_val_loader = DataLoader(dataset=ft_val_data, batch_size=cfg.finetune.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # dataset specific configurations
    if cfg.finetuneDataset == 'aldeghi':
        out_dim = 1 # 1 property
        criterion = nn.MSELoss() # regression
    elif cfg.finetuneDataset == 'diblock':
        out_dim = 5 # 5 classes
        criterion = nn.BCEWithLogitsLoss() # binary multiclass classification
    elif cfg.finetuneDataset == 'zinc':
        out_dim = 1
        criterion = nn.L1Loss()
    else:
        raise ValueError('Invalid dataset name')
    
    # predictor head, that takes the graph embeddings and predicts the property
    predictor = nn.Sequential(
        nn.Linear(cfg.model.hidden_size, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, out_dim)
    ).to(device)

    # if the model has a predictor head, load the weights (only the first two layers)
    if cfg.shouldFinetuneOnPretrainedModel:
        if hasattr(model, 'psuedoLabelPredictor'):
            predictor[0].weight = nn.parameter.Parameter(model.psuedoLabelPredictor[0].weight.clone().detach())
            predictor[0].bias = nn.parameter.Parameter(model.psuedoLabelPredictor[0].bias.clone().detach())
            predictor[2].weight = nn.parameter.Parameter(model.psuedoLabelPredictor[2].weight.clone().detach())
            predictor[2].bias = nn.parameter.Parameter(model.psuedoLabelPredictor[2].bias.clone().detach())
    
    if cfg.frozenWeights:
        print(f'Finetuning while freezing the weights of the model {model_name}')
        optimizer = torch.optim.Adam(predictor.parameters(), lr=cfg.finetune.lr, weight_decay=cfg.finetune.wd)
    else:
        print(f'End-to-End finetuning for model {model_name}')
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=cfg.finetune.lr, weight_decay=cfg.finetune.wd)
        

    for epoch in tqdm(range(cfg.finetune.epochs), desc='Finetuning Epochs'):
        if cfg.frozenWeights:
            model.eval()
        else:
            model.train()

        predictor.train()
        total_train_loss = 0

        all_embeddings, mon_A_type, stoichiometry = torch.tensor([], requires_grad=False, device=device), [], []       

        for data in ft_trn_loader:
            data = data.to(device)
            optimizer.zero_grad()

            if cfg.frozenWeights:
                with torch.no_grad():
                    graph_embeddings = model.encode(data)
            else:
                graph_embeddings = model.encode(data)

            if cfg.finetuneDataset == 'aldeghi':
                all_embeddings = torch.cat((all_embeddings, graph_embeddings), dim=0)
                stoichiometry.extend(data.stoichiometry)
                mon_A_type.extend(data.mon_A_type)

            y_pred_trn = predictor(graph_embeddings).squeeze()

            if cfg.finetuneDataset == 'aldeghi':
                train_loss = criterion(y_pred_trn, data.y_EA.float() if cfg.finetune.property == 'ea' else data.y_IP.float())

            elif cfg.finetuneDataset == 'diblock':
                y_lamellar = torch.tensor(data.y_lamellar, dtype=torch.float32, device=device)
                y_cylinder = torch.tensor(data.y_cylinder, dtype=torch.float32, device=device)
                y_sphere = torch.tensor(data.y_sphere, dtype=torch.float32, device=device)
                y_gyroid = torch.tensor(data.y_gyroid, dtype=torch.float32, device=device)
                y_disordered = torch.tensor(data.y_disordered, dtype=torch.float32, device=device)

                true_labels = torch.stack([y_lamellar, y_cylinder, y_sphere, y_gyroid, y_disordered], dim=1)

                train_loss = criterion(y_pred_trn, true_labels)
            elif cfg.finetuneDataset == 'zinc':
                train_loss = criterion(y_pred_trn, data.y.float())
            else:
                raise ValueError('Invalid dataset name')
            
            wandb.log({'finetune_epoch': epoch, 'finetune_train_loss': train_loss.item()})
            total_train_loss += train_loss
            train_loss.backward()
            optimizer.step()    

        total_train_loss /= len(ft_trn_loader)

        if epoch == cfg.finetune.epochs - 1: # eval only on the last epoch, computation is expensive
            model.eval()
            predictor.eval()
            with torch.no_grad():
                val_loss = 0
                all_y_pred_val = []
                all_true_val = []

                for data in ft_val_loader:
                    data = data.to(device)
                    graph_embeddings = model.encode(data)
                    y_pred_val = predictor(graph_embeddings).squeeze()

                    if cfg.finetuneDataset == 'aldeghi':
                        val_loss += criterion(y_pred_val, data.y_EA.float() if cfg.finetune.property == 'ea' else data.y_IP.float())
                        all_y_pred_val.extend(y_pred_val.detach().cpu().numpy())
                        all_true_val.extend(data.y_EA.detach().cpu().numpy() if cfg.finetune.property == 'ea' else data.y_IP.detach().cpu().numpy())

                    elif cfg.finetuneDataset == 'diblock':
                        y_lamellar = torch.tensor(data.y_lamellar, dtype=torch.float32, device=device)
                        y_cylinder = torch.tensor(data.y_cylinder, dtype=torch.float32, device=device)
                        y_sphere = torch.tensor(data.y_sphere, dtype=torch.float32, device=device)
                        y_gyroid = torch.tensor(data.y_gyroid, dtype=torch.float32, device=device)
                        y_disordered = torch.tensor(data.y_disordered, dtype=torch.float32, device=device)

                        true_labels = torch.stack([y_lamellar, y_cylinder, y_sphere, y_gyroid, y_disordered], dim=1)

                        val_loss += criterion(y_pred_val, true_labels)
                        all_y_pred_val.extend(y_pred_val.detach().cpu().numpy())
                        all_true_val.extend(true_labels.detach().cpu().numpy())

                    elif cfg.finetuneDataset == 'zinc':
                        val_loss += criterion(y_pred_val, data.y.float())
                        all_y_pred_val.extend(y_pred_val.detach().cpu().numpy())
                        all_true_val.extend(data.y.detach().cpu().numpy())
                    else:
                        raise ValueError('Invalid dataset name')
                    
            val_loss /= len(ft_val_loader)

            print(f'Epoch: {epoch+1:03d}, Train Loss: {train_loss:.5f}' f' Val Loss:{val_loss:.5f}')

            os.makedirs(f'Results/{model_name}', exist_ok=True)

            if not cfg.shouldFinetuneOnPretrainedModel: # if we are finetuning on a model that was not pretrained, save hyperparameters
                with open(f'Results/{model_name}/hyperparams.yml', 'w') as f:
                    with redirect_stdout(f): print(cfg.dump())

            percentage = cfg.finetune.aldeghiFTPercentage if cfg.finetuneDataset == 'aldeghi' else cfg.finetune.diblockFTPercentage
            save_folder = f'Results/{model_name}/{cfg.finetuneDataset}_{cfg.modelVersion}_{percentage}'
            metrics = {}
            if cfg.finetuneDataset == 'aldeghi':
                label = 'ea' if cfg.finetune.property == 'ea' else 'ip'
                
                # if cfg.visualize.shouldEmbeddingSpace:
                #     visualeEmbeddingSpace(all_embeddings, mon_A_type, stoichiometry, model_name, epoch, isFineTuning=True)

                R2, RMSE = visualize_aldeghi_results(
                    np.array(all_y_pred_val), 
                    np.array(all_true_val), 
                    label=label, 
                    save_folder=save_folder,
                    epoch=epoch+1,
                    shouldPlotMetrics=cfg.visualize.shouldPlotMetrics
                )
                metrics['R2'] = R2
                metrics['RMSE'] = RMSE
                

            elif cfg.finetuneDataset == 'diblock':
                prc_mean, roc_mean = visualize_diblock_results(
                    np.array(all_y_pred_val), 
                    np.array(all_true_val),
                    save_folder=save_folder,
                    epoch=epoch+1,
                    shouldPlotMetrics=cfg.visualize.shouldPlotMetrics
                )
                metrics['prc_mean'] = prc_mean
                metrics['roc_mean'] = roc_mean
            else:
                raise ValueError('Invalid dataset name')
    
    return train_loss, val_loss, metrics