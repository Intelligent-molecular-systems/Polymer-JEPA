from contextlib import redirect_stdout
import os
import numpy as np
import random
from src.config import cfg
from src.visualize import visualize_aldeghi_results, visualize_diblock_results, visualeEmbeddingSpace
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def finetune(ft_trn_data, ft_val_data, model, model_name, cfg):

    # print(len(ft_val_data))
    # ft_tst_data = ft_data[int(0.9*len(ft_data)):].copy()
    print(f'Finetuning training on: {len(ft_trn_data)} graphs')
    print(f'Finetuning validating on: {len(ft_val_data)} graphs')
    
    if cfg.modelVersion == 'v2':
        # no need to use transform at every data access
        ft_trn_data = [x for x in ft_trn_data]
        

    ft_trn_loader = DataLoader(dataset=ft_trn_data, batch_size=cfg.finetune.batch_size, shuffle=True)
    ft_val_loader = DataLoader(dataset=ft_val_data, batch_size=cfg.finetune.batch_size, shuffle=False)

    # dataset specific configurations
    if cfg.finetuneDataset == 'aldeghi':
        out_dim = 1 # 1 property
        criterion = nn.MSELoss() # regression
    elif cfg.finetuneDataset == 'diblock':
        out_dim = 5 # 5 classes
        # Binary Cross-Entropy With Logits Loss
        # https://discuss.pytorch.org/t/using-bcewithlogisloss-for-multi-label-classification/67011/2
        criterion = nn.BCEWithLogitsLoss() # binary multiclass classification
    elif cfg.finetuneDataset == 'zinc':
        out_dim = 1
        criterion = nn.L1Loss()
    else:
        raise ValueError('Invalid dataset name')
    
    # this is the predictor head, that takes the graph embeddings and predicts the property
    predictor = nn.Sequential(
        nn.Linear(cfg.model.hidden_size, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, out_dim)
    ).to(cfg.device)
    
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

        all_embeddings = torch.tensor([], requires_grad=False, device=cfg.device)
        mon_A_type = []
        stoichiometry = []

        for data in ft_trn_loader:
            data = data.to(cfg.device)
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
                y_lamellar = torch.tensor(data.y_lamellar, dtype=torch.float32, device=cfg.device)
                y_cylinder = torch.tensor(data.y_cylinder, dtype=torch.float32, device=cfg.device)
                y_sphere = torch.tensor(data.y_sphere, dtype=torch.float32, device=cfg.device)
                y_gyroid = torch.tensor(data.y_gyroid, dtype=torch.float32, device=cfg.device)
                y_disordered = torch.tensor(data.y_disordered, dtype=torch.float32, device=cfg.device)

                true_labels = torch.stack([y_lamellar, y_cylinder, y_sphere, y_gyroid, y_disordered], dim=1)

                train_loss = criterion(y_pred_trn, true_labels)
            elif cfg.finetuneDataset == 'zinc':
                train_loss = criterion(y_pred_trn, data.y.float())
            else:
                raise ValueError('Invalid dataset name')
            
            total_train_loss += train_loss
            train_loss.backward()
            optimizer.step()    

        total_train_loss /= len(ft_trn_loader)

        if epoch == cfg.finetune.epochs - 1:
            model.eval()
            predictor.eval()
            with torch.no_grad():
                val_loss = 0
                all_y_pred_val = []
                all_true_val = []

                for data in ft_val_loader:
                    data = data.to(cfg.device)
                    graph_embeddings = model.encode(data)
                    y_pred_val = predictor(graph_embeddings).squeeze()

                    if cfg.finetuneDataset == 'aldeghi':
                        val_loss += criterion(y_pred_val, data.y_EA.float() if cfg.finetune.property == 'ea' else data.y_IP.float())
                        all_y_pred_val.extend(y_pred_val.detach().cpu().numpy())
                        all_true_val.extend(data.y_EA.detach().cpu().numpy() if cfg.finetune.property == 'ea' else data.y_IP.detach().cpu().numpy())

                    elif cfg.finetuneDataset == 'diblock':
                        y_lamellar = torch.tensor(data.y_lamellar, dtype=torch.float32, device=cfg.device)
                        y_cylinder = torch.tensor(data.y_cylinder, dtype=torch.float32, device=cfg.device)
                        y_sphere = torch.tensor(data.y_sphere, dtype=torch.float32, device=cfg.device)
                        y_gyroid = torch.tensor(data.y_gyroid, dtype=torch.float32, device=cfg.device)
                        y_disordered = torch.tensor(data.y_disordered, dtype=torch.float32, device=cfg.device)

                        true_labels = torch.stack([y_lamellar, y_cylinder, y_sphere, y_gyroid, y_disordered], dim=1)

                        # i need to stack the 5 properties in a tensor and use them as the true values
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

            percentage = cfg.pretrain.aldeghiFTPercentage if cfg.finetuneDataset == 'aldeghi' else cfg.pretrain.diblockFTPercentage
            save_folder = f'Results/{model_name}/{cfg.finetuneDataset}_{cfg.modelVersion}_{percentage}'
            if cfg.finetuneDataset == 'aldeghi':
                label = 'ea' if cfg.finetune.property == 'ea' else 'ip'
                
                # if cfg.visualize.shouldEmbeddingSpace:
                #     visualeEmbeddingSpace(all_embeddings, mon_A_type, stoichiometry, model_name, epoch, isFineTuning=True)

                visualize_aldeghi_results(
                    np.array(all_y_pred_val), 
                    np.array(all_true_val), 
                    label=label, 
                    save_folder=save_folder,
                    epoch=epoch+1
                )
                

            elif cfg.finetuneDataset == 'diblock':
                visualize_diblock_results(
                    np.array(all_y_pred_val), 
                    np.array(all_true_val),
                    save_folder=save_folder,
                    epoch=epoch+1
                )
            else:
                raise ValueError('Invalid dataset name')
    
    return train_loss, val_loss