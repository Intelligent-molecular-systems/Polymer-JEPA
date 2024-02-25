import os
import numpy as np
import random
from src.config import cfg
from src.visualize import visualize_aldeghi_results, visualize_diblock_results, visualeEmbeddingSpace
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def finetune(ft_trn_data, ft_val_data, transform, model, model_name, cfg):

    # print(len(ft_val_data))
    # ft_tst_data = ft_data[int(0.9*len(ft_data)):].copy()
    print(f'Finetuning training on: {len(ft_trn_data)} graphs')
    print(f'Finetuning validating on: {len(ft_val_data)} graphs')
    
    if cfg.modelVersion == 'v1': # only model v1 requires the transforms, since finetuning on v2 inputs directly the full graph, no subgraphs
        ft_trn_data.transform = transform
        ft_trn_data = [x for x in ft_trn_data]
        ft_val_data.transform = transform
        ft_val_data = [x for x in ft_val_data]

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
    else:
        raise ValueError('Invalid dataset name')
    
    # this is the predictor head, that takes the graph embeddings and predicts the property
    predictor = nn.Sequential(
        nn.Linear(cfg.model.hidden_size, cfg.model.wdmpnn_hid_dim),
        nn.ReLU(),
        nn.Linear(cfg.model.wdmpnn_hid_dim, out_dim)
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

        all_embeddings = torch.tensor([], device=cfg.device)
        mon_A_type = torch.tensor([], device=cfg.device)
        stoichiometry = []

        for data in ft_trn_loader:
            data = data.to(cfg.device)
            optimizer.zero_grad()
            graph_embeddings = model.encode(data)

            if cfg.finetuneDataset == 'aldeghi':
                mon_A_type = torch.cat((mon_A_type, data.mon_A_type), dim=0)
                all_embeddings = torch.cat((all_embeddings, graph_embeddings), dim=0)
                stoichiometry.extend(data.stoichiometry)

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
            else:
                raise ValueError('Invalid dataset name')
            
            total_train_loss += train_loss
            train_loss.backward()
            optimizer.step()    

        total_train_loss /= len(ft_trn_loader)


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
                else:
                    raise ValueError('Invalid dataset name')
                
        val_loss /= len(ft_val_loader)

        if epoch % 20 == 0 or epoch == cfg.finetune.epochs - 1:
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.5f}' f' Val Loss:{val_loss:.5f}')

            os.makedirs('Results/', exist_ok=True)

            if cfg.finetuneDataset == 'aldeghi':
                if cfg.finetune.property == 'ea':
                    label = 'ea'
                elif cfg.finetune.property == 'ip':
                    label = 'ip'
                else:
                    raise ValueError('Invalid property type')

                visualeEmbeddingSpace(all_embeddings, mon_A_type, stoichiometry, model_name, epoch, isFineTuning=True)

                visualize_aldeghi_results(
                    np.array(all_y_pred_val), 
                    np.array(all_true_val), 
                    label=label, 
                    save_folder=f'Results/{model_name}_aldeghi',
                    epoch=epoch
                )
                

            elif cfg.finetuneDataset == 'diblock':
                visualize_diblock_results(
                    np.array(all_y_pred_val), 
                    np.array(all_true_val), 
                    label='lamellar', 
                    save_folder=f'Results/{model_name}_diblock',
                    epoch=epoch
                )
            else:
                raise ValueError('Invalid dataset name')
    
    return model