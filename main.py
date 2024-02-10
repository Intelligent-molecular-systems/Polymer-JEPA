import os
import numpy as np
import random
import string
from src.config import cfg
from src.data import create_data
from src.infer_and_visualize import infer_by_dataloader, visualize_results
from src.PolymerJEPA import PolymerJEPA
from src.training import train, test
from src.WDNodeMPNN import WDNodeMPNN
import time
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

def run():
    # [RISK]: how to handle the dataset? i am not sure from a dataset instance i can slice it like this
    # https://github.com/pyg-team/pytorch_geometric/issues/4223 
    dataset, transform = create_data(cfg)
    dataset.transform = transform
    # check features dimensions for batches
    # for i in range(0, len(dataset), 20):
    #     try:
    #         Batch.from_data_list(dataset[i:i+20])
    #     except Exception as e:
    #         print(f"An error occurred: {e}")
    #         for data in dataset[i:i+20]:
    #             print(data)
    #             quit()
    # print(type(dataset))
    # 50-50 split for pretraining - fine-tuning data
    pre_data = dataset[:int(0.9*len(dataset.data_list))].copy()

    # 70-20-10 split for pretraining - validation - test data
    pre_trn_data = pre_data[:int(0.7*len(pre_data))].copy()
    pre_val_data = pre_data[int(0.7*len(pre_data)):int(0.9*len(pre_data))].copy()
    pre_tst_data = pre_data[int(0.9*len(pre_data)):].copy()

    ft_data = dataset[int(0.1*len(dataset.data_list)):].copy()
    ft_trn_data = ft_data[:int(0.7*len(ft_data))].copy()
    ft_val_data = ft_data[int(0.7*len(ft_data)):int(0.9*len(ft_data))].copy()
    ft_tst_data = ft_data[int(0.9*len(ft_data)):].copy()

    pre_trn_data.transform = transform
    pre_val_data.transform = transform
    pre_tst_data.transform = transform

    pre_trn_loader = DataLoader(dataset=pre_trn_data, batch_size=cfg.train.batch_size, shuffle=True)
    pre_val_loader = DataLoader(dataset=pre_val_data, batch_size=cfg.train.batch_size, shuffle=False)

    num_node_features = dataset.data_list[0].num_node_features
    num_edge_features = dataset.data_list[0].num_edge_features

    model = PolymerJEPA(
        nfeat_node=num_node_features,
        nfeat_edge=num_edge_features,
        nhid=cfg.model.hidden_size,
        nlayer_mlpmixer=cfg.model.nlayer_mlpmixer,
        gMHA_type=cfg.model.gMHA_type,
        rw_dim=cfg.pos_enc.rw_dim,
        pooling=cfg.model.pool,
        n_patches=cfg.subgraphing.n_patches,
        mlpmixer_dropout=cfg.train.mlpmixer_dropout,
        patch_rw_dim=cfg.pos_enc.patch_rw_dim,
        num_target_patches=cfg.jepa.num_targets
    ).to(cfg.device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.train.lr, 
        weight_decay=cfg.train.wd
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=cfg.train.lr_decay,
        patience=cfg.train.lr_patience,
        verbose=True
    )

    # Create EMA scheduler for target encoder param update
    ipe = len(pre_trn_loader)
    ema_params = [0.996, 1.0]
    momentum_scheduler = (ema_params[0] + i*(ema_params[1]-ema_params[0])/(ipe*cfg.train.epochs)
                        for i in range(int(ipe*cfg.train.epochs)+1))


    random.seed(time.time())
    model_name = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))
    print(f"Model name: {model_name}")
    
    # Pretraining
    for epoch in tqdm(range(cfg.train.epochs), desc='Pretraining Epochs'):
        model.train()
        _, trn_loss = train(
            pre_trn_loader, 
            model, 
            optimizer, 
            device=cfg.device, 
            momentum_weight=next(momentum_scheduler), 
            criterion_type=cfg.jepa.dist,
            regularization=cfg.train.regularization
        )

        model.eval()
        _, val_loss = test(
            pre_val_loader, 
            model,
            device=cfg.device, 
            criterion_type=cfg.jepa.dist,
            regularization=cfg.train.regularization
        )

        if epoch % 20 == 0 or epoch == cfg.train.epochs - 1:
            os.makedirs('Models/Pretrain', exist_ok=True)
            torch.save(model.state_dict(), f'Models/Pretrain/{model_name}.pt')

        scheduler.step(val_loss)

        print(f'Epoch/Fold: {epoch:03d}, Train Loss: {trn_loss:.4f}' f' Test Loss:{val_loss:.4f}')



    # finetune
    model.eval()
    
    ft_trn_data.transform = transform
    ft_val_data.transform = transform

    ft_trn_loader = DataLoader(dataset=ft_trn_data, batch_size=cfg.finetune.batch_size, shuffle=True)
    ft_val_loader = DataLoader(dataset=ft_val_data, batch_size=cfg.finetune.batch_size, shuffle=False)

    X_train, y_train = [], []
    X_test, y_test = [], []
    # get graph embeddings for finetuning, basically we train an MLP on top of the graph (polymer) embeddings
    for data in ft_trn_loader:
        data = data.to(cfg.device)
        with torch.no_grad():
            features = model.encode(data)
            X_train.append(features.detach().cpu().numpy())
            if cfg.finetune.property == 'ea':
                y_train.append(data.y_EA.detach().cpu().numpy())
            elif cfg.finetune.property == 'ip':
                y_train.append(data.y_IP.detach().cpu().numpy())
            else:
                raise ValueError('Invalid property type')
        
    # Concatenate the lists into numpy arrays
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    for data in ft_val_loader:
        data = data.to(cfg.device)
        with torch.no_grad():
            features = model.encode(data)
            X_test.append(features.detach().cpu().numpy())
            if cfg.finetune.property == 'ea':
                y_test.append(data.y_EA.detach().cpu().numpy())
            elif cfg.finetune.property == 'ip':
                y_test.append(data.y_IP.detach().cpu().numpy())
            else:
                raise ValueError('Invalid property type')

    # Concatenate the lists into numpy arrays
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    print("Data shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    X_train = torch.from_numpy(X_train).float().to(cfg.device)
    y_train = torch.from_numpy(y_train).float().to(cfg.device)
    X_test = torch.from_numpy(X_test).float().to(cfg.device)
    y_test = torch.from_numpy(y_test).float().to(cfg.device)

    # this is the predictor head, that takes the graph embeddings and predicts the property
    predictor = nn.Sequential(
        nn.Linear(cfg.model.hidden_size, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    ).to(cfg.device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=cfg.finetune.lr, weight_decay=cfg.finetune.wd)

    for epoch in tqdm(range(cfg.finetune.epochs), desc='Finetuning Epochs'):
        model.eval()
        predictor.train()
        optimizer.zero_grad()
        y_pred_trn = predictor(X_train).squeeze()
        train_loss = criterion(y_pred_trn, y_train)
        train_loss.backward()
        optimizer.step()

        predictor.eval()
        y_pred = predictor(X_test).squeeze()
        val_loss = criterion(y_pred, y_test)

        if epoch % 10 == 0 or epoch == cfg.finetune.epochs - 1:
            os.makedirs('Results/Finetune/', exist_ok=True)

            if cfg.finetune.property == 'ea':
                label = 'ea'
            elif cfg.finetune.property == 'ip':
                label = 'ip'
            else:
                raise ValueError('Invalid property type')
            
            visualize_results(
                y_pred.detach().cpu().numpy(), 
                y_test.detach().cpu().numpy(), 
                label=label, 
                save_folder=f'Results/Finetune/{model_name}',
                epoch=epoch
            )
            
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}' f' Val Loss:{val_loss:.4f}')
        
if __name__ == '__main__':
    run()