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
    pre_data = dataset[:int(0.5*len(dataset.data_list))].copy()

    # 70-20-10 split for pretraining - validation - test data
    pre_trn_data = pre_data[:int(0.7*len(pre_data))].copy()
    pre_val_data = pre_data[int(0.7*len(pre_data)):int(0.9*len(pre_data))].copy()
    pre_tst_data = pre_data[int(0.9*len(pre_data)):].copy()

    ft_data = dataset[int(0.5*len(dataset.data_list)):].copy()
    ft_trn_data = ft_data[:int(0.7*len(ft_data))].copy()
    ft_val_data = ft_data[int(0.7*len(ft_data)):int(0.9*len(ft_data))].copy()
    ft_tst_data = ft_data[int(0.9*len(ft_data)):].copy()

    pre_trn_data.transform = transform
    pre_val_data.transform = transform
    pre_tst_data.transform = transform

    pre_trn_loader = DataLoader(dataset=dataset, batch_size=cfg.train.batch_size, shuffle=True)
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


    # Pretraining
    for epoch in tqdm(range(cfg.train.epochs), desc='Training Epochs'):
        model.train()
        _, trn_loss = train(
            pre_trn_loader, 
            model, 
            optimizer, 
            device=cfg.device, 
            momentum_weight=next(momentum_scheduler), 
            criterion_type=cfg.jepa.dist
        )

        model.eval()
        _, val_loss = test(
            pre_val_loader, 
            model,
            device=cfg.device, 
            criterion_type=cfg.jepa.dist
        )

        scheduler.step(val_loss)

        print(f'Epoch/Fold: {epoch:03d}, Train Loss: {trn_loss:.4f}' f' Test Loss:{val_loss:.4f}')


    # finetune
        
if __name__ == '__main__':
    run()
    # print('Done!')