import os
import numpy as np
import random
import string
from src.config import cfg
from src.data import create_data
from src.infer_and_visualize import infer_by_dataloader, visualize_results
from src.PolymerJEPA import PolymerJEPA
from src.training import train, test
import time
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

def pretrain(pre_data, transform, cfg):
    # 70-20-10 split for pretraining - validation - test data
    pre_trn_data = pre_data[:int(0.7*len(pre_data))].copy()
    pre_val_data = pre_data[int(0.7*len(pre_data)):int(0.9*len(pre_data))].copy()
    pre_tst_data = pre_data[int(0.9*len(pre_data)):].copy()

    pre_trn_data.transform = transform
    pre_val_data.transform = transform
    pre_val_data = [x for x in pre_val_data] # this way we can use the same transform for the validation data all the times

    pre_trn_loader = DataLoader(dataset=pre_trn_data, batch_size=cfg.pretrain.batch_size, shuffle=True)
    pre_val_loader = DataLoader(dataset=pre_val_data, batch_size=cfg.pretrain.batch_size, shuffle=False)

    num_node_features = pre_data.data_list[0].num_node_features
    num_edge_features = pre_data.data_list[0].num_edge_features

    model = PolymerJEPA(
        nfeat_node=num_node_features,
        nfeat_edge=num_edge_features,
        nhid=cfg.model.hidden_size,
        nlayer_mlpmixer=cfg.model.nlayer_mlpmixer,
        gMHA_type=cfg.model.gMHA_type,
        rw_dim=cfg.pos_enc.rw_dim,
        pooling=cfg.model.pool,
        mlpmixer_dropout=cfg.pretrain.mlpmixer_dropout,
        patch_rw_dim=cfg.pos_enc.patch_rw_dim,
        num_target_patches=cfg.jepa.num_targets
    ).to(cfg.device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.pretrain.lr, 
        weight_decay=cfg.pretrain.wd
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=cfg.pretrain.lr_decay,
        patience=cfg.pretrain.lr_patience,
        verbose=True
    )

    # Create EMA scheduler for target encoder param update
    ipe = len(pre_trn_loader)
    ema_params = [0.996, 1.0]
    momentum_scheduler = (ema_params[0] + i*(ema_params[1]-ema_params[0])/(ipe*cfg.pretrain.epochs)
                        for i in range(int(ipe*cfg.pretrain.epochs)+1))


    random.seed(time.time())
    model_name = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))
    print(f"Model name: {model_name}")
    
    # Pretraining
    for epoch in tqdm(range(cfg.pretrain.epochs), desc='Pretraining Epochs'):
        model.train()
        _, trn_loss = train(
            pre_trn_loader, 
            model, 
            optimizer, 
            device=cfg.device, 
            momentum_weight=next(momentum_scheduler), 
            criterion_type=cfg.jepa.dist,
            regularization=cfg.pretrain.regularization
        )

        model.eval()
        _, val_loss = test(
            pre_val_loader, 
            model,
            device=cfg.device, 
            criterion_type=cfg.jepa.dist,
            regularization=cfg.pretrain.regularization
        )

        if epoch % 20 == 0 or epoch == cfg.pretrain.epochs - 1:
            os.makedirs('Models/Pretrain', exist_ok=True)
            torch.save(model.state_dict(), f'Models/Pretrain/{model_name}.pt')

        scheduler.step(val_loss)

        print(f'Epoch/Fold: {epoch:03d}, Train Loss: {trn_loss:.4f}' f' Test Loss:{val_loss:.4f}')
    
    return model, model_name


def finetune(ft_data, transform, model, model_name, cfg):
    ft_trn_data = ft_data[:int(0.7*len(ft_data))].copy()
    ft_val_data = ft_data[int(0.7*len(ft_data)):int(0.9*len(ft_data))].copy()
    ft_tst_data = ft_data[int(0.9*len(ft_data)):].copy()

    
    ft_trn_data.transform = transform
    ft_trn_data = [x for x in ft_trn_data]
    ft_val_data.transform = transform
    ft_val_data = [x for x in ft_val_data]

    ft_trn_loader = DataLoader(dataset=ft_trn_data, batch_size=cfg.finetune.batch_size, shuffle=True)
    ft_val_loader = DataLoader(dataset=ft_val_data, batch_size=cfg.finetune.batch_size, shuffle=False)

    # this is the predictor head, that takes the graph embeddings and predicts the property
    predictor = nn.Sequential(
        nn.Linear(cfg.model.hidden_size, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    ).to(cfg.device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=cfg.finetune.lr, weight_decay=cfg.finetune.wd)

    print(f"Finetuning model {model_name}")

    for epoch in tqdm(range(cfg.finetune.epochs), desc='Finetuning Epochs'):
        model.train()
        predictor.train()
        trn_loss = 0
        for data in ft_trn_loader:
            data = data.to(cfg.device)
            optimizer.zero_grad()
            embeddings = model.encode(data)
            y_pred_trn = predictor(embeddings).squeeze()
            train_loss = criterion(y_pred_trn, data.y_EA.float() if cfg.finetune.property == 'ea' else data.y_IP.float())
            trn_loss += train_loss
            train_loss.backward()
            optimizer.step()    

        trn_loss /= len(ft_trn_loader)

        model.eval()
        predictor.eval()

        with torch.no_grad():
            val_loss = 0
            all_y_pred_val = []
            all_true_val = []
            for data in ft_val_loader:
                data = data.to(cfg.device)
                embeddings = model.encode(data)
                y_pred_val = predictor(embeddings).squeeze()
                val_loss += criterion(y_pred_val, data.y_EA.float() if cfg.finetune.property == 'ea' else data.y_IP.float())
                all_y_pred_val.extend(y_pred_val.detach().cpu().numpy())
                all_true_val.extend(data.y_EA.detach().cpu().numpy() if cfg.finetune.property == 'ea' else data.y_IP.detach().cpu().numpy())
        
        val_loss /= len(ft_val_loader)

        if epoch % 5 == 0 or epoch == cfg.finetune.epochs - 1:
            os.makedirs('Results/Finetune/', exist_ok=True)

            if cfg.finetune.property == 'ea':
                label = 'ea'
            elif cfg.finetune.property == 'ip':
                label = 'ip'
            else:
                raise ValueError('Invalid property type')
            
            visualize_results(
                np.array(all_y_pred_val), 
                np.array(all_true_val), 
                label=label, 
                save_folder=f'Results/Finetune/{model_name}',
                epoch=epoch
            )
            
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}' f' Val Loss:{val_loss:.4f}')


def run():
    # [RISK]: how to handle the dataset? i am not sure from a dataset instance i can slice it like this
    # https://github.com/pyg-team/pytorch_geometric/issues/4223 
    dataset, transform = create_data(cfg)
    
    pre_data = dataset[:int(0.9*len(dataset))].copy()
    ft_data = dataset[int(0.9*len(dataset)):].copy()

    if cfg.shouldPretrain:
        model, model_name = pretrain(pre_data, transform, cfg)
    else:
        # load model from finetuning
        model_name = 'oWTyZivo'
        model = PolymerJEPA(
            nfeat_node=dataset.data_list[0].num_node_features,
            nfeat_edge=dataset.data_list[0].num_edge_features,
            nhid=cfg.model.hidden_size,
            nlayer_mlpmixer=cfg.model.nlayer_mlpmixer,
            gMHA_type=cfg.model.gMHA_type,
            rw_dim=cfg.pos_enc.rw_dim,
            pooling=cfg.model.pool,
            mlpmixer_dropout=cfg.pretrain.mlpmixer_dropout,
            patch_rw_dim=cfg.pos_enc.patch_rw_dim,
            num_target_patches=cfg.jepa.num_targets
        ).to(cfg.device)

        model.load_state_dict(torch.load(f'Models/Pretrain/{model_name}.pt'))
    

    if cfg.shouldFinetune:
        finetune(ft_data, transform, model, model_name, cfg)
    


        
if __name__ == '__main__':
    run()
    # check features dimensions for batches
    # for i in range(0, len(dataset), 20):
    #     try:
    #         Batch.from_data_list(dataset[i:i+20])
    #     except Exception as e:
    #         print(f"An error occurred: {e}")
    #         for data in dataset[i:i+20]:
    #             print(data)
    #             quit()
