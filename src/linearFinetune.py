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
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error

def finetune(ft_trn_data, ft_val_data, model, model_name, cfg):

    print("Finetuning on model: ", model_name)
    print(f'Finetuning training on: {len(ft_trn_data)} graphs')
    print(f'Finetuning validating on: {len(ft_val_data)} graphs')
    
    if cfg.modelVersion == 'v2':
        # no need to use transform at every data access
        ft_trn_data = [x for x in ft_trn_data]

    ft_trn_loader = DataLoader(dataset=ft_trn_data, batch_size=cfg.finetune.batch_size, shuffle=True)
    ft_val_loader = DataLoader(dataset=ft_val_data, batch_size=cfg.finetune.batch_size, shuffle=False)

    # Initialize scikit-learn models
    if cfg.finetuneDataset == 'aldeghi' or cfg.finetuneDataset == 'zinc':
        predictor = Ridge()
    elif cfg.finetuneDataset == 'diblock':
        log_reg = LogisticRegression(max_iter=10000)
        predictor = MultiOutputClassifier(log_reg, n_jobs=-1)
    else:
        raise ValueError('Invalid dataset name')


    X_train, y_train = [], []

    # Collect training data
    for data in ft_trn_loader:
        data = data.to(cfg.device)
        with torch.no_grad():  # Ensure no gradient is computed to save memory
            graph_embeddings = model.encode(data).detach().cpu().numpy()
        X_train.extend(graph_embeddings)

        if cfg.finetuneDataset == 'aldeghi':
            y_train.extend(data.y_EA.detach().cpu().numpy() if cfg.finetune.property == 'ea' else data.y_IP.detach().cpu().numpy())
        elif cfg.finetuneDataset == 'diblock':
            # Need to convert to a format suitable for LogisticRegression
            y_labels = np.stack([data.y_lamellar, data.y_cylinder, data.y_sphere, data.y_gyroid, data.y_disordered], axis=1).argmax(axis=1)
            y_train.extend(y_labels)
        elif cfg.finetuneDataset == 'zinc':
            y_train.extend(data.y.detach().cpu().numpy())
        else:
            raise ValueError('Invalid dataset name')

    # Scale features
    y_train = np.array(y_train)

    # Fit the model
    predictor.fit(X_train, y_train)

    # Evaluation
    X_val, y_val = [], []
    for data in ft_val_loader:
        data = data.to(cfg.device)
        with torch.no_grad():
            graph_embeddings = model.encode(data).detach().cpu().numpy()
        X_val.extend(graph_embeddings)

        if cfg.finetuneDataset == 'aldeghi':
            y_val.extend(data.y_EA.detach().cpu().numpy() if cfg.finetune.property == 'ea' else data.y_IP.detach().cpu().numpy())
        elif cfg.finetuneDataset == 'diblock':
            y_labels = np.stack([data.y_lamellar, data.y_cylinder, data.y_sphere, data.y_gyroid, data.y_disordered], axis=1).argmax(axis=1)
            y_val.extend(y_labels)
        elif cfg.finetuneDataset == 'zinc':
            y_val.extend(data.y.detach().cpu().numpy())
        else:
            raise ValueError('Invalid dataset name')

    X_val = np.array(X_val)
    y_val = np.array(y_val)

    # Predict and evaluate
    y_pred_val = predictor.predict(X_val)

    lin_mae = mean_absolute_error(y_val, y_pred_val)
    print(f'Train R2.: {predictor.score(X_train, y_train)}')
    print(f'Val MAE.: {lin_mae}')

    return model
