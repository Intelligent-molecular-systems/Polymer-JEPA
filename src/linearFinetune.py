import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import mean_absolute_error

def finetune(ft_trn_data, ft_val_data, model, model_name, cfg, device):

    print(f'Number of parameters: {count_parameters(model)}')

    print("Finetuning on model: ", model_name)
    print(f'Finetuning training on: {len(ft_trn_data)} graphs')
    print(f'Finetuning validating on: {len(ft_val_data)} graphs')
    
    
    # no need to use transform at every data access
    ft_trn_data = [x for x in ft_trn_data]
    ft_val_data = [x for x in ft_val_data]

    ft_trn_loader = DataLoader(dataset=ft_trn_data, batch_size=cfg.finetune.batch_size, shuffle=True, num_workers=cfg.num_workers)
    ft_val_loader = DataLoader(dataset=ft_val_data, batch_size=cfg.finetune.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Initialize scikit-learn models
    if cfg.finetuneDataset == 'aldeghi':
        predictor = Ridge(fit_intercept=True, copy_X=True, max_iter=5000)
    elif cfg.finetuneDataset == 'diblock':
        log_reg = LogisticRegression(dual=False, fit_intercept=True, max_iter=5000)
        predictor = MultiOutputClassifier(log_reg, n_jobs=-1)
    else:
        raise ValueError('Invalid dataset name')

    model.eval()
    
    X_train, y_train = [], []

    # Collect training data
    for data in ft_trn_loader:
        data = data.to(device)
        with torch.no_grad():  # Ensure no gradient is computed to save memory
            graph_embeddings = model.encode(data).detach().cpu().numpy()
        X_train.extend(graph_embeddings)

        if cfg.finetuneDataset == 'aldeghi':
            y_train.extend(data.y_EA.detach().cpu().numpy() if cfg.finetune.property == 'ea' else data.y_IP.detach().cpu().numpy())
        elif cfg.finetuneDataset == 'diblock':
            # Need to convert to a format suitable for LogisticRegression
            y_labels = np.stack([data.y_lamellar, data.y_cylinder, data.y_sphere, data.y_gyroid, data.y_disordered], axis=1).argmax(axis=1)
            y_train.extend(y_labels)

        else:
            raise ValueError('Invalid dataset name')

    # Scale features
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # normalize regressor X by subtracting mean and dividing by l2 norm (as it was done in adgcl paper)
    # X_train -= X_train.mean(0)
    # # X_train /= np.linalg.norm(X_train, axis=0)
    # X_train /= X_train.std(0)

    # Fit the model
    predictor.fit(X_train, y_train)

    # Evaluation
    X_val, y_val = [], []
    for data in ft_val_loader:
        data = data.to(device)
        with torch.no_grad():
            graph_embeddings = model.encode(data).detach().cpu().numpy()
        X_val.extend(graph_embeddings)

        if cfg.finetuneDataset == 'aldeghi':
            y_val.extend(data.y_EA.detach().cpu().numpy() if cfg.finetune.property == 'ea' else data.y_IP.detach().cpu().numpy())
        elif cfg.finetuneDataset == 'diblock':
            y_labels = np.stack([data.y_lamellar, data.y_cylinder, data.y_sphere, data.y_gyroid, data.y_disordered], axis=1).argmax(axis=1)
            y_val.extend(y_labels)

        else:
            raise ValueError('Invalid dataset name')

    X_val = np.array(X_val)
    y_val = np.array(y_val)

    # normalize regressor X by subtracting mean and dividing by l2 norm (as it was done in adgcl paper)
    # X_val -= X_val.mean(0)
    # # X_val /= np.linalg.norm(X_val, axis=0)
    # X_val /= X_val.std(0)

    # print("Data shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    # Predict and evaluate
    y_pred_val = predictor.predict(X_val)

    lin_mae = mean_absolute_error(y_val, y_pred_val)
    trn_r2 = predictor.score(X_train, y_train)
    print(f'Train R2.: {trn_r2}')
    print(f'Val MAE.: {lin_mae}')

    metrics = {'train_r2': trn_r2, 'val_mae': lin_mae}
    return metrics

def count_parameters(model):
    # For counting number of parameteres: need to remove unnecessary DiscreteEncoder, and other additional unused params
    return sum(p.numel() for p in model.parameters() if p.requires_grad)