import optuna
import torch
from torch_geometric.loader import DataLoader as pyg_DataLoader
import tqdm
from src.training import train, test
from src.WDNodeMPNN import WDNodeMPNN

# Hyperparameter optimization
def hyperparams_optimization(
        train_datalist,
        test_datalist,
        n_trials=100, 
        num_epochs=30, 
        optimizer_class=torch.optim.Adam
    ):

    def objective(trial):

        node_attr_dim = train_datalist[0].x.shape[1]
        edge_attr_dim = train_datalist[0].edge_attr.shape[1]
     
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        train_dataloader = pyg_DataLoader(train_datalist, batch_size=batch_size, shuffle=True)
        val_dataloader = pyg_DataLoader(test_datalist, batch_size=batch_size, shuffle=False)

        hidden_dimension = trial.suggest_int('hidden_dimension', 50, 500)
        n_message_passing_layers = trial.suggest_int('n_message_passing_layers', 1, 5)
        agg_func = trial.suggest_categorical('aggr_func', ['mean', 'max', 'add'])
        label = trial.suggest_categorical('label', [0, 1])
        model = WDNodeMPNN(
            node_attr_dim=node_attr_dim, 
            edge_attr_dim=edge_attr_dim, 
            n_message_passing_layers=n_message_passing_layers, 
            hidden_dim=hidden_dimension,
            agg_func=agg_func
        )

        print(f"GNN: \n{model}")

        learning_rate = trial.suggest_float('lr', *(0.01, 0.1))
        optimizer = optimizer_class(params=model.parameters(), lr=learning_rate)

        criterion = torch.nn.MSELoss()

        losses = []
        val_losses = []

        for epoch in tqdm.tqdm(range(num_epochs)): #args epochs
            model.train()
            avg_epoch_loss = train(model=model, loader=train_dataloader, label=label, optimizer=optimizer, criterion=criterion)[1]

            model.eval()
            avg_epoch_val_loss = test(model=model, loader=val_dataloader, label=label, criterion=criterion)

            losses.append(avg_epoch_loss)
            val_losses.append(avg_epoch_val_loss)

            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, trn_Loss: {avg_epoch_loss:.3f}, val_loss: {avg_epoch_val_loss:.3f}')
            
            
        
        return avg_epoch_val_loss

    # Optimize hyperparameters
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value
    print(f'Best hyperparameters: {best_params}\nBest validation loss: {best_value}')
    
