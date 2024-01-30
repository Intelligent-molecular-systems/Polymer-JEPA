import torch
import random
import time
from torch_geometric.loader import DataLoader
import string
from src.WDNodeMPNN import WDNodeMPNN
import os
import tqdm
from src.training import get_graphs, train, test
from src.infer_and_visualize import infer_by_dataloader, visualize_results
from src.hyperparam_optim import hyperparams_optimization

# %% Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

hyper_params = {
    'batch_size': 64,
    'epochs': 100,
    'hidden_dimension': 300,
    'learning_rate': 1e-3,
}

labels = { 
    'EA': 0,
    'IP': 1,
}

# %% Load data
graphs = get_graphs(file_csv = 'Data/dataset-poly_chemprop.csv', file_graphs_list = 'Data/Graphs_list.pt')

# shuffle graphs
random.seed(12345)
data_list_shuffle = random.sample(graphs, len(graphs))

# take 80-20 split for training - test data
train_datalist = data_list_shuffle[:int(0.8*len(data_list_shuffle))]
test_datalist = data_list_shuffle[int(0.8*len(data_list_shuffle)):]
num_node_features = train_datalist[0].num_node_features
num_edge_features = train_datalist[0].num_edge_features

# print some statistics
print(f'Number of training graphs: {len(train_datalist)}')
print(f'Number of test graphs: {len(test_datalist)}')
print(f'Number of node features: {num_node_features}')
print(f'Number of edge features:{num_edge_features} ')

batch_size = hyper_params['batch_size']
train_loader = DataLoader(dataset=train_datalist, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_datalist, batch_size=batch_size, shuffle=False)

hidden_dimension = hyper_params['hidden_dimension']


model = WDNodeMPNN(num_node_features, num_edge_features, hidden_dim=hidden_dimension)
random.seed(time.time())
model_name = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))
model.to(device)
print(model_name)
print(model)


optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params['learning_rate'])
criterion = torch.nn.MSELoss()
epochs = hyper_params['epochs']
property = 'IP'
model_save_name = f'{model_name}_{property}'


should_optimize = False

if should_optimize:
    hyperparams_optimization(
        train_datalist=train_datalist,
        test_datalist=test_datalist
    )
else:
    # %% Train model
    for epoch in tqdm.tqdm(range(epochs)):
        model, train_loss = train(model, train_loader, label=labels[property], optimizer=optimizer, criterion=criterion)
        test_loss = test(model, test_loader, label=labels[property], criterion=criterion)

        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        # save model every 20 epochs or last epoch
        if epoch % 10 == 0 or epoch == epochs - 1:
            os.makedirs('Models', exist_ok=True)
            os.makedirs('Results', exist_ok=True)

            pred, ea, ip = infer_by_dataloader(test_loader, model, device)

            if labels[property] == 0:
                visualize_results(pred, ea, label='ea', save_folder=f'Results/{model_save_name}', epoch=epoch)
            else:
                visualize_results(pred, ip, label='ip', save_folder=f'Results/{model_save_name}', epoch=epoch)

            torch.save(model.state_dict(), f'Models/{model_save_name}.pt')
            
    # save latest model
    torch.save(model.state_dict(), f'Models/{model_save_name}.pt')
