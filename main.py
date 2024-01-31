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


def motifs2subgraphs(graph, cliques, cliques_edges, intermonomers_bonds, monomer_mask):
    # [TODO]: Check requirements (size of context and size of targets, etc.)

    # randomly pick one intermonomer bond
    intermonomer_bond = random.choice(intermonomers_bonds)
    
    # create a list of cliques for each monomer
    monomer_cliques = [[], []]
    for clique in cliques:
        monomer_cliques[monomer_mask[clique[0]]].append(clique)
    
    # randomly pick one clique from each monomer the clique must be one of those that contain a node of the intermonomer bond
    monomerA_clique = random.choice([clique for clique in monomer_cliques[monomer_mask[intermonomer_bond[0]]] if intermonomer_bond[0] in clique])
    monomerB_clique = random.choice([clique for clique in monomer_cliques[monomer_mask[intermonomer_bond[1]]] if intermonomer_bond[1] in clique])
    
    # context subgraph: join 2 cliques such that they belong to different monomers and they are connected by an intermonomer bond
    context_subgraph = monomerA_clique + monomerB_clique
    
    # select all other cliques (except the two chosen above) as target subgraphs
    target_subgraphs = [clique for clique in cliques if clique not in [monomerA_clique, monomerB_clique]]

    node_mask, edge_mask = create_masks(graph, context_subgraph, target_subgraphs, len(monomer_mask))

    # # sanity check: control whether the edge mask for the context subgraph is correct
    # for mask_val, edge in zip(edge_mask[0], graph.edge_index.T):
    #     # print mask_val without tensor() and edge without tensor()
    #     print(mask_val.item(), edge.tolist())

    # import networkx as nx
    # from torch_geometric.utils.convert import to_networkx
    # from matplotlib import pyplot as plt
    # G = to_networkx(graph, to_undirected=True)
    # # color context nodes 
    # color_map = []
    # for node in G:
    #     if node not in context_subgraph:
    #         color_map.append('green')
    #     else:
    #          color_map.append('red')
    # fig = nx.draw_networkx(G, node_color=color_map, with_labels=True)
    # plt.show()

    return node_mask, edge_mask

    
def create_masks(graph, context_subgraph, target_subgraphs, n_of_nodes):
    n_of_subgraphs = 1 + len(target_subgraphs)
    node_mask = torch.zeros((n_of_subgraphs, n_of_nodes), dtype=torch.bool)

    # context mask
    for node in context_subgraph:
        node_mask[0, node] = True
    
    # target masks
    for i, target_subgraph in enumerate(target_subgraphs):
        for node in target_subgraph:
            node_mask[i+1, node] = True

    edge_mask = node_mask[:, graph.edge_index[0]] & node_mask[:, graph.edge_index[1]]
    return node_mask, edge_mask



# cliques, cliques_edges, intermonomers_bonds, monomer_mask =  graphs[0].motifs[0], graphs[0].motifs[1], graphs[0].intermonomers_bonds, graphs[0].monomer_mask
# node_mask, edge_mask = motifs2subgraphs(graphs[0], cliques, cliques_edges, intermonomers_bonds, monomer_mask)
# for i in range(15):
#     cliques, cliques_edges, intermonomers_bonds, monomer_mask =  graphs[i].motifs[0], graphs[i].motifs[1], graphs[i].intermonomers_bonds, graphs[i].monomer_mask
#     node_mask, edge_mask = motifs2subgraphs(graphs[i], cliques, cliques_edges, intermonomers_bonds, monomer_mask)
   




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
