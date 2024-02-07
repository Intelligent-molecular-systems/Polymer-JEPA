import os
import pandas as pd
from torch_geometric.data import Data
import torch
from torch_geometric.data import Batch
import tqdm
from src.featurization_utils.featurization import poly_smiles_to_graph
from src.subgraphing_utils.subgraphs_extractor import motifs2subgraphs, metis2subgraphs, randomWalks2subgraphs
from torch_geometric.utils import subgraph

def get_graphs(file_csv = 'Data/dataset-poly_chemprop.csv', file_graphs_list = 'Data/Graphs_list.pt'):
    graphs = []
    # check if graphs_list.pt exists
    if not os.path.isfile(file_graphs_list):
        print('Creating Graphs_list.pt')
        df = pd.read_csv(file_csv)
        # use tqdm to show progress bar
        for i in tqdm.tqdm(range(len(df.loc[:, 'poly_chemprop_input']))):
            poly_strings = df.loc[i, 'poly_chemprop_input']
            poly_labels_EA = df.loc[i, 'EA vs SHE (eV)']
            poly_labels_IP = df.loc[i, 'IP vs SHE (eV)']
            # given the input polymer string, this function returns a pyg data object
            graph = poly_smiles_to_graph(
                poly_strings=poly_strings, 
                poly_labels_EA=poly_labels_EA, 
                poly_labels_IP=poly_labels_IP
            ) 

            
            graphs.append(graph)
            
        torch.save(graphs, file_graphs_list)
        print('Graphs_list.pt saved')
    else:
        print('Loading Graphs_list.pt')
        graphs = torch.load(file_graphs_list)

    return graphs



# simpler approach: precompute masks and them as arguemnts of the Data graph object 

def train(model, model2, loader, label, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
    #     for i, graph in enumerate(batch.to_data_list()):
    #         node_mask, edge_mask = randomWalks2subgraphs(graph)
            # Process the first subgraph through encoder_1 #
            # contextData = Data(x=graph.x, edge_index=graph.edge_index[:, edge_mask[0]], edge_attr=graph.edge_attr[edge_mask[0]], edge_weight=graph.edge_weight[edge_mask[0]], node_weight=graph.node_weight)
            # print(contextData)
            # print(contextData.x)
            # print(contextData.edge_index)
            
            # out1 = model(contextData)
            # print(out1)
        
        batch_list = batch.to_data_list()
        # stack all node masks for all graphs in the batch, each graph has a different number of nodes so pad them
        node_masks = torch.empty((0, max([graph.x.size()[0] for graph in batch_list])), dtype=torch.bool)
        edge_masks = torch.empty((0, max([graph.edge_index.size()[1] for graph in batch_list])), dtype=torch.bool)
        
        subgraphs_per_graph_list = []
        for idx, graph in enumerate(batch_list):
            node_mask, edge_mask = randomWalks2subgraphs(graph)        
            subgraphs_per_graph_list.append(node_mask.size()[0])
            # add padding to the masks
            node_mask = torch.cat([node_mask, torch.zeros((node_mask.size()[0], node_masks.size()[1] - node_mask.size()[1]), dtype=torch.bool)], dim=1)
            edge_mask = torch.cat([edge_mask, torch.zeros((edge_mask.size()[0], edge_masks.size()[1] - edge_mask.size()[1]), dtype=torch.bool)], dim=1)
            
            node_masks = torch.cat([node_masks, node_mask], dim=0)
            edge_masks = torch.cat([edge_masks, edge_mask], dim=0)

        # node_masks = each graph has n entries, where n is the number of subgraphs it has in total, the columns are the nodes of the graph, we consider the graph with more nodes, and then pad
        # batch.x has n rows, where n is the total nodes in the batch graphs, the columns are the features of the nodes
        # for each graph in the batch, apply the mask to the graph to get the context subgraphs
        # batch them together and pass them to the model
        
        context_subgraphs = []
        from copy import deepcopy
        i = 0
        for idx, graph in enumerate(batch_list):
            context_subgraph = deepcopy(graph)
            # we pass the full graph to avoid having issues with the edge index, but then the gnn works only on the masked node indices
            # !!! i might have to pass in the node mask as a paramter to the model, so that the pooling is done only on the nodes that are actually in the subgraph !!!
            context_subgraph.x = graph.x # [node_masks[i][:graph.x.size()[0]]]
            context_subgraph.edge_index = graph.edge_index[:, edge_masks[i][:graph.edge_index.size()[1]]]
            context_subgraph.edge_attr = graph.edge_attr[edge_masks[i][:graph.edge_attr.size()[0]]]
            context_subgraph.edge_weight = graph.edge_weight[edge_masks[i][:graph.edge_weight.size()[0]]]
            context_subgraph.node_weight = graph.node_weight #[node_masks[i][:graph.node_weight.size()[0]]] 
            context_subgraphs.append(context_subgraph)
            i += subgraphs_per_graph_list[idx]
        
        # !!! the issue is that edge index nodes are still the old ones of the full graph !!!
        
        context_batch = Batch.from_data_list(context_subgraphs)
        

        # do the same for the target subgraphs, which are all the other subgraphs that are not in the context subgraphs
        # we can have multiple target subgraphs per graph
        target_subgraphs = []
        target_idx = 0
        for k, graph in enumerate(batch_list):
            for j in range(target_idx +1, target_idx + subgraphs_per_graph_list[k]):
                target_subgraph = deepcopy(graph)
                target_subgraph.x = graph.x # [node_masks[j][:graph.x.size()[0]]]
                target_subgraph.edge_index = graph.edge_index[:, edge_masks[j][:graph.edge_index.size()[1]]]
                target_subgraph.edge_attr = graph.edge_attr[edge_masks[j][:graph.edge_attr.size()[0]]]
                target_subgraph.edge_weight = graph.edge_weight[edge_masks[j][:graph.edge_weight.size()[0]]]
                target_subgraph.node_weight = graph.node_weight # [node_masks[j][:graph.node_weight.size()[0]]]
                target_subgraphs.append(target_subgraph)

            target_idx += subgraphs_per_graph_list[k]
        
        target_batch = Batch.from_data_list(target_subgraphs)

        # print(context_batch)
        # print(target_batch)
        
        # create a batch of graphs            
        out = model(target_batch)
        # Calculate the loss based on the specified label.
        if label == 0: # EA
            loss = criterion(out, target_batch.y_EA.float())
        elif label == 1: # IP
            loss = criterion(out, target_batch.y_IP.float())

        loss.backward()  
        optimizer.step()
        optimizer.zero_grad()

    total_loss += loss.item()

    return model, total_loss / len(loader)

    
def test(model, loader, label, criterion):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        out = model(batch)
        if label == 0: # EA
            loss = criterion(out, batch.y_EA.float())
        elif label == 1: # IP
            loss = criterion(out, batch.y_IP.float())

        total_loss += loss.item()
    return total_loss / len(loader)