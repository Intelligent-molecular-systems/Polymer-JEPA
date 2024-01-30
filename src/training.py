import os
import pandas as pd
import torch
import tqdm
from src.featurization import poly_smiles_to_graph


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


def train(model, loader, label, optimizer, criterion):

    model.train()
    total_loss = 0.0
    for data in loader:
        out = model(data)
        # Calculate the loss based on the specified label.
        if label == 0: # EA
            loss = criterion(out, data.y_EA.float())
        elif label == 1: # IP
            loss = criterion(out, data.y_IP.float())

        loss.backward()  
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return model, total_loss / len(loader)

    
def test(model, loader, label, criterion):
    model.eval()
    total_loss = 0.0
    for data in loader:
        out = model(data)
        if label == 0: # EA
            loss = criterion(out, data.y_EA.float())
        elif label == 1: # IP
            loss = criterion(out, data.y_IP.float())

        total_loss += loss.item()
    return total_loss / len(loader)