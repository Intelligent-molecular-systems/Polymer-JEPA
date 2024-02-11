import os
import pandas as pd
import random
from src.featurization_utils.featurization import poly_smiles_to_graph
from src.transform import PositionalEncodingTransform, GraphJEPAPartitionTransform
import torch
from torch_geometric.data import InMemoryDataset
import tqdm


# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset
# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.InMemoryDataset.html#torch_geometric.data.InMemoryDataset
class MyDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform=None, pre_transform=None):
        self.data_list = data_list
        super().__init__(root, transform, pre_transform)
        # self.load(self.processed_paths[0])
        # For PyG<2.4:
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'dataset.pt'

    # Processes the dataset to the self.processed_dir folder.
    def process(self):
        if self.pre_transform is not None:
            self.data_list = [self.pre_transform(data) for data in self.data_list]
        # self.save(self.data_list, self.processed_paths[0])
        # For PyG<2.4:
        torch.save(self.collate(self.data_list), self.processed_paths[0])


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

    random.seed(12345)
    graphs = random.sample(graphs, len(graphs))
    return graphs


def create_data(cfg):
    pre_transform = PositionalEncodingTransform(rw_dim=cfg.pos_enc.rw_dim, lap_dim=cfg.pos_enc.lap_dim)
    
    transform = GraphJEPAPartitionTransform(
        subgraphing_type=cfg.jepa.subgraphing_type,
        num_targets=cfg.jepa.num_targets,
        n_patches=cfg.subgraphing.n_patches,
        patch_num_diff=cfg.pos_enc.patch_num_diff
    )

    graphs = get_graphs()
    dataset = MyDataset(root='Data', data_list=graphs, pre_transform=pre_transform)

    return dataset, transform