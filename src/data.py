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


def get_graphs(file_csv='Data/aldeghi_coley_ea_ip_dataset.csv', file_graphs_list='Data/aldeghi_graphs_list.pt', dataset='aldeghi'):
    graphs = []
    # check if graphs_list.pt exists
    if not os.path.isfile(file_graphs_list):
        print('Creating graphs pt file...')
        df = pd.read_csv(file_csv)
        # use tqdm to show progress bar
        if dataset == 'aldeghi':
            for i in tqdm.tqdm(range(len(df.loc[:, 'poly_chemprop_input']))):
                poly_strings = df.loc[i, 'poly_chemprop_input']
                ea_values = df.loc[i, 'EA vs SHE (eV)']
                ip_values = df.loc[i, 'IP vs SHE (eV)']
                # given the input polymer string, this function returns a pyg data object
                graph = poly_smiles_to_graph(
                    poly_strings=poly_strings, 
                    y_EA=ea_values, 
                    y_IP=ip_values
                ) 

                graphs.append(graph)
        elif dataset == 'diblock':
            for i in tqdm.tqdm(range(len(df.loc[:, 'poly_chemprop_input']))):
                poly_strings = df.loc[i, 'poly_chemprop_input']
                lamellar_values = df.loc[i, 'lamellar']
                cylinder_values = df.loc[i, 'cylinder']
                sphere_values = df.loc[i, 'sphere']
                gyroid_values = df.loc[i, 'gyroid']
                disordered_values = df.loc[i, 'disordered']
                # given the input polymer string, this function returns a pyg data object
                graph = poly_smiles_to_graph(
                    poly_strings=poly_strings, 
                    y_lamellar=lamellar_values, 
                    y_cylinder=cylinder_values,
                    y_sphere=sphere_values, 
                    y_gyroid=gyroid_values,
                    y_disordered=disordered_values
                ) 
                graphs.append(graph)
        else:
            raise ValueError('Invalid dataset name')
            
        torch.save(graphs, file_graphs_list)
        print('Graphs pt file saved')
    else:
        print('Loading graphs pt file...')
        graphs = torch.load(file_graphs_list)

    random.seed(12345)
    graphs = random.sample(graphs, len(graphs))
    return graphs


def create_data(cfg):
    pre_transform = PositionalEncodingTransform(rw_dim=cfg.pos_enc.rw_dim)
    
    transform = GraphJEPAPartitionTransform(
        subgraphing_type=cfg.subgraphing.type,
        num_targets=cfg.jepa.num_targets,
        n_patches=cfg.subgraphing.n_patches,
        patch_num_diff=cfg.pos_enc.patch_num_diff
    )
    
    graphs = get_graphs(file_csv='Data/aldeghi_coley_ea_ip_dataset.csv', file_graphs_list='Data/aldeghi_graphs_list.pt', dataset='aldeghi')
    dataset = MyDataset(root='Data/aldeghi', data_list=graphs, pre_transform=pre_transform)

    return dataset, transform

# TODO create a function to create the pt files even for the diblock dataset
# if cfg.finetuneDataset == 'aldeghi':
    #     data_path = 'Data/aldeghi_coley_ea_ip_dataset.csv'
    #     graphs_list = 'Data/aldeghi_graphs_list.pt'
        
    # elif cfg.finetuneDataset == 'diblock':
    #     data_path = 'Data/diblock_copolymer_dataset.csv'
    #     graphs_list = 'Data/diblock_graphs_list.pt'
    # else:
    #     raise ValueError('Invalid dataset')