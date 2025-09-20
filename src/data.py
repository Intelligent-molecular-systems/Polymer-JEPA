"""Data loading and preprocessing for Polymer-JEPA.

This module handles:
- Loading polymer datasets (Aldeghi and diblock copolymer)
- Converting SMILES strings to molecular graphs
- Creating PyTorch Geometric datasets with transforms
- Data sampling and statistics

Supported datasets:
- Aldeghi: Conjugated copolymers with EA/IP properties
- Diblock: Diblock copolymers with phase behavior properties
"""

import collections
import os
import random
from typing import List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import tqdm
from rdkit import Chem
from torch_geometric.data import InMemoryDataset

from src.featurization_utils.featurization import poly_smiles_to_graph
from src.transform import PositionalEncodingTransform, GraphJEPAPartitionTransform


class PolymerDataset(InMemoryDataset):
    """Custom PyTorch Geometric dataset for polymer graphs.
    
    This dataset handles in-memory storage of polymer molecular graphs
    with optional pre-processing transforms.
    
    Args:
        root: Root directory for processed data (optional)
        data_list: List of PyTorch Geometric Data objects
        transform: Transform applied on-the-fly during data loading
        pre_transform: Transform applied once during preprocessing
    """
    def __init__(self, root: Optional[str], data_list: List[Any], 
                 transform=None, pre_transform=None):
        self.data_list = data_list
        super().__init__(root or "", transform, pre_transform)
        # self.load(self.processed_paths[0])
        # For PyG<2.4:
        if root is not None:
            self.data, self.slices = torch.load(self.processed_paths[0])
        else: 
            self.data, self.slices = self.collate(data_list)
            

    @property
    def processed_file_names(self) -> str:
        return 'dataset.pt'

    def process(self) -> None:
        """Process the dataset and save to processed directory."""
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            self.data_list = [self.pre_transform(data) for data in self.data_list]
        # self.save(self.data_list, self.processed_paths[0])
        # For PyG<2.4:
        torch.save(self.collate(self.data_list), self.processed_paths[0])


def get_graphs(dataset: str = 'aldeghi') -> Tuple[List[Any], None]:
    """Load or create molecular graphs from polymer datasets.
    
    Args:
        dataset: Dataset name ('aldeghi' or 'diblock')
        
    Returns:
        Tuple of (graph_list, None) where graph_list contains PyG Data objects
        
    Raises:
        ValueError: If dataset name is not supported
    """
    all_graphs = []
    
    file_csv='Data/aldeghi_coley_ea_ip_dataset.csv' if dataset == 'aldeghi' else 'Data/diblock_copolymer_dataset.csv'
    file_graphs_list='Data/aldeghi_graphs_list.pt' if dataset == 'aldeghi' else 'Data/diblock_graphs_list.pt'

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
                    isAldeghiDataset=True,
                    y_EA=ea_values, 
                    y_IP=ip_values
                ) 
            
                all_graphs.append(graph)

        elif dataset == 'diblock':
            for i in tqdm.tqdm(range(len(df.loc[:, 'poly_chemprop_input']))):
                poly_strings = df.loc[i, 'poly_chemprop_input']
                lamellar_values = df.loc[i, 'lamellar']
                cylinder_values = df.loc[i, 'cylinder']
                sphere_values = df.loc[i, 'sphere']
                gyroid_values = df.loc[i, 'gyroid']
                disordered_values = df.loc[i, 'disordered']
                phase1 = df.loc[i, 'phase1']
                # given the input polymer string, this function returns a pyg data object
                graph = poly_smiles_to_graph(
                    poly_strings=poly_strings, 
                    isAldeghiDataset=False,
                    y_lamellar=lamellar_values, 
                    y_cylinder=cylinder_values,
                    y_sphere=sphere_values, 
                    y_gyroid=gyroid_values,
                    y_disordered=disordered_values,
                    phase1=phase1
                ) 
                all_graphs.append(graph)
        else:
            raise ValueError('Invalid dataset name')
    else: 
        all_graphs = torch.load(file_graphs_list)
                
   
    return all_graphs, None
        


def create_data(cfg) -> Tuple[Any, Any, Any]:
    """Create dataset with transforms for training pipeline.
    
    Args:
        cfg: Configuration object containing dataset and transform parameters
        
    Returns:
        Tuple of (dataset, train_transform, val_transform)
        
    Raises:
        ValueError: If dataset name is not supported
    """
    pre_transform = PositionalEncodingTransform(rw_dim=cfg.pos_enc.rw_dim)

    transform_train = GraphJEPAPartitionTransform(
        subgraphing_type=cfg.subgraphing.type,
        num_targets=cfg.jepa.num_targets,
        n_patches=cfg.subgraphing.n_patches,
        patch_rw_dim=cfg.pos_enc.patch_rw_dim,
        patch_num_diff=cfg.pos_enc.patch_num_diff,
        drop_rate=cfg.subgraphing.drop_rate,
        context_size=cfg.subgraphing.context_size,
        target_size=cfg.subgraphing.target_size,
        dataset=cfg.finetuneDataset
    )

    # no dropout for validation
    transform_val = GraphJEPAPartitionTransform(
        subgraphing_type=cfg.subgraphing.type,
        num_targets=cfg.jepa.num_targets,
        n_patches=cfg.subgraphing.n_patches,
        patch_rw_dim=cfg.pos_enc.patch_rw_dim,
        patch_num_diff=cfg.pos_enc.patch_num_diff,
        drop_rate=0.0, 
        context_size=cfg.subgraphing.context_size,
        target_size=cfg.subgraphing.target_size,
        dataset=cfg.finetuneDataset
    )
    
    if cfg.finetuneDataset == 'aldeghi' or cfg.finetuneDataset == 'diblock':
        all_graphs = []
        if cfg.finetuneDataset == 'diblock' and not os.path.isfile('Data/diblock_graphs_list.pt'):
            all_graphs, _ = get_graphs(dataset=cfg.finetuneDataset)
        if not os.path.isfile('Data/aldeghi/processed/dataset.pt'): # avoid loading the graphs, if dataset already exists
            all_graphs, _ = get_graphs(dataset=cfg.finetuneDataset)
        
        dataset = PolymerDataset(root='Data/aldeghi', data_list=all_graphs, pre_transform=pre_transform)
        
        # return full dataset and transforms, split in pretrain/finetune, train/test is done in the training script with k fold
        return dataset, transform_train, transform_val
    else:
        raise ValueError('Invalid dataset name')

def print_dataset_stats(graphs: List[Any]) -> None:
    """Print statistics about the molecular graph dataset.
    
    Args:
        graphs: List of PyTorch Geometric Data objects
    """
    avg_num_nodes = sum([g.num_nodes for g in graphs]) / len(graphs)
    avg_num_edges = sum([g.num_edges for g in graphs]) / len(graphs)
    print(f'Average number of nodes: {avg_num_nodes}')
    print(f'Average number of edges: {avg_num_edges}')

    # print n of graphs with more than 25 nodes
    print(f'Number of graphs with more than 25 nodes: {len([g for g in graphs if g.num_nodes > 25])}')
    print(f'Number of graphs with more than 30 nodes: {len([g for g in graphs if g.num_nodes > 30])}')

    # print max number of nodes
    print(f'Max number of nodes: {max([g.num_nodes for g in graphs])}')
    print(f'min number of nodes: {min([g.num_nodes for g in graphs])}')


def get_random_data(ft_data: Any, size: int, seed: Optional[int] = None) -> List[Any]:
    """Sample random subset of data for training.
    
    Args:
        ft_data: Dataset or list of data points
        size: Number of samples to select
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled data points
    """
    # select 'size' number of random data points from ft_data
    # randomly set torch seed based on the current time
    # torch.manual_seed(int(time.time()))
    # set random seed for python
    
    dataset = ft_data #.shuffle()
    if not isinstance(dataset, list):
        dataset = [x for x in dataset]
    
    random.seed(seed)
    if size != len(ft_data):
        dataset = random.sample(dataset, size)
    else:
        random.shuffle(dataset)

    return dataset

def analyze_diblock_dataset() -> None:
    """Analyze the diblock copolymer dataset structure and composition.
    
    Prints statistics about polymer strings, monomer counts, and displays
    sample molecular structures using RDKit.
    """
    csv_file = 'Data/diblock_copolymer_dataset.csv'
    df = pd.read_csv(csv_file)
    
    # check how many differen entries for the 'poly_chemprop_input' column
    poly_strings = df.loc[:, 'poly_chemprop_input']
    for i in range(len(poly_strings)):
        poly_strings[i] = poly_strings[i].split('|')[0]
    poly_set = set()
    for p in poly_strings:
        poly_set.add(p)
    print(poly_set)
    print('Number of different polymer strings:', len(poly_set))

    # for each polymer string, check how many different monomers are present (sepeartead by a dot)
    n_of_monomers = []
    for p in poly_set:
        n_of_monomers.append(len(p.split('.')))
    print('Number of monomers:', n_of_monomers)
    # count how many times each number of monomers appears
    monomer_dict = collections.defaultdict(int)
    for n in n_of_monomers:
        monomer_dict[n] += 1
    print('Monomer dict:', monomer_dict)

    # use rdkit to plot randomly 9 polymer strings
    for i in range(9):
        m = Chem.MolFromSmiles(poly_strings[i])
        Chem.Draw.MolToImage(m).show()
    

