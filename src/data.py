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
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            self.data_list = [self.pre_transform(data) for data in self.data_list]
        # self.save(self.data_list, self.processed_paths[0])
        # For PyG<2.4:
        torch.save(self.collate(self.data_list), self.processed_paths[0])


def get_graphs(file_csv='Data/aldeghi_coley_ea_ip_dataset.csv', file_graphs_list='Data/train_aldeghi_graphs_list.pt', dataset='aldeghi'):
    train_graphs = []
    test_graphs = []
    
    # 35 monomer B excluded from training set, its a bit more than 2000 graphs
    # test_monomers = {
    #     '[*:3]c1c(O)cc(O)c([*:4])c1O', 
    #     '[*:3]c1cc(C(F)(F)F)nc([*:4])c1N', 
    #     '[*:3]c1cc([*:4])cc(OC)c1', 
    #     '[*:3]c1cc([*:4])cc(C(=O)Cl)c1',  
    #     '[*:3]c1cc([*:4])c2ccc3cc(C(C)(C)C)cc4ccc1c2c43', 
    #     '[*:3]c1cccc([*:4])c1[N+](=O)[O-]', 
    #     '[*:3]c1[nH]nc2c([*:4])cncc12', 
    #     '[*:3]c1cc([*:4])c(N)c(OC(F)(F)F)c1', 
    #     '[*:3]c1cc([*:4])c(Cl)c(C#N)c1N', 
    #     '[*:3]c1cc([*:4])c(F)cc1F', 
    #     '[*:3]c1cc([*:4])c(N)cc1OC', 
    #     '[*:3]c1cc(C=CC(=O)O)cc([*:4])c1OCC', 
    #     '[*:3]c1c(Cl)cc([*:4])c(C)c1Cl', 
    #     '[*:3]c1cc([*:4])c(OC(C)C)c(C=O)c1', 
    #     '[*:3]c1ccc2c(c1)c1cc([*:4])ccc1n2CC1CO1', 
    #     '[*:3]c1c(C)c([N+](=O)[O-])cc([*:4])c1OC', 
    #     '[*:3]c1nc([*:4])cnc1C(=O)OC', 
    #     '[*:3]c1cc(CN)cc([*:4])c1OC', 
    #     '[*:3]c1cc([*:4])cc(C)c1N', 
    #     '[*:3]c1nc([*:4])nn1COC', 
    #     '[*:3]c1cc(F)c(C)c([*:4])c1N',
    #     '[*:3]c1ccc2c(c1)C(=O)c1cc([*:4])ccc1-2',
    #     '[*:3]c1cc(C=O)cc([*:4])c1OC',
    #     '[*:3]c1cc(C(=O)O)c([*:4])cc1C(=O)O',
    #     '[*:3]c1sc([*:4])nc1C',
    #     '[*:3]c1c(N)c(Cl)cc([*:4])c1OC',
    #     '[*:3]c1cc(C(=O)NN)cc([*:4])c1O'
    #     '[*:3]c1cc(C=O)c([*:4])cn1',
    #     '[*:3]c1cn(C)c([*:4])n1',
    #     '[*:3]c1cc([*:4])c(Cl)[nH]c1=O',
    #     '[*:3]c1c(C)ncc([*:4])c1C',
    #     '[*:3]c1cc([*:4])cc([N+](=O)[O-])c1C',
    #     '[*:3]c1cc(C)cc([*:4])c1NCC(=O)NN'
    #     '[*:3]c1ccc2c(c1)C1(CCOCC1)c1cc([*:4])ccc1-2',
    #     '[*:3]c1cc([*:4])c2nc(-c3ccccc3O)ccc2c1',
    #     '[*:3]c1cc([*:4])cnc1CC',
    #     '[*:3]c1cc(N)cc([*:4])c1C' 
    # }
    
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
                    isAldeghiDataset=True,
                    y_EA=ea_values, 
                    y_IP=ip_values
                ) 
                polymer_monomers = set(poly_strings.split('|')[0].split('.'))
                
                # if polymer_monomers.isdisjoint(test_monomers):
                train_graphs.append(graph)
                # else:
                #     test_graphs.append(graph)
        
            print('Number of training graphs:', len(train_graphs))
            print('Number of test graphs:', len(test_graphs))

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
                    isAldeghiDataset=False,
                    y_lamellar=lamellar_values, 
                    y_cylinder=cylinder_values,
                    y_sphere=sphere_values, 
                    y_gyroid=gyroid_values,
                    y_disordered=disordered_values
                ) 
                train_graphs.append(graph)
        else:
            raise ValueError('Invalid dataset name')
        
        random.seed(12345)
        train_graphs = random.sample(train_graphs, len(train_graphs))
        train_graphs = train_graphs[:int(0.8*len(train_graphs))]
        test_graphs = train_graphs[int(0.8*len(train_graphs)):]
        if dataset == 'aldeghi':
            train_file_graphs_list = 'Data/train_aldeghi_graphs_list.pt'
            test_file_graphs_list = 'Data/test_aldeghi_graphs_list.pt'
            torch.save(train_graphs, train_file_graphs_list)
            torch.save(test_graphs, test_file_graphs_list)
            print('Graphs pt file saved')
        else:
            torch.save(train_graphs, file_graphs_list)
            print('Graphs pt file saved')
    else:
        print('Loading graphs pt file...')
        if dataset == 'aldeghi':
            train_graphs = torch.load('Data/train_aldeghi_graphs_list.pt')
            test_graphs = torch.load('Data/test_aldeghi_graphs_list.pt')
        else:
            train_graphs = torch.load(file_graphs_list)

    # monomer split, uncomment to shuffle the graphs
    # train_graphs = random.sample(train_graphs, len(train_graphs))
    # test_graphs = random.sample(test_graphs, len(test_graphs))
    return train_graphs, test_graphs


def create_data(cfg):
    pre_transform = PositionalEncodingTransform(rw_dim=cfg.pos_enc.rw_dim)
    
    transform_train = GraphJEPAPartitionTransform(
        subgraphing_type=cfg.subgraphing.type,
        num_targets=cfg.jepa.num_targets,
        n_patches=cfg.subgraphing.n_patches,
        patch_rw_dim=cfg.pos_enc.patch_rw_dim,
        patch_num_diff=cfg.pos_enc.patch_num_diff,
        drop_rate=cfg.subgraphing.drop_rate,
        context_size=cfg.subgraphing.context_size
    )

    transform_val = GraphJEPAPartitionTransform(
        subgraphing_type=cfg.subgraphing.type,
        num_targets=cfg.jepa.num_targets,
        n_patches=cfg.subgraphing.n_patches,
        patch_rw_dim=cfg.pos_enc.patch_rw_dim,
        patch_num_diff=cfg.pos_enc.patch_num_diff,
        drop_rate=0.0,
        context_size=cfg.subgraphing.context_size
    )
    
    
    train_graphs, test_graphs = get_graphs(file_csv='Data/aldeghi_coley_ea_ip_dataset.csv', file_graphs_list='Data/train_aldeghi_graphs_list.pt', dataset='aldeghi')
    pretrn_graphs = train_graphs[:int(0.5*len(train_graphs))]
    ft_graphs = train_graphs[int(0.5*len(train_graphs)):]

    pretrn_dataset = MyDataset(root='Data/aldeghi/pretrain', data_list=pretrn_graphs, pre_transform=pre_transform, transform=transform_train)
    ft_dataset = MyDataset(root='Data/aldeghi/finetune', data_list=ft_graphs, pre_transform=pre_transform, transform=transform_val)
    val_dataset = MyDataset(root='Data/aldeghi/val', data_list=test_graphs, pre_transform=pre_transform, transform=transform_val)
    val_dataset = [x for x in val_dataset] # keep same transform subgraphs throughout epochs

    return pretrn_dataset, ft_dataset, val_dataset



def printStats(graphs):
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