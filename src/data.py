import collections
import os
import pandas as pd
import random
from rdkit import Chem
from src.featurization_utils.featurization import poly_smiles_to_graph
from src.transform import PositionalEncodingTransform, GraphJEPAPartitionTransform
import time
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import ZINC
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


def get_graphs(dataset='aldeghi'):
    all_graphs = []
    # full_atoms_set = set()
    
    file_csv='Data/aldeghi_coley_ea_ip_dataset.csv' if dataset == 'aldeghi' else 'Data/diblock_copolymer_dataset.csv'
    file_graphs_list='Data/aldeghi_graphs_list.pt' if dataset == 'aldeghi' else 'Data/diblock_graphs_list.pt'

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
                # polymer_monomers = set(poly_strings.split('|')[0].split('.'))

                # m = Chem.MolFromSmiles(graph.smiles['polymer'])
                # full_atoms_set.update(set([atom.GetAtomicNum() for atom in m.GetAtoms()]))
                # if polymer_monomers.isdisjoint(test_monomers):
                all_graphs.append(graph)
                # else:
                #     test_graphs.append(graph)
            
            # with open('full_atoms_list.txt', 'w') as f:
            #     f.write(','.join([str(a) for a in full_atoms_set]))
            # f.close()

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
                all_graphs.append(graph)
        else:
            raise ValueError('Invalid dataset name')
        
        random.seed(12345)
        all_graphs = random.sample(all_graphs, len(all_graphs))
        torch.save(all_graphs, file_graphs_list)
        print('Graphs .pt file saved')
    else:
        print('Loading graphs pt file...')
        all_graphs = torch.load(file_graphs_list)

    return all_graphs


def create_data(cfg):
    pre_transform = PositionalEncodingTransform(rw_dim=cfg.pos_enc.rw_dim)

    transform_train = GraphJEPAPartitionTransform(
        subgraphing_type=cfg.subgraphing.type,
        num_targets=cfg.jepa.num_targets,
        n_patches=cfg.subgraphing.n_patches,
        patch_rw_dim=cfg.pos_enc.patch_rw_dim,
        patch_num_diff=cfg.pos_enc.patch_num_diff,
        drop_rate=cfg.subgraphing.drop_rate,
        context_size=cfg.subgraphing.context_size,
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
        dataset=cfg.finetuneDataset
    )
    
    if cfg.finetuneDataset == 'aldeghi' or cfg.finetuneDataset == 'diblock':
        all_graphs = []
        if not os.path.isfile('Data/aldeghi/processed/dataset.pt'): # avoid loading the graphs, if dataset already exists
            all_graphs = get_graphs(dataset=cfg.finetuneDataset)
        
        dataset = MyDataset(root='Data/aldeghi', data_list=all_graphs, pre_transform=pre_transform)

        # return full dataset and transforms, split in pretrain/finetune, train/test is done in the training script with k fold
        return dataset, transform_train, transform_val
    
    elif cfg.finetuneDataset == 'zinc':
        smiles_dict = {}
        if not os.path.isfile('Data/Zinc/subset/processed/train.pt'):
            sets = ['train', 'val', 'test']
            for set_name in sets:
                smiles_df = pd.read_csv(f'Data/Zinc/{set_name}.txt', header=None)
                print(smiles_df.head())

                with open(f'Data/Zinc/{set_name}.index', 'r') as file:
                    # Read all lines, split by comma and newline, then flatten the list
                    indexes = [int(index) for line in file for index in line.split(',')]

                smiles_list = smiles_df.iloc[indexes, 0].tolist()

                smiles_dict[set_name] = smiles_list
                
        root = 'Data/Zinc'
        train_dataset = ZINC(
            root, subset=True, split='train', pre_transform=pre_transform, transform=transform_train, smiles=smiles_dict)
        
        val_dataset = ZINC(root, subset=True, split='val',
                           pre_transform=pre_transform, transform=transform_val, smiles=smiles_dict)
        
        pretrn_dataset = train_dataset.copy() # [:int(0.5*len(train_dataset))]
        ft_dataset = train_dataset.copy() # [int(0.5*len(train_dataset)):]
    else:
        raise ValueError('Invalid dataset name')
    

    # pretrn_set = [x.full_input_string for x in pretrn_dataset]
    # print(f'Pretraining dataset size: {len(pretrn_dataset)}')
    # pretrn_set = set(pretrn_set)
    # print(f'Pretraining dataset size: {len(pretrn_dataset)}')

    # ft_set = [x.full_input_string for x in ft_dataset]
    # print(f'Finetuning dataset size: {len(ft_dataset)}')
    # ft_set = set(ft_set)
    # print(f'Finetuning dataset size: {len(ft_dataset)}')

    # val_set = set([x.full_input_string for x in val_dataset])
    # print(f'Validation dataset size: {len(val_dataset)}')
    # val_set = set(val_set)
    # print(f'Validation dataset size: {len(val_dataset)}')

    # # check for overlap between datasets
    # print(f'Overlap between pretraining and finetuning datasets: {len(pretrn_set.intersection(ft_set))}')
    # print(f'Overlap between pretraining and validation datasets: {len(pretrn_set.intersection(val_set))}')
    # print(f'Overlap between finetuning and validation datasets: {len(ft_set.intersection(val_set))}')


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


def getFullAtomsList():
    with open('full_atoms_list.txt', 'r') as f:
        full_atoms_list = f.read()
    f.close()

    full_atoms_list = set(full_atoms_list.split(','))

    return full_atoms_list


def getMaximizedVariedData(ft_data, size):
    torch.manual_seed(int(time.time()))
    # shuffle the dataset randomly
    ft_data.shuffle()
    current_size = 0
    i = 0
    # keep track of how many occurrences of each monomerA, if for a monomerA we have 1/9 of the size, we can stop adding that monomerA
    monomerADict = collections.defaultdict(int)
    # stoichiometry and chain architecture dict (same as monomer A), but ratios here are 1/3
    stoichDict = collections.defaultdict(int)
    chainArchDict = collections.defaultdict(int)
    # just make sure that the monomerB is not repeating, so its not in the set
    monomerBSet = set()
    # TODO need to make sure that all possible atoms are included in the dataset, so with monomerB we should keep track of the atoms seen so far, and those that are missing
    # from featurization i can return a list of atoms that are present in the monomerB/polymer
    dataset = []

    full_atoms_list = getFullAtomsList()
    subset_atoms = set()
    while i < len(ft_data) and current_size < size:
        m = Chem.MolFromSmiles(ft_data[i].smiles['polymer'])
        m_atoms = set([atom.GetAtomicNum() for atom in m.GetAtoms()])
        monomerA = ft_data[i].smiles['monomer1']
        monomerB = ft_data[i].smiles['monomer2']
        stoich = ft_data[i].stoichiometry 
        #chainArch = ft_data[i].chain_architecture # TODO: add chain architecture to the data object in featurization.py

        # atoms conditions: if subset_atoms < full_atoms_list and subset_atoms.intersections(m_atoms) == m_atoms then skip current iteration
        # translated means that if we are still missing some atoms in the subset, and the current molecule doesnt bring any new atoms skip it, 
        # in case we already seen all atoms or the moleucle bring some new atoms we add it to the subset

        # chainArchDict[chainArch] >= size // 3 or \
        if \
        monomerADict[monomerA] >= size // 9 or \
        monomerB in monomerBSet or \
        stoichDict[stoich] >= size // 3 or \
        (len(subset_atoms) < len(full_atoms_list) and len(m_atoms) == len(m_atoms.intersection(subset_atoms))):
            i += 1
            continue

        monomerADict[monomerA] += 1
        monomerBSet.add(monomerB)
        subset_atoms.update(m_atoms)
        stoichDict[stoich] += 1
        #chainArchDict[chainArch] += 1
        current_size += 1
        dataset.append(ft_data[i])
        i += 1

    if current_size < size:
        print('Not enough data to reach the desired size')

    # print dataset stats
    # print("\Maximized variation dataset stats:\n")
    # print("Mon A dict:", monomerADict)
    # print("Mon B set length", len(monomerBSet), '; current size:', current_size)
    # print("Stoich dict:", stoichDict)
    # print("Subset atoms length:", len(subset_atoms), "; Full atoms length:", len(full_atoms_list))

    return dataset


# include the least possible number of different monomerA and monomerB while making sure that all possible atoms are present in the dataset
def getLabData(ft_data, size):
    torch.manual_seed(int(time.time()))
    ft_data.shuffle()
    current_size = 0
    i = 0
     # keep track of how many occurrences of each monomerA, if for a monomerA we have 1/9 of the size, we can stop adding that monomerA
    monomerADict = collections.defaultdict(int)
    # stoichiometry and chain architecture dict (same as monomer A), but ratios here are 1/3
    stoichDict = collections.defaultdict(int)
    chainArchDict = collections.defaultdict(int)
    # just make sure that the monomerB is not repeating, so its not in the set
    monomerBDict = collections.defaultdict(int)
   
    dataset = []
    full_atoms_list = getFullAtomsList()
    subset_atoms = set()

    # keep track of how many keys in monomerADict and monomerBSet, 
    while i < len(ft_data) and current_size < size:
        m = Chem.MolFromSmiles(ft_data[i].smiles['polymer'])
        m_atoms = set([atom.GetAtomicNum() for atom in m.GetAtoms()])
        monomerA = ft_data[i].smiles['monomer1']
        monomerB = ft_data[i].smiles['monomer2']
        stoich = ft_data[i].stoichiometry

        # chainArchDict[chainArch] >= size // 3 or \
        if \
        (len(monomerADict) >= 4 and monomerA not in monomerADict) or \
        (len(monomerBDict) >= 20 and monomerB not in monomerBDict) or \
        stoichDict[stoich] >= size // 3 or \
        (len(subset_atoms) < len(full_atoms_list) and len(m_atoms) == len(m_atoms.intersection(subset_atoms))):
            i += 1
            continue

        monomerADict[monomerA] += 1
        monomerBDict[monomerB] += 1
        stoichDict[stoich] += 1
        subset_atoms.update(m_atoms)
        current_size += 1
        dataset.append(ft_data[i])
        i += 1
    
    # print("\Lab dataset stats:\n")
    # print("Size:", current_size)
    # print("Mon A dict:", monomerADict)
    # print("Mon B dict", len(monomerBDict))
    # print("Stoich dict:", stoichDict)
    # print("Subset atoms length:", len(subset_atoms), "; Full atoms length:", len(full_atoms_list))

    return dataset


def getRandomData(ft_data, size, seed=None):
    # select 'size' number of random data points from ft_data
    # randomly set torch seed based on the current time
    # torch.manual_seed(int(time.time()))
    # set random seed for python
    random.seed(seed)
    dataset = ft_data #.shuffle()
    if not isinstance(dataset, list):
        dataset = [x for x in dataset]

    dataset = random.sample(dataset, size)
    # # print dataset stats as in getMaximizedVariedData
    # monomerADict = collections.defaultdict(int)
    # # stoichiometry and chain architecture dict (same as monomer A), but ratios here are 1/3
    # stoichDict = collections.defaultdict(int)
    # chainArchDict = collections.defaultdict(int)
    # # just make sure that the monomerB is not repeating, so its not in the set
    # monomerBDict = collections.defaultdict(int)
    # full_atoms_list = getFullAtomsList()
    # subset_atoms = set()

    # for i in range(len(dataset)):
    #     m = Chem.MolFromSmiles(dataset[i].smiles['polymer'])
    #     m_atoms = set([atom.GetAtomicNum() for atom in m.GetAtoms()])
    #     monomerA = dataset[i].smiles['monomer1']
    #     monomerB = dataset[i].smiles['monomer2']
    #     stoich = dataset[i].stoichiometry

    #     monomerADict[monomerA] += 1
    #     monomerBDict[monomerB] += 1
    #     stoichDict[stoich] += 1
    #     subset_atoms.update(m_atoms)
    
    # print("\nRandom dataset stats:\n")
    # print("Size:", size)
    # print("Mon A dict:", monomerADict)
    # print("Mon B dict", len(monomerBDict))
    # print("Stoich dict:", stoichDict)
    # print("Subset atoms length:", len(subset_atoms), "; Full atoms length:", len(full_atoms_list))

    return dataset

def getTammoData(full_dataset):
    with open('polymers_used.txt', 'r') as f:
        polymer_smiles = f.read()
    f.close()

    polymer_smiles = set(polymer_smiles.split(','))
    print('Number of polymers in polymers_used.txt:', len(polymer_smiles))
    dataset = []
    for g in full_dataset:
        # in case we already have all the polymers we need, stop
        if len(dataset) == len(polymer_smiles):
            break

        if g.full_input_string in polymer_smiles:
            dataset.append(g)

    monomerADict = collections.defaultdict(int)
    # stoichiometry and chain architecture dict (same as monomer A), but ratios here are 1/3
    stoichDict = collections.defaultdict(int)
    chainArchDict = collections.defaultdict(int)
    # just make sure that the monomerB is not repeating, so its not in the set
    monomerBDict = collections.defaultdict(int)
    full_atoms_list = getFullAtomsList()
    subset_atoms = set()

    for i in range(len(dataset)):
        m = Chem.MolFromSmiles(dataset[i].smiles['polymer'])
        m_atoms = set([atom.GetAtomicNum() for atom in m.GetAtoms()])
        monomerA = dataset[i].smiles['monomer1']
        monomerB = dataset[i].smiles['monomer2']
        stoich = dataset[i].stoichiometry

        monomerADict[monomerA] += 1
        monomerBDict[monomerB] += 1
        stoichDict[stoich] += 1
        subset_atoms.update(m_atoms)
    
    print("\nRandom dataset stats:\n")
    print("Size:", len(dataset))
    print("Mon A dict:", monomerADict)
    print("Mon B dict", len(monomerBDict))
    print("Stoich dict:", stoichDict)
    print("Subset atoms length:", len(subset_atoms), "; Full atoms length:", len(full_atoms_list))
    print("Subset atoms:", subset_atoms, "; Full atoms:", full_atoms_list)
    
    return dataset



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