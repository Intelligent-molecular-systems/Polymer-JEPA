import numpy as np
import re
import random
from scipy import sparse as sp
from src.subgraphing_utils.subgraphs_extractor import  motifs2subgraphs, metis2subgraphs, randomWalks2subgraphs
import torch
import torch_geometric
from torch_geometric.data import Data

def cal_coarsen_adj(subgraphs_nodes_mask):
    #a coarse patch adjacency matrix A′ = B*B^T ∈ Rp×p, where each A′ ij contains the node overlap between pi and pj.
    mask = subgraphs_nodes_mask.to(torch.float)
    coarsen_adj = torch.matmul(mask, mask.t()) # element at position (i, j) is the number of nodes that subgraphs i and j have in common.
    return coarsen_adj # create a simplified version of a graph, the purpose of this process is to reduce the complexity of the graph, making it easier to analyze or compute on.

def to_sparse(node_mask, edge_mask):
    subgraphs_nodes = node_mask.nonzero().T
    subgraphs_edges = edge_mask.nonzero().T
    return subgraphs_nodes, subgraphs_edges


def combine_subgraphs(edge_index, subgraphs_nodes, subgraphs_edges, num_selected=None, num_nodes=None):
    if num_selected is None:
        num_selected = subgraphs_nodes[0][-1] + 1

    if num_nodes is None:
        num_nodes = subgraphs_nodes[1].max() + 1

    combined_subgraphs = edge_index[:, subgraphs_edges[1]] 
    node_label_mapper = edge_index.new_full((num_selected, num_nodes), fill_value=-1) 


    node_label_mapper[subgraphs_nodes[0], subgraphs_nodes[1]] = torch.arange(len(subgraphs_nodes[1])) 
    node_label_mapper = node_label_mapper.reshape(-1) 

    inc = torch.arange(num_selected)*num_nodes 
    combined_subgraphs += inc[subgraphs_edges[0]]
    combined_subgraphs = node_label_mapper[combined_subgraphs] 
    return combined_subgraphs


def random_walk(A, n_iter):
    # Geometric diffusion features with Random Walk
    Dinv = A.sum(dim=-1).clamp(min=1).pow(-1).unsqueeze(-1)  # D^-1
    RW = A * Dinv
    M = RW
    M_power = M
    # Iterate
    PE = [torch.diagonal(M)]
    for _ in range(n_iter-1):
        M_power = torch.matmul(M_power, M)
        PE.append(torch.diagonal(M_power))
    PE = torch.stack(PE, dim=-1)
    return PE


def RWSE(edge_index, pos_enc_dim, num_nodes):
    """
        Initializing positional encoding with RWSE
    """
    if edge_index.size(-1) == 0:
        PE = torch.zeros(num_nodes, pos_enc_dim)
    else:
        A = torch_geometric.utils.to_dense_adj(
            edge_index, max_num_nodes=num_nodes)[0]
        PE = random_walk(A, pos_enc_dim)
    return PE





class SubgraphsData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        num_nodes = self.num_nodes
        num_edges = self.edge_index.size(-1)
        if bool(re.search('(combined_subgraphs)', key)):
            return getattr(self, key[:-len('combined_subgraphs')]+'subgraphs_nodes_mapper').size(0)
        elif bool(re.search('(subgraphs_batch)', key)):
            return 1+getattr(self, key)[-1]
        elif bool(re.search('(nodes_mapper)', key)):
            return num_nodes
        elif bool(re.search('(edges_mapper)', key)):
            return num_edges
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if bool(re.search('(combined_subgraphs)', key)):
            return -1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


class PositionalEncodingTransform(object):
    def __init__(self, rw_dim=0):
        super().__init__()
        self.rw_dim = rw_dim

    def __call__(self, data):
        if self.rw_dim > 0:
            data.rw_pos_enc = RWSE(
                data.edge_index, self.rw_dim, data.num_nodes)
        return data

# !!! 
class GraphJEPAPartitionTransform(object):
    def __init__(
            self, 
            subgraphing_type=0,
            num_targets=4,
            n_patches=20,
            patch_num_diff=0
        ):
        super().__init__()
        self.subgraphing_type = subgraphing_type
        # [TODO]: How to handle cases where num_targets is less than the set one?
        self.num_targets = num_targets
        self.n_patches = n_patches # [RISK]: I dont have a n_patches attribute, no fixed n of subgraphs for all graphs, i dont know if this flexibility is allowed
        self.patch_num_diff = patch_num_diff
        
    def _diffuse(self, A):
            # !!! since patch_num_diff is always 0, by default we are using always the regular adjacency matrix !!!
            if self.patch_num_diff == 0:
                return A
            Dinv = A.sum(dim=-1).clamp(min=1).pow(-1).unsqueeze(-1)  # D^-1
            RW = A * Dinv
            M = RW
            M_power = M
            # Iterate
            for _ in range(self.patch_num_diff-1):
                M_power = torch.matmul(M_power, M)
            return M_power
  
    def __call__(self, data):
        data = SubgraphsData(**{k: v for k, v in data})
        if self.subgraphing_type == 0:
            node_masks, edge_masks = motifs2subgraphs(data, self.n_patches, self.num_targets)
        elif self.subgraphing_type == 1:
            node_masks, edge_masks = metis2subgraphs(data, self.n_patches, self.num_targets)
        elif self.subgraphing_type == 2:
            node_masks, edge_masks = randomWalks2subgraphs(data, self.n_patches, self.num_targets)
        else:
            raise ValueError('Invalid subgraphing type')
                     
        subgraphs_nodes, subgraphs_edges = to_sparse(node_masks, edge_masks) 
        
        # [RISK]: I dont have a n_patches attribute, no fixed n of subgraphs for all graphs, i dont know if this flexibility is allowed
        # basically i substitute n_patches with len(node_masks)
        # [CHECKPOINT]: I plotted the combined_subgraphs and it looks good, it means that subgraphs_nodes and subgraphs_edges are good
        combined_subgraphs = combine_subgraphs(
            data.edge_index, 
            subgraphs_nodes, 
            subgraphs_edges,
            num_selected=self.n_patches,
            num_nodes=data.num_nodes
        )

        # this is basically an alternative way to compute the positional encoding based on the RWSE of the patches, i
        # see 4.4. Ablation Studies in the paper
        if self.patch_num_diff > -1:
            coarsen_adj = cal_coarsen_adj(node_masks)
            if self.patch_num_diff > -1: # patch_num_diff = Patch PE diffusion steps
                data.coarsen_adj = self._diffuse(coarsen_adj).unsqueeze(0)

        
        subgraphs_batch = subgraphs_nodes[0] # this is the batch of subgraphs, i.e. the subgraph idxs [0, 0, 1, 1]
        # print('subgraph_batch, ', subgraphs_batch)
        
        mask = torch.zeros(self.n_patches).bool() # if say we have two patches then [False, False]
        mask[subgraphs_batch] = True # if subgraphs_batch = [0, 0, 1, 1] then [True, True]
        # print('mask, ', mask)
        
        # basically mask has the first 20-n elements as False and the remaining n elements as True, n is the number of subgraphs in the graph
        # print(mask) # [RISK]: Check if this is the same in the original code 
        data.subgraphs_batch = subgraphs_batch
        
        data.subgraphs_nodes_mapper = subgraphs_nodes[1] # this is the node idxs [0, 2, 1, 3] (original node idxs)
        data.subgraphs_edges_mapper = subgraphs_edges[1] # this is the edge idxs [0, 1, 2] (original edge idxs)
        data.combined_subgraphs = combined_subgraphs # this is the edge index of th combined subgraph made of disconnected subgraphs, where each subgraph has its own unique node ids
        data.mask = mask.unsqueeze(0) # [True, True] -> [[True, True]]

        # context_subgraph_idx = torch.tensor(0) # use the first subgraph idx as context
        # take the patches in subgraphs_nodes[0].unique()[1:] and shuffle them
        subgraphs = subgraphs_nodes[0].unique()
        context_subgraph_idx = subgraphs[0]
        # print('context_subgraph_idx, ', context_subgraph_idx)
        rand_choice = np.random.choice(subgraphs[1:], self.num_targets, replace=False)
        # target_subgraph_idxs = subgraphs[1:] # torch.tensor(rand_choice) RISK Variable number of targets per graph, if it works i shuld at least normalize the error based on the number of targets
        target_subgraph_idxs = torch.tensor(rand_choice)
        # print('target_subgraph_idxs, ', target_subgraph_idxs)
        data.context_edges_mask = subgraphs_edges[0] == context_subgraph_idx # if context subgraph idx is 0, and subgraphs_edges[0] = [0, 0, 1, 1, 2] then [True, True, False, False, False]
        data.target_edges_mask = torch.isin(subgraphs_edges[0], target_subgraph_idxs) # if target subgraph idxs are [1, 2] then [False, False, True, True, True]
        data.context_nodes_mapper = subgraphs_nodes[1, subgraphs_nodes[0] == context_subgraph_idx] # if context subgraph idx is 0, and subgraphs_nodes[0] (subgraphs indexes) =[0 ,0, 1, 1, 2] subgraphs_nodes[1] (node indexes) = [0, 2, 1, 2, 3] then [0, 2]
        data.target_nodes_mapper = subgraphs_nodes[1, torch.isin(subgraphs_nodes[0], target_subgraph_idxs)] # if target subgraph idxs are [1, 2] then [1, 2, 3]
        data.context_nodes_subgraph = subgraphs_nodes[0, subgraphs_nodes[0] == context_subgraph_idx] # if context subgraph idx is 0, and subgraphs_nodes[0] (subgraphs indexes) =[0 ,0, 1, 1, 2] then [0, 0]
        data.target_nodes_subgraph = subgraphs_nodes[0, torch.isin(subgraphs_nodes[0], target_subgraph_idxs)] # if target subgraph idxs are [1, 2] then [1, 1, 2]
        data.context_subgraph_idx = context_subgraph_idx.tolist() # if context subgraph idx is 0, then[0]
        data.target_subgraph_idxs = target_subgraph_idxs.tolist() # if target subgraph idxs are [1, 2] then [1, 2]
        data.call_n_patches = [self.n_patches] # [RISK] 
        # print('data.call_n_patches, ', data.call_n_patches)

        data.__num_nodes__ = data.num_nodes  # set number of nodes of the current graph
        return data