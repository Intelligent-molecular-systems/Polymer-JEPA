import numpy as np
import re
from src.subgraphing_utils.context_subgraph_extractor import rwContext, motifContext, metis2subgraphs
from src.subgraphing_utils.target_subgraph_extractor import rwTargets, motifTargets

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
        # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html
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


class GraphJEPAPartitionTransform(object):
    def __init__(
            self, 
            subgraphing_type=0,
            num_targets=4,
            n_patches=20,
            patch_rw_dim=0,
            patch_num_diff=0,
            drop_rate=0,
            context_size=0.7,
            target_size=0.15,
            dataset='aldeghi'
        ):

        super().__init__()
        self.subgraphing_type = subgraphing_type
        self.num_targets = num_targets
        self.n_patches = n_patches
        self.patch_rw_dim = patch_rw_dim
        self.patch_num_diff = patch_num_diff
        self.drop_rate = drop_rate
        self.context_size = context_size
        self.target_size = target_size
        self.dataset = dataset
        
    def _diffuse(self, A):
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

        if self.dataset == 'aldeghi' or self.dataset == 'diblock':
            # find the context using one of the 3 options
            if self.subgraphing_type == 0: # motif
                context_node_masks, context_edge_masks, context_subgraphs_used = motifContext(data, sizeContext=self.context_size, n_targets=self.num_targets)
                node_masks, edge_masks = motifTargets(data, n_targets=self.num_targets, n_patches=self.n_patches-1, cliques_used=context_subgraphs_used)
                node_masks = torch.cat([context_node_masks, node_masks], dim=0)
                edge_masks = torch.cat([context_edge_masks, edge_masks], dim=0)

            elif self.subgraphing_type == 1: # metis
                # context_node_masks, context_edge_masks = metisContext(data, sizeContext=self.context_size)
                # node_masks, edge_masks = metisTargets(data, n_patches=self.n_patches-1, drop_rate=self.drop_rate, num_hops=1, is_directed=False)
                node_masks, edge_masks, context_subgraphs_used = metis2subgraphs(data, sizeContext=self.context_size, n_patches=self.n_patches, min_targets=self.num_targets)
            
            elif self.subgraphing_type == 2: # random walk
                context_node_masks, context_edge_masks, rw1, rw2 = rwContext(data, sizeContext=self.context_size)
                context_subgraphs_used = [rw1, rw2]
                node_masks, edge_masks = rwTargets(data, n_patches=self.n_patches-1, n_targets=self.num_targets, rw1=rw1, rw2=rw2, target_size=self.target_size)
                node_masks = torch.cat([context_node_masks, node_masks], dim=0)
                edge_masks = torch.cat([context_edge_masks, edge_masks], dim=0)
            else:
                raise ValueError('Invalid subgraphing type')     
            
   
            
        subgraphs_nodes, subgraphs_edges = to_sparse(node_masks, edge_masks) 
        
        combined_subgraphs = combine_subgraphs(
            data.edge_index, 
            subgraphs_nodes, 
            subgraphs_edges,
            num_selected=self.n_patches,
            num_nodes=data.num_nodes
        )

        if self.patch_num_diff > -1 or self.patch_rw_dim > 0:
            coarsen_adj = cal_coarsen_adj(node_masks)
            if self.patch_rw_dim > 0:
                data.patch_pe = random_walk(coarsen_adj, self.patch_rw_dim)
            if self.patch_num_diff > -1:
                data.coarsen_adj = self._diffuse(coarsen_adj).unsqueeze(0)

        
        subgraphs_batch = subgraphs_nodes[0] # this is the batch of subgraphs, i.e. the subgraph idxs [0, 0, 1, 1]

        mask = torch.zeros(self.n_patches).bool() # if say we have two patches then [False, False]
        mask[subgraphs_batch] = True # if subgraphs_batch = [0, 0, 1, 1] then [True, True]
        
        mask[subgraphs_batch[0]] = False # dont use the context subgraph, so we set it to False since it s always the first, this way the transformer wont pay attention to it

        data.subgraphs_batch = subgraphs_batch        
        data.subgraphs_nodes_mapper = subgraphs_nodes[1] # this is the node idxs [0, 2, 1, 3] (original node idxs)
        data.subgraphs_edges_mapper = subgraphs_edges[1] # this is the edge idxs [0, 1, 2] (original edge idxs)
        data.combined_subgraphs = combined_subgraphs # this is the edge index of th combined subgraph made of disconnected subgraphs, where each subgraph has its own unique node ids
        data.mask = mask.unsqueeze(0)
    
        subgraphs = subgraphs_nodes[0].unique()
        context_subgraph_idx = subgraphs[0]
        # 1+len(context_subgraphs_used) make sure that that the selected targets are not subgraphs that were used for the context subgraph, to minimize overlap and make task less trivial
        # the context subgraphs are still input to the transformer, because we need the full input graph, so all subgraphs found.
        rand_choice = np.random.choice(subgraphs[1+len(context_subgraphs_used):], self.num_targets, replace=False)
        target_subgraph_idxs = torch.tensor(rand_choice)
        
        data.context_subgraph_idx = context_subgraph_idx.tolist() # if context subgraph idx is 0, then[0]
        data.target_subgraph_idxs = sorted(target_subgraph_idxs.tolist()) # if target subgraph idxs are [1, 2] then [1, 2]
        data.call_n_patches = [self.n_patches]  # if target subgraph idxs are [1, 2] then [1, 2]
        data.__num_nodes__ = data.num_nodes  # set number of nodes of the current graph
        
        # these attributes are used only to plot
        data.context_nodes_mapper = subgraphs_nodes[1, subgraphs_nodes[0] == context_subgraph_idx]
        data.target_nodes_mapper = subgraphs_nodes[1, torch.isin(subgraphs_nodes[0], target_subgraph_idxs)]
        data.context_nodes_subgraph = subgraphs_nodes[0, subgraphs_nodes[0] == context_subgraph_idx]
        data.target_nodes_subgraph = subgraphs_nodes[0, torch.isin(subgraphs_nodes[0], target_subgraph_idxs)]
        # plot_from_transform_attributes(data)
        return data