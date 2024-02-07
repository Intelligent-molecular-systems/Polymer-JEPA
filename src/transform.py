import re
import random
import torch
import numpy as np
from torch_geometric.data import Data
from core.transform_utils.subgraph_extractors import metis_subgraph, random_subgraph
from core.data_utils.pe import RWSE, LapPE, random_walk
from torch_geometric.transforms import Compose

from torch_geometric.utils import degree
from torch_geometric.loader import NeighborLoader, GraphSAINTSampler

def cal_coarsen_adj(subgraphs_nodes_mask):
    #a coarse patch adjacency matrix A′ = B*B^T ∈ Rp×p, where each A′ ij contains the node overlap between pi and pj.
    mask = subgraphs_nodes_mask.to(torch.float)
    coarsen_adj = torch.matmul(mask, mask.t()) # element at position (i, j) is the number of nodes that subgraphs i and j have in common.
    return coarsen_adj # create a simplified version of a graph, the purpose of this process is to reduce the complexity of the graph, making it easier to analyze or compute on.


def to_sparse(node_mask, edge_mask):
    # https://pytorch.org/docs/stable/generated/torch.nonzero.html
    subgraphs_nodes = node_mask.nonzero().T
    subgraphs_edges = edge_mask.nonzero().T
    return subgraphs_nodes, subgraphs_edges

# def combine_subgraphs_jepa(edge_index, subgraphs_nodes, subgraphs_edges, 
#                            context_nodes_mask, context_edges_mask, 
#                            target_nodes_mask, target_edges_mask, 
#                            num_selected=None, num_nodes=None):
#     if num_selected is None:
#         num_selected = subgraphs_nodes[0][-1] + 1
#     if num_nodes is None:
#         num_nodes = subgraphs_nodes[1].max() + 1

#     combined_subgraphs = edge_index[:, subgraphs_edges[1]] # Select all the subgraph edges from the global edge index
#     node_label_mapper = edge_index.new_full((num_selected, num_nodes), -1)
#     node_label_mapper[subgraphs_nodes[0], subgraphs_nodes[1]
#                       ] = torch.arange(len(subgraphs_nodes[1])) # For each subgraph, create the new "placeholder" indices
#     node_label_mapper = node_label_mapper.reshape(-1)

#     inc = torch.arange(num_selected)*num_nodes
#     combined_subgraphs += inc[subgraphs_edges[0]]
#     combined_subgraphs = node_label_mapper[combined_subgraphs]

#     return combined_subgraphs


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
    def __init__(self, rw_dim=0, lap_dim=0):
        super().__init__()
        self.rw_dim = rw_dim
        self.lap_dim = lap_dim

    def __call__(self, data):
        if self.rw_dim > 0:
            data.rw_pos_enc = RWSE(
                data.edge_index, self.rw_dim, data.num_nodes)
        if self.lap_dim > 0:
            data.lap_pos_enc = LapPE(
                data.edge_index, self.lap_dim, data.num_nodes)
        return data

# !!! 
class GraphJEPAPartitionTransform(object):
    def __init__(self, n_patches, metis=True, drop_rate=0.0, num_hops=1, is_directed=False, patch_rw_dim=0, patch_num_diff=0, num_context=1, num_targets=4):
        super().__init__()
        self.n_patches = n_patches
        self.drop_rate = drop_rate
        self.num_hops = num_hops
        self.is_directed = is_directed
        self.patch_rw_dim = patch_rw_dim
        self.patch_num_diff = patch_num_diff
        self.metis = metis
        self.num_context = num_context
        self.num_targets = num_targets

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
    
    # The combine_subgraphs operation needs to uniquely identify each node across subgraphs, especially where there is overlap (node 2 in subgraphs 0 and 1, node 4 in subgraphs 1 and 2). Here's how re-indexing works:
    # Each node in each subgraph is assigned a unique global identifier. For example, nodes 0, 1, and 2 in subgraph 0 keep their ids; nodes 2, 3, and 4 in subgraph 1 are re-indexed to avoid overlap with subgraph 0; similarly, node 4 in subgraph 2 is re-indexed to avoid overlap with subgraph 1.
    # Subgraph 0's nodes (0, 1, 2) are indexed as [0, 1, 2].  Subgraph 1's nodes (2, 3, 4), with node 2 overlapping, could be re-indexed as [3, 4, 5]. Subgraph 2's nodes (4, 5), with node 4 overlapping, could be re-indexed as [6, 7].
    def combine_subgraphs(edge_index, subgraphs_nodes, subgraphs_edges, num_selected=None, num_nodes=None):
        # edge_index is full graph
        # subgraphs_nodes: A tuple containing two tensors. The first tensor represents the subgraph indices for each node, and the second tensor represents the original node indices. This structure allows mapping nodes to their respective subgraphs.
        # subgraphs_edges: Similar to subgraphs_nodes, a tuple where the first tensor indicates the subgraph indices for each edge, and the second tensor represents the indices of these edges in the edge_index
        # num_selected: The number of subgraphs selected or generated. If None, it’s inferred from the last subgraph index in subgraphs_nodes[0]
        # num_nodes: The total number of unique nodes across all subgraphs. If None, it’s inferred from the maximum node index in subgraphs_nodes[1]
        
        if num_selected is None:
            num_selected = subgraphs_nodes[0][-1] + 1
        if num_nodes is None:
            num_nodes = subgraphs_nodes[1].max() + 1

        # combined_subgraphs is an edge_index including only the edges that are part of the subgraphs (basically all ?)
        combined_subgraphs = edge_index[:, subgraphs_edges[1]] # Select edges from the global edge_index that are part of the subgraphs, using subgraphs_edges[1] for indexing. This step extracts the edges relevant to the subgraphs being considered.
        # node_label_mapper has shape [n of subgraphs, num_nodes] and is initialized with -1. It will be used to map each node in each subgraph to a unique global identifier across the combined subgraph structure.
        # called on edge_index to use its dtype and device, node_label_mapper is a tensor filled with -1, with dimensions [num_subgraphs, total_num_nodes_in_all_subgraphs]
        node_label_mapper = edge_index.new_full((num_selected, num_nodes), fill_value=-1) # Initialize a tensor (node_label_mapper) that will map each node in each subgraph to a unique global identifier across the combined subgraph structure. This tensor is filled with -1 initially and has dimensions [num_selected, num_nodes], where each subgraph has its own slice of the node namespace.

        # this is basically setting a unique identifier for all subgraph nodes
        # example: subgraphs_nodes[0] = torch.tensor([0, 0, 1, 1, 2, 2]), subgraphs_nodes[1] = torch.tensor([0, 1, 1, 2, 2, 3]) (some nodes belong to multiple subgraphs)
        # torch.arange(len(subgraphs_nodes[1])) = torch.tensor([0, 1, 2, 3, 4, 5])
        # each node occurrence is assigned a unique global identifier based on its order of appearance in subgraphs_nodes[1]
        # node_label_mapper = tensor([[ 0,  1, -1, -1], [-1,  2,  3, -1], [-1, -1,  4,  5]]) -> each subgraph contains a unique identifier for each node, even if the node appears in multiple subgraphs
        node_label_mapper[subgraphs_nodes[0], subgraphs_nodes[1]] = torch.arange(len(subgraphs_nodes[1])) # For each node belonging to a subgraph (as specified in subgraphs_nodes), assign a unique identifier within the combined subgraph context. This is done by replacing the local node indices (subgraphs_nodes[1]) with a sequence of integers (torch.arange(len(subgraphs_nodes[1]))), ensuring that each node across all subgraphs has a unique identifier.
        # node_label_mapper.reshape(-1)= tensor([ 0,  1, -1, -1, -1,  2,  3, -1, -1, -1,  4,  5]) -> reshaping the tensor into a one-dimensional array
        node_label_mapper = node_label_mapper.reshape(-1) # .reshape(-1) will transform it into a one-dimensional array without explicitly specifying the size of that dimension.
        # To ensure the edges refer to the correct nodes in the combined subgraph space, the function increments the node indices in the combined_subgraphs tensor by an offset calculated as num_nodes * subgraph_index
        # calculating offsets when indexing into node_label_mapper.
        inc = torch.arange(num_selected)*num_nodes # [0, 1, ..., num subgraphs-1] * num_nodes = [0, num_nodes, ..., (num subgraphs-1)*num_nodes]
        # applies an offset to the node indices in the combined_subgraphs tensor, ensuring that nodes from different subgraphs are uniquely identified within the overall structure of the combined subgraphs. This is necessary because the same node ID could appear in multiple subgraphs, and simply combining these subgraphs without adjusting the node IDs would result in ambiguity about which subgraph a node belongs to.
        combined_subgraphs += inc[subgraphs_edges[0]] # Increment the node indices in the combined_subgraphs tensor by an offset calculated as num_nodes * subgraph_index. This ensures that the edges refer to the correct nodes in the combined subgraph space.
        # After re-indexing, the combined_subgraphs operation would adjust the edge_index to reflect these new node identifiers, ensuring edges point to the correct, uniquely identified nodes. The edge_index could be transformed as follows, considering the new global identifiers:
        # AKA, the edge indices in combined_subgraphs are remapped through the node_label_mapper, translating them into the unified node index space of the combined subgraph.
        # the operation combined_subgraphs += inc[subgraphs_edges[0]] adjusts node indices for uniqueness across subgraphs, while the operation combined_subgraphs = node_label_mapper[combined_subgraphs] translates these adjusted indices into a final, globally unique and meaningful indexing scheme.
        combined_subgraphs = node_label_mapper[combined_subgraphs] #  remaps these offset-adjusted node indices to their final, unique identifiers across the entire set of combined subgraphs. This remapping is crucial for a few reasons:
        return combined_subgraphs

    def __call__(self, data):
        data = SubgraphsData(**{k: v for k, v in data})
        if self.metis:
            node_masks, edge_masks = metis_subgraph(
                data, n_patches=self.n_patches, drop_rate=self.drop_rate, num_hops=self.num_hops, is_directed=self.is_directed)
        else:
            node_masks, edge_masks = random_subgraph(
                data, n_patches=self.n_patches, num_hops=self.num_hops)
            
        # to_sparse function is used to convert the node_mask and edge_mask tensors into a sparse representation. 
        # This is useful for reducing memory usage and computational requirements when dealing with large graphs.
        subgraphs_nodes, subgraphs_edges = to_sparse(node_masks, edge_masks) # understood till here
        # Subgraphs Nodes:tensor([[0, 0, 1, 1], 
        #                         [0, 2, 1, 3]]) this means that in subgraph 0, nodes 0 and 2 are present, and in subgraph 1, nodes 1 and 3 are present (what about overlap?)
        
        # same for the edges: tensor([[subgraph1idx, subgraph1idx, subgraphs2idx], [edge1, edge2, edge3]])

        # The output, combined_subgraphs, is a tensor of remapped edge indices in this combined subgraph space. Each edge is now associated with a unique node identifier that respects the original subgraph partitioning, enabling the neural network to process information from multiple subgraphs as if it were a single, cohesive structure.
        # This technique allows for handling complex graph structures by dividing them into manageable pieces (subgraphs) and then reassembling their information in a way that preserves the integrity of the original graph while facilitating operations that require global context or comparisons across subgraphs.
        combined_subgraphs = combine_subgraphs(
            data.edge_index, 
            subgraphs_nodes, 
            subgraphs_edges, 
            num_selected=self.n_patches,
            num_nodes=data.num_nodes
        )
        # this is basically an alternative way to compute the positional encoding based on the RWSE of the patches, i
        # see 4.4. Ablation Studies in the paper
        if self.patch_num_diff > -1 or self.patch_rw_dim > 0:
            coarsen_adj = cal_coarsen_adj(node_masks)
            if self.patch_rw_dim > 0: # computed cause default config has patch_rw_dim > 0
                # this is never used, idk why they do position encoding here, while its done in PositionalEncodingTransform using rw_dim property
                data.patch_pe = random_walk(coarsen_adj, self.patch_rw_dim)
            # it is possible to use the RWSE of the patches as conditioning information. Formally, let B ∈ {0, 1}p×N be the patch assignment matrix, such that Bij = 1 if vj ∈ pi. We can calculate a coarse patch adjacency matrix A′ = BBT ∈ Rp×p, where each A′ ij contains the node overlap between pi and pj.
            # computed cause default config has patch_num_diff = 0
            if self.patch_num_diff > -1: # patch_num_diff = Patch PE diffusion steps
                data.coarsen_adj = self._diffuse(coarsen_adj).unsqueeze(0)


        subgraphs_batch = subgraphs_nodes[0] # this is the batch of subgraphs, i.e. the subgraph idxs [0, 0, 1, 1]
        mask = torch.zeros(self.n_patches).bool() # if say we have two patches then [False, False]
        mask[subgraphs_batch] = True # if subgraphs_batch = [0, 0, 1, 1] then [True, True]
        data.subgraphs_batch = subgraphs_batch
        data.subgraphs_nodes_mapper = subgraphs_nodes[1] # this is the node idxs [0, 2, 1, 3] (original node idxs)
        data.subgraphs_edges_mapper = subgraphs_edges[1] # this is the edge idxs [0, 1, 2] (original edge idxs)
        data.combined_subgraphs = combined_subgraphs # this is the edge index of th combined subgraph made of disconnected subgraphs, where each subgraph has its own unique node ids
        data.mask = mask.unsqueeze(0) # [True, True] -> [[True, True]]

        # Pick one subgraph as context and others as targets (at random)
        # Attention computation (mask) fix: Select only from non-empty patches
        # subgraphs_nodes[0].unique() -> pick n subgraph idx from [0, 1, .., n_patches-1], n = num_context + num_targets
        rand_choice = np.random.choice(subgraphs_nodes[0].unique(), self.num_context+self.num_targets, replace=False)
        context_subgraph_idx = rand_choice[0] # use the first subgraph idx as context
        target_subgraph_idxs = torch.tensor(rand_choice[1:]) # use the rest as targets

        data.context_edges_mask = subgraphs_edges[0] == context_subgraph_idx # if context subgraph idx is 0, and subgraphs_edges[0] = [0, 0, 1, 1, 2] then [True, True, False, False, False]
        data.target_edges_mask = torch.isin(subgraphs_edges[0], target_subgraph_idxs) # if target subgraph idxs are [1, 2] then [False, False, True, True, True]
        data.context_nodes_mapper = subgraphs_nodes[1, subgraphs_nodes[0] == context_subgraph_idx] # if context subgraph idx is 0, and subgraphs_nodes[0] (subgraphs indexes) =[0 ,0, 1, 1, 2] subgraphs_nodes[1] (node indexes) = [0, 2, 1, 2, 3] then [0, 2]
        data.target_nodes_mapper = subgraphs_nodes[1, torch.isin(subgraphs_nodes[0], target_subgraph_idxs)] # if target subgraph idxs are [1, 2] then [1, 2, 3]
        data.context_nodes_subgraph = subgraphs_nodes[0, subgraphs_nodes[0] == context_subgraph_idx] # if context subgraph idx is 0, and subgraphs_nodes[0] (subgraphs indexes) =[0 ,0, 1, 1, 2] then [0, 0]
        data.target_nodes_subgraph = subgraphs_nodes[0, torch.isin(subgraphs_nodes[0], target_subgraph_idxs)] # if target subgraph idxs are [1, 2] then [1, 1, 2]
        data.context_subgraph_idx = context_subgraph_idx.tolist() # if context subgraph idx is 0, then[0]
        data.target_subgraph_idxs = target_subgraph_idxs.tolist() # if target subgraph idxs are [1, 2] then [1, 2]
        data.call_n_patches = [self.n_patches]

        data.__num_nodes__ = data.num_nodes  # set number of nodes of the current graph
        return data