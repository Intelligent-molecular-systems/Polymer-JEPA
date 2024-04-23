from copy import deepcopy
import math
from matplotlib import pyplot as plt
import metis
import networkx as nx
import numpy as np
import random
import torch
from torch_geometric.utils import to_networkx
from torch_sparse import SparseTensor  # for propagation
from src.subgraphing_utils.context_subgraph_extractor import expand_one_hop
from src.visualize import plot_subgraphs
### Extracting target subgraph ###

# motif-based target subgraphing
def motifTargets(graph, n_targets, n_patches, cliques_used):
    cliques = {tuple(clique) for clique in graph.motifs[0].copy()}
    cliques_used_set = {tuple(clique) for clique in cliques_used}
    cliques = cliques - cliques_used_set
    cliques = [list(clique) for clique in cliques]

    # do a 1-hop expansion for each clique with less than 3 nodes
    for i, clique in enumerate(cliques):
        if len(clique) < 3:
            cliques[i] = list(expand_one_hop(to_networkx(graph, to_undirected=True), set(clique)))
        
    g = to_networkx(graph)
    while len(cliques) < n_targets:
        # create a subgraph from a random node and 1-hop expansion
        random_node = random.choice(list(g.nodes))
        subgraph = set([random_node])
        subgraph = expand_one_hop(to_networkx(graph, to_undirected=True), subgraph)
        if len(subgraph) >= 3:
            cliques.append(list(subgraph))

    # create node and edge mask, each clique is a subgraph
    node_mask = torch.zeros((n_patches, graph.num_nodes), dtype=torch.bool)
    edge_mask = torch.zeros((n_patches, graph.num_edges), dtype=torch.bool)

    for clique in cliques_used:
        # append the context cliques at the the beginning of the list of all cliques, so that they we can skip them when selecting the target subgraphs by using the index
        for bond in graph.intermonomers_bonds:
            # for all cliques that have an intermonomer bond, add the other node to the clique, to prevent intermonomer edge loss
            if bond[0] in clique and bond[1] not in clique:
                clique.append(bond[1])
            elif bond[1] in clique and bond[0] not in clique:
                clique.append(bond[0])
            else:
                continue
        cliques.insert(0, clique)

    idx = n_patches - len(cliques)
    for target_subgraph in cliques:
        target_mask = torch.zeros(node_mask.shape[1], dtype=torch.bool)
        target_mask[target_subgraph] = True
        node_mask[idx] = target_mask
        idx += 1

    # plot_subgraphs(g, cliques)

    edge_mask = node_mask[:, graph.edge_index[0]] & node_mask[:, graph.edge_index[1]]

    return node_mask, edge_mask 


# random-walk-based target subgraphing
def rwTargets(graph, n_targets, n_patches, rw1, rw2, target_size):
    def random_walk_step(fullGraph, current_node, exclude_nodes):
        neighbors = list(set(fullGraph.neighbors(current_node)) - exclude_nodes)
        if not neighbors:
            neighbors = list(fullGraph.neighbors(current_node))
        return random.choice(neighbors) # if neighbors else None
    
    def random_walk_from_node(fullGraph, start_node, exclude_nodes, total_nodes, size=target_size):
        walk = [start_node]
        while len(walk) / total_nodes < size:
            next_node = random_walk_step(fullGraph=fullGraph, current_node=walk[-1], exclude_nodes=exclude_nodes)
            if next_node:
                walk.append(next_node)
            else:
                break
        return walk
    
    visited_nodes = set()
    visited_nodes.update(rw1)
    visited_nodes.update(rw2)

    rw_walks = []

    # does not guarantee 100% to avoid edge loss, but its unlikely that it will happen, and at each epoch the subgraphs are different so it should be fine, its also a form of data augmentation
    while len(visited_nodes) < graph.num_nodes:
        # pick a random node from the remaining nodes
        remaining_nodes = list(set(range(graph.num_nodes)) - visited_nodes)
        start_node = random.choice(remaining_nodes)
        rw_subgraph = random_walk_from_node(fullGraph=to_networkx(graph, to_undirected=True), start_node=start_node, exclude_nodes=rw1.union(rw2), total_nodes=graph.num_nodes)
        rw_expanded = expand_one_hop(to_networkx(graph, to_undirected=True), rw_subgraph)
        visited_nodes.update(rw_expanded)
        rw_walks.append(rw_expanded)

    while len(rw_walks) < n_targets:
        # create a subgraph from a random node and 1-hop expansion
        start_node = random.choice(list(set(range(graph.num_nodes)).difference(rw1.union(rw2))))
        rw_subgraph = random_walk_from_node(fullGraph=to_networkx(graph, to_undirected=True), start_node=start_node, exclude_nodes=set(), total_nodes=graph.num_nodes)
        rw_walks.append(rw_subgraph)
        

    node_mask = torch.zeros((n_patches, graph.num_nodes), dtype=torch.bool)
    edge_mask = torch.zeros((n_patches, graph.num_edges), dtype=torch.bool)

    # insert the context random walks at the beginning of the list of all random walks, so that they we can skip them when selecting the target subgraphs by using the index
    rw_walks.insert(0, rw1)
    rw_walks.insert(0, rw2)

    # plot_subgraphs(to_networkx(graph, to_undirected=True), rw_walks)

    idx = n_patches - len(rw_walks)
    for target_subgraph in rw_walks:
        target_mask = torch.zeros(node_mask.shape[1], dtype=torch.bool)
        target_mask[list(target_subgraph)] = True
        node_mask[idx] = target_mask
        idx += 1

    edge_mask = node_mask[:, graph.edge_index[0]] & node_mask[:, graph.edge_index[1]]

    return node_mask, edge_mask 



# # metis based subgraphing
# def metisTargets(g, n_patches, drop_rate=0.0, num_hops=1, is_directed=False):
#     def k_hop_subgraph(edge_index, num_nodes, num_hops, is_directed=False):
#         # return k-hop subgraphs for all nodes in the graph
#         if is_directed:
#             row, col = edge_index
#             birow, bicol = torch.cat([row, col]), torch.cat([col, row])
#             edge_index = torch.stack([birow, bicol])
#         else:
#             row, col = edge_index
#         sparse_adj = SparseTensor(
#             row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
#         # each one contains <= i hop masks
#         hop_masks = [torch.eye(num_nodes, dtype=torch.bool,
#                             device=edge_index.device)]
#         hop_indicator = row.new_full((num_nodes, num_nodes), -1)
#         hop_indicator[hop_masks[0]] = 0
#         for i in range(num_hops):
#             next_mask = sparse_adj.matmul(hop_masks[i].float()) > 0
#             hop_masks.append(next_mask)
#             hop_indicator[(hop_indicator == -1) & next_mask] = i+1
#         hop_indicator = hop_indicator.T  # N x N
#         node_mask = (hop_indicator >= 0)  # N x N dense mask matrix
#         return node_mask
    
#     if is_directed:
#         if g.num_nodes < n_patches:
#             membership = torch.arange(g.num_nodes)
#         else:
#             G = to_networkx(g, to_undirected="lower")
#             cuts, membership = metis.part_graph(G, n_patches, recursive=True)
#     else:
#         if g.num_nodes < n_patches:
#             membership = torch.randperm(n_patches)
#         else:
#             # data augmentation
#             adjlist = g.edge_index.t()
#             arr = torch.rand(len(adjlist))
#             selected = arr > drop_rate
#             G = nx.Graph()
#             G.add_nodes_from(np.arange(g.num_nodes))
#             G.add_edges_from(adjlist[selected].tolist())
#             # metis partition
#             cuts, membership = metis.part_graph(G, n_patches, recursive=True)

#     assert len(membership) >= g.num_nodes
#     membership = torch.tensor(np.array(membership[:g.num_nodes]))
#     max_patch_id = torch.max(membership)+1
#     membership = membership+(n_patches-max_patch_id)
#     node_mask = torch.stack([membership == i for i in range(n_patches)])

#     if num_hops > 0:
#         subgraphs_batch, subgraphs_node_mapper = node_mask.nonzero().T
#         k_hop_node_mask = k_hop_subgraph(
#             g.edge_index, g.num_nodes, num_hops, is_directed)
#         node_mask.index_add_(0, subgraphs_batch,
#                              k_hop_node_mask[subgraphs_node_mapper])

#     edge_mask = node_mask[:, g.edge_index[0]] & node_mask[:, g.edge_index[1]]
#     return node_mask, edge_mask