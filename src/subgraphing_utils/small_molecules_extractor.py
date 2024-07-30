from copy import deepcopy
import math
from matplotlib import pyplot as plt
import metis
import networkx as nx
import numpy as np
import random
from rdkit import Chem
from src.subgraphing_utils.motif_subgraphing import get_motifs
import torch
from torch_geometric.utils import to_networkx
from torch_sparse import SparseTensor  # for propagation
from src.visualize import plot_subgraphs


def zincSubgraphs(data, sizeContext, n_patches, n_targets):

    G = to_networkx(data, to_undirected=True)

    smile = data.smile
    mol = Chem.MolFromSmiles(smile)

    cliques, _, _ = get_motifs(mol)

    # filter all cliques to keep only the connected components
    # cliques = [clique for clique in cliques if nx.is_connected(G.subgraph(clique))]

    # 1-hop exp of all cliques with less than 3 nodes
    for i, clique in enumerate(cliques):
        if len(clique) < 2:
            cliques[i] = expand_one_hop(G, clique)


    cliques_used = []
    context_nodes = set()
    
    all_cliques = cliques.copy()
    # select random cliques untl we have enough for context
    while len(context_nodes) / data.num_nodes < sizeContext and all_cliques:
        clique = random.choice(all_cliques)
        if clique in cliques_used:
            continue
        all_cliques.remove(clique)
        cliques_used.append(clique)
        context_nodes.update(clique)
    
    # select random cliques untl we have enough for targets, dont use the same clique used for context
    all_possible_targets = all_cliques
    
    # make sure we have at least n_targets, otherwise select random node, and do 1-hop expansion
    while len(all_possible_targets) < n_targets:
        node = random.choice(list(G.nodes))
        subgraph = set([node])
        subgraph = expand_one_hop(G, subgraph)
        if len(subgraph) >= 3:
            all_possible_targets.append({node})


    for clique in cliques_used:
        all_possible_targets.insert(0, clique)

    context_subgraph = list(context_nodes)
    target_subgraphs = [list(clique) for clique in all_possible_targets]

    # all_subgraphs = [context_subgraph] + target_subgraphs
    # plot_subgraphs(G, all_subgraphs)

    node_mask, edge_mask = create_masks(data, context_subgraph, target_subgraphs, data.num_nodes, n_patches)
    return node_mask, edge_mask, cliques_used


def metisZinc(data, n_patches, sizeContext, n_targets):
    
    # G = to_networkx(data, to_undirected=True)
  
    # line_G = nx.line_graph(G)

    # # Partition the line graph
    # nparts = len(line_G) // 3
    # _, edge_parts = metis.part_graph(line_G, nparts=nparts, contig=True)

    # # Map line graph edges (which are nodes in the line graph) back to original graph edges
    # original_edges = list(line_G.nodes)

    # # Group edges by partition
    # edges_by_partition = {}
    # for edge_index, part in enumerate(edge_parts):
    #     if part not in edges_by_partition:
    #         edges_by_partition[part] = []
    #     edges_by_partition[part].append(original_edges[edge_index])

    # # Create subgraphs from these groups of edges
    # subgraphs = []
    # for part, edges in edges_by_partition.items():
    #     # Initialize an empty subgraph for this partition
    #     subgraph = nx.Graph()
        
    #     # Add edges (and consequently the nodes) to the subgraph
    #     for edge in edges:
    #         subgraph.add_edge(*edge)
        
    #     subgraphs.append(set(subgraph.nodes))

    # # eliminate empty subgraphs
    # subgraphs = [sg for sg in subgraphs if sg]

    # # Ensure all subgraphs are connected components
    # # subgraphs = [sg for sg in expanded_subgraphs if nx.is_connected(G.subgraph(sg))]
    
    # context_subgraph = set()
    # context_subgraphs_used = []
    # while len(context_subgraph) / data.num_nodes < sizeContext and subgraphs:
    #     clique = random.choice(subgraphs)
    #     if clique in context_subgraphs_used:
    #         continue
    #     subgraphs.remove(clique)
    #     context_subgraphs_used.append(clique)
    #     context_subgraph.update(clique)
    
    # # select random cliques untl we have enough for targets, dont use the same clique used for context
    # all_possible_targets = subgraphs
    
    # # make sure we have at least n_targets, otherwise select random node, and do 1-hop expansion
    # while len(all_possible_targets) < n_targets:
    #     node = random.choice(list(G.nodes))
    #     subgraph = set([node])
    #     subgraph = expand_one_hop(G, subgraph)
    #     if len(subgraph) >= 3:
    #         all_possible_targets.append({node})


    # for clique in context_subgraphs_used:
    #     all_possible_targets.insert(0, clique)

    # context_subgraph = list(context_subgraph)
    # all_possible_targets = [list(clique) for clique in all_possible_targets]
    # # Plotting
    # # all_subgraphs = [context_subgraph] + all_possible_targets
    # # plot_subgraphs(G, all_subgraphs)

    # node_mask, edge_mask = create_masks(data, context_subgraph, all_possible_targets, data.num_nodes, n_patches)
    # return node_mask, edge_mask, context_subgraphs_used
    
    # original metis
    n_patches = n_patches - 1
    g = data
    
    if g.num_nodes < n_patches:
        membership = torch.randperm(n_patches)
    else:
        # data augmentation
        adjlist = g.edge_index.t()
        arr = torch.rand(len(adjlist))
        selected = arr > 0.3
        G = nx.Graph()
        G.add_nodes_from(np.arange(g.num_nodes))
        G.add_edges_from(adjlist[selected].tolist())
        # metis partition
        cuts, membership = metis.part_graph(G, n_patches, recursive=True)

    assert len(membership) >= g.num_nodes
    membership = torch.tensor(np.array(membership[:g.num_nodes]))
    
    max_patch_id = torch.max(membership)+1
    membership = membership+(n_patches-max_patch_id)

    node_mask = torch.stack([membership == i for i in range(n_patches)])

    
    subgraphs_batch, subgraphs_node_mapper = node_mask.nonzero().T
    k_hop_node_mask = k_hop_subgraph(
        g.edge_index, g.num_nodes, 1, False)
    node_mask.index_add_(0, subgraphs_batch,
                             k_hop_node_mask[subgraphs_node_mapper])
        
    context_size = math.ceil(sizeContext * g.num_nodes)
    context_subgraph = set()
    context_subgraphs_used = []
    for idx in range(n_patches-max_patch_id, n_patches):
        context_subgraph.update(subgraphs_node_mapper[node_mask[idx]])
        # check if the subgraph is not empty
        if len(subgraphs_node_mapper[node_mask[idx]]) > 0:
            context_subgraphs_used.append(subgraphs_node_mapper[node_mask[idx]])
        if len(context_subgraph) >= context_size:
            break
    
    context_mask = torch.zeros((1, node_mask.shape[1]), dtype=torch.bool)
    context_mask[0, list(context_subgraph)] = True
    # concatenate the context mask
    node_mask = torch.cat([context_mask, node_mask])

    # take the first n subgraphs until reached the context size
    edge_mask = node_mask[:, g.edge_index[0]] & node_mask[:, g.edge_index[1]]
    return node_mask, edge_mask, context_subgraphs_used


def rwZincContext(data, sizeContext):
    if sizeContext == 1:
        return torch.ones((1, data.num_nodes), dtype=torch.bool), torch.ones((1, data.num_edges), dtype=torch.bool)
    
    # Function to perform a single random walk step from a given node
    def random_walk_step(fullGraph, current_node, exclude_nodes):
        neighbors = list(set(fullGraph.neighbors(current_node)) - exclude_nodes)
        return random.choice(neighbors) if neighbors else None
    
    contextRw = set()
    total_nodes = data.num_nodes

    # Perform a random walk from a random node
    start_node = random.choice(list(range(total_nodes)))
    contextRw.add(start_node)
    current_node = start_node
    graph = to_networkx(data, to_undirected=True)
    while len(contextRw) / total_nodes < sizeContext:
        next_node = random_walk_step(graph, current_node, contextRw)
        if next_node is None:
            break
        contextRw.add(next_node)
        current_node = next_node

    # add random nodes until we reach the context size
    counter = 0
    while len(contextRw) / total_nodes < sizeContext:
        random_node = random.choice(list(contextRw))
        next_node = random_walk_step(fullGraph=graph, current_node=random_node, exclude_nodes=contextRw)
        if next_node is not None:
            contextRw.add(next_node)
            counter = 0
        else:
            counter += 1
            if counter > 30:
                break

    node_mask = torch.zeros((1, total_nodes), dtype=torch.bool)
    node_mask[0, list(contextRw)] = True
    edge_mask = node_mask[:, data.edge_index[0]] & node_mask[:, data.edge_index[1]]
    return node_mask, edge_mask, contextRw
    


def rwZincTargets(data, n_patches, n_targets, contextRw, target_size):
    def random_walk_step(fullGraph, current_node, exclude_nodes):
        neighbors = list(set(fullGraph.neighbors(current_node)) - exclude_nodes)
        return random.choice(neighbors) if neighbors else None
    
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
    visited_nodes.update(contextRw)

    rw_walks = []

    # does not guarantee 100% to avoid edge loss, but its unlikely that it will happen, and at each epoch the subgraphs are different so it should be fine, its also a form of data augmentation
    while len(visited_nodes) < data.num_nodes:
        # pick a random node from the remaining nodes
        remaining_nodes = list(set(range(data.num_nodes)) - visited_nodes)
        start_node = random.choice(remaining_nodes)
        rw_subgraph = random_walk_from_node(fullGraph=to_networkx(data, to_undirected=True), start_node=start_node, exclude_nodes=visited_nodes, total_nodes=data.num_nodes)
        rw_expanded = expand_one_hop(to_networkx(data, to_undirected=True), rw_subgraph)
        visited_nodes.update(rw_expanded)
        rw_walks.append(rw_expanded)

    while len(rw_walks) < n_targets:
        # create a subgraph from a random node and 1-hop expansion
        random_node = random.choice(list(visited_nodes))
        subgraph = set([random_node])
        subgraph = expand_one_hop(to_networkx(data, to_undirected=True), subgraph)
        rw_walks.append(subgraph)
    
    node_mask = torch.zeros((n_patches, data.num_nodes), dtype=torch.bool)
    edge_mask = torch.zeros((n_patches, data.num_edges), dtype=torch.bool)

    rw_walks.insert(0, contextRw)

    idx = n_patches - len(rw_walks)
    for target_subgraph in rw_walks:
        target_mask = torch.zeros(node_mask.shape[1], dtype=torch.bool)
        target_mask[list(target_subgraph)] = True
        node_mask[idx] = target_mask
        idx += 1
    
    edge_mask = node_mask[:, data.edge_index[0]] & node_mask[:, data.edge_index[1]]
                                                             
    return node_mask, edge_mask


def expand_one_hop(fullG, subgraph_nodes):
    expanded_nodes = set(subgraph_nodes)
    for node in subgraph_nodes:
        expanded_nodes.update(fullG.neighbors(node))
    return expanded_nodes


def create_masks(graph, context_subgraph, target_subgraphs, n_of_nodes, n_patches):#
    # create always a fixed number of patches, the non existing patches will have all the nodes masked
    node_mask = torch.zeros((n_patches, n_of_nodes), dtype=torch.bool)
    # context mask
    # for node in context_subgraph:
    #     node_mask[start_idx, node] = True
    context_mask = torch.zeros(node_mask.shape[1], dtype=torch.bool)
    context_mask[context_subgraph] = True
    node_mask[0] = context_mask
    
    # actual subgraphs 
    valid_subgraphs = target_subgraphs
    start_idx = n_patches - len(valid_subgraphs) # 20 - 9 = 11: 11, 12, 13, 14, 15, 16, 17, 18, 19 (index range is 0-19, so we are good)
    # target masks
    idx = start_idx 
    for target_subgraph in target_subgraphs:
        target_mask = torch.zeros(node_mask.shape[1], dtype=torch.bool)
        target_mask[target_subgraph] = True
        node_mask[idx] = target_mask
        idx += 1

    edge_mask = node_mask[:, graph.edge_index[0]] & node_mask[:, graph.edge_index[1]]
    return node_mask, edge_mask


def k_hop_subgraph(edge_index, num_nodes, num_hops, is_directed=False):
    # return k-hop subgraphs for all nodes in the graph
    if is_directed:
        row, col = edge_index
        birow, bicol = torch.cat([row, col]), torch.cat([col, row])
        edge_index = torch.stack([birow, bicol])
    else:
        row, col = edge_index
    sparse_adj = SparseTensor(
        row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    # each one contains <= i hop masks
    hop_masks = [torch.eye(num_nodes, dtype=torch.bool,
                           device=edge_index.device)]
    hop_indicator = row.new_full((num_nodes, num_nodes), -1)
    hop_indicator[hop_masks[0]] = 0
    for i in range(num_hops):
        next_mask = sparse_adj.matmul(hop_masks[i].float()) > 0
        hop_masks.append(next_mask)
        hop_indicator[(hop_indicator == -1) & next_mask] = i+1
    hop_indicator = hop_indicator.T  # N x N
    node_mask = (hop_indicator >= 0)  # N x N dense mask matrix
    return node_mask