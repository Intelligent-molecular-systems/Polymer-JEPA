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
        if len(clique) < 3:
            cliques[i] = expand_one_hop(G, clique)


    cliques_used = []
    context_nodes = set()
    
    all_cliques = cliques
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

    all_subgraphs = [context_subgraph] + target_subgraphs
    # plot_subgraphs(G, all_subgraphs)

    node_mask, edge_mask = create_masks(data, context_subgraph, target_subgraphs, data.num_nodes, n_patches)
    return node_mask, edge_mask, cliques_used


def metisZinc(data, n_patches, sizeContext, n_targets):
    G = to_networkx(data, to_undirected=True)
    # apply metis algorithm to the graph
    # idea divide each monomer in two partitions, join two partitions from different monomers and use as a context subgraph
    # otherwise checkout these algorithms: https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/node_clustering.html#overlapping-communities
    # DOC: https://metis.readthedocs.io/en/latest/
    # contig=True ensures that the partitions are connected
    nparts = data.num_nodes // 2 # arbitrary choice, this should make subgraphs of roughly size of 2/3 nodes 
    # if graph.num_nodes < nparts:
    #     parts = torch.randperm(n_patches)
    #     parts = parts[:graph.num_nodes-1]
    # else:
    parts = metis.part_graph(G, nparts=nparts)[1] #, contig=True
    
    # perform a one-hop expansion of each partition to avoid edge loss
    # Create a subgraph for each partition
    # Create subgraphs for each partition
    subgraphs = [set(node for node, part in enumerate(parts) if part == i) for i in range(nparts)]
    # Perform one-hop neighbor expansion for each partition
    # the one-hop expansion on such small and connected subgraphs cause a lot of overlap between at least some partitions
    # this could be non optimal for the prediction task, if the context and target share many nodes, its easy to predict..
    # but usually there are at 2/3 partitions with a small overlap so it should be good
    # alternatives are: 
    # 1. not to expand the subgraphs, 
    # 2. expand in a more sophisticated way (checking only the edges lost and including them in a single subgraph)
    # 3. use a different algorithm to partition the graph that already gives an overlap by default
    subgraphs = [expand_one_hop(G, subgraph) for subgraph in subgraphs if len(subgraph) > 0]

    # Ensure all subgraphs are connected components
    # subgraphs = [sg for sg in expanded_subgraphs if nx.is_connected(G.subgraph(sg))]
    
    context_subgraph = set()
    subgraphs_used = []
    context_subgraphs_used = []
    while len(context_subgraph) / data.num_nodes < sizeContext and subgraphs:
        clique = random.choice(subgraphs)
        if clique in subgraphs_used:
            continue
        subgraphs.remove(clique)
        subgraphs_used.append(clique)
        context_subgraph.update(clique)
    
    # select random cliques untl we have enough for targets, dont use the same clique used for context
    all_possible_targets = subgraphs
    
    # make sure we have at least n_targets, otherwise select random node, and do 1-hop expansion
    while len(all_possible_targets) < n_targets:
        node = random.choice(list(G.nodes))
        subgraph = set([node])
        subgraph = expand_one_hop(G, subgraph)
        if len(subgraph) >= 3:
            all_possible_targets.append({node})


    for clique in subgraphs_used:
        all_possible_targets.insert(0, clique)

    context_subgraph = list(context_subgraph)
    all_possible_targets = [list(clique) for clique in all_possible_targets]
    # Plotting
    # all_subgraphs = [context_subgraph] + target_subgraphs
    # plot_subgraphs(G, all_subgraphs)

    node_mask, edge_mask = create_masks(data, context_subgraph, all_possible_targets, data.num_nodes, n_patches)
    return node_mask, edge_mask, context_subgraphs_used


def expand_one_hop(fullG, subgraph_nodes):
    expanded_nodes = set(subgraph_nodes)
    for node in subgraph_nodes:
        expanded_nodes.update(fullG.neighbors(node))
    return expanded_nodes


def create_masks(graph, context_subgraph, target_subgraphs, n_of_nodes, n_patches):#
    # create always a fixed number of patches, the non existing patches will have all the nodes masked
    node_mask = torch.zeros((n_patches, n_of_nodes), dtype=torch.bool)
    # actual subgraphs 
    valid_subgraphs = target_subgraphs
    start_idx = n_patches - len(valid_subgraphs) # 20 - 9 = 11: 11, 12, 13, 14, 15, 16, 17, 18, 19 (index range is 0-19, so we are good)
    # context mask
    # for node in context_subgraph:
    #     node_mask[start_idx, node] = True
    context_mask = torch.zeros(node_mask.shape[1], dtype=torch.bool)
    context_mask[context_subgraph] = True
    node_mask[0] = context_mask
    
    # target masks
    idx = start_idx 
    for target_subgraph in target_subgraphs:
        target_mask = torch.zeros(node_mask.shape[1], dtype=torch.bool)
        target_mask[target_subgraph] = True
        node_mask[idx] = target_mask
        idx += 1

    edge_mask = node_mask[:, graph.edge_index[0]] & node_mask[:, graph.edge_index[1]]
    return node_mask, edge_mask