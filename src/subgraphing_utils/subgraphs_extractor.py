from copy import deepcopy
import math
from matplotlib import pyplot as plt
import metis
import networkx as nx
import numpy as np
import random
import torch
from torch_geometric.utils.convert import to_networkx


def motifs2subgraphs(graph, n_patches, min_targets):
    cliques, intermonomers_bonds, monomer_mask = graph.motifs[0], graph.intermonomers_bonds, graph.monomer_mask
    # [TODO]: Check requirements (size of context and size of targets, etc.)

    # randomly pick one intermonomer bond
    intermonomer_bond = random.choice(intermonomers_bonds)
    
    # create a list of cliques for each monomer
    monomer_cliques = [[], []]
    for clique in cliques:
        monomer_cliques[monomer_mask[clique[0]]].append(clique)
    
    # randomly pick one clique from each monomer the clique must be one of those that contain a node of the intermonomer bond
    monomerA_clique = random.choice([clique for clique in monomer_cliques[monomer_mask[intermonomer_bond[0]]] if intermonomer_bond[0] in clique])
    monomerB_clique = random.choice([clique for clique in monomer_cliques[monomer_mask[intermonomer_bond[1]]] if intermonomer_bond[1] in clique])
    
    # context subgraph: join 2 cliques such that they belong to different monomers and they are connected by an intermonomer bond
    context_subgraph = monomerA_clique + monomerB_clique
    
    # select all other cliques (except the two chosen above) as target subgraphs
    target_subgraphs = [clique for clique in cliques if clique not in [monomerA_clique, monomerB_clique]]

    # if target subgraphs is smaller than min_targets, add random subgraphs to reach the minimum
    if len(target_subgraphs) < min_targets:
        # available nodes are all nodes that are not in the context subgraph
        available_nodes = set(range(len(monomer_mask))) - set(context_subgraph)
        if not available_nodes:
            available_nodes = set(range(len(monomer_mask)))
        
        G = to_networkx(graph, to_undirected=True)
        while len(target_subgraphs) < min_targets: # RISK infinite loop, but it should not happen
            # pick a random node from the available nodes
            random_node = random.choice(list(available_nodes))
            # expand the node by one hop
            new_subgraph = [random_node]
            new_subgraph = expand_one_hop(G, new_subgraph)
            # check if subgraph is not already in the target subgraphs
            if new_subgraph not in target_subgraphs:
                target_subgraphs.append(list(new_subgraph))
    

    # Plotting
    #convert to nx graphs
    # G = to_networkx(graph, to_undirected=True)
    #all_subgraphs = [context_subgraph] + target_subgraphs
    # plot_subgraphs(G, all_subgraphs)
    
    node_mask, edge_mask = create_masks(graph, context_subgraph, target_subgraphs, len(monomer_mask), n_patches)

    return node_mask, edge_mask


def metis2subgraphs(graph, n_patches, min_targets):
    G = to_networkx(graph, to_undirected=True)
    # apply metis algorithm to the graph
    # idea divide each monomer in two partitions, join two partitions from different monomers and use as a context subgraph
    # otherwise checkout these algorithms: https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/node_clustering.html#overlapping-communities
    # DOC: https://metis.readthedocs.io/en/latest/
    # contig=True ensures that the partitions are connected
    nparts = 6 # arbitrary choice, 6 seems a good number for the dataset considered
    parts = metis.part_graph(G, nparts=nparts, contig=True)[1]
    
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
    expanded_subgraphs = [expand_one_hop(G, subgraph) for subgraph in subgraphs if len(subgraph) > 0]

    # Ensure all subgraphs are connected components
    subgraphs = [sg for sg in expanded_subgraphs if nx.is_connected(G.subgraph(sg))]
    
    context_subgraph = set()
    monomer_mask = graph.monomer_mask
    monomer1_nodes = set([node for node, monomer in enumerate(monomer_mask) if monomer == 0])
    monomer2_nodes = set([node for node, monomer in enumerate(monomer_mask) if monomer == 1])
    # join two partitions from different monomers to form the context subgraph
    # reqs: 
    # 1. process is stochastic (random)
    # 2. the joined subgraphs should be neighboring (i.e share some nodes or be connected by inter subgraph edges)
    # 3. the joined subgraphs should have at least one node from each monomer
    while not context_subgraph:
        subgraph1 = random.choice(subgraphs)
        subgraph2 = random.choice(subgraphs)
        if (subgraph1.intersection(monomer1_nodes) and subgraph2.intersection(monomer2_nodes)) or (subgraph1.intersection(monomer2_nodes) and subgraph2.intersection(monomer1_nodes)):
            if subgraph1.intersection(subgraph2) or any(G.has_edge(node1, node2) for node1 in subgraph1 for node2 in subgraph2):
                context_subgraph = subgraph1.union(subgraph2)
                break
    
    # use the remaining subgraphs as target subgraphs
    target_subgraphs = [subgraph for subgraph in subgraphs if subgraph not in [subgraph1, subgraph2]]

    # if target subgraphs is smaller than min_targets, add random subgraphs to reach the minimum
    # take a non context node, and do a 1-hop expansion
    # this happens in many instances
    # [TODO]: Keep track of how many times this happens, and if it happens too often, consider a different approach
    if len(target_subgraphs) < min_targets:
        #print("tooLittleTargets")
        set_target_subgraphs = set(frozenset(subgraph) for subgraph in target_subgraphs)
        list_possible_nodes = list(monomer1_nodes.union(monomer2_nodes) - context_subgraph) 

        if not list_possible_nodes: 
            list_possible_nodes = list(monomer1_nodes.union(monomer2_nodes))
            #print("fullContexts")
            
        while len(target_subgraphs) < min_targets:
            # pick a random non context node
            random_node = random.choice(list_possible_nodes)
            # expand the node by one hop
            new_subgraph = expand_one_hop(G, {random_node})
            # check if subgraph is not already in the target subgraphs
            if frozenset(new_subgraph) not in set_target_subgraphs:
                target_subgraphs.append(new_subgraph)

    context_subgraph = list(context_subgraph)
    target_subgraphs = [list(subgraph) for subgraph in target_subgraphs]

    # Plotting
    # all_subgraphs = [context_subgraph] + target_subgraphs
    # plot_subgraphs(G, all_subgraphs)

    node_mask, edge_mask = create_masks(graph, context_subgraph, target_subgraphs, len(parts), n_patches)
    return node_mask, edge_mask


def randomWalks2subgraphs(graph, n_patches, min_targets): 
    # [TODO]: Consider edge probabilities (?)
    # Function to perform a single random walk step from a given node
    def random_walk_step(fullGraph, current_node, exclude_nodes):
            neighbors = list(set(fullGraph.neighbors(current_node)) - exclude_nodes)
            return random.choice(neighbors) if neighbors else None
    
    # Function to perform a random walk from a given node
    def random_walk_from_node(fullGraph, start_node, exclude_nodes, total_nodes, size=0.2):
        walk = [start_node]
        while len(walk) / total_nodes < size:
            next_node = random_walk_step(fullGraph=fullGraph, current_node=walk[-1], exclude_nodes=exclude_nodes)
            if next_node:
                walk.append(next_node)
            else:
                break
        return walk
    
    # reqs:
    # 1. use random walks
    # 2. context subgraph should include elements from both monomers
    # 3. every edge of the original graph should be in at least one subgraph (i.e. no edge loss)
    # 4. the context subgraph should be around 50% of the original graph size
    # 5. the target subgraphs should be around 20% of the original graph size

    # randomly pick one intermonomer bond
    intermonomer_bond = random.choice(graph.intermonomers_bonds)
    monomer1root, monomer2root = intermonomer_bond
    # rw includes the two nodes from the different monomers
    context_rw_walk = {monomer1root, monomer2root}
    total_nodes = len(graph.monomer_mask)
    # consider the two monomers alone
    monomer1nodes = [node for node, monomer in enumerate(graph.monomer_mask) if monomer == 0]
    monomer2nodes = [node for node, monomer in enumerate(graph.monomer_mask) if monomer == 1]
    G = to_networkx(graph, to_undirected=True)
    monomer1G = G.subgraph(monomer1nodes)
    monomer2G = G.subgraph(monomer2nodes)

    sizeContext = 0.6
    # do a random walk in each monomer starting from the root node
    lastM1Node = monomer1root
    lastM2Node = monomer2root
    while len(context_rw_walk)/total_nodes <= sizeContext:
        if len(context_rw_walk) % 2 == 0:  # Even steps, expand from monomer1
            next_node = random_walk_step(fullGraph=monomer1G, current_node=lastM1Node, exclude_nodes=context_rw_walk)
        else: # Odd steps, expand from monomer2
            next_node = random_walk_step(fullGraph=monomer2G, current_node=lastM2Node, exclude_nodes=context_rw_walk)

        if next_node:
            if len(context_rw_walk) % 2 == 0:
                lastM1Node = next_node
            else:
                lastM2Node = next_node
            context_rw_walk.add(next_node)
        else:
            break
    
    # add random nodes until reaching desired context subgraph size
    # expansion happens randomly without considering the monomers
    counter = 0
    while len(context_rw_walk)/total_nodes <= sizeContext:
        # pick a random node from the context walk 
        random_node = random.choice(list(context_rw_walk))
        next_node = random_walk_step(fullGraph=G, current_node=random_node, exclude_nodes=context_rw_walk)
        if next_node is not None:
            counter = 0
            context_rw_walk.add(next_node)
        else:
            counter += 1
            if counter > 30:
                # print("Could not reach desired context subgraph size, stopping...")
                break

    # target random walks:
    remaining_nodes = list(set(G.nodes()) - context_rw_walk)
    target_rw_walks = [] # list of sets, each set is a random walk

    # to prevent node or edge loss, start a rw from each remaining node and expand it by one hop
    # updating the excluded nodes by adding every node already visited prevents rws that are extremly overlapping
    # and similar without extra computation. 
    exclude_nodes = deepcopy(context_rw_walk)
    for node in remaining_nodes:
        target_rw_walk = random_walk_from_node(fullGraph=G, start_node=node, exclude_nodes=exclude_nodes, total_nodes=total_nodes)
        if target_rw_walk:
            exclude_nodes.update(target_rw_walk)
            expanded_rw = expand_one_hop(G, target_rw_walk)
            target_rw_walks.append(expanded_rw)
    
    # remove duplicated random walks, the order of the nodes in the walk does not matter
    unique_target_rws = []
    for walk in target_rw_walks:
        if walk not in unique_target_rws:
            unique_target_rws.append(walk)
        # else:
            #print("Duplicated random walk found")

    # [TODO]: Add method to get to min targets if not enough

    target_rw_walks = unique_target_rws
    
    context_subgraph = list(context_rw_walk)
    target_subgraphs = [list(rw) for rw in target_rw_walks]


    # Plotting
    # subgraphs = [context_subgraph] + target_subgraphs
    # plot_subgraphs(G, subgraphs)

    node_mask, edge_mask = create_masks(graph, context_subgraph, target_subgraphs, total_nodes, n_patches)
    return node_mask, edge_mask
    

def create_masks(graph, context_subgraph, target_subgraphs, n_of_nodes, n_patches):
    # create always a fixed number of patches, the non existing patches will have all the nodes masked
    node_mask = torch.zeros((n_patches, n_of_nodes), dtype=torch.bool)
    # actual subgraphs 
    valid_subgraphs = [context_subgraph] + target_subgraphs
    start_idx = n_patches - len(valid_subgraphs) # 20 - 9 = 11: 11, 12, 13, 14, 15, 16, 17, 18, 19 (index range is 0-19, so we are good)
    # context mask
    # for node in context_subgraph:
    #     node_mask[start_idx, node] = True
    context_mask = torch.zeros(node_mask.shape[1], dtype=torch.bool)
    context_mask[context_subgraph] = True
    node_mask[start_idx] = context_mask
    
    # target masks
    idx = start_idx + 1
    for target_subgraph in target_subgraphs:
        target_mask = torch.zeros(node_mask.shape[1], dtype=torch.bool)
        target_mask[target_subgraph] = True
        node_mask[idx] = target_mask
        idx += 1

    edge_mask = node_mask[:, graph.edge_index[0]] & node_mask[:, graph.edge_index[1]]
    return node_mask, edge_mask


def plot_subgraphs(G, subgraphs):
    
    # Calculate the number of rows needed to display all subgraphs with up to 3 per row
    num_rows = math.ceil(len(subgraphs) / 3)
    fig, axes = plt.subplots(num_rows, min(3, len(subgraphs)), figsize=(10, 3 * num_rows))  # Adjust size as needed

    # Flatten the axes array for easy iteration in case of a single row
    if num_rows == 1:
        axes = np.array([axes]).flatten()
    else:
        axes = axes.flatten()

    for ax, subgraph in zip(axes, subgraphs):
        color_map = ['orange' if node in subgraph else 'lightgrey' for node in G.nodes()]
        pos = nx.spring_layout(G, seed=42)  # Fixed seed for consistent layouts across subplots
        nx.draw(G, pos=pos, ax=ax, with_labels=True, node_color=color_map, font_weight='bold')
        ax.set_title(f'Subgraph')

    # If there are more axes than subgraphs, hide the extra axes
    for i in range(len(subgraphs), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# Expand the given set of nodes with their one-hop neighbors
def expand_one_hop(fullG, subgraph_nodes):
    expanded_nodes = set(subgraph_nodes)
    for node in subgraph_nodes:
        expanded_nodes.update(fullG.neighbors(node))
    return expanded_nodes

# # if target subgraphs is smaller than min_targets, add random subgraphs to reach the minimum
    # if len(unique_target_rws) < min_targets:
    #     #print("tooLittleTargets")
    #     while len(unique_target_rws) < min_targets:
    #         # pick a random node from the remaining nodes
    #         random_node = random.choice(remaining_nodes)
    #         # expand the node by one hop
    #         new_subgraph = set([random_node])
    #         new_subgraph = expand_one_hop(G, new_subgraph)
    #         # check if subgraph is not already in the target subgraphs
    #         if new_subgraph not in unique_target_rws:
    #             unique_target_rws.append(new_subgraph)