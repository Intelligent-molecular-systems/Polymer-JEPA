import random
import torch
from torch_geometric.utils import to_networkx


# random walk based context subgraphing
def rwContext(graph, sizeContext=0.85): 
    if sizeContext == 1:
        # return all nodes and edges
        return torch.ones((1, graph.num_nodes), dtype=torch.bool), torch.ones((1, graph.num_edges), dtype=torch.bool)
     
    # Function to perform a single random walk step from a given node
    def random_walk_step(fullGraph, current_node, exclude_nodes):
        neighbors = list(set(fullGraph.neighbors(current_node)) - exclude_nodes)
        return random.choice(neighbors) if neighbors else None
    
    rw1 = set()
    rw2 = set()
    # randomly pick one intermonomer bond
    intermonomer_bond = random.choice(graph.intermonomers_bonds)
    monomer1root, monomer2root = intermonomer_bond
    # rw includes the two nodes from the different monomers
    context_rw_walk = {monomer1root, monomer2root}
    # add the two nodes to the random walks
    rw1.add(monomer1root)
    rw1.add(monomer2root)
    rw2.add(monomer2root)
    rw2.add(monomer1root)
    total_nodes = len(graph.monomer_mask)
    # consider the two monomers alone
    monomer1nodes = [node for node, monomer in enumerate(graph.monomer_mask) if monomer == 0]
    monomer2nodes = [node for node, monomer in enumerate(graph.monomer_mask) if monomer == 1]
    G = to_networkx(graph, to_undirected=True)
    monomer1G = G.subgraph(monomer1nodes)
    monomer2G = G.subgraph(monomer2nodes)

    # do a random walk in each monomer starting from the root node
    lastM1Node = monomer1root
    lastM2Node = monomer2root
    

    while len(context_rw_walk)/total_nodes < sizeContext:
        if len(context_rw_walk) % 2 == 0:  # Even steps, expand from monomer1
            next_node = random_walk_step(fullGraph=monomer1G, current_node=lastM1Node, exclude_nodes=context_rw_walk)
        else: # Odd steps, expand from monomer2
            next_node = random_walk_step(fullGraph=monomer2G, current_node=lastM2Node, exclude_nodes=context_rw_walk)

        if next_node:
            if len(context_rw_walk) % 2 == 0:
                lastM1Node = next_node
                rw1.add(next_node)
            else:
                lastM2Node = next_node
                rw2.add(next_node)
                
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
            if next_node in monomer1nodes:
                rw1.add(next_node)
            elif next_node in monomer2nodes:
                rw2.add(next_node)
            else:
                print("Error: Random walk node not in monomer")
        else:
            counter += 1
            if counter > 30:
                # print("Could not reach desired context subgraph size, stopping...")
                break

    node_mask = torch.zeros((1, total_nodes), dtype=torch.bool)
    node_mask[0, list(context_rw_walk)] = True
    edge_mask = node_mask[:, graph.edge_index[0]] & node_mask[:, graph.edge_index[1]]
    return node_mask, edge_mask, rw1, rw2


# motif-based context subgraphing
def motifContext(graph, sizeContext=0.7, n_targets=4):
    if sizeContext == 1:
        # return all nodes and edges
        return torch.ones((1, graph.num_nodes), dtype=torch.bool), torch.ones((1, graph.num_edges), dtype=torch.bool)
    
    cliques, intermonomers_bonds, monomer_mask = graph.motifs[0].copy(), graph.intermonomers_bonds.copy(), graph.monomer_mask.clone()
    cliques_used = []
    context_nodes = set()

    # randomly pick one intermonomer bond
    intermonomer_bond = random.choice(intermonomers_bonds)
    
    # create a list of cliques for each monomer
    monomer_cliques = [[], []]
    for clique in cliques:
        # add the clique to the list of cliques of the monomer it belongs to
        monomer_cliques[monomer_mask[clique[0]]].append(clique)
    
    while len(intermonomers_bonds) > 0:
        intermonomer_bond = random.choice(intermonomers_bonds)
        # Find all cliques that contain the nodes of the intermonomer bond
        listA = [clique for clique in monomer_cliques[monomer_mask[intermonomer_bond[0]]] if intermonomer_bond[0] in clique]
        listB = [clique for clique in monomer_cliques[monomer_mask[intermonomer_bond[1]]] if intermonomer_bond[1] in clique]
        # If we found at least one clique for each monomer, break the loop
        if listA and listB:
            monomerA_clique = random.choice(listA)
            monomerB_clique = random.choice(listB)
            break
        else:
            intermonomers_bonds.remove(intermonomer_bond)
    
    # Add the 2 cliques to the context subgraph, we now have elements from both monomers
    cliques_used.append(monomerA_clique)
    cliques_used.append(monomerB_clique)
    context_nodes.update(monomerA_clique)
    context_nodes.update(monomerB_clique)

    # context subgraph: join 2 cliques such that they belong to different monomers and they are connected by an intermonomer bond
    context_subgraph = monomerA_clique + monomerB_clique
    # while length of context subgraph is less than context size % of the total nodes, add a random clique if any is available
    cliques.remove(monomerA_clique)
    cliques.remove(monomerB_clique)
    
    while len(context_nodes) / len(monomer_mask) < sizeContext and len(cliques) > n_targets:
        random_clique = random.choice([clique for clique in cliques])
        context_subgraph += random_clique
        cliques.remove(random_clique)
        cliques_used.append(random_clique)
        context_nodes.update(random_clique)

    node_mask = torch.zeros((1, len(monomer_mask)), dtype=torch.bool)
    node_mask[0, context_subgraph] = True
    edge_mask = node_mask[:, graph.edge_index[0]] & node_mask[:, graph.edge_index[1]]

    return node_mask, edge_mask, cliques_used


# metis-based context and target subgraphing
def metis2subgraphs(graph, n_patches, sizeContext, min_targets):
    G = to_networkx(graph, to_undirected=True)

    # use metis ot find subgraphs
    nparts = 7 # arbitrary choice, 6 seems a good number for the dataset considered
    parts = metis.part_graph(G, nparts=nparts, contig=True)[1]
    subgraphs = [set(node for node, part in enumerate(parts) if part == i) for i in range(nparts)]

    # eliminate empty subgraphs
    subgraphs = [sg for sg in subgraphs if sg]
    # Ensure all subgraphs are connected components
    # subgraphs = [sg for sg in expanded_subgraphs if nx.is_connected(G.subgraph(sg))]
    
    context_subgraph = set()
    monomer_mask = graph.monomer_mask
    monomer1_nodes = set([node for node, monomer in enumerate(monomer_mask) if monomer == 0])
    monomer2_nodes = set([node for node, monomer in enumerate(monomer_mask) if monomer == 1])
    # join two partitions from different monomers to form the context subgraph
    # reqs: 
    # 1. process is stochastic (random)
    # 2. the joined subgraphs should be neighboring (i.e share some nodes or be connected by inter subgraph edges)
    # 3. the joined subgraphs should have at least one node from each monomer
    context_subgraphs_used = []
    while not context_subgraph:
        picked_subgraphs = random.sample(subgraphs, 2)
        subgraph1 = picked_subgraphs[0]
        subgraph2 = picked_subgraphs[1]

        if (subgraph1.intersection(monomer1_nodes) and subgraph2.intersection(monomer2_nodes)) or (subgraph1.intersection(monomer2_nodes) and subgraph2.intersection(monomer1_nodes)):
            if subgraph1.intersection(subgraph2) or any(G.has_edge(node1, node2) for node1 in subgraph1 for node2 in subgraph2):
                context_subgraph = subgraph1.union(subgraph2)
                context_subgraphs_used.append(subgraph1)
                context_subgraphs_used.append(subgraph2)
                subgraphs.remove(subgraph1)
                subgraphs.remove(subgraph2)
                break
    
    while len(context_subgraph) / len(monomer_mask) < sizeContext and subgraphs:
        random_subgraph = random.choice(subgraphs)
        context_subgraph = context_subgraph.union(random_subgraph)
        context_subgraphs_used.append(random_subgraph)
        subgraphs.remove(random_subgraph)
        
    
    # use the remaining subgraphs as target subgraphs
    target_subgraphs = subgraphs
    # print(context_subgraphs_used)
    # print(target_subgraphs)

    # if target subgraphs is smaller than min_targets, add random subgraphs to reach the minimum
    # take a non context node, and do a 1-hop expansion
    # this happens in many instances

    if len(target_subgraphs) < min_targets:
        set_target_subgraphs = set(frozenset(subgraph) for subgraph in target_subgraphs)
        list_possible_nodes = list(monomer1_nodes.union(monomer2_nodes) - context_subgraph) 

        if not list_possible_nodes: 
            list_possible_nodes = list(monomer1_nodes.union(monomer2_nodes))
            
        while len(target_subgraphs) < min_targets:
            # pick a random non context node if possible
            random_node = random.choice(list_possible_nodes)
            # expand the node by one hop
            new_subgraph = expand_one_hop(G, {random_node})
            # check if subgraph is not already in the target subgraphs
            if frozenset(new_subgraph) not in set_target_subgraphs:
                target_subgraphs.append(new_subgraph)

    context_subgraph = list(context_subgraph)
    target_subgraphs = [list(subgraph) for subgraph in target_subgraphs]
    
    # append the context subgraphs to the target subgraphs at the beginning
    for subgraph in context_subgraphs_used:
        target_subgraphs.insert(0, list(subgraph))


    # Plotting
    # all_subgraphs = [context_subgraph] + target_subgraphs
    # plot_subgraphs(G, all_subgraphs)

    node_mask, edge_mask = create_masks(graph, context_subgraph, target_subgraphs, graph.num_nodes, n_patches)
    return node_mask, edge_mask, context_subgraphs_used


def create_masks(graph, context_subgraph, target_subgraphs, n_of_nodes, n_patches):#
    # create always a fixed number of patches, the non existing patches will have all the nodes masked
    node_mask = torch.zeros((n_patches, n_of_nodes), dtype=torch.bool)

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


# Expand the given set of nodes with their one-hop neighbors
def expand_one_hop(fullG, subgraph_nodes):
    expanded_nodes = set(subgraph_nodes)
    for node in subgraph_nodes:
        expanded_nodes.update(fullG.neighbors(node))
    return expanded_nodes