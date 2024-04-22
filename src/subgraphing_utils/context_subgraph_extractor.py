from copy import deepcopy
import math
from matplotlib import pyplot as plt
# import metis
import networkx as nx
import numpy as np
import random
from src.visualize import plot_subgraphs
import torch
from torch_geometric.utils import to_networkx
from torch_sparse import SparseTensor  # for propagation


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
    # G = to_networkx(graph, to_undirected=True)

    # # Create the line graph from G
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
    
    # context_subgraph = set()
    # monomer_mask = graph.monomer_mask
    # monomer1_nodes = set([node for node, monomer in enumerate(monomer_mask) if monomer == 0])
    # monomer2_nodes = set([node for node, monomer in enumerate(monomer_mask) if monomer == 1])
    # # join two partitions from different monomers to form the context subgraph
    # # reqs: 
    # # 1. process is stochastic (random)
    # # 2. the joined subgraphs should be neighboring (i.e share some nodes or be connected by inter subgraph edges)
    # # 3. the joined subgraphs should have at least one node from each monomer
    # context_subgraphs_used = []
    # while not context_subgraph:
    #     picked_subgraphs = random.sample(subgraphs, 2)
    #     subgraph1 = picked_subgraphs[0]
    #     subgraph2 = picked_subgraphs[1]

    #     if (subgraph1.intersection(monomer1_nodes) and subgraph2.intersection(monomer2_nodes)) or (subgraph1.intersection(monomer2_nodes) and subgraph2.intersection(monomer1_nodes)):
    #         if subgraph1.intersection(subgraph2) or any(G.has_edge(node1, node2) for node1 in subgraph1 for node2 in subgraph2):
    #             context_subgraph = subgraph1.union(subgraph2)
    #             context_subgraphs_used.append(subgraph1)
    #             context_subgraphs_used.append(subgraph2)
    #             subgraphs.remove(subgraph1)
    #             subgraphs.remove(subgraph2)
    #             break
    
    # while len(context_subgraph) / len(monomer_mask) < sizeContext and subgraphs:
    #     random_subgraph = random.choice(subgraphs)
    #     context_subgraph = context_subgraph.union(random_subgraph)
    #     context_subgraphs_used.append(random_subgraph)
    #     subgraphs.remove(random_subgraph)
        
    
    # # use the remaining subgraphs as target subgraphs
    # target_subgraphs = subgraphs
    # # print(context_subgraphs_used)
    # # print(target_subgraphs)

    # # if target subgraphs is smaller than min_targets, add random subgraphs to reach the minimum
    # # take a non context node, and do a 1-hop expansion
    # # this happens in many instances

    # if len(target_subgraphs) < min_targets:
    #     set_target_subgraphs = set(frozenset(subgraph) for subgraph in target_subgraphs)
    #     list_possible_nodes = list(monomer1_nodes.union(monomer2_nodes) - context_subgraph) 

    #     if not list_possible_nodes: 
    #         list_possible_nodes = list(monomer1_nodes.union(monomer2_nodes))
            
    #     while len(target_subgraphs) < min_targets:
    #         # pick a random non context node if possible
    #         random_node = random.choice(list_possible_nodes)
    #         # expand the node by one hop
    #         new_subgraph = expand_one_hop(G, {random_node})
    #         # check if subgraph is not already in the target subgraphs
    #         if frozenset(new_subgraph) not in set_target_subgraphs:
    #             target_subgraphs.append(new_subgraph)


    # context_subgraph = list(context_subgraph)
    # target_subgraphs = [list(subgraph) for subgraph in target_subgraphs]
    
    # # append the context subgraphs to the target subgraphs at the beginning
    # for subgraph in context_subgraphs_used:
    #     target_subgraphs.insert(0, list(subgraph))


    # # Plotting
    # # all_subgraphs = [context_subgraph] + target_subgraphs
    # # plot_subgraphs(G, all_subgraphs)

    # node_mask, edge_mask = create_masks(graph, context_subgraph, target_subgraphs, graph.num_nodes, n_patches)
    # return node_mask, edge_mask, context_subgraphs_used
    pass


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



# def metisContext(graph, sizeContext=0.7):
#     if sizeContext == 1:
#         # return all nodes and edges
#         return torch.ones((1, graph.num_nodes), dtype=torch.bool), torch.ones((1, graph.num_edges), dtype=torch.bool)
    
#     G = to_networkx(graph, to_undirected=True)
#     nparts = 7 # arbitrary choice, 6 seems a good number for the dataset considered
#     parts = metis.part_graph(G, nparts=nparts, contig=True)[1]

#     subgraphs = [set(node for node, part in enumerate(parts) if part == i) for i in range(nparts)]    
#     expanded_subgraphs = [expand_one_hop(G, subgraph) for subgraph in subgraphs if len(subgraph) > 0]

#     # Ensure all subgraphs are connected components
#     subgraphs = [sg for sg in expanded_subgraphs if nx.is_connected(G.subgraph(sg))]
    
#     context_subgraph = set()
#     monomer_mask = graph.monomer_mask
#     monomer1_nodes = set([node for node, monomer in enumerate(monomer_mask) if monomer == 0])
#     monomer2_nodes = set([node for node, monomer in enumerate(monomer_mask) if monomer == 1])
  
#     while not context_subgraph:
#         subgraph1 = random.choice(subgraphs)
#         subgraph2 = random.choice(subgraphs)
#         if (subgraph1.intersection(monomer1_nodes) and subgraph2.intersection(monomer2_nodes)) or (subgraph1.intersection(monomer2_nodes) and subgraph2.intersection(monomer1_nodes)):
#             if subgraph1.intersection(subgraph2) or any(G.has_edge(node1, node2) for node1 in subgraph1 for node2 in subgraph2):
#                 context_subgraph = subgraph1.union(subgraph2)
#                 break
    
#     # add a metis subgraph until desired size, if any subgraph is available
#     while len(context_subgraph) / len(monomer_mask) < sizeContext and subgraphs:
#         random_subgraph = random.choice(subgraphs)
#         context_subgraph = context_subgraph.union(random_subgraph)
#         subgraphs.remove(random_subgraph)
        

#     node_mask = torch.zeros((1, len(monomer_mask)), dtype=torch.bool)
#     node_mask[0, list(context_subgraph)] = True
#     edge_mask = node_mask[:, graph.edge_index[0]] & node_mask[:, graph.edge_index[1]]
#     return node_mask, edge_mask


# def motifs2subgraphs(graph, n_patches, min_targets):
#     cliques, intermonomers_bonds, monomer_mask = graph.motifs[0], graph.intermonomers_bonds, graph.monomer_mask
#     # [TODO]: Check requirements (size of context and size of targets, etc.)

#     # randomly pick one intermonomer bond
#     intermonomer_bond = random.choice(intermonomers_bonds)
    
#     # create a list of cliques for each monomer
#     monomer_cliques = [[], []]
#     for clique in cliques:
#         monomer_cliques[monomer_mask[clique[0]]].append(clique)
    
#     # randomly pick one clique from each monomer the clique must be one of those that contain a node of the intermonomer bond
#     monomerA_clique = random.choice([clique for clique in monomer_cliques[monomer_mask[intermonomer_bond[0]]] if intermonomer_bond[0] in clique])
#     monomerB_clique = random.choice([clique for clique in monomer_cliques[monomer_mask[intermonomer_bond[1]]] if intermonomer_bond[1] in clique])
    
#     # context subgraph: join 2 cliques such that they belong to different monomers and they are connected by an intermonomer bond
#     context_subgraph = monomerA_clique + monomerB_clique
    
#     # select all other cliques (except the two chosen above) as target subgraphs
#     target_subgraphs = [clique for clique in cliques if clique not in [monomerA_clique, monomerB_clique]]

#     # if target subgraphs is smaller than min_targets, add random subgraphs to reach the minimum
#     if len(target_subgraphs) < min_targets:
#         # available nodes are all nodes that are not in the context subgraph
#         available_nodes = set(range(len(monomer_mask))) - set(context_subgraph)
#         if not available_nodes:
#             available_nodes = set(range(len(monomer_mask)))
        
#         G = to_networkx(graph, to_undirected=True)
#         while len(target_subgraphs) < min_targets: # RISK infinite loop, but it should not happen
#             # pick a random node from the available nodes
#             random_node = random.choice(list(available_nodes))
#             # expand the node by one hop
#             new_subgraph = [random_node]
#             new_subgraph = expand_one_hop(G, new_subgraph)
#             # check if subgraph is not already in the target subgraphs
#             if new_subgraph not in target_subgraphs:
#                 target_subgraphs.append(list(new_subgraph))
    

#     # Plotting
#     #convert to nx graphs
#     # G = to_networkx(graph, to_undirected=True)
#     #all_subgraphs = [context_subgraph] + target_subgraphs
#     # plot_subgraphs(G, all_subgraphs)
    
#     node_mask, edge_mask = create_masks(graph, context_subgraph, target_subgraphs, len(monomer_mask), n_patches)

#     return node_mask, edge_mask



# def randomWalks2subgraphs(graph, n_patches, min_targets): 
#     # [TODO]: Consider edge probabilities (?)
#     # Function to perform a single random walk step from a given node
#     def random_walk_step(fullGraph, current_node, exclude_nodes):
#             neighbors = list(set(fullGraph.neighbors(current_node)) - exclude_nodes)
#             return random.choice(neighbors) if neighbors else None
    
#     # Function to perform a random walk from a given node
#     def random_walk_from_node(fullGraph, start_node, exclude_nodes, total_nodes, size=0.2):
#         walk = [start_node]
#         while len(walk) / total_nodes < size:
#             next_node = random_walk_step(fullGraph=fullGraph, current_node=walk[-1], exclude_nodes=exclude_nodes)
#             if next_node:
#                 walk.append(next_node)
#             else:
#                 break
#         return walk
    
#     # reqs:
#     # 1. use random walks
#     # 2. context subgraph should include elements from both monomers
#     # 3. every edge of the original graph should be in at least one subgraph (i.e. no edge loss)
#     # 4. the context subgraph should be around 50% of the original graph size
#     # 5. the target subgraphs should be around 20% of the original graph size

#     # randomly pick one intermonomer bond
#     intermonomer_bond = random.choice(graph.intermonomers_bonds)
#     monomer1root, monomer2root = intermonomer_bond
#     # rw includes the two nodes from the different monomers
#     context_rw_walk = {monomer1root, monomer2root}
#     total_nodes = len(graph.monomer_mask)
#     # consider the two monomers alone
#     monomer1nodes = [node for node, monomer in enumerate(graph.monomer_mask) if monomer == 0]
#     monomer2nodes = [node for node, monomer in enumerate(graph.monomer_mask) if monomer == 1]
#     G = to_networkx(graph, to_undirected=True)
#     monomer1G = G.subgraph(monomer1nodes)
#     monomer2G = G.subgraph(monomer2nodes)

#     sizeContext = 0.85
#     # do a random walk in each monomer starting from the root node
#     lastM1Node = monomer1root
#     lastM2Node = monomer2root
#     while len(context_rw_walk)/total_nodes <= sizeContext:
#         if len(context_rw_walk) % 2 == 0:  # Even steps, expand from monomer1
#             next_node = random_walk_step(fullGraph=monomer1G, current_node=lastM1Node, exclude_nodes=context_rw_walk)
#         else: # Odd steps, expand from monomer2
#             next_node = random_walk_step(fullGraph=monomer2G, current_node=lastM2Node, exclude_nodes=context_rw_walk)

#         if next_node:
#             if len(context_rw_walk) % 2 == 0:
#                 lastM1Node = next_node
#             else:
#                 lastM2Node = next_node
#             context_rw_walk.add(next_node)
#         else:
#             break
    
#     # add random nodes until reaching desired context subgraph size
#     # expansion happens randomly without considering the monomers
#     counter = 0
#     while len(context_rw_walk)/total_nodes <= sizeContext:
#         # pick a random node from the context walk 
#         random_node = random.choice(list(context_rw_walk))
#         next_node = random_walk_step(fullGraph=G, current_node=random_node, exclude_nodes=context_rw_walk)
#         if next_node is not None:
#             counter = 0
#             context_rw_walk.add(next_node)
#         else:
#             counter += 1
#             if counter > 30:
#                 # print("Could not reach desired context subgraph size, stopping...")
#                 break

#     # target random walks:
#     remaining_nodes = list(set(G.nodes()) - context_rw_walk)
#     target_rw_walks = [] # list of sets, each set is a random walk

#     # to prevent node or edge loss, start a rw from each remaining node and expand it by one hop
#     # updating the excluded nodes by adding every node already visited prevents rws that are extremly overlapping
#     # and similar without extra computation. 
#     exclude_nodes = deepcopy(context_rw_walk)
#     for node in remaining_nodes:
#         target_rw_walk = random_walk_from_node(fullGraph=G, start_node=node, exclude_nodes=exclude_nodes, total_nodes=total_nodes)
#         if target_rw_walk:
#             exclude_nodes.update(target_rw_walk)
#             expanded_rw = expand_one_hop(G, target_rw_walk)
#             target_rw_walks.append(expanded_rw)
    
#     # remove duplicated random walks, the order of the nodes in the walk does not matter
#     unique_target_rws = []
#     for walk in target_rw_walks:
#         if walk not in unique_target_rws:
#             unique_target_rws.append(walk)
#         # else:
#             #print("Duplicated random walk found")

#     # If n of target subgraphs is smaller than min_targets, add random subgraphs to reach the minimum
#     # take a non context node, and do a 1-hop expansion
#     if len(unique_target_rws) < min_targets:
#         #print("tooLittleTargets")
#         availableNodes = remaining_nodes
#         if not availableNodes:
#             availableNodes = list(G.nodes())
#             #print("fullContexts")
#         while len(unique_target_rws) < min_targets:
#             # pick a random node from the remaining nodes
#             random_node = random.choice(remaining_nodes)
#             # expand the node by one hop
#             new_subgraph = set([random_node])
#             new_subgraph = expand_one_hop(G, new_subgraph)
#             # check if subgraph is not already in the target subgraphs
#             if new_subgraph not in unique_target_rws:
#                 unique_target_rws.append(new_subgraph)

#     target_rw_walks = unique_target_rws
    
#     context_subgraph = list(context_rw_walk)
#     target_subgraphs = [list(rw) for rw in target_rw_walks]


#     # Plotting
#     # subgraphs = [context_subgraph] + target_subgraphs
#     # plot_subgraphs(G, subgraphs)

#     node_mask, edge_mask = create_masks(graph, context_subgraph, target_subgraphs, total_nodes, n_patches)
#     return node_mask, edge_mask


# def newImprovedSubgraphing(graph, n_patches, min_targets): 
#     # Function to perform a single random walk step from a given node
#     def random_walk_step(fullGraph, current_node, exclude_nodes):
#             neighbors = list(set(fullGraph.neighbors(current_node)) - exclude_nodes)
#             return random.choice(neighbors) if neighbors else None
    
#     # Function to perform a random walk from a given node
#     def random_walk_from_node(fullGraph, start_node, exclude_nodes, total_nodes, size=0.2):
#         walk = [start_node]
#         while len(walk) / total_nodes < size:
#             next_node = random_walk_step(fullGraph=fullGraph, current_node=walk[-1], exclude_nodes=exclude_nodes)
#             if next_node:
#                 walk.append(next_node)
#             else:
#                 break
#         return walk
    
#     # reqs:
#     # 1. use random walks
#     # 2. context subgraph should include elements from both monomers
#     # 3. every edge of the original graph should be in at least one subgraph (i.e. no edge loss)
#     # 4. the context subgraph should be around 50% of the original graph size
#     # 5. the target subgraphs should be around 20% of the original graph size  
#     G = to_networkx(graph, to_undirected=True)
#     total_nodes = len(graph.monomer_mask)

#     # Initialize context and target subgraphs
#     context_nodes = set()  # Now using a set for all context nodes
#     all_possible_target_subgraphs = []  # Will remain as a list of sets

#     # 1. Start with intermonomer bonds
#     for bond in graph.intermonomers_bonds:
#         # Add both nodes of the bond to the context
#         context_nodes.update(bond)
    
#     # take a random node from context and expand it by one hop
#     # this is to avoid edge loss and at the same time to avoid too much overlap or little diversity between the subgraphs
#     # random_context_node = random.choice(list(context_nodes))
#     # context_nodes.update(expand_one_hop(G, {random_context_node}))
        
#     exclude_nodes = set(context_nodes)
#     # remaining_nodes = list(set(G.nodes()) - exclude_nodes)
#     # 2. Perform random walks to generate new subgraphs, excluding nodes already in context
#     while len(exclude_nodes) <= total_nodes:
#         remaining_nodes = list(set(G.nodes()) - exclude_nodes)
#         if not remaining_nodes:
#             break
#         start_node = random.choice(remaining_nodes)
#         rw_subgraph = random_walk_from_node(fullGraph=G, start_node=start_node, exclude_nodes=exclude_nodes, total_nodes=total_nodes, size=0.02)
#         rw_expanded = expand_one_hop(G, rw_subgraph)
#         exclude_nodes.update(rw_expanded)
#         all_possible_target_subgraphs.append(rw_expanded)
    
#     # all_subgraphs = [context_nodes] + all_possible_target_subgraphs
#     # plot_subgraphs(G, all_subgraphs)

#     # pick randomly 4 target subgraphs, put the nodes in the remaining subgraphs not selected as targets, inside the context
#     if len(all_possible_target_subgraphs) < min_targets:
#         # print("tooLittleTargets")
#         remaining_nodes = list(set(G.nodes()) - context_nodes)
#         random.shuffle(remaining_nodes)
#         for _ in range(min_targets - len(all_possible_target_subgraphs)):
#             node = remaining_nodes.pop()
#             target_subgraph = expand_one_hop(G, {node})
#             all_possible_target_subgraphs.append(target_subgraph)

#     random.shuffle(all_possible_target_subgraphs)

#     # plt_context_subgraph = list(context_nodes)
#     # plt_all_possible_target_subgraphs = [list(subgraph) for subgraph in all_possible_target_subgraphs]

#     # subgraphs = [plt_context_subgraph] + plt_all_possible_target_subgraphs
#     # plot_subgraphs(G, subgraphs)


#     selected_target_subgraphs = all_possible_target_subgraphs[:min_targets]

#     # # THIS IS NEEDED TO ENSURE THE CONTEXT HAS ENOUGH ELEMENTS FROM BOTH MONOMERS
#     # make sure that selected_target_subgraphs contain elemetns from both monomers
#     monomer1nodes = {node for node, monomer in enumerate(graph.monomer_mask) if monomer == 0}
#     monomer2nodes = {node for node, monomer in enumerate(graph.monomer_mask) if monomer == 1}
#     flagMonomer1 = False
#     flagMonomer2 = False 
#     for subgraph in selected_target_subgraphs:
#         if subgraph.intersection(monomer1nodes):
#             flagMonomer1 = True
#         if subgraph.intersection(monomer2nodes):
#             flagMonomer2 = True
    
#     if not flagMonomer1:
#         # remove the first subgraph and add a random one till we have a subgraph that contains nodes from both monomers
#         temp = selected_target_subgraphs.pop(0)
#         while not flagMonomer1:
#             random_idx = random.choice(range(len(all_possible_target_subgraphs[min_targets:])))
#             random_subgraph = all_possible_target_subgraphs[min_targets:][random_idx]
#             if random_subgraph.intersection(monomer1nodes):
#                 selected_target_subgraphs.append(random_subgraph)
#                 all_possible_target_subgraphs[0] = random_subgraph
#                 all_possible_target_subgraphs[random_idx + min_targets] = temp
#                 flagMonomer1 = True
    
#     if not flagMonomer2:
#         # remove the first subgraph and add a random one till we have a subgraph that contains nodes from both monomers
#         temp = selected_target_subgraphs.pop(0)
#         while not flagMonomer2:
#             random_idx = random.choice(range(len(all_possible_target_subgraphs[min_targets:])))
#             random_subgraph = all_possible_target_subgraphs[min_targets:][random_idx]
#             if random_subgraph.intersection(monomer2nodes):
#                 selected_target_subgraphs.append(random_subgraph)
#                 all_possible_target_subgraphs[0] = random_subgraph
#                 all_possible_target_subgraphs[random_idx + min_targets] = temp
#                 flagMonomer2 = True

#     # this prevents any kind of edge loss but leads to higher overlap. It also make the context subgraph bigger (which could be good)
#     context_nodes = expand_one_hop(G, context_nodes)

#     # find a list of all nodes in selected_target_subgraphs
#     target_subgraphs_nodes = set()
#     for subgraph in selected_target_subgraphs:
#         target_subgraphs_nodes.update(subgraph)

#     # add the nodes of the remaining subgraphs to the context
#     for subgraph in all_possible_target_subgraphs[min_targets:]:
#         for node in subgraph:
#             if node not in target_subgraphs_nodes:
#                 context_nodes.add(node)
        
    
#     # Continue with your processing, such as creating masks or other operations based on context and targets
#     context_subgraph = list(context_nodes)
#     all_possible_target_subgraphs = [list(subgraph) for subgraph in all_possible_target_subgraphs]

#     # subgraphs = [context_subgraph] + all_possible_target_subgraphs
#     # plot_subgraphs(G, subgraphs)

#     node_mask, edge_mask = create_masks(graph, context_subgraph, all_possible_target_subgraphs, total_nodes, n_patches)
#     return node_mask, edge_mask

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





# def motifs2subgraphs_gjepa(graph, n_patches, min_targets):
#     cliques, intermonomers_bonds, monomer_mask = graph.motifs[0], graph.intermonomers_bonds, graph.monomer_mask
#     # [TODO]: Check requirements (size of context and size of targets, etc.)

#     # randomly pick one intermonomer bond
#     # 

#     # if target subgraphs is smaller than min_targets, add random subgraphs to reach the minimum
#     if len(cliques) < min_targets:
#         # available nodes are all nodes that are not in the context subgraph
#         available_nodes = set(range(len(monomer_mask))) - set(cliques)
#         if not available_nodes:
#             available_nodes = set(range(len(monomer_mask)))
        
#         G = to_networkx(graph, to_undirected=True)
#         while len(cliques) < min_targets: # RISK infinite loop, but it should not happen
#             # pick a random node from the available nodes
#             random_node = random.choice(list(available_nodes))
#             # expand the node by one hop
#             new_subgraph = [random_node]
#             new_subgraph = expand_one_hop(G, new_subgraph)
#             # check if subgraph is not already in the target subgraphs
#             if new_subgraph not in cliques:
#                 cliques.append(list(new_subgraph))
    

#     # Plotting
#     #convert to nx graphs
#     # G = to_networkx(graph, to_undirected=True)
#     #all_subgraphs = [context_subgraph] + target_subgraphs
#     # plot_subgraphs(G, all_subgraphs)
    
#     node_mask, edge_mask = create_masks(graph, context_subgraph, target_subgraphs, len(monomer_mask), n_patches)

#     return node_mask, edge_mask

# def metis_subgraph_gjepa(g, n_patches, drop_rate=0.0, num_hops=1, is_directed=False):
#     import torch
#     from torch_sparse import SparseTensor  # for propagation
#     import numpy as np
#     import metis
#     import torch_geometric
#     import networkx as nx
#     if is_directed:
#         if g.num_nodes < n_patches:
#             # assigns each node to its own partition
#             membership = torch.arange(g.num_nodes) # each node is assigned to its own partition
#         else:
#             # https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.to_networkx
#             G = torch_geometric.utils.to_networkx(g, to_undirected="lower") #  If set to "lower", the undirected graph will only correspond to the lower triangle of the input adjacency matrix.
#             cuts, membership = metis.part_graph(G, n_patches, recursive=True) #  n_patches= The target number of partitions. You might get fewer.
#             # i.e. membership[i] is the partition ID of node i
#     else:
#         if g.num_nodes < n_patches: # basically each node a different partition
#             # in this case membership is longer than g.num_nodes, but we only need the first g.num_nodes elements
#             membership = torch.randperm(n_patches) # torch.randperm(4) = tensor([2, 1, 0, 3])
#         else:
#             # data augmentation
#             # this is about dropping some edges in the grpah to ensure patches are different at each epoch
#             adjlist = g.edge_index.t()
#             arr = torch.rand(len(adjlist))
#             selected = arr > drop_rate
#             G = nx.Graph()
#             G.add_nodes_from(np.arange(g.num_nodes))
#             G.add_edges_from(adjlist[selected].tolist())
#             # metis partition
#             cuts, membership = metis.part_graph(G, n_patches, recursive=True)

#     assert len(membership) >= g.num_nodes 
#     # take only the first g.num_nodes elements and convert membership to tensor
#     membership = torch.tensor(np.array(membership[:g.num_nodes])) # i think this is useful in the randperm case above
#     max_patch_id = torch.max(membership)+1
#     # membership = tensor([0, 2, 1, 3]), max_patch_id = 4, n_patches = 32, membership+(n_patches-max_patch_id) = tensor([0, 2, 1, 3]) + 32-4 = tensor([28, 30, 29, 31])
#     # tensor([10, 19,  3, 30, 17,  1, 26, 16, 14, 15, 13, 21, 11, 28, 22, 24, 20,  2,
#     #      9,  6,  8, 23, 29, 27, 25])
#     old_membership = membership
#     membership = membership+(n_patches-max_patch_id)
#     # tensor([11, 20,  4, 31, 18,  2, 27, 17, 15, 16, 14, 22, 12, 29, 23, 25, 21,  3,
#     #     10,  7,  9, 24, 30, 28, 26])
    # # This stacks the list of tensors along a new dimension. The result is a 2D tensor, where each row 
    # # corresponds to a subgraph, and each column corresponds to a node. 
    # # The element at position (i, j) is True if node j belongs to subgraph i, and False otherwise.
    # # !!! the node mask has always n_patches rows !!!
    # node_mask = torch.stack([membership == i for i in range(n_patches)])
    # # in this case the node mask is a 32xN tensor, where N is the number of nodes in the graph
    # # in practice we have single node subgraphs in most of the instances, so each row in node_mask
    # # has all elements false but one of them which is true.
    # # each row of the tensor is a mask for a subgraph
    # # each column of the tensor is a node, and the element at position (i, j) is True if node j belongs to subgraph i, and False otherwise.
    # def k_hop_subgraph(edge_index, num_nodes, num_hops, is_directed=False):
    #     # return k-hop subgraphs for all nodes in the graph
    #     if is_directed:
    #         row, col = edge_index
    #         birow, bicol = torch.cat([row, col]), torch.cat([col, row])
    #         edge_index = torch.stack([birow, bicol])
    #     else:
    #         row, col = edge_index
    #     sparse_adj = SparseTensor(
    #         row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    #     # each one contains <= i hop masks
    #     hop_masks = [torch.eye(num_nodes, dtype=torch.bool,
    #                         device=edge_index.device)]
    #     hop_indicator = row.new_full((num_nodes, num_nodes), -1)
    #     hop_indicator[hop_masks[0]] = 0
    #     for i in range(num_hops):
    #         next_mask = sparse_adj.matmul(hop_masks[i].float()) > 0
    #         hop_masks.append(next_mask)
    #         hop_indicator[(hop_indicator == -1) & next_mask] = i+1
    #     hop_indicator = hop_indicator.T  # N x N
    #     node_mask = (hop_indicator >= 0)  # N x N dense mask matrix
    #     return node_mask

    # if num_hops > 0:
    #     subgraphs_batch, subgraphs_node_mapper = node_mask.nonzero().T
    #     k_hop_node_mask = k_hop_subgraph(
    #         g.edge_index, g.num_nodes, num_hops, is_directed)
    #     node_mask.index_add_(0, subgraphs_batch,
    #                             k_hop_node_mask[subgraphs_node_mapper])
        
    # # After this one hop expansion, each row in node_mask has multiple elements set to True
        
        
    # torch.set_printoptions(threshold=10_000)

    # # print(node_mask)
    # # quit(0)
    # # DIFFERENTLY from my code, most of the rows have some elements set to True, its only when there s a shift
    # # that the first n rows that were shifted will be false

    # # so eventually this code works like mine, only thign is that i have more empty subgraphs

    # # if not torch.equal(membership, old_membership):
    # #     print(old_membership)
    # #     print(membership)
    # #     print(node_mask)
    # #     quit(0)

    # edge_mask = node_mask[:, g.edge_index[0]] & node_mask[:, g.edge_index[1]]
    # return node_mask, edge_mask