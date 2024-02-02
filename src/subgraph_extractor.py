from matplotlib import pyplot as plt
import metis
import networkx as nx
import random
import torch
from torch_geometric.utils.convert import to_networkx

def motifs2subgraphs(graph):
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

    node_mask, edge_mask = create_masks(graph, context_subgraph, target_subgraphs, len(monomer_mask))

    # # sanity check: control whether the edge mask for the context subgraph is correct
    # for mask_val, edge in zip(edge_mask[0], graph.edge_index.T):
    #     # print mask_val without tensor() and edge without tensor()
    #     print(mask_val.item(), edge.tolist())

    # import networkx as nx
    # from torch_geometric.utils.convert import to_networkx
    # from matplotlib import pyplot as plt
    # G = to_networkx(graph, to_undirected=True)
    # # color context nodes 
    # color_map = []
    # for node in G:
    #     if node not in context_subgraph:
    #         color_map.append('green')
    #     else:
    #          color_map.append('red')
    # fig = nx.draw_networkx(G, node_color=color_map, with_labels=True)
    # plt.show()

    return node_mask, edge_mask


def metis2subgraphs(graph):
    G = to_networkx(graph, to_undirected=True)
    # apply metis algorithm to the graph
    # idea divide each monomer in two partitions, join two partitions from different monomers and use as a context subgraph
    # otherwise checkout these algorithms: https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/node_clustering.html#overlapping-communities
    # DOC: https://metis.readthedocs.io/en/latest/
    # contig=True ensures that the partitions are connected
    nparts = 6
    parts = metis.part_graph(G, nparts=nparts, contig=True)[1]
    # print(parts)

    # # plot and color the partitions
    # color_map = []
    # for node in G:
    #     if parts[node] == 0:
    #         color_map.append('green')
    #     elif parts[node] == 1:
    #         color_map.append('red')
    #     elif parts[node] == 2:
    #         color_map.append('blue')
    #     elif parts[node] == 3:
    #         color_map.append('yellow')
    #     elif parts[node] == 4:
    #         color_map.append('orange')
    #     elif parts[node] == 5:
    #         color_map.append('purple')
    #     else:
    #         color_map.append('black')
       
    # import networkx as nx
    # from matplotlib import pyplot as plt
    # fig = nx.draw_networkx(G, node_color=color_map, with_labels=True)
    # plt.show()    

    # perform a one-hop expansion of each partition to avoid edge loss
    # Create a subgraph for each partition
    subgraphs = [G.subgraph([node for node, part in enumerate(parts) if part == i]) for i in range(nparts)]
    # remove all empty subgraphs
    subgraphs = [subgraph for subgraph in subgraphs if subgraph.number_of_nodes() > 0]
    # Perform one-hop neighbor expansion for each partition
    # the one-hop expansion on such small and connected subgraphs cause a lot of overlap between at least some partitions
    # this could be non optimal for the prediction task, if the context and target share many nodes, its easy to predict..
    # but usually there are at 2/3 partitions with a small overlap so it should be good
    # alternatives are: 
    # 1. not to expand the subgraphs, 
    # 2. expand in a more sophisticated way (checking only the edges lost and including them in a single subgraph)
    # 3. use a different algorithm to partition the graph that already gives an overlap by default
    expanded_subgraphs = []
    for i in range(len(subgraphs)):
        partition_nodes = subgraphs[i].nodes()
        neighbors = set()
        for node in partition_nodes:
            neighbors.update(G.neighbors(node))
        expanded_subgraph_nodes = set(partition_nodes).union(neighbors)
        expanded_subgraphs.append(G.subgraph(expanded_subgraph_nodes))

    # check if all subgraphs are connected components
    for subgraph in expanded_subgraphs:
        assert nx.is_connected(subgraph)

    # create a list of all subgraphs where each subgraph is a set
    subgraphs = [set(subgraph.nodes()) for subgraph in expanded_subgraphs]

    # join two partitions from different monomers to form the context subgraph
    context_subgraph = None
    target_subgraphs = []

    monomer_mask = graph.monomer_mask

    monomer1_nodes = set([node for node, monomer in enumerate(monomer_mask) if monomer == 0])
    monomer2_nodes = set([node for node, monomer in enumerate(monomer_mask) if monomer == 1])

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
    
    # use the remaining subgraphs as target subgraphs
    target_subgraphs = [subgraph for subgraph in subgraphs if subgraph not in [subgraph1, subgraph2]]

    # print(context_subgraph)
    # print(target_subgraphs)
    # color_map = []
    # for node in G:
    #     if node in target_subgraphs[0]:
    #         color_map.append('green')
    #     else:
    #         color_map.append('yellow')
    # fig = nx.draw_networkx(G, node_color=color_map, with_labels=True)
    # plt.show()    

    context_subgraph = list(context_subgraph)
    target_subgraphs = [list(subgraph) for subgraph in target_subgraphs]
    node_mask, edge_mask = create_masks(graph, context_subgraph, target_subgraphs, len(parts))
    return node_mask, edge_mask



def randomWalks2subgraphs(graph): 
    # reqs:
    # 1. use random walks
    # 2. context subgraph should include elements from both monomers
    # 3. every edge of the original graph should be in at least one subgraph (i.e. no edge loss)
    # 4. the context subgraph should be around 50% of the original graph size
    # 5. the target subgraphs should be around 20% of the original graph size

    # randomly pick one intermonomer bond
    intermonomer_bond = random.choice(graph.intermonomers_bonds)
    monomer1root = intermonomer_bond[0]
    monomer2root = intermonomer_bond[1]
    context_rw_walk = [monomer1root, monomer2root]
    totat_nodes = len(graph.monomer_mask)
    # consider the two monomers alone
    monomer1nodes = [node for node, monomer in enumerate(graph.monomer_mask) if monomer == 0]
    monomer2nodes = [node for node, monomer in enumerate(graph.monomer_mask) if monomer == 1]
    monomer1G = to_networkx(graph, to_undirected=True).subgraph(monomer1nodes)
    monomer2G = to_networkx(graph, to_undirected=True).subgraph(monomer2nodes)

    # do a random walk in each monomer starting from the root node
    while len(context_rw_walk)/totat_nodes <= 0.55:
        if len(context_rw_walk) % 2 == 0:  # Even steps, expand from monomer1
            neighbors_monomer1 = monomer1G.neighbors(context_rw_walk[-2])
            # remove the nodes that are already in the walk 
            neighbors_monomer1 = list(set(neighbors_monomer1) - set(context_rw_walk))
            if neighbors_monomer1:
                next_node = random.choice(neighbors_monomer1)
                context_rw_walk.append(next_node)
            else:
                break
        
        else:
            neighbors_monomer2 = monomer2G.neighbors(context_rw_walk[-2])
            # remove the nodes that are already in the walk
            neighbors_monomer2 = list(set(neighbors_monomer2) - set(context_rw_walk))
            if neighbors_monomer2:
                next_node = random.choice(neighbors_monomer2)
                context_rw_walk.append(next_node)
            else:
                break
    

    # target random walks:
    # consider the nodes of the graph taht are not part of the context subgraph
    # start a random walk from each node that is not in the context subgraph
    # do a one-hop expansion of each of such random walks to avoid edge loss
    # remove random walks that are duplicated
    # each rw should be no more than 20% of the original graph size
    remaining_nodes = [node for node in range(totat_nodes) if node not in context_rw_walk]
    target_rw_walks = []
    G = to_networkx(graph, to_undirected=True)
    for node in remaining_nodes:
        target_rw_walk = [node]
        while len(target_rw_walk)/totat_nodes < 0.15:
            # neighbors should not be in the context subgraph
            neighbors = list(set(G.neighbors(target_rw_walk[-1])) - set(context_rw_walk))
            if neighbors:
                target_rw_walk.append(random.choice(neighbors))
            else:
                break

        # do a one-hop expansion of the random walk to avoid edge loss
        rw_neighbors = set()
        expanded_rw = set()
        for node in target_rw_walk:
            rw_neighbors.update(G.neighbors(node))
        expanded_rw = set(target_rw_walk).union(rw_neighbors)
        target_rw_walks.append(expanded_rw)
    
    # remove duplicated random walks, the order of the nodes in the walk does not matter
    for i, rw in enumerate(target_rw_walks):
        for rw2 in target_rw_walks[i+1:]:
            if rw == rw2:
                target_rw_walks.remove(rw2)
    
    context_subgraph = list(context_rw_walk)
    target_subgraphs = [list(rw) for rw in target_rw_walks]

    node_mask, edge_mask = create_masks(graph, context_subgraph, target_subgraphs, totat_nodes)
    return node_mask, edge_mask

    
def create_masks(graph, context_subgraph, target_subgraphs, n_of_nodes):
    n_of_subgraphs = 1 + len(target_subgraphs)
    node_mask = torch.zeros((n_of_subgraphs, n_of_nodes), dtype=torch.bool)

    # context mask
    for node in context_subgraph:
        node_mask[0, node] = True
    
    # target masks
    for i, target_subgraph in enumerate(target_subgraphs):
        for node in target_subgraph:
            node_mask[i+1, node] = True

    edge_mask = node_mask[:, graph.edge_index[0]] & node_mask[:, graph.edge_index[1]]
    return node_mask, edge_mask



# cliques, cliques_edges, intermonomers_bonds, monomer_mask =  graphs[0].motifs[0], graphs[0].motifs[1], graphs[0].intermonomers_bonds, graphs[0].monomer_mask
# node_mask, edge_mask = motifs2subgraphs(graphs[0], cliques, cliques_edges, intermonomers_bonds, monomer_mask)
# for i in range(15):
#     cliques, cliques_edges, intermonomers_bonds, monomer_mask =  graphs[i].motifs[0], graphs[i].motifs[1], graphs[i].intermonomers_bonds, graphs[i].monomer_mask
#     node_mask, edge_mask = motifs2subgraphs(graphs[i], cliques, cliques_edges, intermonomers_bonds, monomer_mask)
   