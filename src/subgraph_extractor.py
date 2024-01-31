import random
import torch


def motifs2subgraphs(graph, cliques, cliques_edges, intermonomers_bonds, monomer_mask):
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
   