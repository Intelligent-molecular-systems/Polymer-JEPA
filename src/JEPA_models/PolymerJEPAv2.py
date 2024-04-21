import numpy as np
from src.JEPA_models.model_utils.elements import MLP  
from src.JEPA_models.WDNodeMPNN import WDNodeMPNN
from src.visualize import plot_from_transform_attributes
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter


class PolymerJEPAv2(nn.Module):
    def __init__(self,
        nfeat_node, 
        nfeat_edge,
        nhid, 
        nlayer_gnn,
        rw_dim=0,
        patch_rw_dim=0,
        pooling='mean',
        num_target_patches=4,
        should_share_weights=False,
        regularization=False,
        shouldUse2dHyperbola=False,
        shouldUseNodeWeights=False
    ):
        
        super().__init__()

        self.pooling = pooling
        self.nhid = nhid
        self.num_target_patches = num_target_patches
        self.regularization = regularization

        self.rw_encoder = MLP(rw_dim, nhid, 1)
        self.patch_rw_encoder = MLP(patch_rw_dim, nhid, 1)

        self.input_encoder = nn.Linear(nfeat_node, nhid)

        # Context and Target Encoders are both WDNodeMPNN
        self.context_encoder = WDNodeMPNN(nhid, nfeat_edge, n_message_passing_layers=nlayer_gnn, hidden_dim=nhid, shouldUseNodeWeights=shouldUseNodeWeights)
        
        if regularization and should_share_weights:
            self.target_encoder = self.context_encoder
        else:
            self.target_encoder = WDNodeMPNN(nhid, nfeat_edge, n_message_passing_layers=nlayer_gnn, hidden_dim=nhid, shouldUseNodeWeights=shouldUseNodeWeights)
        
        self.shouldUse2dHyperbola = shouldUse2dHyperbola
        
        self.target_predictor = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.LayerNorm(nhid),
            nn.ReLU(),
            nn.Linear(nhid, 2 if self.shouldUse2dHyperbola else nhid)
        )
  
        # as suggested in JEPA original paper, we apply vicReg not directly on embeddings, but on the expanded embeddings
        # The role of the expander is twofold: (1) eliminate the information by which the two representations differ, (2) expand the dimension in a non-linear fashion so that decorrelating the embedding variables will reduce the dependencies (not just the correlations) between the variables of the representation vector.
        if self.regularization: 
            self.expander_dim = 256
            self.context_expander = nn.Sequential(
                nn.Linear(nhid, self.expander_dim),
                nn.LayerNorm(self.expander_dim),
                nn.ReLU(),
                nn.Linear(self.expander_dim, self.expander_dim)
            )

            self.target_expander = nn.Sequential(
                nn.Linear(nhid, self.expander_dim),
                nn.LayerNorm(self.expander_dim),
                nn.ReLU(),
                nn.Linear(self.expander_dim, self.expander_dim)
            )


    def forward(self, data):
        # Embed node features and edge attributes
        x = self.input_encoder(data.x).squeeze()
        x += self.rw_encoder(data.rw_pos_enc)
        # x = torch.cat([data.x, data.rw_pos_enc], dim=1)
        x = x[data.subgraphs_nodes_mapper]
        node_weights = data.node_weight[data.subgraphs_nodes_mapper]
        edge_index = data.combined_subgraphs # the new edge index is the one that consider the graph of disconnected subgraphs, with unique node indices       
        edge_attr = data.edge_attr[data.subgraphs_edges_mapper] # edge attributes again based on the subgraphs_edges_mapper, so we have the correct edge attributes for each subgraph
        edge_weights = data.edge_weight[data.subgraphs_edges_mapper]
        batch_x = data.subgraphs_batch # i.e. the subgraph idxs [0, 0, 1, 1, ...]

        ### JEPA - Context Encoder ###
        # initial encoder, encode all the subgraphs, then consider only the context subgraphs
        x = self.context_encoder(x, edge_index, edge_attr, edge_weights, node_weights)
        embedded_subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling) # batch_size*call_n_patches x nhid

        batch_indexer = torch.tensor(np.cumsum(data.call_n_patches)) # cumsum: return the cumulative sum of the elements along a given axis.
        batch_indexer = torch.hstack((torch.tensor(0), batch_indexer[:-1])).to(data.y_EA.device)

        context_subgraph_idx = data.context_subgraph_idx + batch_indexer
        embedded_context_x = embedded_subgraph_x[context_subgraph_idx] # Extract context subgraph embedding
        
        # Add its patch positional encoding
        # context_pe = data.patch_pe[context_subgraph_idx]
        # embedded_context_x += self.patch_rw_encoder(context_pe) #  modifying embedded_context_x after it is created from embedded_subgraph_x does not modify embedded_subgraph_x, because they do not share storage for their data.     
        vis_context_embedding = embedded_context_x.detach().clone() # for visualization
        embedded_context_x = embedded_context_x.unsqueeze(1)

        ### JEPA - Target Encoder ###
        # full graph nodes embedding (original full graph)
        full_x = self.input_encoder(data.x).squeeze()
        full_x += self.rw_encoder(data.rw_pos_enc)
        parameters = (full_x, data.edge_index, data.edge_attr, data.edge_weight, data.node_weight)

        if not self.regularization:
            # in case of EMA update to avoid collapse
            with torch.no_grad():
                # work on the original full graph
                full_graph_nodes_embedding = self.target_encoder(*parameters)
        else:
            # in case of vicReg to avoid collapse we have regularization
            full_graph_nodes_embedding = self.target_encoder(*parameters)

        with torch.no_grad():
            # pool the node embeddings to get the full graph embedding
            vis_graph_embedding = global_mean_pool(full_graph_nodes_embedding.detach().clone(), data.batch)
            
        # map it as we do for x at the beginning
        full_graph_nodes_embedding = full_graph_nodes_embedding[data.subgraphs_nodes_mapper]

        # pool the embeddings found for the full graph, this will produce the subgraphs embeddings for all subgraphs (context and target subgraphs)
        subgraphs_x_from_full = scatter(full_graph_nodes_embedding, batch_x, dim=0, reduce=self.pooling) # batch_size*call_n_patches x nhid

        # Compute the target indexes to find the target subgraphs embeddings
        target_subgraphs_idx = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs]).to(data.y_EA.device)
        target_subgraphs_idx += batch_indexer.unsqueeze(1)

        # n_context_nodes = [torch.sum(data.subgraphs_batch == idx).item() for idx in context_subgraph_idx]
        # print('n of nodes in the context_subgraph_idx:', n_context_nodes)

        # # Example for target subgraphs; adjust according to actual data structure
        # n_target_nodes = [torch.sum(data.subgraphs_batch == idx).item() for idx_list in target_subgraphs_idx for idx in idx_list]
        # print('n of nodes in the target_subgraphs_idx:', n_target_nodes)

        # for graph in data.to_data_list():
        #     plot_from_transform_attributes(graph)

        # target subgraphs nodes embedding
        # Construct context and target PEs frome the node pes of each subgraph
        embedded_target_x = subgraphs_x_from_full[target_subgraphs_idx.flatten()]

        embedded_target_x = embedded_target_x.reshape(-1, self.num_target_patches, self.nhid) # batch_size x num_target_patches x nhid
        vis_target_embeddings = embedded_target_x[:, 0, :].detach().clone() # for visualization

        expanded_context_embeddings = torch.tensor([]) # save the embeddings for regularization
        expanded_target_embeddings = torch.tensor([])
        if self.regularization: 
            input_context_x = embedded_context_x.reshape(-1, self.nhid)
            expanded_context_embeddings = self.context_expander(input_context_x)

            input_target_x = embedded_target_x[:, 0, :].reshape(-1, self.nhid) # take only the first patch to avoid overweighting the target embeddings
            expanded_target_embeddings = self.target_expander(input_target_x)

        if self.shouldUse2dHyperbola:
            x_coord = torch.cosh(embedded_target_x.mean(-1).unsqueeze(-1))
            y_coord = torch.sinh(embedded_target_x.mean(-1).unsqueeze(-1))
            embedded_target_x = torch.cat([x_coord, y_coord], dim=-1) # target_x shape: batch_size x num_target_patches x 2
        
        target_pes = data.patch_pe[target_subgraphs_idx.flatten()]
        encoded_tpatch_pes = self.patch_rw_encoder(target_pes)

        embedded_context_x_pe_conditioned = embedded_context_x + encoded_tpatch_pes.reshape(-1, self.num_target_patches, self.nhid) # B n_targets d
        predicted_target_embeddings = self.target_predictor(embedded_context_x_pe_conditioned)
        return embedded_target_x, predicted_target_embeddings, expanded_context_embeddings, expanded_target_embeddings,torch.tensor([], requires_grad=False, device=data.y_EA.device), torch.tensor([], requires_grad=False, device=data.y_EA.device), vis_context_embedding, vis_target_embeddings, vis_graph_embedding


    def encode(self, data):
        full_x = self.input_encoder(data.x).squeeze()

        if hasattr(data, 'rw_pos_enc'):
            full_x += self.rw_encoder(data.rw_pos_enc)
       
        node_embeddings = self.target_encoder(
            full_x, 
            data.edge_index, 
            data.edge_attr, 
            data.edge_weight, 
            data.node_weight
        )

        graph_embedding = global_mean_pool(node_embeddings, data.batch)
        return graph_embedding
    

     # torch.set_printoptions(threshold=10_000)
        # print(data.subgraphs_batch.shape)
        # print(data.subgraphs_nodes_mapper.shape)
        # quit()
        # x = data.x
        # with torch.no_grad():
        #     x = self.linearTry(x)
        # print('x before:', x.shape)
        # for i, row in enumerate(x):
        #     # Round each element in the row to 2 decimal places
        #     rounded_row = [round(element, 2) for element in row.tolist()]
        #     print(i, rounded_row)

        # print('data.subgraphs_nodes_mapper:', data.subgraphs_nodes_mapper)    
        # x = x[data.subgraphs_nodes_mapper]
        # print('x after:', x.shape)
        # for i, row in enumerate(x):
        #     # Again, round each element in the row to 2 decimal places
        #     rounded_row = [round(element, 2) for element in row.tolist()]
        #     print(i, rounded_row)


# from collections import Counter

        # # Step 1: Count the occurrences of each node in the mapper
        # node_occurrences = Counter(data.subgraphs_nodes_mapper.tolist())

        # # Step 2: Identify nodes that are repeated (appear more than once)
        # repeated_nodes = {node: count for node, count in node_occurrences.items() if count > 1}

        # print("Repeated Nodes and their counts:", repeated_nodes)

        # node_to_subgraphs = {node: [] for node in repeated_nodes.keys()}
        # for idx, node in enumerate(data.subgraphs_nodes_mapper.tolist()):
        #     if node in node_to_subgraphs:
        #         subgraph_idx = data.subgraphs_batch[idx].item()
        #         node_to_subgraphs[node].append(subgraph_idx)
        # i = 0
        # for node, subgraphs in node_to_subgraphs.items():
        #     if i < 4:
        #         print(f"Node {node} belongs to subgraphs: {subgraphs}")
        #     i += 1
        # print('target_subgraphs_idx:', target_subgraphs_idx)
        # # Debug print n of nodes in the context and target subgraphs (already with the new index keeping the batch
        # # indexer into account) they always match with the n of nodes
        # # we can see in the plot from the original datat
        # n_context_nodes = [torch.sum(data.subgraphs_batch == idx).item() for idx in context_subgraph_idx]
        # print('n of nodes in the context_subgraph_idx:', n_context_nodes)

        # # # Example for target subgraphs; adjust according to actual data structure
        # n_target_nodes = [torch.sum(data.subgraphs_batch == idx).item() for idx_list in target_subgraphs_idx for idx in idx_list]
        # print('n of nodes in the target_subgraphs_idx:', n_target_nodes)
        # for graph in data.to_data_list():
        #     # plot_from_transform_attributes(graph)
        #     # graph = data[0]
        #     import networkx as nx
        #     edge_index = graph.combined_subgraphs
        #     # plot 
        #     G_context = nx.Graph()
        #     G_context.add_edges_from(edge_index.T.cpu().numpy())
        #     nx.draw(G_context, with_labels=True, node_color='skyblue')
        #     import matplotlib.pyplot as plt
        #     plt.show()
        # print("\n\n")