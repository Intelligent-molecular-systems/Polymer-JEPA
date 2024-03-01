import numpy as np
from src.model_utils.elements import MLP  
from src.WDNodeMPNN import WDNodeMPNN
from src.transform import plot_from_transform_attributes
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter


class PolymerJEPAv2(nn.Module):
    def __init__(self,
        nfeat_node, 
        nfeat_edge,
        nhid, 
        rw_dim=0,
        pooling='mean',
        num_target_patches=4,
        should_share_weights = False,
        regularization = False,
        n_hid_wdmpnn=300,
        shouldUse2dHyperbola=False,
        shouldLayerNorm=False
    ):
        
        super().__init__()

        self.pooling = pooling
        self.nhid = nhid
        self.num_target_patches = num_target_patches
        self.regularization = regularization

        self.rw_encoder = MLP(rw_dim, nhid, 1)

        # Context and Target Encoders are both WDNodeMPNN
        self.context_encoder = WDNodeMPNN(nfeat_node, nfeat_edge, n_message_passing_layers=3, hidden_dim=n_hid_wdmpnn)
        self.contextLinearTransform = nn.Linear(n_hid_wdmpnn, nhid)
        if regularization and should_share_weights:
            self.target_encoder = self.context_encoder
            self.targetLinearTransform = self.contextLinearTransform
        else:
            self.target_encoder = WDNodeMPNN(nfeat_node, nfeat_edge, n_message_passing_layers=3, hidden_dim=n_hid_wdmpnn)
            self.targetLinearTransform = nn.Linear(n_hid_wdmpnn, nhid)
        
        # Predictor MLP for the target subgraphs
        # according to g-jepa when using hyperbolic space, the target predictor should be a linear layer
        # otherwise representation collapse
        #self.target_predictor = nn.Linear(nhid, 2) # V2: Directly predict (depends on the definition of self.target_predictor)
        # Use this if you wish to do euclidean or poincaré embeddings in the latent space
        self.shouldUse2dHyperbola = shouldUse2dHyperbola
        # consider reshaping the input to batch_size*num_target_patches x nhid (while now is batch_size x num_target_patches x nhid) in order to use batch norm
        self.target_predictor = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.LayerNorm(nhid) if shouldLayerNorm else nn.BatchNorm1d(nhid),
            nn.ReLU(),
            nn.Linear(nhid, 2 if self.shouldUse2dHyperbola else nhid)
        )
  
        
        # as suggested in JEPA original paper, we apply vicReg not directly on embeddings, but on the expanded embeddings
        # The role of the expander is twofold: (1) eliminate the information by which the two representations differ, (2) expand the dimension in a non-linear fashion so that decorrelating the embedding variables will reduce the dependencies (not just the correlations) between the variables of the representation vector.
        if self.regularization: 
            self.expander_dim = 256
            self.context_expander = nn.Sequential(
                nn.Linear(nhid, self.expander_dim),
                nn.BatchNorm1d(self.expander_dim),
                nn.ReLU(),
                nn.Linear(self.expander_dim, self.expander_dim)
            )

            self.target_expander = nn.Sequential(
                nn.Linear(nhid, self.expander_dim),
                nn.BatchNorm1d(self.expander_dim),
                nn.ReLU(),
                nn.Linear(self.expander_dim, self.expander_dim)
            )
        

    def forward(self, data):
        x = data.x[data.subgraphs_nodes_mapper]
        # x = torch.rand(x.shape)
        node_weights = data.node_weight[data.subgraphs_nodes_mapper]
        # node_weights = torch.rand(node_weights.shape)

        # the new edge index is the one that consider the graph of disconnected subgraphs, with unique node indices
        edge_index = data.combined_subgraphs        
        # edge attributes again based on the subgraphs_edges_mapper, so we have the correct edge attributes for each subgraph
        edge_attr = data.edge_attr[data.subgraphs_edges_mapper]
        # edge_attr = torch.rand(edge_attr.shape)
        edge_weights = data.edge_weight[data.subgraphs_edges_mapper]
        # edge_weights = torch.rand(edge_weights.shape)

        # print('x:', data.x.shape)
        # x = data.x[data.subgraphs_nodes_mapper]
        # # print('x:', x.shape)
        # # quit()
        # node_weights = data.node_weight[data.subgraphs_nodes_mapper]

        # # the new edge index is the one that consider the graph of disconnected subgraphs, with unique node indices
        # edge_index = data.combined_subgraphs        
        # # edge attributes again based on the subgraphs_edges_mapper, so we have the correct edge attributes for each subgraph
        # edge_attr = data.edge_attr[data.subgraphs_edges_mapper]
        # edge_weights = data.edge_weight[data.subgraphs_edges_mapper]
        batch_x = data.subgraphs_batch # this is the batch of subgraphs, i.e. the subgraph idxs [0, 0, 1, 1, ...]
        pes = data.rw_pos_enc[data.subgraphs_nodes_mapper]
        patch_pes = scatter(pes, batch_x, dim=0, reduce='max') 
        # initial encoder, encode all the subgraphs, then consider only the context subgraphs
        x = self.context_encoder(x, edge_index, edge_attr, edge_weights, node_weights)
        x = self.contextLinearTransform(x)
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling) # batch_size*call_n_patches x nhid

        batch_indexer = torch.tensor(np.cumsum(data.call_n_patches)) # cumsum: return the cumulative sum of the elements along a given axis.
        batch_indexer = torch.hstack((torch.tensor(0), batch_indexer[:-1])).to(data.y_EA.device) # [TODO]: adapt this to work with different ys
        # print('batch_indexer:', batch_indexer)
        context_subgraph_idx = data.context_subgraph_idx + batch_indexer
        # print('context_subgraph_idx:', context_subgraph_idx)
        context_subgraphs_x = subgraph_x[context_subgraph_idx]
        context_pe = patch_pes[context_subgraph_idx] 
        context_subgraphs_x += self.rw_encoder(context_pe)
        context_embedding = context_subgraphs_x.unsqueeze(1)  # batch_size x 1 x nhid
        # print('context_x:', context_x.shape)

        # full graph nodes embedding (original full graph)
        parameters = (data.x, data.edge_index, data.edge_attr, data.edge_weight, data.node_weight)

        if not self.regularization:
            # in case of EMA update to avoid collapse
            with torch.no_grad():
                # work on the original full graph
                full_graph_nodes_embedding = self.target_encoder(*parameters)
                full_graph_nodes_embedding = self.targetLinearTransform(full_graph_nodes_embedding)
        else:
            # in case of vicReg to avoid collapse we have regularization
            full_graph_nodes_embedding = self.target_encoder(*parameters)
            full_graph_nodes_embedding = self.targetLinearTransform(full_graph_nodes_embedding)

        # map it as we do for x at the beginning
        full_graph_nodes_embedding = full_graph_nodes_embedding[data.subgraphs_nodes_mapper]

        # pool the embeddings found for the full graph, this will produce the subgraphs embeddings for all subgraphs (context and target subgraphs)
        subgraphs_x_from_full = scatter(full_graph_nodes_embedding, batch_x, dim=0, reduce=self.pooling) # batch_size*call_n_patches x nhid

        target_subgraphs_idx = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs]).to(data.y_EA.device)
        # Similar to context subgraphs, target_subgraphs_idx += batch_indexer.unsqueeze(1) adjusts the indices of target subgraphs. This operation is necessary because the target subgraphs can span multiple graphs within a batch, and their indices need to be corrected to reflect their actual positions in the batched data.
        target_subgraphs_idx += batch_indexer.unsqueeze(1)

        # target subgraphs nodes embedding
        # Construct context and target PEs frome the node pes of each subgraph
        target_subgraphs = subgraphs_x_from_full[target_subgraphs_idx.flatten()] 
        target_pes = patch_pes[target_subgraphs_idx.flatten()]
        encoded_tpatch_pes = self.rw_encoder(target_pes)
        target_embeddings = target_subgraphs.reshape(-1, self.num_target_patches, self.nhid) # batch_size x num_target_patches x nhid
        
        expanded_context_embeddings = torch.tensor([]) # save the embeddings for regularization
        expanded_target_embeddings = torch.tensor([])
        if self.regularization: 
            input_context_x = context_embedding.reshape(-1, self.nhid)
            expanded_context_embeddings = self.context_expander(input_context_x)#.reshape(-1, self.expander_dim)
            # expanded_target_x = self.target_expander(target_x)
            input_target_x = target_embeddings.reshape(-1, self.nhid)
            expanded_target_embeddings = self.target_expander(input_target_x)#.reshape(-1, self.expander_dim)
            # expanded_context_embeddings = torch.vstack([expanded_context_x.reshape(-1, self.expander_dim), expanded_target_x.reshape(-1, self.expander_dim)])

        if self.shouldUse2dHyperbola:
            x_coord = torch.cosh(target_embeddings.mean(-1).unsqueeze(-1))
            y_coord = torch.sinh(target_embeddings.mean(-1).unsqueeze(-1))
            target_embeddings = torch.cat([x_coord, y_coord], dim=-1) # target_x shape: batch_size x num_target_patches x 2
        

        # Make predictions using the target predictor: for each target subgraph, we use the context + the target PE
        target_prediction_embeddings = context_embedding + encoded_tpatch_pes.reshape(-1, self.num_target_patches, self.nhid) # batch_size x num_target_patches x nhid
        # print('target_x:', target_x.shape)
        # print('context_x:', context_x.shape)
        # print('encoded_tpatch_pes:', encoded_tpatch_pes.shape)
        # print('target_prediction_embeddings:', target_prediction_embeddings.shape)
        target_prediction_embeddings = target_prediction_embeddings.reshape(-1, self.nhid)

        predicted_target_embeddings = self.target_predictor(target_prediction_embeddings) # V1: Directly predict (depends on the definition of self.target_predictor)
        out_dim = 2 if self.shouldUse2dHyperbola else self.nhid
        predicted_target_embeddings = predicted_target_embeddings.reshape(-1, self.num_target_patches, out_dim)
        # print(target_x.requires_grad)
        # print(target_y.requires_grad)
        # target_y shape: batch_size x num_target_patches x 2
        # return the predicted target (via context + PE) and the true target obtained via the target encoder.
        return target_embeddings, predicted_target_embeddings, expanded_context_embeddings, expanded_target_embeddings


    def encode(self, data):
        # pass data through the target encoder that already acts on the full graph
        # x = torch.rand(data.x.shape)
        # edge_attr = torch.rand(data.edge_attr.shape) 
        # edge_weight = torch.rand(data.edge_weight.shape)
        # node_weight = torch.rand(data.node_weight.shape)
        node_embeddings = self.target_encoder(
            data.x, 
            data.edge_index, 
            data.edge_attr, 
            data.edge_weight, 
            data.node_weight
        )

        node_embeddings = self.targetLinearTransform(node_embeddings)

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