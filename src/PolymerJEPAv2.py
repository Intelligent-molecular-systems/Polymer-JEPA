import numpy as np
from src.model_utils.elements import MLP  
from src.WDNodeMPNN import WDNodeMPNN
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
        patch_rw_dim=0,
        num_target_patches=4,
        should_share_weights = False,
        regularization = False
    ):
        
        super().__init__()

        self.pooling = pooling
        self.patch_rw_dim = patch_rw_dim
        self.nhid = nhid
        self.num_target_patches = num_target_patches
        self.regularization = regularization

        if rw_dim > 0: #[TODO] Understand what is this for, right now it is not used
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        
        if self.patch_rw_dim > 0:
            self.patch_rw_encoder = MLP(self.patch_rw_dim, nhid, 1)

        # Context and Target Encoders are both WDNodeMPNN
        # TODO: for now i must keep the context and target models equal for EMA update, with vicReg (no weight sharing case obviously) i can change this
        # i think n_message_passing_layers could be lower for context encoder
        self.context_encoder = WDNodeMPNN(nfeat_node, nfeat_edge, n_message_passing_layers=3, out_dim=nhid)
        if should_share_weights:
            self.target_encoder = self.context_encoder
        else:
            self.target_encoder = WDNodeMPNN(nfeat_node, nfeat_edge, n_message_passing_layers=3, out_dim=nhid)
        
        # Predictor MLP for the target subgraphs
        # according to g-jepa when using hyperbolic space, the target predictor should be a linear layer
        # otherwise representation collapse
        self.target_predictor = nn.Linear(nhid, 2) # V2: Directly predict (depends on the definition of self.target_predictor)
        # self.target_predictor = MLP(
        #     nin=nhid, 
        #     nout=2, 
        #     nlayer=3, 
        #     with_final_activation=False, 
        #     with_norm=False
        # )
        

    def forward(self, data):
        x = data.x[data.subgraphs_nodes_mapper]
        node_weights = data.node_weight[data.subgraphs_nodes_mapper]
        # the new edge index is the one that consider the graph of disconnected subgraphs, with unique node indices
        edge_index = data.combined_subgraphs        
        # edge attributes again based on the subgraphs_edges_mapper, so we have the correct edge attributes for each subgraph
        edge_attr = data.edge_attr[data.subgraphs_edges_mapper]
        edge_weights = data.edge_weight[data.subgraphs_edges_mapper]
        batch_x = data.subgraphs_batch # this is the batch of subgraphs, i.e. the subgraph idxs [0, 0, 1, 1, ...]

        pes = data.rw_pos_enc[data.subgraphs_nodes_mapper]
        patch_pes = scatter(pes, batch_x, dim=0, reduce='max') 

        # initial encoder, encode all the subgraphs, then consider only the context subgraphs
        x = self.context_encoder(x, edge_index, edge_attr, edge_weights, node_weights)
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)


        batch_indexer = torch.tensor(np.cumsum(data.call_n_patches)) # cumsum: return the cumulative sum of the elements along a given axis.
        batch_indexer = torch.hstack((torch.tensor(0), batch_indexer[:-1])).to(data.y_EA.device) # [TODO]: adapt this to work with different ys

        context_subgraph_idx = data.context_subgraph_idx + batch_indexer
        # print('context_subgraph_idx:', context_subgraph_idx)
        context_subgraphs_x = subgraph_x[context_subgraph_idx]
        context_pe = patch_pes[context_subgraph_idx] 
        context_subgraphs_x += self.patch_rw_encoder(context_pe)
        context_x = context_subgraphs_x.unsqueeze(1)  # batch_size x num_target_patches x nhid
        
        
        # full graph nodes embedding (original full graph)
        parameters = (data.x, data.edge_index, data.edge_attr, data.edge_weight, data.node_weight)

        if not self.regularization:
            # in case of EMA update to avoid collapse
            with torch.no_grad():
                # work on the original full graph
                full_graph_nodes_embedding = self.target_encoder(*parameters)
        else:
            # in case of vicReg to avoid collapse we have regularization
            full_graph_nodes_embedding = self.target_encoder(*parameters)

        # map it as we do for x at the beginning
        full_graph_nodes_embedding = full_graph_nodes_embedding[data.subgraphs_nodes_mapper]
        # pool the embeddings found for the full graph, this will produce the pooled subgraphs embeddings for all subgraphs (context and target subgraphs)
        subgraphs_x_from_full = scatter(full_graph_nodes_embedding, batch_x, dim=0, reduce=self.pooling)

        target_subgraphs_idx = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs]).to(data.y_EA.device)
        # Similar to context subgraphs, target_subgraphs_idx += batch_indexer.unsqueeze(1) adjusts the indices of target subgraphs. This operation is necessary because the target subgraphs can span multiple graphs within a batch, and their indices need to be corrected to reflect their actual positions in the batched data.
        target_subgraphs_idx += batch_indexer.unsqueeze(1)

        # n_context_nodes = [torch.sum(data.subgraphs_batch == idx).item() for idx in context_subgraph_idx]
        # print('n of nodes in the context_subgraph_idx:', n_context_nodes)

        # # Example for target subgraphs; adjust according to actual data structure
        # n_target_nodes = [torch.sum(data.subgraphs_batch == idx).item() for idx_list in target_subgraphs_idx for idx in idx_list]
        # print('n of nodes in the target_subgraphs_idx:', n_target_nodes)
        # target subgraphs nodes embedding
        # Construct context and target PEs frome the node pes of each subgraph
        target_subgraphs = subgraphs_x_from_full[target_subgraphs_idx.flatten()] 
        target_pes = patch_pes[target_subgraphs_idx.flatten()]
        encoded_tpatch_pes = self.patch_rw_encoder(target_pes)
        target_x = target_subgraphs.reshape(-1, self.num_target_patches, self.nhid) # batch_size x num_target_patches x nhid
        embeddings = torch.vstack([context_x.reshape(-1, self.nhid), target_x.reshape(-1, self.nhid)])
        x_coord = torch.cosh(target_x.mean(-1).unsqueeze(-1))
        y_coord = torch.sinh(target_x.mean(-1).unsqueeze(-1))
        target_x = torch.cat([x_coord, y_coord], dim=-1) # target_x shape: batch_size x num_target_patches x 2

        # Make predictions using the target predictor: for each target subgraph, we use the context + the target PE
        target_prediction_embeddings = context_x + encoded_tpatch_pes.reshape(-1, self.num_target_patches, self.nhid) # batch_size x num_target_patches x nhid
        # print('target_x:', target_x.shape)
        # print('context_x:', context_x.shape)
        # print('encoded_tpatch_pes:', encoded_tpatch_pes.shape)
        # print('target_prediction_embeddings:', target_prediction_embeddings.shape)
        target_y = self.target_predictor(target_prediction_embeddings) # V1: Directly predict (depends on the definition of self.target_predictor)
        # print('target_y:', target_y.shape)
        # target_y shape: batch_size x num_target_patches x 2
        # return the predicted target (via context + PE) and the true target obtained via the target encoder.
        return target_x, target_y, embeddings


    def encode(self, data):
        # pass data through the target encoder that already acts on the full graph
        node_embeddings = self.target_encoder(
            data.x, 
            data.edge_index, 
            data.edge_attr, 
            data.edge_weight, 
            data.node_weight
        )

        graph_embedding = global_mean_pool(node_embeddings, data.batch)
        return graph_embedding
    