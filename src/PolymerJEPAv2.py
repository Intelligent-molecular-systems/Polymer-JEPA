import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter
from src.WDNodeMPNN import WDNodeMPNN  # Ensure this import matches your project structure
from src.model_utils.elements import MLP  # Adjust imports as necessary

class PolymerJEPAv2(nn.Module):
    def __init__(self,
                 nfeat_node, 
                 nfeat_edge,
                 nhid, 
                 rw_dim=0,
                 pooling='mean',
                 patch_rw_dim=0,
                 num_target_patches=4):
        
        super().__init__()

        self.use_rw = rw_dim > 0
        self.pooling = pooling
        self.patch_rw_dim = patch_rw_dim
        self.nhid = nhid
        self.num_target_patches = num_target_patches

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        
        if self.patch_rw_dim > 0:
            self.patch_rw_encoder = MLP(self.patch_rw_dim, nhid, 1)

        # Context and Target Encoders are both WDNodeMPNN
        self.context_encoder = WDNodeMPNN(nfeat_node, nfeat_edge, out_dim=nhid)
        self.target_encoder = WDNodeMPNN(nfeat_node, nfeat_edge, out_dim=nhid)
        
        # Predictor MLP for the target subgraphs
        self.target_predictor = MLP(
            nhid, 2, nlayer=3, with_final_activation=False, with_norm=False)

    def forward(self, data):
        context_node_features = data.x[data.context_nodes_mapper]
        context_node_weights = data.node_weight[data.context_nodes_mapper]
        context_edge_index = data.combined_subgraphs[:, data.context_edges_mask]
        general_edge_attr = data.edge_attr[data.subgraphs_edges_mapper]
        context_edge_attr = general_edge_attr[data.context_edges_mask]
        general_edge_weights = data.edge_weight[data.subgraphs_edges_mapper]
        context_edge_weights = general_edge_weights[data.context_edges_mask]
        
        context_subgraphs_nodes_embedding = self.context_encoder(
            context_node_features, 
            context_edge_index, 
            context_edge_attr, 
            context_edge_weights, 
            context_node_weights
        )
        # TODO test this functions, not sure about data.batch, need to debug and trial and error
        # pool the context subgraph nodes embedding
        context_subgraph_embeddings = scatter(
            context_subgraphs_nodes_embedding, 
            data.batch[data.context_nodes_mapper], #data.context_subgraph_idx, 
            dim=0, 
            reduce=self.pooling
        )

        # full graph nodes embedding
        full_graph_nodes_embedding = self.target_encoder(
            data.x, 
            data.edge_index, 
            data.edge_attr, 
            data.edge_weight, 
            data.node_weight
        )

        # extract target nodes from teh full graph nodes embeddings
        # pool the target subgraph nodes embedding
        target_subgraph_embeddings = scatter(
            full_graph_nodes_embedding, 
            data.batch[data.target_nodes_mapper], 
            dim=0, 
            reduce=self.pooling
        )

        quit()


    def encode(self, data):
        # pass data through the target encoder that already acts on the full graph
        return self.target_encoder(
            data.x, 
            data.edge_index, 
            data.edge_attr, 
            data.edge_weight, 
            data.node_weight
        )
