import numpy as np
from src.JEPA_models.model_utils.elements import MLP  
from src.JEPA_models.model_utils.gnn import GNN
from src.visualize import plot_from_transform_attributes
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter


class GeneralJEPAv2(nn.Module):
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
        shouldUse2dHyperbola=False
    ):
        
        super().__init__()

        self.pooling = pooling
        self.nhid = nhid
        self.num_target_patches = num_target_patches
        self.regularization = regularization

        self.rw_encoder = MLP(rw_dim, nhid, 1)
        self.patch_rw_encoder = MLP(patch_rw_dim, nhid, 1)

        self.input_encoder = nn.Embedding(nfeat_node, nhid)
        self.edge_encoder = nn.Embedding(nfeat_edge, nhid)

        
        self.contextGNNs = nn.ModuleList([GNN(nin=nhid, nout=nhid, nlayer_gnn=1, gnn_type="GINEConv",
                        bn=True, dropout=0.1, res=True) for _ in range(nlayer_gnn)])
                                            
        self.contextU = nn.ModuleList(
                [MLP(nhid, nhid, nlayer=1, with_final_activation=True) for _ in range(nlayer_gnn-1)])
        
        # useful for EMA in training.py
        self.context_encoder = nn.ModuleList([self.contextGNNs, self.contextU])
        
        if regularization and should_share_weights:
            self.target_encoder = self.context_encoder
        else:
            self.targetGNNs = nn.ModuleList([GNN(nin=nhid, nout=nhid, nlayer_gnn=1, gnn_type="GINEConv",
                        bn=True, dropout=0.1, res=True) for _ in range(nlayer_gnn)])
                                            
            self.targetU = nn.ModuleList(
                    [MLP(nhid, nhid, nlayer=1, with_final_activation=True) for _ in range(nlayer_gnn-1)])
            
            self.target_encoder = nn.ModuleList([self.targetGNNs, self.targetU])
        
        self.shouldUse2dHyperbola = shouldUse2dHyperbola
        
        self.target_predictor = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.BatchNorm1d(nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
            nn.BatchNorm1d(nhid),
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
        # Embed node features and edge attributes
        x = self.input_encoder(data.x).squeeze()
        edge_attr = self.edge_encoder(data.edge_attr)

        x += self.rw_encoder(data.rw_pos_enc)
        x = x[data.subgraphs_nodes_mapper]
        edge_index = data.combined_subgraphs # the new edge index is the one that consider the graph of disconnected subgraphs, with unique node indices       
        edge_attr = edge_attr[data.subgraphs_edges_mapper] # edge attributes again based on the subgraphs_edges_mapper, so we have the correct edge attributes for each subgraph
        batch_x = data.subgraphs_batch # i.e. the subgraph idxs [0, 0, 1, 1, ...]

        ### JEPA - Context Encoder ###
        # initial encoder, encode all the subgraphs, then consider only the context subgraphs
        for i, gnn in enumerate(self.contextGNNs):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                x = x + self.contextU[i-1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper,
                            dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, edge_attr)

        embedded_subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling) # pool each subgraph node embeddings to obtain an embedding for each subgraph/patch

        batch_indexer = torch.tensor(np.cumsum(data.call_n_patches)) # cumsum: return the cumulative sum of the elements along a given axis.
        batch_indexer = torch.hstack((torch.tensor(0), batch_indexer[:-1])).to(data.y.device)

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
        edge_attr = self.edge_encoder(data.edge_attr)
        full_x += self.rw_encoder(data.rw_pos_enc)
        parameters = (full_x, data.edge_index, edge_attr)

        if not self.regularization:
            # in case of EMA update to avoid collapse
            with torch.no_grad():
                # work on the original full graph
                for i, gnn in enumerate(self.targetGNNs):
                    if i > 0:
                        # global mean pooling here we use the original full graph, than map it to n of nodes
                        graphEmbedding = global_mean_pool(x, data.batch)[data.batch]
                        x = x + self.targetU[i-1](graphEmbedding)
                       
                    if i == 0:
                        x = gnn(*parameters)
                    else:
                        x = gnn(x, data.edge_index, edge_attr)
                full_graph_nodes_embedding = x
        else:
            # in case of vicReg to avoid collapse we have regularization
            for i, gnn in enumerate(self.targetGNNs):
                if i > 0:
                    # global mean pooling here we use the original full graph, than map it to n of nodes
                    graphEmbedding = global_mean_pool(x, data.batch)[data.batch]
                    x = x + self.targetU[i-1](graphEmbedding)
                    
                if i == 0:
                    x = gnn(*parameters)
                else:
                    x = gnn(x, data.edge_index, edge_attr)
            full_graph_nodes_embedding = x

        with torch.no_grad():
            vis_graph_embedding = global_mean_pool(full_graph_nodes_embedding.detach().clone(), data.batch)

        # map it as we do for x at the beginning
        full_graph_nodes_embedding = full_graph_nodes_embedding[data.subgraphs_nodes_mapper]

        # pool the embeddings found for the full graph, this will produce the subgraphs embeddings for all subgraphs (context and target subgraphs)
        subgraphs_x_from_full = scatter(full_graph_nodes_embedding, batch_x, dim=0, reduce=self.pooling) # batch_size*call_n_patches x nhid

        target_subgraphs_idx = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs]).to(data.y.device)
        target_subgraphs_idx += batch_indexer.unsqueeze(1)

        embedded_target_x = subgraphs_x_from_full[target_subgraphs_idx.flatten()]

        embedded_target_x = embedded_target_x.reshape(-1, self.num_target_patches, self.nhid) # batch_size x num_target_patches x nhid
        vis_target_embeddings = embedded_target_x[:, 0, :].detach().clone()

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
        # convert to B x n_targets x d for batch norm
        embedded_context_x_pe_conditioned = embedded_context_x_pe_conditioned.reshape(-1, self.nhid)
        predicted_target_embeddings = self.target_predictor(embedded_context_x_pe_conditioned)
        # convert back to B x n_targets x d
        predicted_target_embeddings = predicted_target_embeddings.reshape(-1, self.num_target_patches, self.nhid)
        return embedded_target_x, predicted_target_embeddings, expanded_context_embeddings, expanded_target_embeddings, torch.tensor([]), torch.tensor([]), vis_context_embedding, vis_target_embeddings, vis_graph_embedding


    def encode(self, data):
        full_x = self.input_encoder(data.x).squeeze()
        full_x += self.rw_encoder(data.rw_pos_enc)

        edge_attr = self.edge_encoder(data.edge_attr)
       
        parameters = (full_x, data.edge_index, edge_attr)

        for i, gnn in enumerate(self.targetGNNs):
            if i > 0:
                graphEmbedding = global_mean_pool(full_x, data.batch)[data.batch]
                full_x = full_x + self.targetU[i-1](graphEmbedding)
                
            if i == 0:
                full_x = gnn(*parameters)
            else:
                full_x = gnn(full_x, data.edge_index, edge_attr)

        node_embeddings = full_x


        graph_embedding = global_mean_pool(node_embeddings, data.batch)
        return graph_embedding