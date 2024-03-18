from einops.layers.torch import Rearrange
import numpy as np
from src.model_utils.elements import MLP
import src.model_utils.gMHA_wrapper as gMHA_wrapper
from src.model_utils.gnn import GNN
from src.visualize import plot_from_transform_attributes
from src.WDNodeMPNNLayer import WDNodeMPNNLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_mean

class PolymerJEPA(nn.Module):

    def __init__(self,
                 nfeat_node, 
                 nfeat_edge,
                 nhid,
                 nlayer_gnn,
                 nlayer_mlpmixer,
                 gMHA_type='Hadamard',
                 rw_dim=0,
                 patch_rw_dim=0,
                 mlpmixer_dropout=0,
                 pooling='mean',
                 n_patches=32,
                 num_target_patches=4,
                 should_share_weights=False,
                 regularization=False,
                 shouldUse2dHyperbola=False,
                 shouldUseNodeWeights=False
        ):

        super().__init__()
        self.pooling = pooling
        self.nhid = nhid
        self.num_target_patches=num_target_patches
        self.regularization=regularization

        self.rw_encoder = MLP(rw_dim, nhid, 1)
        self.patch_rw_encoder = MLP(patch_rw_dim, nhid, 1)
        
        self.input_encoder = nn.Linear(nfeat_node, nhid)
        # self.edge_encoder = nn.Linear(nfeat_edge, nhid)

        self.wdmpnns = nn.ModuleList()
        self.wdmpnns.append(WDNodeMPNNLayer(nhid, nfeat_edge, hidden_dim=nhid, isFirstLayer=True, shouldUseNodeWeights=False))
        for _ in range(nlayer_gnn-2):
            self.wdmpnns.append(WDNodeMPNNLayer(nhid, nfeat_edge, hidden_dim=nhid, shouldUseNodeWeights=False))
        self.wdmpnns.append(WDNodeMPNNLayer(nhid, nfeat_edge, hidden_dim=nhid, isLastLayer=True, shouldUseNodeWeights=shouldUseNodeWeights))

        self.U = nn.ModuleList(
            [MLP(nhid, nhid, nlayer=1, with_final_activation=True) for _ in range(nlayer_gnn-1)])
        
        # in case we use reg and we dont share weights, we dont need an additional context encoder, we already have the initial gnn.
        if not regularization or should_share_weights:
            self.context_encoder = getattr(gMHA_wrapper, 'Hadamard')(
                nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer) # , n_patches=n_patches, nhead=8
        
        if regularization and should_share_weights:
            self.target_encoder = self.context_encoder
        else:
            self.target_encoder = getattr(gMHA_wrapper, gMHA_type)(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer) # n_patches=n_patches
            
        self.shouldUse2dHyperbola = shouldUse2dHyperbola
        self.shouldShareWeights = should_share_weights

        self.target_predictor = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.LayerNorm(nhid),
            nn.ReLU(),
            nn.Linear(nhid, 2 if self.shouldUse2dHyperbola else nhid)
        )

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


    def forward(self, data, epoch=0):
        # Embed node features and edge attributes
        x = self.input_encoder(data.x).squeeze()
        # edge_attr = self.edge_encoder(data.edge_attr)
        
        # add node PE to the node initial embeddings
        x += self.rw_encoder(data.rw_pos_enc)

        ### Patch Encoder ###
        # x = torch.cat([data.x, data.rw_pos_enc], dim=1) # add node PE to the node initial embeddings
        x = x[data.subgraphs_nodes_mapper]
        node_weights = data.node_weight[data.subgraphs_nodes_mapper]
        edge_index = data.combined_subgraphs # the new edge index is the one that consider the graph of disconnected subgraphs, with unique node indices       
        edge_attr = data.edge_attr[data.subgraphs_edges_mapper] # edge attributes again based on the subgraphs_edges_mapper, so we have the correct edge attributes for each subgraph
        edge_weights = data.edge_weight[data.subgraphs_edges_mapper]
        batch_x = data.subgraphs_batch # i.e. the subgraph idxs [0, 0, 1, 1, ...]

        for i, wdmpnn in enumerate(self.wdmpnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                
                x = x + self.U[i-1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper, dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            
            if i == 0:
                x, h0 = wdmpnn(x, edge_index, edge_attr, edge_weights, node_weights)
            else:
                x, _ = wdmpnn(x, edge_index, edge_attr, edge_weights, node_weights, h0)

        embedded_subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling) # (B * n_patches) d   pool each subgraph node embeddings to obtain an embedding for each subgraph/patch
        
        ### JEPA - Context Encoder ###
        # Create the correct indexer for each subgraph given the batching procedure
        batch_indexer = torch.tensor(np.cumsum(data.call_n_patches)) # cumsum: return the cumulative sum of the elements along a given axis.
        batch_indexer = torch.hstack((torch.tensor(0), batch_indexer[:-1])).to(data.y_EA.device)
       
        # Find correct idxs for target subgraphs
        target_subgraphs_idx = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs]).to(data.y_EA.device)
        target_subgraphs_idx += batch_indexer.unsqueeze(1) # Similar to context subgraphs, target_subgraphs_idx += batch_indexer.unsqueeze(1) adjusts the indices of target subgraphs. This operation is necessary because the target subgraphs can span multiple graphs within a batch, and their indices need to be corrected to reflect their actual positions in the batched data.
        vis_initial_target_embedding = embedded_subgraph_x[target_subgraphs_idx.flatten()].reshape(-1, self.num_target_patches, self.nhid)[:, 0, :].detach().clone()

        # # Find correct idxs for context subgraph
        # context_subgraph_idx = data.context_subgraph_idx + batch_indexer # Get idx of context and target subgraphs according to masks, adjusts the context subgraph indices based on their position in the batch, ensuring each index points to the correct subgraph within the batched data structure.
        # embedded_context_x = embedded_subgraph_x[context_subgraph_idx].clone() # Extract context subgraph embedding

        # # Add its patch positional encoding
        # # context_pe = data.patch_pe[context_subgraph_idx]
        # # embedded_context_x += self.patch_rw_encoder(context_pe) #  modifying embedded_context_x after it is created from embedded_subgraph_x does not modify embedded_subgraph_x, because they do not share storage for their data.     
        # vis_initial_context_embeddings = embedded_context_x.detach().clone() # for visualization
        # embedded_context_x = embedded_context_x.unsqueeze(1) # # 'B d ->  B 1 d'


        # in case we use reg and we dont share weights, we dont need an additional context encoder, we already have the initial gnn.
        if not self.regularization or self.shouldShareWeights:
            # input all subgraphs into context encoder but attend and pool only the context subgraphs
            context_embedded_subgraph_x = embedded_subgraph_x.clone()
            context_embedded_subgraph_x += self.patch_rw_encoder(data.patch_pe)
            context_mixer_x = context_embedded_subgraph_x.reshape(len(data.call_n_patches), data.call_n_patches[0][0], -1) # (B * p) d ->  B p d Prepare input (all subgraphs) for target encoder (transformer)
            # mask to attend only context subgraph
            # context_mask = data.mask.flatten()[context_subgraph_idx.flatten()].reshape(-1, 1) # USELESS IN THEORY SINCE INPUT OF ENCODER IS A SINGLE ELEMENT Given that there's only one element the attention operation "won't do anything", This is simply for commodity of the EMA (need same weights so same model) between context and target encoders
            embedded_context_x = self.context_encoder(context_mixer_x, coarsen_adj=data.coarsen_adj, mask=~data.context_mask)
            embedded_context_x = (embedded_context_x * data.context_mask.unsqueeze(-1)).sum(1) / data.context_mask.sum(1, keepdim=True) # B d
            
        else:
            # Find correct idxs for context subgraph
            # context_subgraph_idx = data.context_subgraph_idx + batch_indexer # Get idx of context and target subgraphs according to masks, adjusts the context subgraph indices based on their position in the batch, ensuring each index points to the correct subgraph within the batched data structure.
            # print(data.context_subgraph_idxs)
            
            # Add its patch positional encoding
            # context_pe = data.patch_pe[context_subgraph_idx_flatten]
            # embedded_context_x += self.patch_rw_encoder(context_pe) #  modifying embedded_context_x after it is created from embedded_subgraph_x does not modify embedded_subgraph_x, because they do not share storage for their data.     
            
            # initial_context_embeddings = embedded_context_x.detach().clone() # for visualization
            # embedded_context_x = embedded_context_x.unsqueeze(1) # # 'B d ->  B 1 d'
            context_subgraph_idx = torch.vstack([torch.tensor(dc) for dc in data.context_subgraph_idxs]).to(data.y_EA.device)
            context_subgraph_idx += batch_indexer.unsqueeze(1)
            context_subgraph_idx_flatten = context_subgraph_idx.flatten()
            # remove padding elements (i.e. negative indices)       
            context_subgraph_idx_flatten = context_subgraph_idx_flatten[context_subgraph_idx_flatten >= 0]
            embedded_context_x = embedded_subgraph_x[context_subgraph_idx_flatten].clone() # Extract context subgraph embedding
            graph_indices = torch.arange(len(data.n_context)).repeat_interleave(data.n_context).to(data.y_EA.device)
            # pool the context subgraphs embeddings of each graph to obtain a single context embedding for each graph
            embedded_context_x = scatter_mean(embedded_context_x, graph_indices, dim=0, dim_size=len(data.n_context)) # B d
   
        embedded_context_x = embedded_context_x.unsqueeze(1) # 'B d ->  B 1 d'
        vis_context_embedding = embedded_context_x.squeeze().detach().clone() # for visualization
        vis_initial_context_embeddings = vis_context_embedding # TODO fix this to be the initial wdmpnn output, pick randomly n of batch context subgraph
    
        # ### JEPA - Target Encoder ###
        target_mixer_x = embedded_subgraph_x.reshape(len(data.call_n_patches), data.call_n_patches[0][0], -1) # (B * p) d ->  B p d Prepare input (all subgraphs) for target encoder (transformer)
        # print("bef")
        # print(target_mixer_x.data_ptr() == embedded_context_x.data_ptr())
        # print("are equal", torch.equal(target_mixer_x, embedded_context_x))
        if not self.regularization:
            # in case of EMA weights update to avoid collapse, the target forward step musn't store gradients, since the target encoder is optimized via EMA
            with torch.no_grad():
                target_mixer_x = self.target_encoder(target_mixer_x, coarsen_adj=data.coarsen_adj, mask=~data.mask) # Don't attend to empty patches when doing the target encoding, nor to context patch
        else:
            mixer_x = self.target_encoder(mixer_x, coarsen_adj=data.coarsen_adj, mask=~data.mask)

        with torch.no_grad():
            vis_graph_embedding = (mixer_x * data.mask.unsqueeze(-1)).sum(1) / data.mask.sum(1, keepdim=True) # for visualization
      
        mixer_x = mixer_x.reshape(-1, self.nhid) # B p d -> (B * p) d
        embedded_target_x = mixer_x[target_subgraphs_idx.flatten()] # (B * n_targets) d

        embedded_target_x = embedded_target_x.reshape(-1, self.num_target_patches, self.nhid) # (B * n_targets) d ->  B n_targets d
        vis_target_embeddings = embedded_target_x[:, 0, :].detach().clone()  # take a single target for each graph for visualization, so element [0] at position 1 (n_targets)

        expanded_context_embeddings = torch.tensor([]) # save the embeddings for regularization
        expanded_target_embeddings = torch.tensor([])
        if self.regularization: 
            input_context_x = embedded_context_x.reshape(-1, self.nhid).clone()
            expanded_context_embeddings = self.context_expander(input_context_x)#.reshape(-1, self.expander_dim)

            input_target_x = embedded_target_x[:, 0, :].reshape(-1, self.nhid).clone() # take only the first patch to avoid overrepresenting the target embeddings
            expanded_target_embeddings = self.target_expander(input_target_x)#.reshape(-1, self.expander_dim)

        if self.shouldUse2dHyperbola:
            # Predict the coordinates of the patches in the unit hyperbola
            x_coord = torch.cosh(embedded_target_x.mean(-1).unsqueeze(-1))
            y_coord = torch.sinh(embedded_target_x.mean(-1).unsqueeze(-1))
            embedded_target_x = torch.cat([x_coord, y_coord], dim=-1)

        target_pes = data.patch_pe[target_subgraphs_idx.flatten()]
        encoded_tpatch_pes = self.patch_rw_encoder(target_pes)

        embedded_context_x_pe_conditioned = embedded_context_x + encoded_tpatch_pes.reshape(-1, self.num_target_patches, self.nhid) # B n_targets d
        predicted_target_embeddings = self.target_predictor(embedded_context_x_pe_conditioned)

        return embedded_target_x, predicted_target_embeddings, expanded_context_embeddings, expanded_target_embeddings, vis_initial_context_embeddings, vis_initial_target_embedding, vis_context_embedding, vis_target_embeddings, vis_graph_embedding

    

    def encode(self, data):
        x = self.input_encoder(data.x).squeeze()
        # edge_attr = self.edge_encoder(data.edge_attr)

        # add node PE to the node initial embeddings
        x += self.rw_encoder(data.rw_pos_enc)

        ### Patch Encoder ###
        # x = torch.cat([data.x, data.rw_pos_enc], dim=1) # add node PE to the node initial embeddings
        x = x[data.subgraphs_nodes_mapper]
        node_weights = data.node_weight[data.subgraphs_nodes_mapper]
        edge_index = data.combined_subgraphs
        edge_attr = data.edge_attr[data.subgraphs_edges_mapper]
        edge_weights = data.edge_weight[data.subgraphs_edges_mapper]
        batch_x = data.subgraphs_batch

        for i, wdmpnn in enumerate(self.wdmpnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                
                x = x + self.U[i-1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper, dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
                
            if i == 0: 
                x, h0 = wdmpnn(x, edge_index, edge_attr, edge_weights, node_weights)
            else:
                x, _ = wdmpnn(x, edge_index, edge_attr, edge_weights, node_weights, h0)
               
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)
        
        mixer_x = subgraph_x.reshape(len(data.call_n_patches), data.call_n_patches[0][0], -1)
        # Eval via target encoder
        mixer_x = self.target_encoder(mixer_x, data.coarsen_adj, ~data.mask) # Don't attend to empty patches when doing the final encoding, nor to context patch
        
        graph_embedding = (mixer_x * data.mask.unsqueeze(-1)).sum(1) / data.mask.sum(1, keepdim=True)
        return graph_embedding






    # plotting
    # import networkx as nx
    #     import matplotlib.pyplot as plt
    #     from torch_geometric.utils import subgraph

    #     # Assuming context_subgraph_idx, target_subgraphs_idx, and data are defined as per your setup

    #     # Initialize a figure with 3 subplots
    #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    #     # Context Subgraph
    #     contextPlot = context_subgraph_idx[0]
    #     context_nodes = data.subgraphs_nodes_mapper[contextPlot == data.subgraphs_batch]
    #     context_edge_index, _ = subgraph(context_nodes, edge_index)
    #     G_context = nx.Graph()
    #     G_context.add_edges_from(context_edge_index.T.cpu().numpy())
    #     nx.draw(G_context, with_labels=True, ax=axes[0], node_color='skyblue')
    #     axes[0].set_title("Context Subgraph")

    #     # First Target Subgraph
    #     targetPlot = target_subgraphs_idx.flatten()[0]
    #     target_nodes = data.subgraphs_nodes_mapper[targetPlot == data.subgraphs_batch]
    #     target_edge_index, _ = subgraph(target_nodes, edge_index)
    #     G_target1 = nx.Graph()
    #     G_target1.add_edges_from(target_edge_index.T.cpu().numpy())
    #     nx.draw(G_target1, with_labels=True, ax=axes[1], node_color='lightgreen')
    #     axes[1].set_title("Target Subgraph 1")

    #     # Second Target Subgraph
    #     targetPlot2 = target_subgraphs_idx.flatten()[1]
    #     target_nodes2 = data.subgraphs_nodes_mapper[targetPlot2 == data.subgraphs_batch]
    #     target_edge_index2, _ = subgraph(target_nodes2, edge_index)
    #     G_target2 = nx.Graph()
    #     G_target2.add_edges_from(target_edge_index2.T.cpu().numpy())
    #     nx.draw(G_target2, with_labels=True, ax=axes[2], node_color='salmon')
    #     axes[2].set_title("Target Subgraph 2")

    #     plt.tight_layout()
    #     plt.show()


    # n_context_nodes = [torch.sum(data.subgraphs_batch == idx).item() for idx in context_subgraph_idx]
        # print('n of nodes in the context_subgraph_idx:', n_context_nodes)

        # # Example for target subgraphs; adjust according to actual data structure
        # n_target_nodes = [torch.sum(data.subgraphs_batch == idx).item() for idx_list in target_subgraphs_idx for idx in idx_list]
        # print('n of nodes in the target_subgraphs_idx:', n_target_nodes)

        # for graph in data.to_data_list():
        #     plot_from_transform_attributes(graph)
    

    # n_context_nodes = [torch.sum(data.subgraphs_batch == idx).item() for idx in context_subgraph_idx]
    #     print('n of nodes in the context_subgraph_idx:', n_context_nodes)

    #     # Example for target subgraphs; adjust according to actual data structure
    #     n_target_nodes = [torch.sum(data.subgraphs_batch == idx).item() for idx_list in target_subgraphs_idx for idx in idx_list]
    #     print('n of nodes in the target_subgraphs_idx:', n_target_nodes)

    #     for graph in data.to_data_list():
    #         plot_from_transform_attributes(graph)