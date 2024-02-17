from einops.layers.torch import Rearrange
import numpy as np
from src.model_utils.elements import MLP
import src.model_utils.gMHA_wrapper as gMHA_wrapper
from src.model_utils.gnn import GNN
from src.WDNodeMPNN import WDNodeMPNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

class PolymerJEPA(nn.Module):

    def __init__(self,
                 nfeat_node, 
                 nfeat_edge,
                 nhid, 
                 nlayer_mlpmixer,
                 gMHA_type='MLPMixer',
                 rw_dim=0,
                 mlpmixer_dropout=0,
                 pooling='mean',
                 patch_rw_dim=0,
                 num_target_patches=4,
                 should_share_weights = False,
                 regularization = False
        ):

        super().__init__()
        self.use_rw = rw_dim > 0
        self.pooling = pooling
        self.patch_rw_dim = patch_rw_dim
        self.nhid = nhid
        self.num_target_patches=num_target_patches
        self.regularization=regularization

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        
        if self.patch_rw_dim > 0:
            self.patch_rw_encoder = MLP(self.patch_rw_dim, nhid, 1)

        # Input Encoder
        self.wdmpnn = WDNodeMPNN(nfeat_node, nfeat_edge, n_message_passing_layers=2, out_dim=nhid)
        
        self.context_encoder = getattr(gMHA_wrapper, 'Standard')(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer)
        
        if should_share_weights:
            self.target_encoder = self.context_encoder
        else:
            self.target_encoder = getattr(gMHA_wrapper, gMHA_type)(
                nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer)
            
        # in g-jepa they say 2d hyperbolic should use linear predictor, g-jepa code does not use this though
        self.target_predictor = nn.Linear(nhid, 2) # V2: Directly predict (depends on the definition of self.target_predictor)
        # Use this if you wish to do euclidean or poincaré embeddings in the latent space
        # self.target_predictor = MLP(
        #     nhid, 2, nlayer=3, with_final_activation=False, with_norm=False)

        if self.regularization:
            self.expander_dim = 512
            self.context_expander = nn.Sequential(
                nn.Linear(nhid, self.expander_dim),
                nn.ReLU(),
                nn.Linear(self.expander_dim, self.expander_dim)
            )

            self.target_expander = nn.Sequential(
                nn.Linear(nhid, self.expander_dim),
                nn.ReLU(),
                nn.Linear(self.expander_dim, self.expander_dim)
            )

    def forward(self, data):
        # x = self.input_encoder(data.x).squeeze() RISK?
        # The model first encodes node features data.x and edge attributes data.edge_attr using respective encoders (self.input_encoder and self.edge_encoder)
        # if edge_attr is None:
        #     edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1)).float().unsqueeze(-1)
        # edge_attr = self.edge_encoder(edge_attr)

        # Patch Encoder
        # Nodes (data.subgraphs_nodes_mapper) and edges (data.combined_subgraphs) from specific subgraphs are selected based on precomputed mappings. 
        # This selection is critical for focusing the model's attention on relevant parts of the graph.
        # if x has shape N X F, and data.subgraphs_nodes_mapper is shape M (all subgraphs), then x[data.subgraphs_nodes_mapper] has shape M X F
        # If a node index appears more than once in data.subgraphs_nodes_mapper, its feature vector is duplicated in the resulting tensor. If a node index does not appear in data.subgraphs_nodes_mapper, its feature vector is excluded from the result. (Never happens in our case, but it's good to know)
        x = data.x[data.subgraphs_nodes_mapper]
        node_weights = data.node_weight[data.subgraphs_nodes_mapper]
        # the new edge index is the one that consider the graph of disconnected subgraphs, with unique node indices
        edge_index = data.combined_subgraphs        
        # edge attributes again based on the subgraphs_edges_mapper, so we have the correct edge attributes for each subgraph
        edge_attr = data.edge_attr[data.subgraphs_edges_mapper]
        edge_weights = data.edge_weight[data.subgraphs_edges_mapper]
        batch_x = data.subgraphs_batch # this is the batch of subgraphs, i.e. the subgraph idxs [0, 0, 1, 1, ...]
        # Positional encodings (data.rw_pos_enc) are used to enhance the node features by providing spatial information. These are aggregated per subgraph (patch_pes) using a scatter operation with a 'max' reduction to capture the most significant positional signal
        # pes contains the positional encodings for each node in the graph, again using mapper has the same effect as for x (few lines above)
        pes = data.rw_pos_enc[data.subgraphs_nodes_mapper]
        # knowing which nodes belong to which subgraph (batch_x), and the pos encoding for each node (pes), we can aggregate the pes for each subgraph:
        # the pos encoding for each patch, is the max between each patch node PE
        patch_pes = scatter(pes, batch_x, dim=0, reduce='max') 

        # initial encoder
        # gnn is working on the combined graph (graph of disconnected subgraphs)
        x = self.wdmpnn(x, edge_index, edge_attr, edge_weights, node_weights)
        
        # this is the final operation (pooling) to obtain an embedding for each subgraph/patch from the subgraph nodes embeddings
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)


        ######################## Graph-JEPA ########################
        # Create the correct indexer for each subgraph given the batching procedure
        # The model calculates indexes for context and target subgraphs based on batching information (data.call_n_patches) and adds these to respective subgraph indices (data.context_subgraph_idx and data.target_subgraph_idxs) to correctly reference subgraphs across potentially multiple input graphs.
        # np.cumsum(data.call_n_patches) computes the cumulative sum of the number of patches (subgraphs) across batches. This is useful for indexing when multiple graphs are batched together, and you need to keep track of the starting index of subgraphs for each graph in the batch.
        batch_indexer = torch.tensor(np.cumsum(data.call_n_patches)) # cumsum: return the cumulative sum of the elements along a given axis.
        # torch.hstack((torch.tensor(0), batch_indexer[:-1])) prepends a 0 to the cumulative sum, shifting the indices to correctly reference the start of each graph's subgraphs within a batched setup
        batch_indexer = torch.hstack((torch.tensor(0), batch_indexer[:-1])).to(data.y_EA.device) # [TODO]: adapt this to work with different ys

        # Get idx of context and target subgraphs according to masks
        # Adjusts the context subgraph indices based on their position in the batch, ensuring each index points to the correct subgraph within the batched data structure.
        # the context subgraphs is always the first subgraph for each graph    
        # batch_indexer = [0, 20, 40, 60...], according to my understanding, data.context_subgraph_idx is always 0 for all graphs
        # so the correct context_subgraph_idx is always 0, 20, 40, 60... while the right index should be at 20-n of subgraphs found
        context_subgraph_idx = data.context_subgraph_idx + batch_indexer
        # print('(forward) context_subgraph_idx:', context_subgraph_idx)
        target_subgraphs_idx = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs]).to(data.y_EA.device)
        # Similar to context subgraphs, target_subgraphs_idx += batch_indexer.unsqueeze(1) adjusts the indices of target subgraphs. This operation is necessary because the target subgraphs can span multiple graphs within a batch, and their indices need to be corrected to reflect their actual positions in the batched data.
        target_subgraphs_idx += batch_indexer.unsqueeze(1)
        # print('(forward) target_subgraphs_idx:', target_subgraphs_idx)
        
        # ...(your existing code)...

        # n of nodes in the context_subgraph_idx
        # Example for context subgraph; adjust according to actual data structure
        # DEBUG: to check whether indexing is correct i print the number of nodes in the context and target subgraphs.
        # indexing is correct, empty subgraphs are never considered, all contexts size are much larger than targets so its correct.
        n_context_nodes = [torch.sum(data.subgraphs_batch == idx).item() for idx in context_subgraph_idx]
        #print('n of nodes in the context_subgraph_idx:', n_context_nodes)

        # Example for target subgraphs; adjust according to actual data structure
        n_target_nodes = [torch.sum(data.subgraphs_batch == idx).item() for idx_list in target_subgraphs_idx for idx in idx_list]
        #print('n of nodes in the target_subgraphs_idx:', n_target_nodes)

        # import matplotlib.pyplot as plt
        # from torch_geometric.utils import subgraph
        # import networkx as nx
        # # Assuming context_subgraph_idx, target_subgraphs_idx, and data are defined as per your setup

        # # Initialize a figure with 3 subplots
        # fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # # Context Subgraph
        # contextPlot = context_subgraph_idx[0]
        # context_nodes = data.subgraphs_nodes_mapper[contextPlot == data.subgraphs_batch]
        # context_edge_index, _ = subgraph(context_nodes, edge_index)
        # G_context = nx.Graph()
        # G_context.add_edges_from(context_edge_index.T.cpu().numpy())
        # nx.draw(G_context, with_labels=True, ax=axes[0], node_color='skyblue')
        # axes[0].set_title("Context Subgraph")
        # plt.tight_layout()
        # plt.show()
        
        # Extract context and target subgraph (mpnn) embeddings
        context_subgraphs = subgraph_x[context_subgraph_idx]
        target_subgraphs = subgraph_x[target_subgraphs_idx.flatten()] 
     
        # Construct context and target PEs frome the node pes of each subgraph
        target_pes = patch_pes[target_subgraphs_idx.flatten()]
        context_pe = patch_pes[context_subgraph_idx] 
        # Positional encodings (patch_pes) are added to both context and target subgraph embeddings to incorporate structural information into the embeddings. self.patch_rw_encoder is applied to positional encodings before addition, indicating that these encodings are processed (likely transformed) before being incorporated into the subgraph embeddings.
        context_subgraphs += self.patch_rw_encoder(context_pe)
        encoded_tpatch_pes = self.patch_rw_encoder(target_pes)
        
        # Prepare inputs for MHA
        target_x = target_subgraphs.reshape(-1, self.num_target_patches, self.nhid)
        context_x = context_subgraphs.unsqueeze(1)


        # Given that there's only one element the attention operation "won't do anything"
        # This is simply for commodity of the EMA (need same weights so same model) between context and target encoders
        context_mask = data.mask.flatten()[context_subgraph_idx].reshape(-1, 1) # this should be -1 x num context which is always 1
        # print('context_mask:', context_mask) # RISK this prints a tensor n of graphs in batch x 1, with all Trues
        
        # pass context subgraph through context encoder
        context_x = self.context_encoder(
            context_x, 
            data.coarsen_adj if hasattr(data, 'coarsen_adj') else None, 
            ~context_mask
        )


        # RISK in the original code these lines are put under the torch.grad but i dont think is necessary
        if hasattr(data, 'coarsen_adj'): # if we have a coarsen adj, we use it to encode the target subgraphs
            subgraph_incides = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs])
            patch_adj = data.coarsen_adj[
                torch.arange(target_x.shape[0]).unsqueeze(1).unsqueeze(2),  # Batch dimension
                subgraph_incides.unsqueeze(1),  # Row dimension
                subgraph_incides.unsqueeze(2)   # Column dimension
            ]
            parameters = (target_x, patch_adj, None)
        else:
            parameters = (target_x, None, None)

        
        expanded_embeddings = torch.tensor([]) # save the embeddings for regularization

        if not self.regularization:
        # in case of EMA update to avoid collapse, the target forward step musn't store gradients, since the target encoder is optimized via EMA
            with torch.no_grad():
                target_x = self.target_encoder(*parameters)
                # save embeddings for visualization of embedding space
                # embeddings = torch.vstack([context_x.reshape(-1, self.nhid), target_x.reshape(-1, self.nhid)])

                # Predict the coordinates of the patches in the Q1 hyperbola
                # Remove this part if you wish to do euclidean or poincaré embeddings in the latent space
                x_coord = torch.cosh(target_x.mean(-1).unsqueeze(-1))
                y_coord = torch.sinh(target_x.mean(-1).unsqueeze(-1))
                target_x = torch.cat([x_coord, y_coord], dim=-1)
        else:
            # in case of vicReg to avoid collapse we have regularization
            target_x = self.target_encoder(*parameters)
            # save embeddings for visualization of embedding space
            # embeddings = torch.vstack([context_x.reshape(-1, self.nhid), target_x.reshape(-1, self.nhid)])
            
            # Regularization
            expanded_context_x = self.context_expander(context_x)
            expanded_target_x = self.target_expander(target_x)
            expanded_embeddings = torch.vstack([expanded_context_x.reshape(-1, self.expander_dim), expanded_target_x.reshape(-1, self.expander_dim)])
            
            # Predict the coordinates of the patches in the Q1 hyperbola
            x_coord = torch.cosh(target_x.mean(-1).unsqueeze(-1))
            y_coord = torch.sinh(target_x.mean(-1).unsqueeze(-1))
            target_x = torch.cat([x_coord, y_coord], dim=-1)


        
        # Make predictions using the target predictor: for each target subgraph, we use the context + the target PE
        target_prediction_embeddings = context_x + encoded_tpatch_pes.reshape(-1, self.num_target_patches, self.nhid)
        target_y = self.target_predictor(target_prediction_embeddings) # V1: Directly predict (depends on the definition of self.target_predictor)
        # return the predicted target (via context + PE) and the true target obtained via the target encoder.
        return target_x, target_y, expanded_embeddings
    

    def encode(self, data):
        x = data.x[data.subgraphs_nodes_mapper]
        node_weights = data.node_weight[data.subgraphs_nodes_mapper]
        edge_index = data.combined_subgraphs
        edge_attr = data.edge_attr[data.subgraphs_edges_mapper]
        edge_weights = data.edge_weight[data.subgraphs_edges_mapper]
        batch_x = data.subgraphs_batch
        pes = data.rw_pos_enc[data.subgraphs_nodes_mapper]
        patch_pes = scatter(pes, batch_x, dim=0, reduce='mean')

        # encode all subgraphs separately
        x = self.wdmpnn(x, edge_index, edge_attr, edge_weights, node_weights)

        # pool the subgraph node embeddings to find the subgraph embeddings
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)
        subgraph_x += self.patch_rw_encoder(patch_pes)
        
        # Handles different patch sizes based on the data object for multiscale training
        mixer_x = subgraph_x.reshape(len(data.call_n_patches), data.call_n_patches[0][0], -1)

        # Eval via target encoder
        mixer_x = self.target_encoder(
            mixer_x, 
            data.coarsen_adj if hasattr(data, 'coarsen_adj') else None, 
            ~data.mask
        ) # Don't attend to empty patches when doing the final encoding
        
        # Global Average Pooling
        embeddings = (mixer_x * data.mask.unsqueeze(-1)).sum(1) / data.mask.sum(1, keepdim=True)
        return embeddings








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