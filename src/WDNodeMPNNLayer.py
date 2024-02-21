import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

# node-centered message passing
class WDNodeMPNNLayer(nn.Module):
    class_name = "WDNodeMPNNLayer"

    def __init__(
            self, 
            node_attr_dim, 
            edge_attr_dim,
            hidden_dim=300,
            isFirstLayer=False,
            isLastLayer=False,
        ):

        super().__init__()
        self.node_attr_dim = node_attr_dim
        self.edge_attr_dim = edge_attr_dim
        self.isFirstLayer = isFirstLayer
        self.isLastLayer = isLastLayer
        if isFirstLayer:
            self.lin0 = nn.Linear(node_attr_dim + edge_attr_dim, hidden_dim)

        self.mp_layer = MessagePassingLayer(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    add_residual=True
                )
        
        # concatenate the node features with the embedding from the message passing layers
        # self.final_message_passing_layer = MessagePassingLayer(
        #     input_dim= hidden_dim + node_attr_dim,
        #     hidden_dim=out_dim,
        #     add_residual=False
        # )   

    def forward(self, x, edge_index, edge_attr, edge_weight, node_weight, h0=None):
        # x, edge_index, edge_attr, edge_weight, node_weight = data.x, data.edge_index, data.edge_attr, data.edge_weight, data.node_weight    
        # print('x:', x.shape)
        # print('edge_index:', edge_index.shape)
        # print('edge_attr:', edge_attr.shape)
        # print('edge_weight:', edge_weight.shape)
        # print('node_weight:', node_weight.shape)
        if self.isFirstLayer:
            incoming_edges_weighted_sum = torch.zeros(x.size()[0], edge_attr.size()[1])
            # print('incoming_edges_weighted_sum:', incoming_edges_weighted_sum.shape)
            edge_index_reshaped = edge_index[1].view(-1, 1)
            # print('edge_index_reshaped:', edge_index_reshaped.shape)
            # min_index_value = torch.max(edge_index_reshaped)
            # print("maximum index value:", min_index_value.item())
            
            # the issue might be related to the fact that the target indices in edge_index_reshaped are not within the valid range for incoming_edges_weighted_sum. The size of incoming_edges_weighted_sum is (765, 14), and the target indices in edge_index_reshaped should be within the range [0, 764] to access valid indices along dimension 0.
            # sum over the rows (edges), index is the target node (i want to sum all edges where the target node is the same), src = attributes weighted
            incoming_edges_weighted_sum.scatter_add_(0, edge_index_reshaped.expand_as(edge_attr), edge_weight.view(-1, 1) * edge_attr)
            concat_features = torch.cat([x, incoming_edges_weighted_sum], dim=1)

            h0 = F.relu(self.lin0(concat_features))
            # for layer in self.message_passing_layers:
            h = self.mp_layer(h0, edge_index, edge_weight, h0)
        else:
            h = self.mp_layer(x, edge_index, edge_weight, h0)
            h0 = None
            
        # concatenate the node features with the embedding from the message passing layers
        # h = self.final_message_passing_layer(torch.cat([h, x], dim=1), edge_index, edge_weight, h0)

        # multiply the node features by the node weights
        if self.isLastLayer:
            h = h * node_weight.view(-1, 1)
        # h is be the (weighted) embeddings of each node of the graph
        return h, h0
    

    
# https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html#the-messagepassing-base-class
class MessagePassingLayer(MessagePassing):
    def __init__(
            self, 
            input_dim,
            hidden_dim,
            add_residual=True,
            aggr='mean', 
            flow='source_to_target'
        ):
        
        super().__init__(aggr=aggr,flow=flow)
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.add_residual = add_residual


    def forward(self, h_t, edge_index, edge_weight, h0):
        # propage will call the message function, then the aggregate (i.e. mean) function, and finally the update function.
        return self.propagate(edge_index, x=h_t, edge_weight=edge_weight, h0=h0)

    
    # take the features of the source nodes, weight them by the edge weight, and return the weighted features
    # Constructs messages to node i for each edge (j,i) if flow="source_to_target"
    # https://github.com/pyg-team/pytorch_geometric/issues/1489
    def message(self, x_i, x_j, edge_weight): 
        # x_i contains the node features of the target nodes 'i's for each edge (j,i). [num_edges, node_i_feats]   
        # x_j contains the node features of the source nodes 'j's for each edge (j, i). [num_edges, node_j_feats]
        # weight each edge by its probability, use x_j since i am interested in the sources nodes.
        return edge_weight.unsqueeze(1) * x_j

    # aggregates (take mean) of the incoming nodes weighted features (for node 'i' consider all nodes js where (j, i))
    # aggr_out has shape [num_nodes, node_hidden_channels]
    def update(self, aggr_out, h0):
        # aggr_out contains the output of aggregation. [num_nodes, node_hidden_channels]
        if self.add_residual:
            return F.relu(h0 + self.linear(aggr_out)) # [num_nodes, node_hidden_channels]
        else:
            return F.relu(self.linear(aggr_out))