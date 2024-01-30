import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_max_pool, global_add_pool, global_mean_pool

# node-centered message passing
class WDNodeMPNN(nn.Module):
    class_name = "WDNodeMPNN"

    def __init__(
            self, 
            node_attr_dim, 
            edge_attr_dim,
            n_message_passing_layers=3,
            hidden_dim=300,
            agg_func="mean"
        ):

        super().__init__()
        self.node_attr_dim = node_attr_dim
        self.edge_attr_dim = edge_attr_dim
        self.n_message_passing_layers = n_message_passing_layers
        self.message_passing_layers = nn.ModuleList()
        self.lin0 = nn.Linear(node_attr_dim + edge_attr_dim, hidden_dim)

        assert agg_func in ['mean', 'max', 'add']
        self.agg_func = {
           'mean': global_mean_pool,
           'max': global_max_pool,
           'add': global_add_pool
        }[agg_func]

        for _ in range(n_message_passing_layers):
            self.message_passing_layers.append(
                MessagePassingLayer(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    add_residual=True
                )
            )
        
        # concatenate the node features with the embedding from the message passing layers
        self.final_message_passing_layer = MessagePassingLayer(
            input_dim= hidden_dim + node_attr_dim,
            hidden_dim=hidden_dim,
            add_residual=False
        )   
        
        # final mlp to predict the label
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )


    def forward(self, data):
        x, edge_index, edge_attr, edge_weight, node_weight = data.x, data.edge_index, data.edge_attr, data.edge_weight, data.node_weight    
        
        incoming_edges_weighted_sum = torch.zeros(x.size()[0], edge_attr.size()[1])
        edge_index_reshaped = edge_index[1].view(-1, 1)
        # sum over the rows (edges), index is the target node (i want to sum all edges where the target node is the same), src = attributes weighted
        incoming_edges_weighted_sum.scatter_add_(0, edge_index_reshaped.expand_as(edge_attr), edge_weight.view(-1, 1) * edge_attr)
        concat_features = torch.cat([x, incoming_edges_weighted_sum], dim=1)
      
        h0 = F.relu(self.lin0(concat_features))
        h = h0
        for layer in self.message_passing_layers:
            h = layer(h, edge_index, edge_weight, h0)
            
        # concatenate the node features with the embedding from the message passing layers
        h = self.final_message_passing_layer(torch.cat([h, x], dim=1), edge_index, edge_weight, h0)


        graph_embedding = self.agg_func(h * node_weight.view(-1, 1), data.batch)

        out = self.final_mlp(graph_embedding)
        return out.squeeze(1)

    
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