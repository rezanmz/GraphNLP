from typing import List
from torch import nn
from torch_geometric.nn.dense import DenseGCNConv

class EdgeConstruction(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()

        # Output size is always 1 (model decides whether there should be an edge or not)
        hidden_dims.append(1)
        
        self.linear = nn.ModuleList([nn.Linear(input_dim * 2, hidden_dims[0])])
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.linear.append(nn.ReLU())
            self.linear.append(nn.Linear(in_dim, out_dim))

        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        for layer in self.linear:
            x = layer(x)
        return self.activation(x)


class FeatureConstruction(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()

        hidden_dims.append(output_dim)
        
        self.linear = nn.ModuleList([nn.Linear(input_dim, hidden_dims[0])])
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.linear.append(nn.ReLU())
            self.linear.append(nn.Linear(in_dim, out_dim))

    
    def forward(self, x):
        for layer in self.linear:
            x = layer(x)
        return x


class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()

        hidden_dims.append(output_dim)
        
        self.gcn_conv = nn.ModuleList([DenseGCNConv(input_dim, hidden_dims[0])])
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.gcn_conv.append(DenseGCNConv(in_dim, out_dim))
        
        self.activation = nn.ReLU()
    
    def forward(self, adj, features):
        num_nodes = features.shape[0]
        adj = adj.view(num_nodes, num_nodes) 
        for layer in self.gcn_conv[:-1]:
            features = layer(features, adj)
            features = self.activation(features)
        features = self.gcn_conv[-1](features, adj)
        return features