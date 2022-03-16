from torch import nn
from graph_models import EdgeConstruction, FeatureConstruction, GCN
from typing import List

class StudentModel(nn.Module):
    def __init__(
        self,
        num_feats: int,
        edge_construction_hidden_dims: List[int],
        feature_construction_hidden_dims: List[int],
        gcn_hidden_dims: List[int],
        feature_construction_output_dim: int,
        gcn_output_dim: int
    ):
        super().__init__()

        self.edge_construction = EdgeConstruction(num_feats, edge_construction_hidden_dims)
        self.feature_construction = FeatureConstruction(num_feats, feature_construction_hidden_dims, feature_construction_output_dim)
        self.gcn = GCN(feature_construction_output_dim, gcn_hidden_dims, gcn_output_dim)

    def forward(self, features):
        features = features.squeeze()
        pairwise_features = torch.cartesian_prod(torch.tensor(range(features.shape[0])), torch.tensor(range(features.shape[0])))
        pairwise_features = torch.stack([torch.stack([features[idx[0]], features[idx[1]]]).flatten() for idx in pairwise_features])
        
        edge_construction_output = self.edge_construction(pairwise_features)
        feature_construction_output = self.feature_construction(features)

        gcn_output = self.gcn(edge_construction_output, feature_construction_output)

        return gcn_output.mean(1)