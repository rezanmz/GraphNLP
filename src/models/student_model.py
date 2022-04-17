import torch
from torch import nn
from models.graph_models import EdgeConstruction, FeatureConstruction, GCN
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

        self.edge_construction = EdgeConstruction(
            num_feats, edge_construction_hidden_dims)
        self.feature_construction = FeatureConstruction(
            num_feats, feature_construction_hidden_dims, feature_construction_output_dim)
        self.gcn = GCN(feature_construction_output_dim,
                       gcn_hidden_dims, gcn_output_dim)

    def forward(self, features, attention_mask):
        n_samples = features.size(0)
        n_tokens = features.size(1)
        embedding_dim = features.size(2)

        pairwise_features = torch.cartesian_prod(torch.tensor(
            range(n_tokens)), torch.tensor(range(n_tokens)))
        pairwise_features = torch.concat([features[sample][pairwise_features].view(
            pairwise_features.size(0), -1) for sample in range(n_samples)])

        edge_construction_output = self.edge_construction(pairwise_features)
        edge_construction_output = edge_construction_output.view(
            n_samples, n_tokens, n_tokens)

        features = features.view(-1, embedding_dim)
        feature_construction_output = self.feature_construction(features)
        feature_construction_output = feature_construction_output.view(
            n_samples, n_tokens, -1)

        gcn_output = self.gcn(
            edge_construction_output,
            feature_construction_output,
            attention_mask
        )

        return torch.tanh(gcn_output)
