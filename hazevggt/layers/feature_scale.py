"""Channel-wise response scaling for conditioned geometry features."""

import torch
from torch import nn


class FeatureResponseScale(nn.Module):
    def __init__(self, embedding_dimension, initial_value):
        super().__init__()
        self.gamma = nn.Parameter(
            initial_value * torch.ones(embedding_dimension)
        )

    def forward(self, geometry_features):
        return geometry_features * self.gamma
