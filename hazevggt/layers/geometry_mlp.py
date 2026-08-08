"""Feed-forward projection for Haze-VGGT visual-geometry tokens."""

from torch import nn


class GeometryFeedForward(nn.Module):
    def __init__(
        self, input_features, hidden_features, output_features=None
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(
            hidden_features, output_features or input_features
        )

    def forward(self, geometry_tokens):
        return self.fc2(self.act(self.fc1(geometry_tokens)))
