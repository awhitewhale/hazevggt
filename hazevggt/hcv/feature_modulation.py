"""Feature-normalization primitives for haze-conditioned representation learning."""

from torch import nn


class HCVFeatureNorm(nn.Module):
    def __init__(self, feature_channels):
        super().__init__()
        self.norm = nn.GroupNorm(
            32, feature_channels, eps=1e-6, affine=True
        )

    def forward(self, features):
        return self.norm(features)


class HCVActivation(nn.Module):
    def __init__(self, activation_name):
        super().__init__()
        self.func = (
            nn.SiLU(True)
            if activation_name == "silu"
            else nn.LeakyReLU(0.2, True)
        )

    def forward(self, features):
        return self.func(features)


class HCVResidualBlock(nn.Module):
    """Preserve local haze evidence while updating condition features."""

    def __init__(
        self, input_channels, output_channels, activation_name="leakyrelu"
    ):
        super().__init__()
        self.conv = nn.Sequential(
            HCVFeatureNorm(input_channels),
            HCVActivation(activation_name),
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            HCVFeatureNorm(output_channels),
            HCVActivation(activation_name),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
        )
        self.conv_res = (
            nn.Conv2d(output_channels, output_channels, 1, bias=False)
            if input_channels != output_channels
            else None
        )

    def forward(self, condition_features):
        residual_features = condition_features
        updated_features = self.conv(condition_features)
        if self.conv_res is not None:
            residual_features = self.conv_res(updated_features)
        return updated_features + residual_features


class HCVPostProjection(nn.Module):
    def __init__(self, feature_channels):
        super().__init__()
        self.conv = nn.Conv2d(feature_channels, feature_channels, 3, 1, 1)

    def forward(self, haze_embedding):
        return self.conv(haze_embedding)
