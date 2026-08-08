"""HCV token estimation, confidence prediction, and FiLM-style fusion."""

import torch
from torch import nn

from .condition_codebook import HazeConditionCodec
from .condition_transformer import HCVResidualTransformer
from .feature_modulation import HCVResidualBlock


class HCVTokenEstimator(nn.Module):
    """Estimate latent degradation tokens from haze-observation features."""

    def __init__(self):
        super().__init__()
        self.condition_blocks = nn.ModuleList(
            HCVResidualTransformer() for _ in range(4)
        )
        self.output_norm = nn.LayerNorm(256)
        self.token_projection = nn.Sequential(nn.Linear(256, 1024, bias=False))

    def forward(self, haze_features):
        batch_size, feature_channels, feature_height, feature_width = (
            haze_features.shape
        )
        condition_tokens = haze_features.reshape(
            batch_size, feature_channels, feature_height * feature_width
        ).transpose(1, 2)
        for condition_block in self.condition_blocks:
            condition_tokens = condition_block(
                condition_tokens, (feature_height, feature_width)
            )
        return self.token_projection(self.output_norm(condition_tokens))


class HCVFeatureModulation(nn.Module):
    """Apply HCV-derived channel scaling and bias to decoder features."""

    def __init__(self, feature_channels):
        super().__init__()
        self.condition_encoder = HCVResidualBlock(
            2 * feature_channels, feature_channels
        )
        self.film_scale = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
        )
        self.film_bias = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
        )

    def forward(self, observation_features, conditioned_features, hcv_gate):
        haze_embedding = self.condition_encoder(
            torch.cat((observation_features, conditioned_features), 1)
        )
        modulated_residual = hcv_gate * (
            conditioned_features * self.film_scale(haze_embedding)
            + self.film_bias(haze_embedding)
        )
        return conditioned_features + modulated_residual


class HCVConfidenceHead(nn.Module):
    """Predict the confidence gate used during latent HCV refinement."""

    def __init__(self):
        super().__init__()
        self.confidence_blocks = nn.ModuleList(
            HCVResidualTransformer() for _ in range(2)
        )
        self.output_norm = nn.LayerNorm(256)
        self.hcv_embedding = nn.Embedding(1024, 256)
        self.confidence_projection = nn.Sequential(nn.Linear(256, 1))

    def forward(self, condition_tokens, feature_height, feature_width):
        haze_embedding = self.hcv_embedding(condition_tokens)
        for confidence_block in self.confidence_blocks:
            haze_embedding = confidence_block(
                haze_embedding, (feature_height, feature_width)
            )
        return self.confidence_projection(
            self.output_norm(haze_embedding)
        ).squeeze(-1)


class HazeConditionNetwork(nn.Module):
    """Compact HCV front end used by the paper-facing inference entry point."""

    def __init__(self):
        super().__init__()
        self.hcv_resolution = 256
        self.condition_depth = 2
        self.hcv_codec = HazeConditionCodec()
        self.hcv_estimator = HCVTokenEstimator()
        self.film_blocks = nn.ModuleDict(
            {
                "64": HCVFeatureModulation(256),
                "128": HCVFeatureModulation(128),
            }
        )
