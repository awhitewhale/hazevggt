"""Transformer block shared by the alternating Haze-VGGT backbone."""

from functools import partial

from torch import nn

from .feature_scale import FeatureResponseScale
from .geometry_attention import GeometryAttention
from .geometry_mlp import GeometryFeedForward


class GeometryTransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dimension,
        attention_heads,
        mlp_ratio=4,
        initial_feature_scale=None,
        normalize_query_key=False,
        spatial_rope=None,
        normalization_epsilon=1e-5,
    ):
        super().__init__()
        geometry_norm = partial(nn.LayerNorm, eps=normalization_epsilon)
        self.norm1 = geometry_norm(embedding_dimension)
        self.attn = GeometryAttention(
            embedding_dimension,
            attention_heads,
            normalize_query_key,
            normalization_epsilon,
            spatial_rope,
        )
        self.ls1 = (
            FeatureResponseScale(embedding_dimension, initial_feature_scale)
            if initial_feature_scale
            else nn.Identity()
        )
        self.norm2 = geometry_norm(embedding_dimension)
        self.mlp = GeometryFeedForward(
            embedding_dimension, int(embedding_dimension * mlp_ratio)
        )
        self.ls2 = (
            FeatureResponseScale(embedding_dimension, initial_feature_scale)
            if initial_feature_scale
            else nn.Identity()
        )

    def forward(self, geometry_tokens, positions=None):
        geometry_tokens = geometry_tokens + self.ls1(
            self.attn(self.norm1(geometry_tokens), positions)
        )
        return geometry_tokens + self.ls2(
            self.mlp(self.norm2(geometry_tokens))
        )
