"""Self-attention for frame-wise and global visual-geometry interaction."""

import torch.nn.functional as F
from torch import nn


class GeometryAttention(nn.Module):
    def __init__(
        self,
        embedding_dimension,
        attention_heads,
        normalize_query_key=False,
        normalization_epsilon=1e-5,
        spatial_rope=None,
    ):
        super().__init__()
        self.num_heads = attention_heads
        self.head_dim = embedding_dimension // attention_heads
        self.qkv = nn.Linear(embedding_dimension, embedding_dimension * 3)
        self.q_norm = (
            nn.LayerNorm(self.head_dim, eps=normalization_epsilon)
            if normalize_query_key
            else nn.Identity()
        )
        self.k_norm = (
            nn.LayerNorm(self.head_dim, eps=normalization_epsilon)
            if normalize_query_key
            else nn.Identity()
        )
        self.proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.rope = spatial_rope

    def forward(self, geometry_tokens, positions=None):
        batch_size, token_count, embedding_dimension = geometry_tokens.shape
        query_key_value = self.qkv(geometry_tokens).reshape(
            batch_size, token_count, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = query_key_value.unbind(0)
        query, key = self.q_norm(query), self.k_norm(key)
        if self.rope is not None:
            query = self.rope(query, positions)
            key = self.rope(key, positions)
        attended_tokens = F.scaled_dot_product_attention(query, key, value)
        attended_tokens = attended_tokens.transpose(1, 2).reshape(
            batch_size, token_count, embedding_dimension
        )
        return self.proj(attended_tokens)
