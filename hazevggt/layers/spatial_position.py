"""Spatial coordinates and rotary embeddings for cross-view geometry tokens."""

import torch
import torch.nn.functional as F
from torch import nn


class SpatialPositionGrid:
    def __init__(self):
        self.cache = {}

    def __call__(self, batch_size, grid_height, grid_width, device):
        cache_key = (grid_height, grid_width, device)
        if cache_key not in self.cache:
            vertical_positions = torch.arange(grid_height, device=device)
            horizontal_positions = torch.arange(grid_width, device=device)
            self.cache[cache_key] = torch.cartesian_prod(
                vertical_positions, horizontal_positions
            )
        return self.cache[cache_key].view(
            1, grid_height * grid_width, 2
        ).expand(batch_size, -1, -1)


class SpatialRotaryEmbedding(nn.Module):
    """Retain 2D spatial alignment during frame and global attention."""

    def __init__(self, spatial_frequency):
        super().__init__()
        self.frequency = spatial_frequency
        self.cache = {}

    def _frequency_components(
        self, embedding_dimension, sequence_length, device, data_type
    ):
        cache_key = (
            embedding_dimension,
            sequence_length,
            device,
            data_type,
        )
        if cache_key not in self.cache:
            frequency_exponent = (
                torch.arange(0, embedding_dimension, 2, device=device).float()
                / embedding_dimension
            )
            inverse_frequency = 1 / self.frequency**frequency_exponent
            spatial_index = torch.arange(
                sequence_length,
                device=device,
                dtype=inverse_frequency.dtype,
            )
            rotary_angle = torch.einsum(
                "i,j->ij", spatial_index, inverse_frequency
            ).to(data_type)
            rotary_angle = torch.cat((rotary_angle, rotary_angle), -1)
            self.cache[cache_key] = (
                rotary_angle.cos(),
                rotary_angle.sin(),
            )
        return self.cache[cache_key]

    @staticmethod
    def _rotate(spatial_tokens):
        first_half, second_half = spatial_tokens.chunk(2, -1)
        return torch.cat((-second_half, first_half), -1)

    def _apply_spatial_rope(
        self, spatial_tokens, positions, cosine, sine
    ):
        cosine = F.embedding(positions, cosine)[:, None]
        sine = F.embedding(positions, sine)[:, None]
        return spatial_tokens * cosine + self._rotate(spatial_tokens) * sine

    def forward(self, geometry_tokens, spatial_positions):
        embedding_dimension = geometry_tokens.shape[-1] // 2
        cosine, sine = self._frequency_components(
            embedding_dimension,
            int(spatial_positions.max()) + 1,
            geometry_tokens.device,
            geometry_tokens.dtype,
        )
        vertical_tokens, horizontal_tokens = geometry_tokens.chunk(2, -1)
        vertical_tokens = self._apply_spatial_rope(
            vertical_tokens, spatial_positions[..., 0], cosine, sine
        )
        horizontal_tokens = self._apply_spatial_rope(
            horizontal_tokens, spatial_positions[..., 1], cosine, sine
        )
        return torch.cat((vertical_tokens, horizontal_tokens), -1)
