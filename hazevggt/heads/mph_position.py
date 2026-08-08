"""Normalized image-plane coordinates for dense geometric prediction."""

import torch


def encode_geometry_grid(view_grid, embedding_dimension):
    grid_height, grid_width, _ = view_grid.shape
    flattened_grid = view_grid.reshape(-1, 2)
    horizontal_embedding = _encode_spatial_axis(
        embedding_dimension // 2, flattened_grid[:, 0]
    )
    vertical_embedding = _encode_spatial_axis(
        embedding_dimension // 2, flattened_grid[:, 1]
    )
    return torch.cat((horizontal_embedding, vertical_embedding), -1).view(
        grid_height, grid_width, embedding_dimension
    )


def _encode_spatial_axis(embedding_dimension, spatial_positions):
    angular_frequency = torch.arange(
        embedding_dimension // 2,
        dtype=torch.double,
        device=spatial_positions.device,
    )
    angular_frequency = 1 / 100 ** (
        angular_frequency / (embedding_dimension / 2)
    )
    phase = torch.einsum(
        "m,d->md", spatial_positions.reshape(-1), angular_frequency
    )
    return torch.cat((phase.sin(), phase.cos()), 1).float()


def create_normalized_view_grid(
    grid_width, grid_height, aspect_ratio, data_type, device
):
    view_diagonal = (aspect_ratio**2 + 1) ** 0.5
    horizontal_span = aspect_ratio / view_diagonal
    vertical_span = 1 / view_diagonal
    horizontal_positions = torch.linspace(
        -horizontal_span * (grid_width - 1) / grid_width,
        horizontal_span * (grid_width - 1) / grid_width,
        grid_width,
        dtype=data_type,
        device=device,
    )
    vertical_positions = torch.linspace(
        -vertical_span * (grid_height - 1) / grid_height,
        vertical_span * (grid_height - 1) / grid_height,
        grid_height,
        dtype=data_type,
        device=device,
    )
    return torch.stack(
        torch.meshgrid(
            horizontal_positions, vertical_positions, indexing="xy"
        ),
        -1,
    )
