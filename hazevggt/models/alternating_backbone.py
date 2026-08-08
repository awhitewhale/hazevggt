"""Alternating frame-wise and global attention backbone for Haze-VGGT."""

import torch
from torch import nn

from hazevggt.layers.geometry_transformer import GeometryTransformerBlock
from hazevggt.layers.spatial_position import SpatialPositionGrid, SpatialRotaryEmbedding
from hazevggt.layers.visual_token_encoder import build_visual_token_encoder


class AlternatingViewBackbone(nn.Module):
    """Fuse local structure and cross-view evidence over 24 alternating layers."""

    def __init__(self):
        super().__init__()
        self.visual_encoder = build_visual_token_encoder()
        self.spatial_rope = SpatialRotaryEmbedding(100)
        self.position_grid = SpatialPositionGrid()
        self.frame_attention = nn.ModuleList(
            GeometryTransformerBlock(
                1024,
                16,
                initial_feature_scale=0.01,
                normalize_query_key=True,
                spatial_rope=self.spatial_rope,
            )
            for _ in range(24)
        )
        self.global_attention = nn.ModuleList(
            GeometryTransformerBlock(
                1024,
                16,
                initial_feature_scale=0.01,
                normalize_query_key=True,
                spatial_rope=self.spatial_rope,
            )
            for _ in range(24)
        )
        self.reference_camera_token = nn.Parameter(torch.randn(1, 2, 1, 1024))
        self.geometry_register_tokens = nn.Parameter(torch.randn(1, 2, 4, 1024))
        self.visual_token_start = 5
        nn.init.normal_(self.reference_camera_token, std=1e-6)
        nn.init.normal_(self.geometry_register_tokens, std=1e-6)
        self.register_buffer(
            "_image_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_image_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, conditioned_views):
        batch_size, view_count, color_channels, image_height, image_width = (
            conditioned_views.shape
        )
        normalized_views = (
            (conditioned_views - self._image_mean) / self._image_std
        ).view(batch_size * view_count, color_channels, image_height, image_width)
        visual_tokens = self.visual_encoder(normalized_views)[
            "normalized_visual_tokens"
        ]
        camera_tokens = _expand_reference_tokens(
            self.reference_camera_token, batch_size, view_count
        )
        register_tokens = _expand_reference_tokens(
            self.geometry_register_tokens, batch_size, view_count
        )
        geometry_tokens = torch.cat(
            (camera_tokens, register_tokens, visual_tokens), 1
        )

        spatial_positions = self.position_grid(
            batch_size * view_count,
            image_height // 14,
            image_width // 14,
            normalized_views.device,
        ) + 1
        spatial_positions = torch.cat(
            (
                torch.zeros(
                    batch_size * view_count,
                    5,
                    2,
                    device=normalized_views.device,
                    dtype=spatial_positions.dtype,
                ),
                spatial_positions,
            ),
            1,
        )

        token_count, embedding_dimension = geometry_tokens.shape[1:]
        alternating_features = []
        for layer_index in range(24):
            geometry_tokens = geometry_tokens.view(
                batch_size * view_count, token_count, embedding_dimension
            )
            frame_positions = spatial_positions.view(
                batch_size * view_count, token_count, 2
            )
            frame_features = self.frame_attention[layer_index](
                geometry_tokens, positions=frame_positions
            )

            geometry_tokens = frame_features.view(
                batch_size, view_count * token_count, embedding_dimension
            )
            global_positions = spatial_positions.view(
                batch_size, view_count * token_count, 2
            )
            global_features = self.global_attention[layer_index](
                geometry_tokens, positions=global_positions
            )
            geometry_tokens = global_features
            alternating_features.append(
                torch.cat(
                    (
                        frame_features.view(
                            batch_size,
                            view_count,
                            token_count,
                            embedding_dimension,
                        ),
                        global_features.view(
                            batch_size,
                            view_count,
                            token_count,
                            embedding_dimension,
                        ),
                    ),
                    -1,
                )
            )
        return alternating_features, self.visual_token_start


def _expand_reference_tokens(reference_tokens, batch_size, view_count):
    """Anchor the first-view coordinate system and expand the remaining views."""
    first_view_tokens = reference_tokens[:, :1].expand(
        batch_size, 1, *reference_tokens.shape[2:]
    )
    remaining_view_tokens = reference_tokens[:, 1:].expand(
        batch_size, view_count - 1, *reference_tokens.shape[2:]
    )
    return torch.cat((first_view_tokens, remaining_view_tokens), 1).view(
        batch_size * view_count, *reference_tokens.shape[2:]
    )
