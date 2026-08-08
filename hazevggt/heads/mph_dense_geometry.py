"""MPH dense-geometry branch with per-pixel reliability prediction."""

import torch
import torch.nn.functional as F
from torch import nn

from .mph_activation import decode_depth_reliability
from .mph_position import create_normalized_view_grid, encode_geometry_grid


class MPHDenseGeometryHead(nn.Module):
    """Decode multi-scale backbone features into depth and reliability maps."""

    def __init__(self):
        super().__init__()
        self.geometry_norm = nn.LayerNorm(2048)
        self.scale_projections = nn.ModuleList(
            nn.Conv2d(2048, feature_channels, 1)
            for feature_channels in (256, 512, 1024, 1024)
        )
        self.scale_alignment = nn.ModuleList(
            (
                nn.ConvTranspose2d(256, 256, 4, 4),
                nn.ConvTranspose2d(512, 512, 2, 2),
                nn.Identity(),
                nn.Conv2d(1024, 1024, 3, 2, 1),
            )
        )
        self.dense_decoder = _build_dense_geometry_decoder()

    def forward(
        self, alternating_features, conditioned_views, visual_token_start
    ):
        view_count = conditioned_views.shape[1]
        depth_sequence = []
        reliability_sequence = []
        for view_start in range(0, view_count, 8):
            view_end = min(view_start + 8, view_count)
            depth_chunk, reliability_chunk = self._predict_view_chunk(
                alternating_features,
                conditioned_views[:, view_start:view_end],
                visual_token_start,
                view_start,
                view_end,
            )
            depth_sequence.append(depth_chunk)
            reliability_sequence.append(reliability_chunk)
        return torch.cat(depth_sequence, 1), torch.cat(reliability_sequence, 1)

    def _predict_view_chunk(
        self,
        alternating_features,
        conditioned_views,
        visual_token_start,
        view_start,
        view_end,
    ):
        batch_size, view_count, _, image_height, image_width = (
            conditioned_views.shape
        )
        patch_height, patch_width = image_height // 14, image_width // 14
        geometry_scales = []
        for scale_index, layer_index in enumerate((4, 11, 17, 23)):
            scale_tokens = alternating_features[layer_index][
                :, view_start:view_end, visual_token_start:
            ]
            scale_tokens = self.geometry_norm(
                scale_tokens.reshape(
                    batch_size * view_count, -1, scale_tokens.shape[-1]
                )
            )
            scale_features = scale_tokens.permute(0, 2, 1).reshape(
                batch_size * view_count,
                2048,
                patch_height,
                patch_width,
            )
            scale_features = self.scale_projections[scale_index](scale_features)
            scale_features = self._inject_spatial_positions(
                scale_features, image_width, image_height
            )
            geometry_scales.append(
                self.scale_alignment[scale_index](scale_features)
            )

        dense_features = self._fuse_dense_scales(geometry_scales)
        dense_features = F.interpolate(
            dense_features,
            (image_height, image_width),
            mode="bilinear",
            align_corners=True,
        )
        dense_features = self._inject_spatial_positions(
            dense_features, image_width, image_height
        )
        depth_maps, geometry_reliability = decode_depth_reliability(
            self.dense_decoder.depth_reliability_projection(dense_features)
        )
        return (
            depth_maps.view(
                batch_size, view_count, *depth_maps.shape[1:]
            ),
            geometry_reliability.view(
                batch_size, view_count, *geometry_reliability.shape[1:]
            ),
        )

    def _inject_spatial_positions(
        self, geometry_features, image_width, image_height
    ):
        view_grid = create_normalized_view_grid(
            geometry_features.shape[-1],
            geometry_features.shape[-2],
            image_width / image_height,
            geometry_features.dtype,
            geometry_features.device,
        )
        spatial_embedding = encode_geometry_grid(
            view_grid, geometry_features.shape[1]
        )
        spatial_embedding = spatial_embedding.permute(2, 0, 1)[None].expand(
            geometry_features.shape[0], -1, -1, -1
        )
        return geometry_features + spatial_embedding * 0.1

    def _fuse_dense_scales(self, geometry_scales):
        scale1, scale2, scale3, scale4 = geometry_scales
        scale1 = self.dense_decoder.scale1_projection(scale1)
        scale2 = self.dense_decoder.scale2_projection(scale2)
        scale3 = self.dense_decoder.scale3_projection(scale3)
        scale4 = self.dense_decoder.scale4_projection(scale4)
        fused_geometry = self.dense_decoder.fusion_stage4(
            scale4, size=scale3.shape[2:]
        )
        fused_geometry = self.dense_decoder.fusion_stage3(
            fused_geometry, scale3, size=scale2.shape[2:]
        )
        fused_geometry = self.dense_decoder.fusion_stage2(
            fused_geometry, scale2, size=scale1.shape[2:]
        )
        fused_geometry = self.dense_decoder.fusion_stage1(
            fused_geometry, scale1
        )
        return self.dense_decoder.dense_projection(fused_geometry)


def _build_dense_geometry_decoder():
    dense_decoder = nn.Module()
    dense_decoder.scale1_projection = nn.Conv2d(
        256, 256, 3, 1, 1, bias=False
    )
    dense_decoder.scale2_projection = nn.Conv2d(
        512, 256, 3, 1, 1, bias=False
    )
    dense_decoder.scale3_projection = nn.Conv2d(
        1024, 256, 3, 1, 1, bias=False
    )
    dense_decoder.scale4_projection = nn.Conv2d(
        1024, 256, 3, 1, 1, bias=False
    )
    dense_decoder.fusion_stage1 = DenseFeatureFusion(True)
    dense_decoder.fusion_stage2 = DenseFeatureFusion(True)
    dense_decoder.fusion_stage3 = DenseFeatureFusion(True)
    dense_decoder.fusion_stage4 = DenseFeatureFusion(False)
    dense_decoder.dense_projection = nn.Conv2d(256, 128, 3, 1, 1)
    dense_decoder.depth_reliability_projection = nn.Sequential(
        nn.Conv2d(128, 32, 3, 1, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 2, 1),
    )
    return dense_decoder


class ResidualGeometryUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.geometry_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.geometry_conv2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.geometry_activation = nn.ReLU(inplace=True)

    def forward(self, geometry_features):
        residual_features = self.geometry_conv1(
            self.geometry_activation(geometry_features)
        )
        residual_features = self.geometry_conv2(
            self.geometry_activation(residual_features)
        )
        return residual_features + geometry_features


class DenseFeatureFusion(nn.Module):
    """Fuse DPT features while preserving reliable multi-scale geometry."""

    def __init__(self, use_residual):
        super().__init__()
        self.geometry_projection = nn.Conv2d(256, 256, 1)
        if use_residual:
            self.residual_geometry = ResidualGeometryUnit()
        self.use_residual = use_residual
        self.output_geometry = ResidualGeometryUnit()

    def forward(
        self, geometry_features, residual_features=None, output_size=None, size=None
    ):
        target_size = output_size if output_size is not None else size
        if self.use_residual:
            geometry_features = geometry_features + self.residual_geometry(
                residual_features
            )
        geometry_features = self.output_geometry(geometry_features)
        geometry_features = F.interpolate(
            geometry_features,
            size=target_size,
            scale_factor=None if target_size is not None else 2,
            mode="bilinear",
            align_corners=True,
        )
        return self.geometry_projection(geometry_features)
