"""MPH camera branch for the shared first-view reference frame."""

import torch
from torch import nn

from hazevggt.heads.mph_activation import decode_camera_parameters
from hazevggt.layers.geometry_mlp import GeometryFeedForward
from hazevggt.layers.geometry_transformer import GeometryTransformerBlock


class MPHCameraHead(nn.Module):
    """Iteratively decode intrinsic and extrinsic camera parameters."""

    def __init__(self):
        super().__init__()
        self.camera_transformer = nn.Sequential(
            *(
                GeometryTransformerBlock(
                    2048, 16, initial_feature_scale=0.01
                )
                for _ in range(4)
            )
        )
        self.camera_token_norm = nn.LayerNorm(2048)
        self.camera_output_norm = nn.LayerNorm(2048)
        self.reference_pose_tokens = nn.Parameter(torch.zeros(1, 1, 9))
        self.pose_embedding = nn.Linear(9, 2048)
        self.camera_film = nn.Sequential(nn.SiLU(), nn.Linear(2048, 6144))
        self.modulation_norm = nn.LayerNorm(
            2048, elementwise_affine=False, eps=1e-6
        )
        self.camera_projection = GeometryFeedForward(2048, 1024, 9)

    def forward(self, alternating_features):
        camera_tokens = self.camera_token_norm(
            alternating_features[-1][:, :, 0]
        )
        batch_size, view_count, _ = camera_tokens.shape
        camera_parameters = None
        for _ in range(4):
            pose_source = (
                self.reference_pose_tokens.expand(batch_size, view_count, -1)
                if camera_parameters is None
                else camera_parameters
            )
            film_bias, film_scale, confidence_gate = self.camera_film(
                self.pose_embedding(pose_source)
            ).chunk(3, -1)
            conditioned_camera_tokens = confidence_gate * (
                self.modulation_norm(camera_tokens) * (1 + film_scale)
                + film_bias
            )
            camera_update = self.camera_projection(
                self.camera_output_norm(
                    self.camera_transformer(
                        conditioned_camera_tokens + camera_tokens
                    )
                )
            )
            camera_parameters = (
                camera_update
                if camera_parameters is None
                else camera_parameters + camera_update
            )
        return decode_camera_parameters(camera_parameters)
