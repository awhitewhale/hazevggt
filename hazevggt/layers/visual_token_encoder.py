"""Large visual-token encoder used by the Haze-VGGT geometry backbone."""

import math

import torch
from torch import nn

from .geometry_transformer import GeometryTransformerBlock
from .view_token_embedding import ViewTokenEmbedding


class VisualGeometryEncoder(nn.Module):
    """Encode each conditioned view before alternating multi-view fusion."""

    def __init__(self):
        super().__init__()
        self.patch_embed = ViewTokenEmbedding()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1024))
        self.pos_embed = nn.Parameter(torch.zeros(1, 37 * 37 + 1, 1024))
        self.register_tokens = nn.Parameter(torch.zeros(1, 4, 1024))
        self.blocks = nn.ModuleList(
            GeometryTransformerBlock(
                1024,
                16,
                initial_feature_scale=1,
                normalization_epsilon=1e-6,
            )
            for _ in range(24)
        )
        self.norm = nn.LayerNorm(1024, eps=1e-6)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.register_tokens, std=1e-6)
        for encoder_module in self.modules():
            if isinstance(encoder_module, nn.Linear):
                nn.init.trunc_normal_(encoder_module.weight, std=0.02)
                if encoder_module.bias is not None:
                    nn.init.zeros_(encoder_module.bias)

    def _interpolate_spatial_embedding(
        self, visual_tokens, image_height, image_width
    ):
        patch_count = visual_tokens.shape[1] - 1
        if patch_count == 37 * 37 and image_height == image_width:
            return self.pos_embed
        class_position = self.pos_embed[:, :1]
        patch_positions = self.pos_embed[:, 1:]
        patch_grid_side = int(math.sqrt(patch_positions.shape[1]))
        patch_positions = nn.functional.interpolate(
            patch_positions.reshape(
                1, patch_grid_side, patch_grid_side, 1024
            ).permute(0, 3, 1, 2),
            size=(image_height // 14, image_width // 14),
            mode="bicubic",
            antialias=True,
        )
        patch_positions = patch_positions.permute(0, 2, 3, 1).reshape(
            1, -1, 1024
        )
        return torch.cat((class_position, patch_positions), 1).to(
            visual_tokens.dtype
        )

    def forward(self, conditioned_views):
        batch_size, _, image_height, image_width = conditioned_views.shape
        visual_tokens = self.patch_embed(conditioned_views)
        visual_tokens = torch.cat(
            (self.cls_token.expand(batch_size, -1, -1), visual_tokens), 1
        )
        visual_tokens = visual_tokens + self._interpolate_spatial_embedding(
            visual_tokens, image_height, image_width
        )
        visual_tokens = torch.cat(
            (
                visual_tokens[:, :1],
                self.register_tokens.expand(batch_size, -1, -1),
                visual_tokens[:, 1:],
            ),
            1,
        )
        for geometry_block in self.blocks:
            visual_tokens = geometry_block(visual_tokens)
        visual_tokens = self.norm(visual_tokens)
        return {"normalized_visual_tokens": visual_tokens[:, 5:]}


def build_visual_token_encoder():
    return VisualGeometryEncoder()
