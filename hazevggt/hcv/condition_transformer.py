"""Windowed Transformer used to estimate spatially varying haze conditions."""

import torch
from torch import nn


class HCVFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 1024)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(1024, 256)

    def forward(self, condition_tokens):
        return self.fc2(self.act(self.fc1(condition_tokens)))


def _partition_condition_windows(condition_features):
    batch_size, feature_height, feature_width, feature_channels = (
        condition_features.shape
    )
    condition_features = condition_features.view(
        batch_size,
        feature_height // 8,
        8,
        feature_width // 8,
        8,
        feature_channels,
    )
    return (
        condition_features.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, 8, 8, feature_channels)
    )


def _merge_condition_windows(condition_windows, feature_height, feature_width):
    batch_size = int(
        condition_windows.shape[0] / (feature_height * feature_width / 64)
    )
    condition_features = condition_windows.view(
        batch_size,
        feature_height // 8,
        feature_width // 8,
        8,
        8,
        -1,
    )
    return (
        condition_features.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(batch_size, feature_height, feature_width, -1)
    )


class HCVWindowAttention(nn.Module):
    """Aggregate local degradation evidence within an HCV feature window."""

    def __init__(self):
        super().__init__()
        self.scale = (256 // 8) ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros(225, 8))

        grid_coordinates = torch.stack(
            torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
        )
        grid_coordinates = torch.flatten(grid_coordinates, 1)
        relative_coordinates = (
            grid_coordinates[:, :, None] - grid_coordinates[:, None, :]
        )
        relative_coordinates = relative_coordinates.permute(1, 2, 0).contiguous()
        relative_coordinates[:, :, 0] += 7
        relative_coordinates[:, :, 1] += 7
        relative_coordinates[:, :, 0] *= 15
        self.register_buffer(
            "relative_position_index", relative_coordinates.sum(-1)
        )

        self.qkv = nn.Linear(256, 768)
        self.proj = nn.Linear(256, 256)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, condition_tokens, attention_mask):
        batch_size, token_count, feature_channels = condition_tokens.shape
        query_key_value = self.qkv(condition_tokens).reshape(
            batch_size, token_count, 3, 8, 32
        ).permute(2, 0, 3, 1, 4)
        query, key, value = query_key_value.unbind(0)
        attention_logits = (query * self.scale) @ key.transpose(-2, -1)
        position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(64, 64, 8).permute(2, 0, 1)
        attention_logits = attention_logits + position_bias.unsqueeze(0)
        if attention_mask is not None:
            window_count = attention_mask.shape[0]
            attention_logits = attention_logits.view(
                -1, window_count, 8, 64, 64
            )
            attention_logits = attention_logits + attention_mask.unsqueeze(
                1
            ).unsqueeze(0)
            attention_logits = attention_logits.view(-1, 8, 64, 64)
        attention_weights = self.softmax(attention_logits)
        conditioned_tokens = (attention_weights @ value).transpose(1, 2).reshape(
            batch_size, token_count, feature_channels
        )
        return self.proj(conditioned_tokens)


class HCVTransformerBlock(nn.Module):
    def __init__(self, shift_size):
        super().__init__()
        self.input_resolution = (32, 32)
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(256)
        self.attn = HCVWindowAttention()
        self.norm2 = nn.LayerNorm(256)
        self.mlp = HCVFeedForward()
        self.register_buffer(
            "attn_mask", self._make_mask(32, 32) if shift_size else None
        )

    def _make_mask(self, feature_height, feature_width):
        region_mask = torch.zeros((1, feature_height, feature_width, 1))
        shifted_regions = (slice(0, -8), slice(-8, -4), slice(-4, None))
        region_index = 0
        for height_slice in shifted_regions:
            for width_slice in shifted_regions:
                region_mask[:, height_slice, width_slice] = region_index
                region_index += 1
        condition_windows = _partition_condition_windows(region_mask).view(-1, 64)
        attention_mask = condition_windows.unsqueeze(1) - condition_windows.unsqueeze(2)
        return attention_mask.masked_fill(attention_mask != 0, -100).masked_fill(
            attention_mask == 0, 0
        )

    def forward(self, condition_tokens, feature_size):
        feature_height, feature_width = feature_size
        batch_size, _, feature_channels = condition_tokens.shape
        residual_tokens = condition_tokens
        condition_tokens = self.norm1(condition_tokens).view(
            batch_size, feature_height, feature_width, feature_channels
        )
        if self.shift_size:
            condition_tokens = torch.roll(
                condition_tokens, shifts=(-4, -4), dims=(1, 2)
            )
        condition_windows = _partition_condition_windows(condition_tokens).view(
            -1, 64, feature_channels
        )
        attention_mask = (
            self.attn_mask
            if feature_size == self.input_resolution
            else self._make_mask(feature_height, feature_width).to(
                condition_tokens.device
            )
        )
        condition_windows = self.attn(
            condition_windows, attention_mask
        ).view(-1, 8, 8, feature_channels)
        condition_tokens = _merge_condition_windows(
            condition_windows, feature_height, feature_width
        )
        if self.shift_size:
            condition_tokens = torch.roll(
                condition_tokens, shifts=(4, 4), dims=(1, 2)
            )
        condition_tokens = residual_tokens + condition_tokens.view(
            batch_size, feature_height * feature_width, feature_channels
        )
        return condition_tokens + self.mlp(self.norm2(condition_tokens))


class HCVTransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            HCVTransformerBlock(0 if block_index % 2 == 0 else 4)
            for block_index in range(6)
        )

    def forward(self, condition_tokens, feature_size):
        for transformer_block in self.blocks:
            condition_tokens = transformer_block(condition_tokens, feature_size)
        return condition_tokens


class HCVPatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, condition_features):
        return condition_features.flatten(2).transpose(1, 2)


class HCVPatchRecovery(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, condition_tokens, feature_size):
        batch_size = condition_tokens.shape[0]
        return condition_tokens.transpose(1, 2).view(
            batch_size, 256, feature_size[0], feature_size[1]
        )


class HCVResidualTransformer(nn.Module):
    """Residual window attention block for compact condition reasoning."""

    def __init__(self):
        super().__init__()
        self.residual_group = HCVTransformerLayer()
        self.conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.patch_embed = HCVPatchEmbedding()
        self.patch_unembed = HCVPatchRecovery()

    def forward(self, condition_tokens, feature_size):
        residual_tokens = self.residual_group(condition_tokens, feature_size)
        residual_features = self.conv(
            self.patch_unembed(residual_tokens, feature_size)
        )
        return self.patch_embed(residual_features) + condition_tokens
