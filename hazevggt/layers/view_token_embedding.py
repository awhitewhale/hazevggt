"""Patch embedding for haze-degraded multi-view RGB observations."""

from torch import nn


class ViewTokenEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 1024, kernel_size=14, stride=14)
        self.norm = nn.Identity()

    def forward(self, multi_view_images):
        return self.norm(
            self.proj(multi_view_images).flatten(2).transpose(1, 2)
        )
