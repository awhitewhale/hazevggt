"""Discrete HCV codebook and multi-scale haze-observation codec."""

from torch import nn

from .feature_modulation import HCVPostProjection, HCVResidualBlock


class HCVCodebook(nn.Module):
    """Map discrete degradation indices to latent condition embeddings."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1024, 256)

    def get_codebook_entry(self, hcv_indices):
        batch_size, _, feature_height, feature_width = hcv_indices.shape
        haze_embedding = self.embedding(hcv_indices.flatten())
        return (
            haze_embedding.view(
                batch_size, feature_height, feature_width, 256
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )


class HazeObservationEncoder(nn.Module):
    """Extract the multi-scale degradation evidence consumed by the HCV path."""

    def __init__(self):
        super().__init__()
        self.in_conv = nn.Conv2d(3, 64, 4, padding=1)
        self.blocks = nn.ModuleList(
            (
                nn.Sequential(
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    HCVResidualBlock(128, 128, "silu"),
                    HCVResidualBlock(128, 128, "silu"),
                ),
                nn.Sequential(
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    HCVResidualBlock(256, 256, "silu"),
                    HCVResidualBlock(256, 256, "silu"),
                ),
            )
        )

    def forward(self, hazy_view):
        observation_pyramid = []
        haze_features = self.in_conv(hazy_view)
        for degradation_block in self.blocks:
            haze_features = degradation_block(haze_features)
            observation_pyramid.append(haze_features)
        return observation_pyramid


class ConditionDecoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            HCVResidualBlock(output_channels, output_channels, "silu"),
            HCVResidualBlock(output_channels, output_channels, "silu"),
        )

    def forward(self, conditioned_features):
        return self.block(conditioned_features)


class HazeConditionCodec(nn.Module):
    """Encode observations, inject the HCV embedding, and decode RGB tensors."""

    def __init__(self):
        super().__init__()
        self.observation_encoder = HazeObservationEncoder()
        self.condition_decoders = nn.ModuleList(
            (
                ConditionDecoderBlock(256, 128),
                ConditionDecoderBlock(128, 64),
            )
        )
        self.rgb_projection = nn.Conv2d(64, 3, 3, 1, 1)
        self.hcv_codebook = HCVCodebook()
        self.hcv_projection = nn.Conv2d(256, 256, 1)
        self.hcv_post_projection = HCVPostProjection(256)
