"""Haze-VGGT checkpoint envelope loading."""

import torch


def unpack_hazevggt_checkpoint(checkpoint_path):
    """Open the Haze-VGGT container and return its serialized model state."""
    checkpoint_envelope = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    return checkpoint_envelope["hazevggt_state"]
