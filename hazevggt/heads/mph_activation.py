"""Output parameterizations for Haze-VGGT multi-task prediction heads."""

import torch
import torch.nn.functional as F


def decode_camera_parameters(camera_parameters):
    return torch.cat(
        (camera_parameters[..., :7], F.relu(camera_parameters[..., 7:])), -1
    )


def decode_depth_reliability(dense_geometry_output):
    dense_geometry_output = dense_geometry_output.permute(0, 2, 3, 1)
    depth_maps = torch.exp(dense_geometry_output[..., :-1])
    geometry_reliability = 1 + torch.exp(dense_geometry_output[..., -1])
    return depth_maps, geometry_reliability
