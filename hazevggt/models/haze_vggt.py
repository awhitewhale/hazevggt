"""Haze-VGGT multi-task geometry model."""

from torch import nn

from hazevggt.heads.mph_camera import MPHCameraHead
from hazevggt.heads.mph_dense_geometry import MPHDenseGeometryHead
from hazevggt.models.alternating_backbone import AlternatingViewBackbone


class HazeVGGT(nn.Module):
    """Predict cameras, dense geometry, and reliability from conditioned views."""

    def __init__(self):
        super().__init__()
        self.alternating_backbone = AlternatingViewBackbone()
        self.mph_camera = MPHCameraHead()
        self.mph_dense_geometry = MPHDenseGeometryHead()

    def forward(self, conditioned_views):
        if conditioned_views.ndim == 4:
            conditioned_views = conditioned_views.unsqueeze(0)
        backbone_features, visual_token_start = self.alternating_backbone(
            conditioned_views
        )
        camera_parameters = self.mph_camera(backbone_features)
        depth_maps, geometry_reliability = self.mph_dense_geometry(
            backbone_features, conditioned_views, visual_token_start
        )
        return {
            "camera_parameters": camera_parameters,
            "depth_maps": depth_maps,
            "geometry_reliability": geometry_reliability,
            "conditioned_views": conditioned_views,
        }
