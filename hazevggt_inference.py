"""Haze-VGGT inference for haze-degraded multi-view 3D reconstruction."""

import colorsys
import gc
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from hazevggt.checkpoint_envelope import unpack_hazevggt_checkpoint
from hazevggt.hcv.hcv_network import HCVConfidenceHead, HazeConditionNetwork
from hazevggt.models.haze_vggt import HazeVGGT


PROJECT_ROOT = Path(__file__).resolve().parent
CUDA_DEVICE = torch.device("cuda")
COMMON_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _align_checkpoint_schema(weights, aliases):
    """Translate published checkpoint keys to the paper-facing module hierarchy."""
    aligned = {}
    for legacy_key, value in weights.items():
        paper_key = legacy_key
        for legacy_name, paper_name in aliases:
            paper_key = paper_key.replace(legacy_name, paper_name)
        aligned[paper_key] = value
    return aligned


def _sample_hcv_tokens(
    unknown_hcv_tokens,
    haze_observation_features,
    hcv_estimator,
    hcv_confidence_head,
    hcv_codebook,
):
    """Resolve latent haze tokens with confidence-aware iterative sampling."""
    batch_size, feature_channels, feature_height, feature_width = (
        haze_observation_features.shape
    )
    current_hcv_tokens = unknown_hcv_tokens
    sampled_hcv_sequence = unknown_hcv_tokens.unsqueeze(1).repeat(1, 8, 1)
    unresolved_token_count = (unknown_hcv_tokens == -1).sum(-1)

    for refinement_step in range(8):
        unresolved_mask = current_hcv_tokens == -1
        known_hcv_features = hcv_codebook(
            (current_hcv_tokens * ~unresolved_mask).reshape(
                batch_size, 1, feature_height, feature_width
            )
        )
        unresolved_features = unresolved_mask.reshape(
            batch_size, 1, feature_height, feature_width
        ).expand(-1, feature_channels, -1, -1)
        hcv_logits = hcv_estimator(
            known_hcv_features * ~unresolved_features
            + haze_observation_features * unresolved_features
        )
        hcv_probabilities = F.softmax(hcv_logits, -1)
        sampled_hcv_tokens = torch.multinomial(
            hcv_probabilities.reshape(-1, hcv_probabilities.shape[-1]), 1
        ).reshape(batch_size, -1)
        sampled_hcv_sequence[:, refinement_step] = sampled_hcv_tokens

        hcv_confidence = torch.sigmoid(
            hcv_confidence_head(
                sampled_hcv_tokens, feature_height, feature_width
            )
        )
        unresolved_ratio = math.cos(math.pi * (refinement_step + 1) / 16)
        unresolved_length = torch.floor(
            unresolved_token_count * unresolved_ratio
        ).unsqueeze(1)
        unresolved_length = unresolved_length.clamp(
            torch.ones_like(unresolved_length),
            unresolved_mask.sum(-1, keepdim=True) - 1,
        ).long()
        ranked_confidence = hcv_confidence.sort(dim=-1, descending=True).values
        confidence_gate = ranked_confidence.gather(-1, unresolved_length)
        current_hcv_tokens = torch.where(
            hcv_confidence >= confidence_gate, -1, sampled_hcv_tokens
        )

    return sampled_hcv_sequence


class HazeConditionAdapter:
    """Map hazy RGB observations to condition-adapted tensors entirely in memory."""

    def __init__(self):
        self.condition_network = HazeConditionNetwork()
        estimator_weights = unpack_hazevggt_checkpoint(
            PROJECT_ROOT / "hazevggt_checkpoints/hcv_token_estimator.pth"
        )["params"]
        estimator_aliases = (
            ("vqgan.", "hcv_codec."),
            ("transformer.", "hcv_estimator."),
            ("fuse_convs_dict.", "film_blocks."),
            ("swin_blks.", "condition_blocks."),
            ("norm_out.", "output_norm."),
            ("idx_pred_layer.", "token_projection."),
            ("multiscale_encoder.", "observation_encoder."),
            ("decoder_group.", "condition_decoders."),
            ("out_conv.", "rgb_projection."),
            ("quantize.", "hcv_codebook."),
            ("before_quant.", "hcv_projection."),
            ("after_quant.", "hcv_post_projection."),
            ("encode_enc.", "condition_encoder."),
            (".scale.", ".film_scale."),
            (".shift.", ".film_bias."),
        )
        self.condition_network.load_state_dict(
            _align_checkpoint_schema(estimator_weights, estimator_aliases)
        )

        self.confidence_head = HCVConfidenceHead()
        confidence_weights = unpack_hazevggt_checkpoint(
            PROJECT_ROOT / "hazevggt_checkpoints/hcv_confidence_head.pth"
        )["params"]
        confidence_aliases = (
            ("swin_blks.", "confidence_blocks."),
            ("norm_out.", "output_norm."),
            ("tok_emb.", "hcv_embedding."),
            ("idx_pred_layer.", "confidence_projection."),
        )
        self.confidence_head.load_state_dict(
            _align_checkpoint_schema(confidence_weights, confidence_aliases)
        )

        self.condition_network.eval().requires_grad_(False).to(CUDA_DEVICE)
        self.confidence_head.eval().requires_grad_(False).to(CUDA_DEVICE)

    @torch.inference_mode()
    def __call__(self, hazy_view):
        hazy_view = hazy_view.unsqueeze(0).to(CUDA_DEVICE)
        _, _, original_height, original_width = hazy_view.shape

        requires_visibility_resize = original_height * original_width >= 1500**2
        if requires_visibility_resize:
            visibility_scale = 1500 / max(original_height, original_width)
            hazy_view = torch.nn.UpsamplingBilinear2d(
                scale_factor=visibility_scale
            )(hazy_view)

        _, _, view_height, view_width = hazy_view.shape
        hcv_height = (view_height // 32 + 1) * 32
        hcv_width = (view_width // 32 + 1) * 32
        hazy_view = torch.cat((hazy_view, hazy_view.flip(2)), 2)[:, :, :hcv_height]
        hazy_view = torch.cat((hazy_view, hazy_view.flip(3)), 3)[:, :, :, :hcv_width]

        observation_pyramid = self.condition_network.hcv_codec.observation_encoder(
            hazy_view
        )[::-1]
        deepest_observation = observation_pyramid[0]
        batch_size, _, feature_height, feature_width = deepest_observation.shape
        haze_observation_features = self.condition_network.hcv_codec.hcv_projection(
            deepest_observation
        )
        unknown_hcv_tokens = torch.full(
            (batch_size, feature_height * feature_width),
            -1,
            device=CUDA_DEVICE,
            dtype=torch.long,
        )

        sampled_hcv_sequence = _sample_hcv_tokens(
            unknown_hcv_tokens,
            haze_observation_features,
            self.condition_network.hcv_estimator,
            self.confidence_head,
            self.condition_network.hcv_codec.hcv_codebook.get_codebook_entry,
        )
        hcv_embedding = (
            self.condition_network.hcv_codec.hcv_codebook.get_codebook_entry(
                sampled_hcv_sequence[:, -1].reshape(
                    batch_size, 1, feature_height, feature_width
                )
            )
        )
        conditioned_features = self.condition_network.hcv_codec.hcv_post_projection(
            hcv_embedding
        )

        # FiLM-style feature fusion propagates the haze condition across scales.
        for condition_level in range(self.condition_network.condition_depth):
            condition_resolution = (
                self.condition_network.hcv_resolution
                // 2**self.condition_network.condition_depth
                * 2**condition_level
            )
            conditioned_features = self.condition_network.film_blocks[
                str(condition_resolution)
            ](
                observation_pyramid[condition_level], conditioned_features, 1
            )
            conditioned_features = self.condition_network.hcv_codec.condition_decoders[
                condition_level
            ](conditioned_features)

        conditioned_view = self.condition_network.hcv_codec.rgb_projection(
            conditioned_features
        )[..., :view_height, :view_width]
        if requires_visibility_resize:
            conditioned_view = torch.nn.UpsamplingBilinear2d(
                (original_height, original_width)
            )(conditioned_view)
        return conditioned_view.squeeze(0).clamp(0, 1).cpu()


def _load_hazy_view(view_path):
    hazy_rgb = cv2.cvtColor(cv2.imread(str(view_path)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(hazy_rgb.copy()).permute(2, 0, 1).float().div_(255)


def _collect_hazy_views(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    return sorted(
        view_path
        for view_path in input_path.iterdir()
        if view_path.suffix.lower() in COMMON_IMAGE_SUFFIXES
    )


def _encode_visual_geometry_views(conditioned_views):
    """Normalize condition-adapted observations into Haze-VGGT visual tokens."""
    geometry_views = []
    for conditioned_view in conditioned_views:
        _, view_height, view_width = conditioned_view.shape
        geometry_height = round(view_height * 518 / view_width / 14) * 14
        conditioned_view = F.interpolate(
            conditioned_view.unsqueeze(0),
            size=(geometry_height, 518),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).clamp(0, 1)
        if geometry_height > 518:
            crop_start = (geometry_height - 518) // 2
            conditioned_view = conditioned_view[
                :, crop_start : crop_start + 518
            ]
        geometry_views.append(conditioned_view)

    shared_height = max(view.shape[1] for view in geometry_views)
    aligned_views = []
    for geometry_view in geometry_views:
        vertical_padding = shared_height - geometry_view.shape[1]
        aligned_views.append(
            F.pad(
                geometry_view,
                (
                    0,
                    0,
                    vertical_padding // 2,
                    vertical_padding - vertical_padding // 2,
                ),
                value=1,
            )
        )
    return torch.stack(aligned_views)


def _estimate_haze_condition_views(hazy_view_paths):
    condition_adapter = HazeConditionAdapter()
    conditioned_views = [
        condition_adapter(_load_hazy_view(view_path))
        for view_path in hazy_view_paths
    ]
    del condition_adapter
    gc.collect()
    torch.cuda.empty_cache()
    return _encode_visual_geometry_views(conditioned_views)


def _decode_camera_rotation(camera_quaternion):
    quaternion_i, quaternion_j, quaternion_k, quaternion_real = (
        camera_quaternion.unbind(-1)
    )
    quaternion_scale = 2 / camera_quaternion.square().sum(-1)
    camera_rotation = torch.stack(
        (
            1 - quaternion_scale * (quaternion_j * quaternion_j + quaternion_k * quaternion_k),
            quaternion_scale * (quaternion_i * quaternion_j - quaternion_k * quaternion_real),
            quaternion_scale * (quaternion_i * quaternion_k + quaternion_j * quaternion_real),
            quaternion_scale * (quaternion_i * quaternion_j + quaternion_k * quaternion_real),
            1 - quaternion_scale * (quaternion_i * quaternion_i + quaternion_k * quaternion_k),
            quaternion_scale * (quaternion_j * quaternion_k - quaternion_i * quaternion_real),
            quaternion_scale * (quaternion_i * quaternion_k - quaternion_j * quaternion_real),
            quaternion_scale * (quaternion_j * quaternion_k + quaternion_i * quaternion_real),
            1 - quaternion_scale * (quaternion_i * quaternion_i + quaternion_j * quaternion_j),
        ),
        -1,
    )
    return camera_rotation.reshape(camera_quaternion.shape[:-1] + (3, 3))


def _decode_mph_cameras(camera_parameters, image_size):
    image_height, image_width = image_size
    camera_rotation = _decode_camera_rotation(camera_parameters[..., 3:7])
    camera_extrinsics = torch.cat(
        (camera_rotation, camera_parameters[..., :3, None]), -1
    )
    camera_intrinsics = torch.zeros(
        camera_parameters.shape[:2] + (3, 3), device=camera_parameters.device
    )
    camera_intrinsics[..., 0, 0] = (image_width / 2) / torch.tan(
        camera_parameters[..., 8] / 2
    )
    camera_intrinsics[..., 1, 1] = (image_height / 2) / torch.tan(
        camera_parameters[..., 7] / 2
    )
    camera_intrinsics[..., 0, 2] = image_width / 2
    camera_intrinsics[..., 1, 2] = image_height / 2
    camera_intrinsics[..., 2, 2] = 1
    return camera_extrinsics, camera_intrinsics


def _unproject_dense_geometry(depth_maps, camera_extrinsics, camera_intrinsics):
    point_maps = []
    for depth_map, extrinsic, intrinsic in zip(
        depth_maps[..., 0], camera_extrinsics, camera_intrinsics
    ):
        image_height, image_width = depth_map.shape
        pixel_u, pixel_v = np.meshgrid(
            np.arange(image_width), np.arange(image_height)
        )
        camera_points = np.stack(
            (
                (pixel_u - intrinsic[0, 2]) * depth_map / intrinsic[0, 0],
                (pixel_v - intrinsic[1, 2]) * depth_map / intrinsic[1, 1],
                depth_map,
            ),
            -1,
        ).astype(np.float32)
        camera_center = -extrinsic[:, :3].T @ extrinsic[:, 3]
        point_maps.append(camera_points @ extrinsic[:, :3] + camera_center)
    return np.stack(point_maps)


def _load_hazevggt_backbone():
    haze_vggt = HazeVGGT()
    geometry_weights = unpack_hazevggt_checkpoint(
        PROJECT_ROOT / "hazevggt_checkpoints/hazevggt_geometry_backbone.pt"
    )
    geometry_aliases = (
        ("aggregator.", "alternating_backbone."),
        ("camera_head.", "mph_camera."),
        ("depth_head.", "mph_dense_geometry."),
        ("alternating_backbone.patch_embed.", "alternating_backbone.visual_encoder."),
        ("alternating_backbone.frame_blocks.", "alternating_backbone.frame_attention."),
        ("alternating_backbone.global_blocks.", "alternating_backbone.global_attention."),
        ("alternating_backbone.camera_token", "alternating_backbone.reference_camera_token"),
        ("alternating_backbone.register_token", "alternating_backbone.geometry_register_tokens"),
        ("mph_camera.trunk.", "mph_camera.camera_transformer."),
        ("mph_camera.token_norm.", "mph_camera.camera_token_norm."),
        ("mph_camera.trunk_norm.", "mph_camera.camera_output_norm."),
        ("mph_camera.empty_pose_tokens", "mph_camera.reference_pose_tokens"),
        ("mph_camera.embed_pose.", "mph_camera.pose_embedding."),
        ("mph_camera.poseLN_modulation.", "mph_camera.camera_film."),
        ("mph_camera.adaln_norm.", "mph_camera.modulation_norm."),
        ("mph_camera.pose_branch.", "mph_camera.camera_projection."),
        ("mph_dense_geometry.norm.", "mph_dense_geometry.geometry_norm."),
        ("mph_dense_geometry.projects.", "mph_dense_geometry.scale_projections."),
        ("mph_dense_geometry.resize_layers.", "mph_dense_geometry.scale_alignment."),
        ("mph_dense_geometry.scratch.", "mph_dense_geometry.dense_decoder."),
        (".layer1_rn.", ".scale1_projection."),
        (".layer2_rn.", ".scale2_projection."),
        (".layer3_rn.", ".scale3_projection."),
        (".layer4_rn.", ".scale4_projection."),
        (".refinenet1.", ".fusion_stage1."),
        (".refinenet2.", ".fusion_stage2."),
        (".refinenet3.", ".fusion_stage3."),
        (".refinenet4.", ".fusion_stage4."),
        (".output_conv1.", ".dense_projection."),
        (".output_conv2.", ".depth_reliability_projection."),
        (".resConfUnit1.", ".residual_geometry."),
        (".resConfUnit2.", ".output_geometry."),
        (".out_conv.", ".geometry_projection."),
        (".conv1.", ".geometry_conv1."),
        (".conv2.", ".geometry_conv2."),
    )
    haze_vggt.load_state_dict(
        _align_checkpoint_schema(geometry_weights, geometry_aliases)
    )
    return haze_vggt.eval().requires_grad_(False).to(CUDA_DEVICE)


def _predict_hazevggt_geometry(conditioned_views):
    haze_vggt = _load_hazevggt_backbone()
    conditioned_views = conditioned_views.to(CUDA_DEVICE)
    inference_dtype = (
        torch.bfloat16
        if torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=inference_dtype):
        mph_outputs = haze_vggt(conditioned_views)
        camera_extrinsics, camera_intrinsics = _decode_mph_cameras(
            mph_outputs["camera_parameters"], conditioned_views.shape[-2:]
        )

    depth_maps = mph_outputs["depth_maps"].float().cpu().numpy().squeeze(0)
    geometry_reliability = (
        mph_outputs["geometry_reliability"].float().cpu().numpy().squeeze(0)
    )
    view_colors = mph_outputs["conditioned_views"].float().cpu().numpy().squeeze(0)
    camera_extrinsics = camera_extrinsics.float().cpu().numpy().squeeze(0)
    camera_intrinsics = camera_intrinsics.float().cpu().numpy().squeeze(0)
    point_maps = _unproject_dense_geometry(
        depth_maps, camera_extrinsics, camera_intrinsics
    )

    del mph_outputs, conditioned_views, haze_vggt
    gc.collect()
    torch.cuda.empty_cache()
    return point_maps, geometry_reliability, view_colors, camera_extrinsics


def _create_camera_frustum_faces(camera_cone):
    frustum_faces = []
    cone_vertex_count = len(camera_cone.vertices)
    for cone_face in camera_cone.faces:
        if 0 in cone_face:
            continue
        vertex_a, vertex_b, vertex_c = cone_face
        vertex_a1, vertex_b1, vertex_c1 = cone_face + cone_vertex_count
        vertex_a2, vertex_b2, vertex_c2 = cone_face + 2 * cone_vertex_count
        frustum_faces.extend(
            (
                (vertex_a, vertex_b, vertex_b1),
                (vertex_a, vertex_a1, vertex_c),
                (vertex_c1, vertex_b, vertex_c),
                (vertex_a, vertex_b, vertex_b2),
                (vertex_a, vertex_a2, vertex_c),
                (vertex_c2, vertex_b, vertex_c),
            )
        )
    frustum_faces += [
        (vertex_c, vertex_b, vertex_a)
        for vertex_a, vertex_b, vertex_c in frustum_faces
    ]
    return np.array(frustum_faces)


def _add_reference_camera(
    reconstruction_scene, camera_transform, camera_color, scene_scale
):
    frustum_height = scene_scale * 0.1
    camera_cone = trimesh.creation.cone(
        scene_scale * 0.05, frustum_height, sections=4
    )
    diagonal_rotation = math.sqrt(0.5)
    camera_alignment = np.array(
        (
            (diagonal_rotation, -diagonal_rotation, 0, 0),
            (diagonal_rotation, diagonal_rotation, 0, 0),
            (0, 0, 1, -frustum_height),
            (0, 0, 0, 1),
        )
    )
    opengl_coordinates = np.diag((1, -1, -1, 1))
    outline_angle = math.radians(2)
    outline_rotation = np.array(
        (
            (math.cos(outline_angle), -math.sin(outline_angle), 0, 0),
            (math.sin(outline_angle), math.cos(outline_angle), 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
    )
    frustum_vertices = np.concatenate(
        (
            camera_cone.vertices,
            camera_cone.vertices * 0.95,
            trimesh.transform_points(camera_cone.vertices, outline_rotation),
        )
    )
    camera_mesh = trimesh.Trimesh(
        vertices=trimesh.transform_points(
            frustum_vertices,
            camera_transform @ opengl_coordinates @ camera_alignment,
        ),
        faces=_create_camera_frustum_faces(camera_cone),
    )
    camera_mesh.visual.face_colors[:, :3] = camera_color
    reconstruction_scene.add_geometry(camera_mesh)


def _assemble_uncertainty_aware_scene(
    point_maps, geometry_reliability, conditioned_views, camera_extrinsics
):
    """Fuse reliable dense geometry and camera poses into the final GLB scene."""
    scene_points = point_maps.reshape(-1, 3)
    scene_colors = (
        conditioned_views.transpose(0, 2, 3, 1).reshape(-1, 3) * 255
    ).astype(np.uint8)
    geometry_reliability = geometry_reliability.reshape(-1)
    reliability_gate = (
        geometry_reliability >= np.percentile(geometry_reliability, 50)
    ) & (geometry_reliability > 1e-5)
    scene_points = scene_points[reliability_gate]
    scene_colors = scene_colors[reliability_gate]

    geometry_lower_bound = np.percentile(scene_points, 5, axis=0)
    geometry_upper_bound = np.percentile(scene_points, 95, axis=0)
    scene_scale = np.linalg.norm(geometry_upper_bound - geometry_lower_bound)
    reconstruction_scene = trimesh.Scene(
        trimesh.PointCloud(vertices=scene_points, colors=scene_colors)
    )

    camera_matrices = np.zeros((len(camera_extrinsics), 4, 4))
    camera_matrices[:, :3] = camera_extrinsics
    camera_matrices[:, 3, 3] = 1
    for view_index, world_to_camera in enumerate(camera_matrices):
        camera_rgb = colorsys.hsv_to_rgb(view_index / len(camera_matrices), 1, 1)
        _add_reference_camera(
            reconstruction_scene,
            np.linalg.inv(world_to_camera),
            tuple(int(color_channel * 255) for color_channel in camera_rgb),
            scene_scale,
        )

    opengl_coordinates = np.diag((1, -1, -1, 1))
    reference_alignment = np.diag((-1, 1, -1, 1))
    reconstruction_scene.apply_transform(
        np.linalg.inv(camera_matrices[0])
        @ opengl_coordinates
        @ reference_alignment
    )
    return reconstruction_scene


def reconstruct_hazy_multiview_scene(input_path, output_path):
    """Run the full HCV-to-MPH path and export one Haze-VGGT GLB."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    conditioned_views = _estimate_haze_condition_views(
        _collect_hazy_views(input_path)
    )
    reconstruction_scene = _assemble_uncertainty_aware_scene(
        *_predict_hazevggt_geometry(conditioned_views)
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reconstruction_scene.export(str(output_path))


def run_hazevggt_inference():
    reconstruct_hazy_multiview_scene(
        "hazy_multiview", "hazevggt_outputs/hazevggt_reconstruction.glb"
    )


if __name__ == "__main__":
    from urllib.error import URLError
    from urllib.request import urlopen

    checkpoint_directory = PROJECT_ROOT / "hazevggt_checkpoints"
    checkpoint_directory.mkdir(exist_ok=True)
    checkpoint_names = (
        "hazevggt_geometry_backbone.pt",
        "hcv_confidence_head.pth",
        "hcv_token_estimator.pth",
    )
    checkpoint_sources = (
        "https://huggingface.co/awhitewhale/hazevggt/resolve/main/",
        "https://hf-mirror.com/awhitewhale/hazevggt/resolve/main/",
    )

    for checkpoint_name in checkpoint_names:
        checkpoint_path = checkpoint_directory / checkpoint_name
        if checkpoint_path.exists():
            continue
        temporary_path = checkpoint_path.with_suffix(
            checkpoint_path.suffix + ".download"
        )
        for checkpoint_source in checkpoint_sources:
            try:
                with urlopen(
                    checkpoint_source + checkpoint_name, timeout=30
                ) as response, temporary_path.open("wb") as checkpoint_file:
                    while checkpoint_chunk := response.read(1024 * 1024):
                        checkpoint_file.write(checkpoint_chunk)
                temporary_path.replace(checkpoint_path)
                break
            except (OSError, TimeoutError, URLError):
                temporary_path.unlink(missing_ok=True)
        else:
            raise RuntimeError(
                f"Unable to download {checkpoint_name} from Hugging Face or hf-mirror."
            )

    run_hazevggt_inference()
