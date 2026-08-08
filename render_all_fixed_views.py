#!/usr/bin/env python3
"""
Render the 20 fixed input views of all nine garden GLB reconstructions.

The current predictions.npz contains the camera parameters for the 20 images
DSC07956, DSC07958, ..., DSC07994.  The files in resized_* are used only as a
safe filename manifest when present.  Rendering and PSNR evaluation always use
the full-resolution files in images/, haze/, and dehaze/.

The GLB files contain no glTF camera nodes.  Their mesh-node transform is used
to recover first-camera coordinates, after which the relative camera poses in
predictions.npz are applied.  VGGT point clouds are Z-buffer splatted and
DUSt3R triangle meshes are rasterized with a Z-buffer.

Examples
--------
    python render_all_fixed_views.py
    python render_all_fixed_views.py --only DSC07956 DSC07994
    python render_all_fixed_views.py --models vggtclear1 dust3rclear
    python render_all_fixed_views.py --output-resolution network
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
SHARED_RENDERER = (
    ROOT.parents[1] / "real_capture" / "scan40" / "render_all_fixed_views.py"
)


def load_shared_renderer():
    """Load the already validated GLB parser and CPU Z-buffer renderer."""
    if not SHARED_RENDERER.is_file():
        raise FileNotFoundError(
            "Shared rendering kernel was not found: "
            f"{SHARED_RENDERER}\n"
            "Keep this repository layout, or copy the scan40 renderer back."
        )
    spec = importlib.util.spec_from_file_location(
        "_fixed_view_render_kernel", SHARED_RENDERER
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {SHARED_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses inspects sys.modules while creating Camera.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R = load_shared_renderer()

MODEL_FILES = {
    "vggtclear1": ROOT / "vggtclear1.glb",
    "vggtclear2": ROOT / "vggtclear2.glb",
    "vggtdehaze1": ROOT / "vggtdehaze1.glb",
    "vggtdehaze2": ROOT / "vggtdehaze2.glb",
    "vggthaze1": ROOT / "vggthaze1.glb",
    "vggthaze2": ROOT / "vggthaze2.glb",
    "dust3rclear": ROOT / "dust3rclear.glb",
    "dust3rdehaze": ROOT / "dust3rdehaze.glb",
    "dust3rhaze": ROOT / "dust3rhaze.glb",
}

# Robust trimmed-ICP alignment from each DUSt3R GLB's recovered first-camera
# coordinates to vggtdehaze1 coordinates.  DUSt3R has no cameras.npz in this
# dataset, and its three GLB viewer transforms differ slightly.  Applying these
# small rigid corrections makes all nine reconstructions use the same physical
# camera height/pitch instead of merely sharing the same numeric pose matrices.
DUST_ALIGNMENT_TO_VGGT = {
    "dust3rclear": {
        "rotation": np.asarray(
            [
                [0.9993970198, 0.0342452845, 0.0057321252],
                [-0.0341608083, 0.9993149248, -0.0142379842],
                [-0.0062157821, 0.0140335849, 0.9998822043],
            ],
            dtype=np.float64,
        ),
        "translation": np.asarray(
            [-0.0091859061, -0.0073211173, -0.0046091443],
            dtype=np.float64,
        ),
    },
    "dust3rdehaze": {
        "rotation": np.asarray(
            [
                [0.9999979481, -0.0017212222, 0.0010682837],
                [0.0017149958, 0.9999816961, 0.0058022694],
                [-0.0010782512, -0.0058004254, 0.9999825961],
            ],
            dtype=np.float64,
        ),
        "translation": np.asarray(
            [0.0006214338, 0.0163763434, 0.0066307866],
            dtype=np.float64,
        ),
    },
    "dust3rhaze": {
        "rotation": np.asarray(
            [
                [0.9993147764, 0.0368444798, 0.0035301415],
                [-0.0368378652, 0.9993194095, -0.0019208197],
                [-0.0035985105, 0.0017894606, 0.9999919242],
            ],
            dtype=np.float64,
        ),
        "translation": np.asarray(
            [-0.0090810425, -0.0427115038, -0.0059924452],
            dtype=np.float64,
        ),
    },
}

# Remaining vertical principal-point bias measured at the 518x336 network
# raster after 3D alignment.  Negative values move the DUSt3R render upward.
# The correction is scaled automatically for full-resolution output.
DUST_CY_OFFSET_NETWORK = {
    "dust3rclear": -15.0,
    "dust3rdehaze": -15.0,
    "dust3rhaze": -15.0,
}

# The clear and haze DUSt3R reconstructions used a camera-reference frame
# shifted by three entries relative to predictions.npz/dehaze.  Visual and
# geometric verification gives:
#
#   clear/haze camera index 2 (DSC07960) == dehaze index 5 (DSC07966)
#
# This is corrected in 3D below.  Renaming files would lose DSC07956 and would
# still apply the wrong camera trajectory, so it is deliberately not used.
DUST_VIEW_FRAME_ANCHORS = {
    "dust3rclear": (2, 5),
    "dust3rhaze": (2, 5),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_camera_data(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    with np.load(path, allow_pickle=False) as data:
        extrinsics = np.asarray(data["extrinsic"], dtype=np.float64)
        intrinsics = np.asarray(data["intrinsic"], dtype=np.float64)
        stored_images_shape = tuple(data["images"].shape)

    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (3, 4):
        raise ValueError(f"Unexpected extrinsic shape: {extrinsics.shape}")
    if intrinsics.shape != (len(extrinsics), 3, 3):
        raise ValueError(f"Unexpected intrinsic shape: {intrinsics.shape}")
    if len(stored_images_shape) != 4:
        raise ValueError(
            f"Unexpected predictions image shape: {stored_images_shape}"
        )

    network_height = int(stored_images_shape[-2])
    network_width = int(stored_images_shape[-1])
    relative_poses = R.relative_extrinsics(extrinsics)
    return (
        relative_poses,
        intrinsics,
        network_width,
        network_height,
        len(extrinsics),
    )


def select_view_names(view_count: int, manifest_dir: Path) -> list[str]:
    """
    Resolve camera order without using resized images as render/evaluation data.

    The existing resized_images folder was produced from the exact 20-frame
    reconstruction input list, so its basenames are the most reliable manifest.
    A deterministic garden fallback is retained for clean reruns.
    """
    if manifest_dir.is_dir():
        manifest = image_files(manifest_dir)
        if len(manifest) == view_count:
            return [p.name for p in manifest]

    expected = [
        f"DSC{7956 + 2 * index:05d}.JPG" for index in range(view_count)
    ]
    if view_count == 20:
        return expected
    raise ValueError(
        f"predictions.npz has {view_count} cameras, but {manifest_dir} does "
        "not contain the same number of manifest entries. Supply "
        "--manifest-dir with the ordered input filenames."
    )


def resolve_originals(directory: Path, names: list[str]) -> list[Path]:
    by_lower_name = {p.name.lower(): p for p in image_files(directory)}
    missing = [name for name in names if name.lower() not in by_lower_name]
    if missing:
        raise FileNotFoundError(
            f"{directory} is missing camera input images: {missing}"
        )
    return [by_lower_name[name.lower()] for name in names]


def parse_color(value: str) -> np.ndarray:
    return R.parse_color(value)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render fixed views for 6 VGGT and 3 DUSt3R garden GLBs."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_FILES) + ("all",),
        default=["all"],
    )
    parser.add_argument(
        "--predictions", type=Path, default=ROOT / "predictions.npz"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=ROOT / "images",
        help="full-resolution clear images; also used as default PSNR GT",
    )
    parser.add_argument(
        "--haze-dir", type=Path, default=ROOT / "haze"
    )
    parser.add_argument(
        "--dehaze-dir", type=Path, default=ROOT / "dehaze"
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=ROOT / "resized_images",
        help=(
            "ordered filename manifest only; its pixel data is never loaded"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "fixed_views"
    )
    parser.add_argument(
        "--output-resolution",
        choices=("original", "network"),
        default="original",
        help="original=5187x3361 source resolution; network=518x336",
    )
    parser.add_argument(
        "--vggt-raster-resolution",
        choices=("native", "output"),
        default="native",
        help=(
            "native renders VGGT at its depth-grid resolution before a "
            "high-quality resize, avoiding a dotted 10x-upsampled point grid"
        ),
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=ROOT / "images",
        help="full-resolution GT directory for PSNR; use 'none' to disable",
    )
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument(
        "--dust-render-mode", choices=("mesh", "points"), default="mesh"
    )
    parser.add_argument("--max-triangle-pixels", type=int, default=400)
    parser.add_argument(
        "--background",
        type=parse_color,
        default=parse_color("FFFFFF"),
        metavar="RRGGBB",
    )
    parser.add_argument("--near", type=float, default=1e-6)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="optional camera filenames or stems",
    )
    args = parser.parse_args()
    if str(args.reference_dir).lower() == "none":
        args.reference_dir = None
    if args.point_radius < 0:
        parser.error("--point-radius must be >= 0")
    if args.max_triangle_pixels <= 0:
        parser.error("--max-triangle-pixels must be positive")
    return args


def compute_psnr(
    reference: np.ndarray,
    rendered: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    if mask is not None:
        if not np.any(mask):
            return float("nan")
        difference = (
            reference[mask].astype(np.float64)
            - rendered[mask].astype(np.float64)
        )
    else:
        difference = (
            reference.astype(np.float64) - rendered.astype(np.float64)
        )
    mse = float(np.mean(difference * difference))
    return float("inf") if mse == 0 else 10 * math.log10(255.0**2 / mse)


def main() -> None:
    args = get_args()
    for model_path in MODEL_FILES.values():
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")

    poses, intrinsics, net_width, net_height, view_count = load_camera_data(
        args.predictions
    )
    view_names = select_view_names(view_count, args.manifest_dir)

    # Validate all three full-resolution sequences.  They share camera names.
    clear_paths = resolve_originals(args.image_dir, view_names)
    resolve_originals(args.haze_dir, view_names)
    resolve_originals(args.dehaze_dir, view_names)

    with Image.open(clear_paths[0]) as first:
        original_width, original_height = first.size
    for path in clear_paths[1:]:
        with Image.open(path) as image:
            if image.size != (original_width, original_height):
                raise ValueError(
                    f"Inconsistent original image size: {path}={image.size}"
                )

    if args.output_resolution == "original":
        output_width, output_height = original_width, original_height
    else:
        output_width, output_height = net_width, net_height

    selected_models = (
        list(MODEL_FILES)
        if "all" in args.models
        else list(dict.fromkeys(args.models))
    )
    selected_stems = None
    if args.only:
        selected_stems = {Path(value).stem.lower() for value in args.only}
        available = {Path(name).stem.lower() for name in view_names}
        missing = selected_stems - available
        if missing:
            raise ValueError(f"Unknown requested views: {sorted(missing)}")

    # predictions.npz was produced from the dehazed sequence.  Its reconstruction
    # therefore defines the trajectory translation unit.
    reference_xyz, _, _, _ = R.read_glb_model(
        MODEL_FILES["vggtdehaze1"], "vggt"
    )
    reference_depth_scale = R.robust_scene_scale(reference_xyz)
    del reference_xyz

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float]] = []
    print(
        f"Cameras: {view_count}; network={net_width}x{net_height}; "
        f"output={output_width}x{output_height}"
    )
    print("Views:", ", ".join(Path(name).stem for name in view_names))
    print("Image pixels come from: images/, haze/, dehaze/ (not resized_*)")

    for model_number, model_name in enumerate(selected_models, start=1):
        family = "vggt" if model_name.startswith("vggt") else "dust3r"
        xyz, rgb, faces, primitive_mode = R.read_glb_model(
            MODEL_FILES[model_name], family
        )
        geometry_scale = (
            R.robust_scene_scale(xyz) / reference_depth_scale
        )
        if family == "dust3r":
            # Convert DUSt3R geometry into the VGGT reference unit and apply
            # the per-model rigid correction.  Camera translations can then
            # use scale 1 exactly, just like vggtdehaze1.
            correction = DUST_ALIGNMENT_TO_VGGT[model_name]
            xyz = (
                (xyz / geometry_scale)
                @ correction["rotation"].T
                + correction["translation"]
            )
            if model_name in DUST_VIEW_FRAME_ANCHORS:
                source_index, target_index = DUST_VIEW_FRAME_ANCHORS[
                    model_name
                ]
                # Existing source-index render corresponds to target-index
                # view. Convert the model into the target/dehaze world frame:
                # X_target = inv(E_target) @ E_source @ X_source.
                frame_correction = (
                    np.linalg.inv(poses[target_index])
                    @ poses[source_index]
                )
                xyz = (
                    xyz @ frame_correction[:3, :3].T
                    + frame_correction[:3, 3]
                )
            geometry_scale = 1.0
        if (
            family == "vggt"
            and args.vggt_raster_resolution == "native"
        ):
            raster_width, raster_height = net_width, net_height
        else:
            raster_width, raster_height = output_width, output_height
        cameras = R.make_cameras(
            poses,
            intrinsics,
            geometry_scale,
            raster_width,
            raster_height,
            net_width,
            net_height,
        )
        if family == "dust3r":
            cy_offset = (
                DUST_CY_OFFSET_NETWORK[model_name]
                * raster_height
                / net_height
            )
            cameras = [
                R.Camera(
                    width=camera.width,
                    height=camera.height,
                    fx=camera.fx,
                    fy=camera.fy,
                    cx=camera.cx,
                    cy=camera.cy + cy_offset,
                    world_to_camera=camera.world_to_camera,
                )
                for camera in cameras
            ]

        model_dir = args.output_dir / model_name
        mask_dir = model_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        primitive = "POINTS" if primitive_mode == 0 else "TRIANGLES"
        print(
            f"\n[{model_number}/{len(selected_models)}] {model_name}: "
            f"{len(xyz):,} vertices, {primitive}, "
            f"camera_translation_scale={geometry_scale:.8g}"
        )

        for index, (name, camera) in enumerate(zip(view_names, cameras)):
            stem = Path(name).stem
            if selected_stems is not None and stem.lower() not in selected_stems:
                continue
            if (
                family == "dust3r"
                and args.dust_render_mode == "mesh"
                and faces is not None
            ):
                rendered, coverage = R.render_mesh(
                    xyz,
                    rgb,
                    faces,
                    camera,
                    args.background,
                    args.near,
                    args.max_triangle_pixels,
                )
            else:
                rendered, coverage = R.render_points(
                    xyz,
                    rgb,
                    camera,
                    args.point_radius,
                    args.background,
                    args.near,
                )

            if (raster_width, raster_height) != (
                output_width,
                output_height,
            ):
                rendered = np.asarray(
                    Image.fromarray(rendered, "RGB").resize(
                        (output_width, output_height),
                        Image.Resampling.LANCZOS,
                    )
                )
                coverage = np.asarray(
                    Image.fromarray(
                        coverage.astype(np.uint8) * 255, "L"
                    ).resize(
                        (output_width, output_height),
                        Image.Resampling.NEAREST,
                    )
                ) > 127

            output_path = model_dir / f"{stem}.png"
            Image.fromarray(rendered, "RGB").save(output_path)
            Image.fromarray(
                coverage.astype(np.uint8) * 255, "L"
            ).save(mask_dir / output_path.name)

            coverage_percent = 100.0 * float(coverage.mean())
            row: dict[str, str | float] = {
                "model": model_name,
                "view": output_path.name,
                "coverage_percent": coverage_percent,
                "camera_translation_scale": geometry_scale,
            }
            message = (
                f"  [{index:02d}] {output_path.name}: "
                f"coverage={coverage_percent:.2f}%"
            )
            if args.reference_dir is not None:
                reference_path = resolve_originals(
                    args.reference_dir, [name]
                )[0]
                with Image.open(reference_path) as image:
                    reference = np.asarray(
                        image.convert("RGB").resize(
                            (output_width, output_height),
                            Image.Resampling.LANCZOS,
                        )
                    )
                full_psnr = compute_psnr(reference, rendered)
                covered_psnr = compute_psnr(
                    reference, rendered, coverage
                )
                row["psnr_full_db"] = full_psnr
                row["psnr_covered_db"] = covered_psnr
                message += (
                    f", PSNR(full)={full_psnr:.3f} dB, "
                    f"PSNR(covered)={covered_psnr:.3f} dB"
                )
            rows.append(row)
            print(message)

        del xyz, rgb, faces

    fieldnames = [
        "model",
        "view",
        "coverage_percent",
        "camera_translation_scale",
    ]
    if args.reference_dir is not None:
        fieldnames += ["psnr_full_db", "psnr_covered_db"]
    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDone. Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
