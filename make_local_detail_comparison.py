#!/usr/bin/env python3
"""
Create paper-style local-detail comparisons for garden views DSC07956 and
DSC07966.

Inputs (12 groups)
------------------
1. images/                        (clear GT)
2. haze/                          (hazy input)
3. dehaze/                        (dehazed input)
4-9. fixed_views/vggt{clear,dehaze,haze}{1,2}/
10-12. fixed_views/dust3r{clear,dehaze,haze}/

For each view, the script creates:
* one enlarged ROI crop for every group;
* a new full-resolution GT image with the ROI marked by a red rectangle;
* a 4 x 3 labeled comparison panel suitable for a paper figure;
* a JSON manifest recording the exact ROI and source paths.

The same pixel ROI is applied to every group within a view, ensuring a fair
local-detail comparison.

Examples
--------
    python make_local_detail_comparison.py

    python make_local_detail_comparison.py --zoom-scale 3

    python make_local_detail_comparison.py \
        --roi-DSC07956 1994 1231 3194 2131 \
        --roi-DSC07966 1994 1231 3194 2131
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ComparisonGroup:
    label: str
    slug: str
    directory: Path
    is_gt: bool = False


GROUPS = (
    ComparisonGroup("GT", "gt", ROOT / "images", is_gt=True),
    ComparisonGroup("Hazy Input", "hazy_input", ROOT / "haze"),
    ComparisonGroup("Dehazed Input", "dehazed_input", ROOT / "dehaze"),
    ComparisonGroup(
        "VGGT-Clear-1",
        "vggtclear1",
        ROOT / "fixed_views" / "vggtclear1",
    ),
    ComparisonGroup(
        "VGGT-Clear-2",
        "vggtclear2",
        ROOT / "fixed_views" / "vggtclear2",
    ),
    ComparisonGroup(
        "VGGT-Dehaze-1",
        "vggtdehaze1",
        ROOT / "fixed_views" / "vggtdehaze1",
    ),
    ComparisonGroup(
        "VGGT-Dehaze-2",
        "vggtdehaze2",
        ROOT / "fixed_views" / "vggtdehaze2",
    ),
        ComparisonGroup(
        "VGGT-Dehaze-3",
        "vggtdehaze3",
        ROOT / "fixed_views" / "vggtdehaze3",
    ),
    ComparisonGroup(
        "VGGT-Haze-1",
        "vggthaze1",
        ROOT / "fixed_views" / "vggthaze1",
    ),
    ComparisonGroup(
        "VGGT-Haze-2",
        "vggthaze2",
        ROOT / "fixed_views" / "vggthaze2",
    ),
    ComparisonGroup(
        "DUSt3R-Clear",
        "dust3rclear",
        ROOT / "fixed_views" / "dust3rclear",
    ),
    ComparisonGroup(
        "DUSt3R-Dehaze",
        "dust3rdehaze",
        ROOT / "fixed_views" / "dust3rdehaze",
    ),
    ComparisonGroup(
        "DUSt3R-Haze",
        "dust3rhaze",
        ROOT / "fixed_views" / "dust3rhaze",
    ),
)


# Coordinates use Pillow's crop convention: (left, top, right, bottom), where
# right and bottom are exclusive.
DEFAULT_ROIS = {
    # Both 1200x900 ROIs are exactly centered in the 5187x3361 originals.
    # They cover the central vase/table region in these garden views.
    "DSC07956": (1994, 1231, 3194, 2131),
    "DSC07966": (1994, 1231, 3194, 2131),
}


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                Path("C:/Windows/Fonts/arialbd.ttf"),
                Path("C:/Windows/Fonts/calibrib.ttf"),
            ]
        )
    candidates.extend(
        [
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/calibri.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
            if bold
            else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def validate_roi(
    roi: tuple[int, int, int, int],
    width: int,
    height: int,
    view_name: str,
) -> None:
    left, top, right, bottom = roi
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError(
            f"ROI for {view_name}={roi} is outside image size "
            f"{width}x{height}"
        )
    if right <= left or bottom <= top:
        raise ValueError(f"ROI for {view_name} has no positive area: {roi}")


def red_width(image_width: int, requested_width: int) -> int:
    """Keep the rectangle visible if a different-resolution image is used."""
    return max(2, min(requested_width, round(image_width / 200)))


def draw_rectangle_inside(
    image: Image.Image,
    roi: tuple[int, int, int, int],
    color: tuple[int, int, int],
    width: int,
) -> Image.Image:
    """Draw a rectangle fully inside the image/ROI boundary."""
    output = image.copy()
    draw = ImageDraw.Draw(output)
    left, top, right, bottom = roi
    # Pillow rectangle endpoints are inclusive; subtract one from crop's
    # exclusive right/bottom coordinates.
    for offset in range(width):
        box = (
            left + offset,
            top + offset,
            right - 1 - offset,
            bottom - 1 - offset,
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            break
        draw.rectangle(box, outline=color, width=1)
    return output


def make_zoom_crop(
    image: Image.Image,
    roi: tuple[int, int, int, int],
    zoom_scale: int,
) -> Image.Image:
    crop = image.crop(roi)
    return crop.resize(
        (crop.width * zoom_scale, crop.height * zoom_scale),
        Image.Resampling.LANCZOS,
    )


def centered_text_position(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    center_x: int,
    top: int,
) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    text_width = box[2] - box[0]
    return center_x - text_width // 2, top


def make_comparison_panel(
    crops: list[tuple[ComparisonGroup, Image.Image]],
    columns: int,
    gap: int,
    label_height: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    if not crops:
        raise ValueError("No crops supplied for comparison panel")
    tile_width = crops[0][1].width
    tile_height = crops[0][1].height
    for _, crop in crops:
        if crop.size != (tile_width, tile_height):
            raise ValueError("All comparison crops must have identical size")

    rows = (len(crops) + columns - 1) // columns
    panel_width = columns * tile_width + (columns - 1) * gap
    panel_height = rows * (tile_height + label_height) + (rows - 1) * gap
    panel = Image.new("RGB", (panel_width, panel_height), "white")
    draw = ImageDraw.Draw(panel)

    for index, (group, crop) in enumerate(crops):
        row, column = divmod(index, columns)
        x = column * (tile_width + gap)
        y = row * (tile_height + label_height + gap)
        panel.paste(crop, (x, y))
        label_y = y + tile_height + max(2, label_height // 8)
        label_position = centered_text_position(
            draw,
            group.label,
            font,
            x + tile_width // 2,
            label_y,
        )
        draw.text(label_position, group.label, fill="black", font=font)
    return panel


def view_image_path(group: ComparisonGroup, view: str) -> Path:
    # Original inputs are JPG while fixed-view renders are PNG.
    for suffix in (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"):
        candidate = group.directory / f"{view}{suffix}"
        if candidate.is_file():
            return candidate
    return group.directory / f"{view}.png"


def process_view(
    view: str,
    roi: tuple[int, int, int, int],
    output_root: Path,
    zoom_scale: int,
    rectangle_width: int,
    columns: int,
    panel_gap: int,
    label_height: int,
    label_font: ImageFont.ImageFont,
) -> dict:
    output_dir = output_root / f"view_{view}"
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_path = view_image_path(GROUPS[0], view)
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)
    with Image.open(gt_path) as gt_file:
        gt = gt_file.convert("RGB")
    validate_roi(roi, gt.width, gt.height, view)

    marked_gt = draw_rectangle_inside(
        gt,
        roi,
        color=(255, 0, 0),
        width=red_width(gt.width, rectangle_width),
    )
    marked_gt_path = output_dir / f"{view}_GT_marked.png"
    marked_gt.save(marked_gt_path)

    comparison_crops: list[tuple[ComparisonGroup, Image.Image]] = []
    source_records = []
    expected_size = gt.size

    for group in GROUPS:
        source_path = view_image_path(group, view)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        with Image.open(source_path) as image_file:
            image = image_file.convert("RGB")
        original_size = image.size
        resized_to_gt = original_size != expected_size
        if image.size != expected_size:
            print(
                f"[resize] {source_path}: {image.size[0]}x{image.size[1]} "
                f"-> {expected_size[0]}x{expected_size[1]}"
            )
            # Stretch to the exact GT dimensions so every group uses the same
            # pixel ROI.  Width and height are intentionally scaled
            # independently when the aspect ratios differ.
            image = image.resize(
                expected_size,
                Image.Resampling.LANCZOS,
            )

        crop = make_zoom_crop(
            image,
            roi,
            zoom_scale=zoom_scale,
        )
        crop_path = output_dir / f"{view}_{group.slug}_detail_x{zoom_scale}.png"
        crop.save(crop_path)
        comparison_crops.append((group, crop))
        source_records.append(
            {
                "label": group.label,
                "slug": group.slug,
                "source": str(source_path.resolve()),
                "detail_crop": str(crop_path.resolve()),
                "is_gt": group.is_gt,
                "original_size": list(original_size),
                "comparison_size": list(image.size),
                "resized_to_gt": resized_to_gt,
            }
        )

    panel = make_comparison_panel(
        comparison_crops,
        columns=columns,
        gap=panel_gap,
        label_height=label_height,
        font=label_font,
    )
    panel_path = output_dir / f"{view}_local_detail_comparison.png"
    panel.save(panel_path)

    return {
        "view": view,
        "image_size": [gt.width, gt.height],
        "roi_xyxy": list(roi),
        "roi_size": [roi[2] - roi[0], roi[3] - roi[1]],
        "zoom_scale": zoom_scale,
        "gt_marked": str(marked_gt_path.resolve()),
        "comparison_panel": str(panel_path.resolve()),
        "groups": source_records,
    }


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create garden local-detail crops for DSC07956 and DSC07966."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "local_detail_comparison",
    )
    parser.add_argument(
        "--roi-DSC07956",
        dest="roi_dsc07956",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        default=DEFAULT_ROIS["DSC07956"],
    )
    parser.add_argument(
        "--roi-DSC07966",
        dest="roi_dsc07966",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        default=DEFAULT_ROIS["DSC07966"],
    )
    parser.add_argument("--zoom-scale", type=int, default=2)
    parser.add_argument(
        "--rectangle-width",
        type=int,
        default=8,
        help="red rectangle width on the full GT image",
    )
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--panel-gap", type=int, default=24)
    parser.add_argument("--label-height", type=int, default=64)
    parser.add_argument("--font-size", type=int, default=38)
    args = parser.parse_args()

    if args.zoom_scale < 1:
        parser.error("--zoom-scale must be >= 1")
    if args.rectangle_width < 1:
        parser.error("--rectangle-width must be >= 1")
    if args.columns < 1:
        parser.error("--columns must be >= 1")
    if args.panel_gap < 0 or args.label_height < 1 or args.font_size < 1:
        parser.error("panel/font dimensions must be positive")
    return args


def main() -> None:
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_font = load_font(args.font_size, bold=True)
    view_rois = {
        "DSC07956": tuple(args.roi_dsc07956),
        "DSC07966": tuple(args.roi_dsc07966),
    }

    manifest = {
        "description": (
            "Paper-style local-detail comparison using identical ROIs for "
            "GT, hazy/dehazed inputs, six VGGT outputs, and three DUSt3R "
            "outputs. Only the full-size GT image has a red ROI rectangle."
        ),
        "root": str(ROOT),
        "views": [],
    }
    for view, roi in view_rois.items():
        record = process_view(
            view=view,
            roi=roi,
            output_root=args.output_dir,
            zoom_scale=args.zoom_scale,
            rectangle_width=args.rectangle_width,
            columns=args.columns,
            panel_gap=args.panel_gap,
            label_height=args.label_height,
            label_font=label_font,
        )
        manifest["views"].append(record)
        print(
            f"{view}: ROI={roi}, panel={record['comparison_panel']}"
        )

    manifest_path = args.output_dir / "roi_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
