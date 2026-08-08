#!/usr/bin/env python3
"""计算各子文件夹图像相对于 gt 的 PSNR、SSIM 和 LPIPS。"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


IMAGE_NAMES = ("013.png", "035.png")


def load_rgb(path: Path) -> np.ndarray:
    """以 RGB、float32、[0, 1] 范围读取图像。"""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def to_lpips_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """将 HWC [0, 1] 图像转换为 LPIPS 所需的 NCHW [-1, 1] 张量。"""
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return (tensor * 2.0 - 1.0).to(device)


def calculate_one(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    lpips_model: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"图像尺寸不一致：预测图 {prediction.shape}，GT {ground_truth.shape}"
        )

    psnr = float(peak_signal_noise_ratio(ground_truth, prediction, data_range=1.0))
    ssim = float(
        structural_similarity(
            ground_truth,
            prediction,
            channel_axis=2,
            data_range=1.0,
        )
    )
    with torch.inference_mode():
        distance = lpips_model(
            to_lpips_tensor(prediction, device),
            to_lpips_tensor(ground_truth, device),
        )
    return psnr, ssim, float(distance.item())


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def format_number(value: float) -> str:
    if math.isnan(value):
        return "N/A"
    if math.isinf(value):
        return "inf"
    return f"{value:.6f}"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="计算子文件夹内 013.png、035.png 相对于 gt 同名图像的指标。"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=script_dir,
        help="数据根目录（默认：脚本所在目录）",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="LPIPS 运行设备（默认：auto）",
    )
    parser.add_argument(
        "--net",
        choices=("alex", "vgg", "squeeze"),
        default="alex",
        help="LPIPS 主干网络（默认：alex）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV 输出路径（默认：根目录/metrics_results.csv）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    gt_dir = root / "gt"
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "metrics_results.csv"
    )

    if not gt_dir.is_dir():
        raise FileNotFoundError(f"找不到 GT 目录：{gt_dir}")
    missing_gt = [name for name in IMAGE_NAMES if not (gt_dir / name).is_file()]
    if missing_gt:
        raise FileNotFoundError(f"GT 目录缺少：{', '.join(missing_gt)}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法使用 CUDA。")
    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )

    print(f"根目录: {root}")
    print(f"LPIPS: net={args.net}, device={device}")
    lpips_model = lpips.LPIPS(net=args.net).to(device).eval()
    gt_images = {name: load_rgb(gt_dir / name) for name in IMAGE_NAMES}

    folders = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.lower() != "gt" and not path.name.startswith(".")
    )
    rows: list[dict[str, str]] = []

    header = f"{'folder':<20} {'image':<10} {'PSNR':>12} {'SSIM':>12} {'LPIPS':>12}"
    print("\n" + header)
    print("-" * len(header))

    for folder in folders:
        folder_values: dict[str, list[float]] = {
            "psnr": [],
            "ssim": [],
            "lpips": [],
        }

        for image_name in IMAGE_NAMES:
            prediction_path = folder / image_name
            if not prediction_path.is_file():
                print(
                    f"{folder.name:<20} {image_name:<10} "
                    f"{'MISSING':>12} {'MISSING':>12} {'MISSING':>12}"
                )
                rows.append(
                    {
                        "folder": folder.name,
                        "image": image_name,
                        "psnr": "",
                        "ssim": "",
                        "lpips": "",
                        "status": "missing",
                    }
                )
                continue

            try:
                prediction = load_rgb(prediction_path)
                psnr, ssim, lpips_value = calculate_one(
                    prediction, gt_images[image_name], lpips_model, device
                )
            except Exception as exc:
                print(f"{folder.name:<20} {image_name:<10} ERROR: {exc}")
                rows.append(
                    {
                        "folder": folder.name,
                        "image": image_name,
                        "psnr": "",
                        "ssim": "",
                        "lpips": "",
                        "status": f"error: {exc}",
                    }
                )
                continue

            folder_values["psnr"].append(psnr)
            folder_values["ssim"].append(ssim)
            folder_values["lpips"].append(lpips_value)
            print(
                f"{folder.name:<20} {image_name:<10} "
                f"{format_number(psnr):>12} {format_number(ssim):>12} "
                f"{format_number(lpips_value):>12}"
            )
            rows.append(
                {
                    "folder": folder.name,
                    "image": image_name,
                    "psnr": format_number(psnr),
                    "ssim": format_number(ssim),
                    "lpips": format_number(lpips_value),
                    "status": "ok",
                }
            )

        if folder_values["psnr"]:
            avg_psnr = mean_or_nan(folder_values["psnr"])
            avg_ssim = mean_or_nan(folder_values["ssim"])
            avg_lpips = mean_or_nan(folder_values["lpips"])
            print(
                f"{folder.name:<20} {'AVERAGE':<10} "
                f"{format_number(avg_psnr):>12} {format_number(avg_ssim):>12} "
                f"{format_number(avg_lpips):>12}"
            )
            rows.append(
                {
                    "folder": folder.name,
                    "image": "AVERAGE",
                    "psnr": format_number(avg_psnr),
                    "ssim": format_number(avg_ssim),
                    "lpips": format_number(avg_lpips),
                    "status": "ok",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("folder", "image", "psnr", "ssim", "lpips", "status"),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n结果已保存到：{output_path}")


if __name__ == "__main__":
    main()
