# Haze-VGGT

Haze-VGGT: Adapting a Visual Geometry Foundation Model to Haze-Degraded Multi-View Data.

## Abstract

Visual geometry foundation models can fuse multiple images to infer cameras, depth, and three-dimensional structure, but their reliability deteriorates when the input views are jointly degraded by haze. Atmospheric scattering produces low-quality observations whose feature strength and geometric reliability vary with depth, image region, and viewpoint; unconditional fusion can consequently propagate unreliable evidence. We present Haze-VGGT, a condition-aware adaptation of a FastVGGT-based visual geometry foundation model for multi-view reconstruction from hazy images. A Haze Condition Vector (HCV) encodes scattering strength, effective visibility, spatial non-uniformity, and latent degradation factors, while a lightweight estimator predicts the condition and its confidence for real images. The HCV embedding drives Feature-wise Linear Modulation (FiLM) and Conditional Attention Bias (CAB) to adapt feature responses and cross-view associations, and a per-pixel uncertainty output represents unreliable dense geometry. Haze-VGGT retains FastVGGT's inherited Token Merging mechanism for scalable fusion over longer view sequences. On DTU-Haze, Real-Haze, and MipNeRF-360-Haze, Haze-VGGT obtains the best value among the compared methods for every metric reported in the respective benchmark tables. On DTU-Haze, it reaches 28.421 dB PSNR, 0.944 SSIM, 0.041 LPIPS, and 1.732 Chamfer Distance. Component ablations, HCV sensitivity tests, and view-count experiments characterize the effects of adaptive conditioning and scalable fusion. These results support condition-aware foundation-model adaptation as a practical route to geometric learning from haze-degraded multi-view data.

## Architecture

[Figure 1: Overview of Haze-VGGT](fig1.pdf)

The inference path consists of an HCV front end, an alternating frame-wise and global attention backbone, and multi-task prediction heads for cameras and dense geometry. Intermediate condition-adapted views remain as PyTorch tensors in memory. The program writes only the final GLB reconstruction.

## Repository Layout

```text
hazevggt/
    hcv/                 Haze-condition estimation and feature modulation
    heads/               Multi-task prediction heads
    layers/              Attention, embedding, and Transformer layers
    models/              Haze-VGGT model definitions
hazevggt_checkpoints/     Model checkpoint directory
hazy_multiview/           Input hazy multi-view images
hazevggt_outputs/         Generated reconstruction output
fig1.pdf                  Architecture overview
hazevggt_inference.py     Inference entry point
requirements.txt          Python dependencies
```

## Installation

An NVIDIA GPU and a CUDA-enabled PyTorch installation are required. The inference code uses CUDA directly.

Create a Python 3.10 environment:

```text
conda create -n hazevggt python=3.10 -y
conda activate hazevggt
```

Install the packages imported by the source code:

```text
pip install -r requirements.txt
```

If the installed PyTorch package does not support the local CUDA configuration, install a matching CUDA-enabled PyTorch build before running the program.

## Model Checkpoints

The inference script checks the following files before loading the models:

```text
hazevggt_checkpoints/
    hcv_token_estimator.pth
    hcv_confidence_head.pth
    hazevggt_geometry_backbone.pt
```

If a file is missing, `hazevggt_inference.py` downloads it automatically. The official repository is:

[https://huggingface.co/awhitewhale/hazevggt/](https://huggingface.co/awhitewhale/hazevggt/)

Direct file links:

- [hazevggt_geometry_backbone.pt](https://huggingface.co/awhitewhale/hazevggt/resolve/main/hazevggt_geometry_backbone.pt)
- [hcv_confidence_head.pth](https://huggingface.co/awhitewhale/hazevggt/resolve/main/hcv_confidence_head.pth)
- [hcv_token_estimator.pth](https://huggingface.co/awhitewhale/hazevggt/resolve/main/hcv_token_estimator.pth)

The official Hugging Face source is tried first. If the connection times out or fails, the script switches to the domestic mirror automatically:

[https://hf-mirror.com/awhitewhale/hazevggt/](https://hf-mirror.com/awhitewhale/hazevggt/)

The corresponding domestic direct links are:

- [hazevggt_geometry_backbone.pt](https://hf-mirror.com/awhitewhale/hazevggt/resolve/main/hazevggt_geometry_backbone.pt)
- [hcv_confidence_head.pth](https://hf-mirror.com/awhitewhale/hazevggt/resolve/main/hcv_confidence_head.pth)
- [hcv_token_estimator.pth](https://hf-mirror.com/awhitewhale/hazevggt/resolve/main/hcv_token_estimator.pth)

The files may also be downloaded manually and placed in `hazevggt_checkpoints/`. An incomplete download is written with a temporary `.download` suffix and is not used as a checkpoint.

The checkpoint files use the Haze-VGGT envelope format. The inference program opens the envelope and extracts the serialized model state automatically.

The final directory layout is:

```text
hazevggt_checkpoints/
    hcv_token_estimator.pth
    hcv_confidence_head.pth
    hazevggt_geometry_backbone.pt
```

## Input Images

Place the hazy views in `hazy_multiview/`. The default input directory already contains example JPEG images. JPEG, PNG, BMP, and WebP images are accepted.

Image files are sorted by filename before reconstruction. Use a consistent naming sequence when view order matters.

## Inference

Run the complete reconstruction pipeline:

```text
python hazevggt_inference.py
```

The result is written to:

```text
hazevggt_outputs/hazevggt_reconstruction.glb
```

The program performs haze-condition adaptation, multi-view geometry prediction, reliability-aware point selection, camera visualization, and GLB export in one run. No intermediate image files are created.
