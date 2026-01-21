# ABPN Patch-wise Training (2048px) with Mask Prediction

This repository contains a **patch-wise multi-GPU training pipeline** for an improved **ABPN (Adaptive Blend Pyramid Network)** architecture.  
The model is trained for **high-resolution image retouching / inpainting tasks** with an **explicit mask prediction branch** and multiple perceptual + structural losses.

The setup is designed to handle **very large images** by training on overlapping **2048×2048 patches**.

---

## Key Features

- Patch-wise training on full-resolution images (2048 px)
- Multi-GPU Distributed Data Parallel (DDP)
- Automatic Mixed Precision (AMP)
- Explicit **mask prediction branch** trained with Dice loss
- Multiple inpainting-focused losses:
  - Masked L1
  - Global L1
  - Masked edge (Sobel)
  - Masked high-frequency (Laplacian)
  - Perceptual loss (VGG16, memory-optimized)
- Robust distributed training with timeout handling
- Epoch-wise visual sample saving
- Weights & Biases logging

---

## Project Structure

| trainer.py # Full training pipeline (DDP, dataset, losses, logging)
| model.py # ABPN architecture (retouch + mask prediction)
| README.md



---

## Model Overview

The **ABPN model** outputs **two things**:

1. **Retouched image** (same resolution as input patch)
2. **Predicted mask** (lower resolution, resized during training)

The predicted mask is trained jointly using **Dice loss**, encouraging the network to learn *where* to retouch, not just *how*.

---

## Dataset Assumptions

You need **paired images** with identical base filenames:

/workspace/original/
image_001.jpg
image_002.jpg

/workspace/retouch/
image_001.jpg
image_002.jpg



Only files with matching base names are used.

---

## Patch-wise Training Strategy

- Patch size: **2048×2048**
- Stride: **1536** (512 px overlap)
- Each image produces multiple overlapping patches
- Dataset length = total number of patches (not images)

Padding:
- Border patches are padded using `cv2.BORDER_REFLECT`

---

## Ground Truth Mask Generation

Ground-truth masks are **auto-generated** from input vs GT images:

- Convert both images to HSV
- Use **V channel (brightness) difference**
- Amplify difference (`magnifier`)
- Threshold to isolate retouched regions
- Morphological opening + dilation
- Gaussian blur for soft edges

This produces a **soft supervision mask**, resized to half resolution for training.

---

## Loss Functions

### Image Losses

| Loss | Purpose |
|----|----|
| Masked L1 | Focus learning on retouched regions |
| Global L1 | Preserve overall image consistency |
| Masked Edge (Sobel) | Preserve edges and structure |
| Masked High-Frequency (Laplacian) | Preserve fine details |
| Perceptual (VGG16) | Texture and semantic similarity |

### Mask Loss

- **Dice Loss** between predicted mask and GT mask

### Default Loss Weights

- Masked L1        = 1.3
- Masked Edge      = 1.0
- Perceptual       = 0.5
- High-Frequency   = 0.7
- Global L1        = 1.0
- Dice (Mask)      = 1.0


Distributed Training
- Uses PyTorch DDP
- NCCL configured for stability on large patches
- Timeout extended to 30 minutes
- Safe all_reduce with timeout protection
- find_unused_parameters=True for mask branch safety

Mixed Precision (AMP)
- AMP is enabled by default:
- torch.amp.autocast
- torch.amp.GradScaler
- Reduces memory usage significantly for 2048 px patches


Training Configuration

Defined inside main():

```args = {
    'image_dir': '/workspace/original',
    'gt_dir': '/workspace/retouch/',
    'num_epochs': 200,
    'batch_size': 16,
    'learning_rate': 0.0005,
    'save_dir': 'dc_newarchnew/',
    'patch_size': 2048,
    'stride': 1536,
    'pyramid_levels': 1,
    'mask_threshold': 3,
    'sample_save_freq': 1,
    'pretrained_path': ''
}

Adjust batch size carefully.
2048 px patches are extremely memory-heavy.

Running Training:

```python trainer.py


