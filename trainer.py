#!/usr/bin/env python3
"""
Modified ABPN Training Script - Updated for improved architecture with better mask prediction
"""

import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torchvision.models import vgg16
import torch.nn.functional as F
import cv2
import numpy as np 
import wandb
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import random
import torchvision.transforms.functional as TF
from contextlib import contextmanager
import time
import datetime

# Import IMPROVED ABPN model
from model import ABPN


def setup_distributed(rank, world_size, port=12355):
    """Setup distributed training with longer timeout"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    # Increase timeout to 30 minutes
    os.environ['NCCL_TIMEOUT'] = '1800'
    
    # Initialize with longer timeout
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=torch.distributed.default_pg_timeout * 3  # 30 minutes
    )


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except:
            pass


@contextmanager
def suppress_stderr():
    """Suppress stderr for libpng warnings"""
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


class PerceptualLoss(nn.Module):
    """Memory-efficient VGG-based perceptual loss - downsamples large images"""
    def __init__(self, max_size=512):
        super(PerceptualLoss, self).__init__()
        self.max_size = max_size  # Maximum dimension for perceptual loss computation
        try:
            vgg = vgg16(pretrained=True).features
            self.slice1 = nn.Sequential(*list(vgg.children())[:4])
            self.slice2 = nn.Sequential(*list(vgg.children())[4:9])
            self.slice3 = nn.Sequential(*list(vgg.children())[9:16])
            self.slice4 = nn.Sequential(*list(vgg.children())[16:23])
            
            # Freeze parameters
            for param in self.parameters():
                param.requires_grad = False
                
            # Set to eval mode
            self.eval()
            self.use_vgg = True
        except Exception as e:
            print(f"Warning: Failed to load VGG16: {e}")
            self.use_vgg = False

    def forward(self, x, y):
        if not self.use_vgg:
            return F.mse_loss(x, y)
            
        try:
            # Downsample to max_size if images are too large (saves memory!)
            if x.shape[2] > self.max_size or x.shape[3] > self.max_size:
                x = F.interpolate(x, size=(self.max_size, self.max_size), mode='bilinear', align_corners=False)
                y = F.interpolate(y, size=(self.max_size, self.max_size), mode='bilinear', align_corners=False)
            
            # Ensure minimum size for VGG (64x64)
            if x.shape[2] < 64 or x.shape[3] < 64:
                return F.mse_loss(x, y)
            
            # Normalize from [-1,1] to [0,1] then to ImageNet stats
            x = (x + 1) / 2
            y = (y + 1) / 2
            
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
            
            x = (x - mean) / std
            y = (y - mean) / std
            
            # Progressive feature extraction with error handling
            h1_x = self.slice1(x)
            h1_y = self.slice1(y)
            loss = F.mse_loss(h1_x, h1_y)
            
            if h1_x.shape[2] >= 32 and h1_x.shape[3] >= 32:
                h2_x = self.slice2(h1_x)
                h2_y = self.slice2(h1_y)
                loss += F.mse_loss(h2_x, h2_y)
                
                if h2_x.shape[2] >= 16 and h2_x.shape[3] >= 16:
                    h3_x = self.slice3(h2_x)
                    h3_y = self.slice3(h2_y)
                    loss += F.mse_loss(h3_x, h3_y)
                    
                    if h3_x.shape[2] >= 8 and h3_x.shape[3] >= 8:
                        h4_x = self.slice4(h3_x)
                        h4_y = self.slice4(h3_y)
                        loss += F.mse_loss(h4_x, h4_y)
            
            return loss
            
        except Exception as e:
            print(f"Warning: PerceptualLoss failed, using MSE: {e}")
            return F.mse_loss(x, y)


def rgb_to_grayscale_tensor(x: torch.Tensor) -> torch.Tensor:
    """Convert a BCHW RGB tensor to grayscale B1HW using standard weights."""
    # x expected in shape (B,3,H,W)
    r, g, b = x[:, 0:1, ...], x[:, 1:2, ...], x[:, 2:3, ...]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edge magnitude for a B1HW tensor. Returns B1HW."""
    # kernels
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    grad = torch.sqrt(gx * gx + gy * gy + 1e-6)
    return grad


def laplacian(x: torch.Tensor) -> torch.Tensor:
    """Compute Laplacian (high-frequency) response for BCHW tensor. Returns BCHW."""
    # Laplacian kernel applied per-channel
    k = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]], device=x.device).view(1, 1, 3, 3)
    # apply per-channel by grouping
    B, C, H, W = x.shape
    x_reshaped = x.view(B * C, 1, H, W)
    resp = F.conv2d(x_reshaped, k, padding=1)
    resp = resp.view(B, C, H, W)
    return resp


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked L1 loss averaged over masked pixels. pred/target BCHW, mask B1HW or BHW in [0,1]."""
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    # broadcast mask to channels
    mask3 = mask.repeat(1, pred.shape[1], 1, 1)
    loss_map = torch.abs(pred - target) * mask3

    denom = mask3.sum()
    if denom.item() <= 0:
        # fallback to global mean if mask empty
        return loss_map.mean()
    return loss_map.sum() / (denom + 1e-6)


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute L1 loss for the whole image"""
    loss_map = torch.abs(pred - target)
    return loss_map.mean()


def masked_edge_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked edge loss using Sobel on grayscale images."""
    pred_gray = rgb_to_grayscale_tensor(pred)
    tgt_gray = rgb_to_grayscale_tensor(target)
    pred_e = sobel_edges(pred_gray)
    tgt_e = sobel_edges(tgt_gray)
    # mask is single channel
    if mask.dim() == 3:
        mask1 = mask.unsqueeze(1)
    else:
        mask1 = mask
    loss_map = torch.abs(pred_e - tgt_e) * mask1
    denom = mask1.sum()
    if denom.item() <= 0:
        return loss_map.mean()
    return loss_map.sum() / (denom + 1e-6)


# def masked_highfreq_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#     """Compute masked high-frequency loss using Laplacian per-channel."""
#     pred_h = laplacian(pred)
#     tgt_h = laplacian(target)
#     if mask.dim() == 3:
#         mask = mask.unsqueeze(1)
#     mask3 = mask.repeat(1, pred.shape[1], 1, 1)
#     loss_map = torch.abs(pred_h - tgt_h) * mask3
#     denom = mask3.sum()
#     if denom.item() <= 0:
#         return loss_map.mean()
#     return loss_map.sum() / (denom + 1e-6)


# def generate_ground_truth_mask(input_image, target_image, threshold=0.1):
#     """Generate ground truth mask from input and target images"""
#     input_norm = input_image.astype(np.float32) / 255.0
#     target_norm = target_image.astype(np.float32) / 255.0
    
#     diff = np.abs(input_norm - target_norm)
#     diff_gray = np.mean(diff, axis=2)
#     mask = (diff_gray > threshold).astype(np.float32)
    
#     # Clean up mask
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
#     return mask[..., np.newaxis]

def generate_ground_truth_mask(input_image, target_image, magnifier=1.0, threshold=12):
    """Stray hair specialist - brightness only."""
    hsv1 = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV)
    hsv2 = cv2.cvtColor(target_image, cv2.COLOR_RGB2HSV)
    
    v1 = hsv1[:, :, 2].astype(np.float32)
    v2 = hsv2[:, :, 2].astype(np.float32)
    
    diff = np.abs(v1 - v2) * magnifier  # Amplifies faint hairs
    binary_mask = (diff > threshold).astype(np.uint8) * 255
    blurred = cv2.GaussianBlur(binary_mask, (5, 5), 0)  # Soft edges
    mask_float = blurred.astype(np.float32) / 255.0
    return mask_float[..., np.newaxis]


def masked_highfreq_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked high-frequency loss using Laplacian per-channel."""
    pred_h = laplacian(pred)
    tgt_h = laplacian(target)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask3 = mask.repeat(1, pred.shape[1], 1, 1)
    loss_map = torch.abs(pred_h - tgt_h) * mask3
    denom = mask3.sum()
    if denom.item() <= 0:
        return loss_map.mean()
    return loss_map.sum() / (denom + 1e-6)

def generate_ground_truth_mask(input_image, target_image, magnifier=6.0, threshold=30):
    """
    Improved mask generation with morphological cleaning to remove background noise.
    
    Args:
        input_image: Source RGB
        target_image: Ground truth RGB
        magnifier: Higher multiplier makes wrinkles stand out (Try 3.0)
        threshold: Higher threshold removes faint ghosting in shadows (Try 25-30)
    """
    # Convert to HSV and extract V channel (Brightness)
    v1 = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV)[:, :, 2].astype(np.float32)
    v2 = cv2.cvtColor(target_image, cv2.COLOR_RGB2HSV)[:, :, 2].astype(np.float32)
    
    # 1. Calculate absolute difference and amplify
    diff = np.abs(v1 - v2) * magnifier
    
    # 2. Initial binary mask with higher threshold
    _, binary_mask = cv2.threshold(diff.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
    
    # 3. CLEANING STEP: Morphological Opening
    # This removes the tiny white dots (noise) shown in your right-hand image
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 4. Optional: Small dilation to ensure full coverage of dewrinkled area
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    
    # 5. Soften edges
    blurred = cv2.GaussianBlur(binary_mask, (5, 5), 0)
    mask_float = blurred.astype(np.float32) / 255.0
    
    return mask_float[..., np.newaxis]



def get_files_by_base_name(directory, valid_extensions=('.jpg', '.jpeg', '.png')):
    """Get files mapped by base name"""
    file_dict = {}
    for filename in os.listdir(directory):
        base_name, ext = os.path.splitext(filename)
        if ext.lower() in valid_extensions:
            file_dict[base_name] = filename
    return file_dict


class ABPNDataset(Dataset):
    """ABPN Dataset with PATCH-WISE training - extracts multiple 2048px patches from full-res images"""
    def __init__(self, image_dir, gt_dir, patch_size=2048, stride=1536, mask_threshold=3, pyramid_levels=1):
        """
        Args:
            image_dir: Directory with input images (any resolution)
            gt_dir: Directory with GT retouched images
            patch_size: Size of patches to extract (2048)
            stride: Stride for patch extraction (1536 = 512px overlap)
            mask_threshold: Threshold for mask generation
            pyramid_levels: Pyramid levels for ABPN
        """
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.patch_size = patch_size
        self.stride = stride
        self.mask_threshold = mask_threshold

        self.mask_size = (patch_size // 2, patch_size // 2)

        image_files = get_files_by_base_name(image_dir)
        gt_files = get_files_by_base_name(gt_dir)

        self.bases = sorted(set(image_files) & set(gt_files))
        if not self.bases:
            raise RuntimeError("No matching image/gt pairs found")

        self.image_paths = [
            (os.path.join(image_dir, image_files[b]),
             os.path.join(gt_dir, gt_files[b]))
            for b in self.bases
        ]

        # Precompute patch grid size using ONE image
        sample_img = cv2.imread(self.image_paths[0][0])
        h, w = sample_img.shape[:2]

        self.num_patches_y = max(1, (h - patch_size + stride) // stride)
        self.num_patches_x = max(1, (w - patch_size + stride) // stride)
        self.patches_per_image = self.num_patches_y * self.num_patches_x

        print(
            f"LazyDataset: {len(self.image_paths)} images, "
            f"{self.patches_per_image} patches/image"
        )

    def __len__(self):
        return len(self.image_paths) * self.patches_per_image

    def _patch_coords(self, patch_idx, h, w):
        py = patch_idx // self.num_patches_x
        px = patch_idx % self.num_patches_x

        y = min(py * self.stride, h - self.patch_size)
        x = min(px * self.stride, w - self.patch_size)

        return max(0, y), max(0, x)

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image

        img_path, gt_path = self.image_paths[img_idx]

        image = cv2.imread(img_path)
        gt = cv2.imread(gt_path)

        if image is None or gt is None:
            raise RuntimeError(f"Failed to load {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        y, x = self._patch_coords(patch_idx, h, w)

        patch_image = image[y:y+self.patch_size, x:x+self.patch_size]
        patch_gt = gt[y:y+self.patch_size, x:x+self.patch_size]

        if patch_image.shape[:2] != (self.patch_size, self.patch_size):
            patch_image = cv2.copyMakeBorder(
                patch_image,
                0, self.patch_size - patch_image.shape[0],
                0, self.patch_size - patch_image.shape[1],
                cv2.BORDER_REFLECT
            )
            patch_gt = cv2.copyMakeBorder(
                patch_gt,
                0, self.patch_size - patch_gt.shape[0],
                0, self.patch_size - patch_gt.shape[1],
                cv2.BORDER_REFLECT
            )

        mask_full = generate_ground_truth_mask(
            patch_image, patch_gt, self.mask_threshold
        )

        mask_low = cv2.resize(mask_full.squeeze(), self.mask_size)[..., None]

        patch_image = (patch_image.astype(np.float32) / 255.0) * 2 - 1
        patch_gt = (patch_gt.astype(np.float32) / 255.0) * 2 - 1

        patch_image = torch.from_numpy(patch_image).permute(2, 0, 1)
        patch_gt = torch.from_numpy(patch_gt).permute(2, 0, 1)
        mask = torch.from_numpy(mask_low).permute(2, 0, 1)

        return {
            "image": patch_image,
            "gt": patch_gt,
            "mask": mask,
            "filename": f"{os.path.basename(img_path)}_{patch_idx}"
        }


def dice_loss(pred_mask, target_mask):
    """Dice loss for mask prediction - handles size mismatches"""
    smooth = 1e-8
    
    # Handle size mismatches by resizing predicted mask to match target
    if pred_mask.shape != target_mask.shape:
        pred_mask = F.interpolate(pred_mask, size=target_mask.shape[2:], mode='bilinear', align_corners=False)
    
    pred_flat = pred_mask.view(-1)
    target_flat = target_mask.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1 - dice


def save_epoch_samples(model, data_loader, device, epoch, save_dir, num_samples=4):
    """Save sample results after epoch completion"""
    model.eval()
    results_dir = os.path.join(save_dir, 'epoch_results', f'epoch_{epoch+1}')
    os.makedirs(results_dir, exist_ok=True)
    
    saved_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if saved_count >= num_samples:
                break
                
            images = batch['image'].to(device, non_blocking=True)
            gt_images = batch['gt'].to(device, non_blocking=True)
            gt_masks = batch['mask'].to(device, non_blocking=True)
            
            # Forward pass
            pred_retouched, pred_mask = model(images)
            
            # Save samples from this batch
            batch_size = images.size(0)
            samples_to_save = min(num_samples - saved_count, batch_size)
            
            for i in range(samples_to_save):
                try:
                    # Convert tensors to numpy arrays and denormalize
                    orig_img = ((images[i].detach().cpu().permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
                    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
                    
                    gt_img = ((gt_images[i].detach().cpu().permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
                    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
                    
                    pred_img = ((pred_retouched[i].detach().cpu().permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
                    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
                    
                    # Predicted mask (resize to match image size for visualization)
                    if pred_mask[i].shape[1:] != orig_img.shape[:2]:
                        pred_mask_resized = F.interpolate(pred_mask[i:i+1], size=orig_img.shape[:2], mode='bilinear', align_corners=False)
                        pred_mask_img = (pred_mask_resized[0].detach().cpu().squeeze().numpy() * 255).astype(np.uint8)
                    else:
                        pred_mask_img = (pred_mask[i].detach().cpu().squeeze().numpy() * 255).astype(np.uint8)
                    
                    # Ground truth mask (resize to match image size for visualization)
                    if gt_masks[i].shape[1:] != orig_img.shape[:2]:
                        gt_mask_resized = F.interpolate(gt_masks[i:i+1], size=orig_img.shape[:2], mode='bilinear', align_corners=False)
                        gt_mask_img = (gt_mask_resized[0].detach().cpu().squeeze().numpy() * 255).astype(np.uint8)
                    else:
                        gt_mask_img = (gt_masks[i].detach().cpu().squeeze().numpy() * 255).astype(np.uint8)
                    
                    # Create filename
                    filename = batch['filename'][i]
                    base_name = os.path.splitext(filename)[0]
                    sample_idx = saved_count + i + 1
                    
                    # Save images
                    cv2.imwrite(f"{results_dir}/sample{sample_idx:02d}_{base_name}_original.jpg", orig_img)
                    cv2.imwrite(f"{results_dir}/sample{sample_idx:02d}_{base_name}_gt.jpg", gt_img)
                    cv2.imwrite(f"{results_dir}/sample{sample_idx:02d}_{base_name}_predicted.jpg", pred_img)
                    cv2.imwrite(f"{results_dir}/sample{sample_idx:02d}_{base_name}_pred_mask.jpg", pred_mask_img)
                    cv2.imwrite(f"{results_dir}/sample{sample_idx:02d}_{base_name}_gt_mask.jpg", gt_mask_img)
                    
                except Exception as e:
                    print(f"Error saving sample {saved_count + i + 1}: {e}")
                    continue
            
            saved_count += samples_to_save
    
    model.train()  # Switch back to training mode
    print(f"Saved {saved_count} sample results for epoch {epoch+1}")


def validate_model(model, val_loader, device, criterion_perceptual,
                   weight_l1=1.0, weight_edge=1.0, weight_perc=0.5, weight_highfreq=0.3,
                   weight_img=1.0, weight_dice=0.5, use_amp=True, compute_lpips=False, compute_fid=False):
    """Validation with INPAINTING LOSSES"""
    model.eval()
    
    total_val_loss = 0.0
    total_l1 = 0.0
    total_edge = 0.0
    total_perc = 0.0
    total_hf = 0.0
    total_img = 0.0
    total_dice = 0.0
    num_val_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch['image'].to(device, non_blocking=True)
            gt_images = batch['gt'].to(device, non_blocking=True)
            gt_masks = batch['mask'].to(device, non_blocking=True)
            
            # Mixed Precision Forward pass
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred_retouched, pred_mask = model(images)
                
                # Resize mask to prediction size
                mask_resized = F.interpolate(gt_masks, size=pred_retouched.shape[2:], 
                                            mode='bilinear', align_corners=False)
                
                # Compute all inpainting losses
                l1 = masked_l1_loss(pred_retouched, gt_images, mask_resized)
                edge = masked_edge_loss(pred_retouched, gt_images, mask_resized)
                perc = criterion_perceptual(pred_retouched, gt_images)
                hf = masked_highfreq_loss(pred_retouched, gt_images, mask_resized)
                img = l1_loss(pred_retouched, gt_images)
                dice = dice_loss(pred_mask, gt_masks)
                
                total_loss = (
                    weight_l1 * l1 
                    + weight_edge * edge 
                    + weight_perc * perc 
                    + weight_highfreq * hf 
                    + weight_img * img
                    + weight_dice * dice
                )
            
            total_val_loss += total_loss.item()
            total_l1 += l1.item()
            total_edge += edge.item()
            total_perc += perc.item()
            total_hf += hf.item()
            total_img += img.item()
            total_dice += dice.item()
            num_val_batches += 1
    
    # Average losses
    avg_total_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    avg_l1 = total_l1 / num_val_batches if num_val_batches > 0 else 0.0
    avg_edge = total_edge / num_val_batches if num_val_batches > 0 else 0.0
    avg_perc = total_perc / num_val_batches if num_val_batches > 0 else 0.0
    avg_hf = total_hf / num_val_batches if num_val_batches > 0 else 0.0
    avg_img = total_img / num_val_batches if num_val_batches > 0 else 0.0
    avg_dice = total_dice / num_val_batches if num_val_batches > 0 else 0.0
    
    model.train()
    
    return avg_total_loss, avg_l1, avg_edge, avg_perc, avg_hf, avg_img, avg_dice, None, None




def safe_distributed_reduce(tensor, world_size):
    """FIXED: Safely reduce tensor across processes with timeout handling"""
    try:
        # FIXED: Create the work handle first
        work = dist.all_reduce(tensor, async_op=True)
        work.wait(timeout=datetime.timedelta(seconds=120))
        return tensor.item() / world_size
    except Exception as e:
        print(f"Warning: Distributed reduce failed: {e}")
        return tensor.item()  # Return local value if reduce fails


def train_abpn(local_rank, world_size, model, train_loader, test_loader, args):
    """Training function with INPAINTING LOSSES from your script"""
    
    device = torch.device(f'cuda:{local_rank}')
    model = model.to(device)
    
    # Use find_unused_parameters=True for better stability
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # Loss function - Perceptual Loss with memory optimization
    criterion_perceptual = PerceptualLoss(max_size=512).to(device)  # Downsample to 512 for memory efficiency
    
    if local_rank == 0:
        print("✓ Initialized Inpainting Losses:")
        print("  - Masked L1 Loss (masked regions)")
        print("  - Global L1 Loss (whole image)")
        print("  - Masked Edge Loss (Sobel)")
        print("  - Masked High-Frequency Loss (Laplacian)")
        print("  - Perceptual Loss (VGG16, downsampled to 512)")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Mixed Precision Training
    use_amp = args.get('use_amp', True)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if local_rank == 0 and use_amp:
        print("✓ Using Automatic Mixed Precision (AMP) training")
    
    # ===== INPAINTING LOSS WEIGHTS (from your script) =====
    weight_l1 = 1.3          # Masked L1 loss
    weight_edge = 1.0        # Masked edge loss
    weight_perc = 0.5        # Perceptual loss
    weight_highfreq = 0.7    # Masked high-frequency loss
    weight_img = 1.0         # Global L1 loss
    weight_dice = 1.0       # Dice loss for mask prediction

    if local_rank == 0:
        print(f"\n✓ Loss Weights:")
        print(f"  Masked L1: {weight_l1}")
        print(f"  Masked Edge: {weight_edge}")
        print(f"  Perceptual: {weight_perc}")
        print(f"  Masked High-Freq: {weight_highfreq}")
        print(f"  Global L1: {weight_img}")
        print(f"  Dice (Mask): {weight_dice}")
    
    best_loss = float('inf')
    
    for epoch in range(args['num_epochs']):
        # Training
        train_loader.sampler.set_epoch(epoch)
        model.train()
        
        # Track ALL losses separately
        epoch_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_edge_loss = 0.0
        epoch_perc_loss = 0.0
        epoch_highfreq_loss = 0.0
        epoch_img_loss = 0.0
        epoch_dice_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()
        
        if local_rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args['num_epochs']}")
        else:
            pbar = train_loader
        
        for batch_idx, batch in enumerate(pbar):
            try:
                images = batch['image'].to(device, non_blocking=True)
                gt_images = batch['gt'].to(device, non_blocking=True)
                gt_masks = batch['mask'].to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Mixed Precision Forward pass
                with torch.amp.autocast('cuda', enabled=use_amp):
                    # Forward pass - model returns prediction and predicted mask
                    pred_retouched, pred_mask = model(images)
                    
                    # Resize mask to prediction size if needed
                    mask_resized = F.interpolate(gt_masks, size=pred_retouched.shape[2:], 
                                                mode='bilinear', align_corners=False)
                    
                    # ===== COMPUTE ALL INPAINTING LOSSES =====
                    
                    # 1. Masked L1 Loss (wrinkle/defect regions)
                    loss_l1 = masked_l1_loss(pred_retouched, gt_images, mask_resized)
                    
                    # 2. Masked Edge Loss (preserve sharp boundaries)
                    loss_edge = masked_edge_loss(pred_retouched, gt_images, mask_resized)
                    
                    # 3. Perceptual Loss (texture/structure matching)
                    loss_perc = criterion_perceptual(pred_retouched, gt_images)
                    
                    # 4. Masked High-Frequency Loss (fine details)
                    loss_highfreq = masked_highfreq_loss(pred_retouched, gt_images, mask_resized)
                    
                    # 5. Global L1 Loss (overall image quality)
                    loss_img = l1_loss(pred_retouched, gt_images)
                    
                    # 6. Dice Loss for mask prediction
                    loss_dice = dice_loss(pred_mask, gt_masks)
                    
                    # Total weighted loss
                    total_loss = (
                        weight_l1 * loss_l1
                        + weight_edge * loss_edge
                        + weight_perc * loss_perc
                        + weight_highfreq * loss_highfreq
                        + weight_img * loss_img
                        + weight_dice * loss_dice
                    )
                
                # Mixed Precision Backward pass
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                # Track all losses
                epoch_loss += total_loss.item()
                epoch_l1_loss += loss_l1.item()
                epoch_edge_loss += loss_edge.item()
                epoch_perc_loss += loss_perc.item()
                epoch_highfreq_loss += loss_highfreq.item()
                epoch_img_loss += loss_img.item()
                epoch_dice_loss += loss_dice.item()
                num_batches += 1
                
                if local_rank == 0:
                    pbar.set_postfix({
                        'Loss': f'{total_loss.item():.4f}',
                        'L1': f'{loss_l1.item():.4f}',
                        'Edge': f'{loss_edge.item():.4f}',
                        'Perc': f'{loss_perc.item():.4f}',
                        'HF': f'{loss_highfreq.item():.4f}',
                        'Img': f'{loss_img.item():.4f}',
                        'Dice': f'{loss_dice.item():.4f}'
                    })
                
                # Clear cache periodically (important for 2048px patches!)
                if batch_idx % 20 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"[GPU {local_rank}] Error in batch {batch_idx}: {e}")
                continue
        
        # Calculate average epoch losses
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_l1_loss = epoch_l1_loss / num_batches if num_batches > 0 else 0.0
        avg_edge_loss = epoch_edge_loss / num_batches if num_batches > 0 else 0.0
        avg_perc_loss = epoch_perc_loss / num_batches if num_batches > 0 else 0.0
        avg_highfreq_loss = epoch_highfreq_loss / num_batches if num_batches > 0 else 0.0
        avg_img_loss = epoch_img_loss / num_batches if num_batches > 0 else 0.0
        avg_dice_loss = epoch_dice_loss / num_batches if num_batches > 0 else 0.0
        
        # Create tensors for distributed reduce
        loss_tensor = torch.tensor(avg_epoch_loss, device=device)
        l1_loss_tensor = torch.tensor(avg_l1_loss, device=device)
        edge_loss_tensor = torch.tensor(avg_edge_loss, device=device)
        perc_loss_tensor = torch.tensor(avg_perc_loss, device=device)
        highfreq_loss_tensor = torch.tensor(avg_highfreq_loss, device=device)
        img_loss_tensor = torch.tensor(avg_img_loss, device=device)
        dice_loss_tensor = torch.tensor(avg_dice_loss, device=device)
        
        # Safe distributed reduce for ALL losses
        final_loss = safe_distributed_reduce(loss_tensor, world_size)
        final_l1_loss = safe_distributed_reduce(l1_loss_tensor, world_size)
        final_edge_loss = safe_distributed_reduce(edge_loss_tensor, world_size)
        final_perc_loss = safe_distributed_reduce(perc_loss_tensor, world_size)
        final_highfreq_loss = safe_distributed_reduce(highfreq_loss_tensor, world_size)
        final_img_loss = safe_distributed_reduce(img_loss_tensor, world_size)
        final_dice_loss = safe_distributed_reduce(dice_loss_tensor, world_size)
        
        # Validation and logging on main process only
        if local_rank == 0:
            epoch_time = time.time() - epoch_start_time
            print(
                f"Epoch {epoch+1}: Total={final_loss:.4f}, L1={final_l1_loss:.4f}, "
                f"Edge={final_edge_loss:.4f}, Perc={final_perc_loss:.4f}, "
                f"HF={final_highfreq_loss:.4f}, Img={final_img_loss:.4f}, Dice={final_dice_loss:.4f}, Time={epoch_time:.1f}s"
            )
            
            # Validation
            val_total_loss, val_l1, val_edge, val_perc, val_hf, val_img, val_dice, _, _ = validate_model(
                model, test_loader, device,
                criterion_perceptual,
                weight_l1, weight_edge, weight_perc, weight_highfreq, weight_img, weight_dice,
                use_amp,
                compute_lpips=False,
                compute_fid=False
            )
            
            print(f"Validation Loss: {val_total_loss:.4f}")
            
            # Log to W&B
            wandb.log({
                "epoch": epoch + 1,
                "train_loss_total": final_loss,
                "train_loss_l1": final_l1_loss,
                "train_loss_edge": final_edge_loss,
                "train_loss_perceptual": final_perc_loss,
                "train_loss_highfreq": final_highfreq_loss,
                "train_loss_img": final_img_loss,
                "train_loss_dice": final_dice_loss,
                "val_loss_total": val_total_loss,
                "val_loss_l1": val_l1,
                "val_loss_edge": val_edge,
                "val_loss_perceptual": val_perc,
                "val_loss_highfreq": val_hf,
                "val_loss_img": val_img,
                "val_loss_dice": val_dice,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch_time": epoch_time
            })
            
            # Save best model
            if val_total_loss < best_loss:
                best_loss = val_total_loss
                torch.save(model.module.state_dict(), f"{args['save_dir']}/best_model.pth")
                print(f"✓ New best model saved! Val Loss: {best_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss,
                }, f"{args['save_dir']}/checkpoint_epoch_{epoch + 1}.pth")
            
            # Save epoch samples
            if (epoch + 1) % args['sample_save_freq'] == 0:
                print(f"Saving epoch {epoch+1} samples...")
                save_epoch_samples(model, test_loader, device, epoch, args['save_dir'], 
                                 num_samples=20)
        
        scheduler.step()
    
    if local_rank == 0:
        torch.save(model.module.state_dict(), f"{args['save_dir']}/final_model.pth")
        print("✓ Training completed!")

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume the checkpoint is directly the state dict
        model.load_state_dict(checkpoint)
    
    return model

def main_worker(local_rank, world_size, args):
    """Main worker for distributed training with improved error handling"""
    
    try:
        # Setup distributed training
        setup_distributed(local_rank, world_size, args['port'])
        torch.cuda.set_device(local_rank)
        
        if local_rank == 0:
            print("Initializing W&B...")
            wandb.init(project='abpn_patchwise_2048', entity=args['entity_name'], config=args)
        
        # Create PATCH-WISE dataset
        dataset = ABPNDataset(
            args['image_dir'],
            args['gt_dir'],
            patch_size=args['patch_size'],
            stride=args['stride'],
            mask_threshold=args['mask_threshold']
        )
        
        # Split dataset (same as before)
        total_size = len(dataset)  # This is now TOTAL PATCHES, not images
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        if local_rank == 0:
            print(f"Dataset split: Train={train_size}, Val={val_size}")
        
        # Create data loaders
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args['batch_size'],
            sampler=train_sampler,
            num_workers=16,
            pin_memory=False,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args['batch_size'], 
            sampler=val_sampler, 
            num_workers=16,
            pin_memory=False,
            drop_last=True
        )
        
        # Create model
        model = ABPN(in_channels=3, pyramid_levels=args['pyramid_levels'])
        if args['pretrained_path']:
            if local_rank == 0:
                print(f"Loading pretrained model from {args['pretrained_path']}")
                model = load_checkpoint(model, args['pretrained_path'])
        
        if local_rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Improved ABPN Model parameters: {total_params:,}")
        
        # Start training
        train_abpn(local_rank, world_size, model, train_loader, val_loader, args)
        
    except Exception as e:
        print(f"[GPU {local_rank}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        cleanup_distributed()
        if local_rank == 0:
            wandb.finish()


def main():
    """Main function"""
    print("=== Improved ABPN Multi-GPU Training ===")
    
    args = {
        'image_dir': '/workspace/original',
        'gt_dir': '/workspace/retouch/',
        'num_epochs': 200,
        'batch_size': 16,
        'learning_rate': 0.0005,
        'save_dir': 'dc_newarchnew/',
        'port': 12355,
        'entity_name': 'anmol-d-aftershoot',
        'patch_size': 2048,
        'stride': 1536,
        'pyramid_levels': 1,
        'mask_threshold': 3,
        'sample_save_freq': 1,
        'pretrained_path' : ''

    }
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return
    
    world_size = torch.cuda.device_count()
    print(f"Available GPUs: {world_size}")
    
    # Create save directory
    os.makedirs(args['save_dir'], exist_ok=True)
    os.makedirs(os.path.join(args['save_dir'], 'epoch_results'), exist_ok=True)
    
    # Print configuration
    print("\nConfiguration:")
    for key, value in args.items():
        print(f"  {key}: {value}")
    
    try:
        mp.set_start_method('spawn', force=True)
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, args), join=True)
        print("✓ Training completed successfully!")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        cleanup_distributed()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        cleanup_distributed()
        sys.exit(1)