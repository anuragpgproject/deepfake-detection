"""
src/explainability/attention_viz.py

Attention visualization for the ViT branch. Uses the last transformer
block's [CLS] token attention over image patches to produce a coarse
saliency map showing which regions of the face the ViT branch attended
to most when forming its global-context representation.
"""

import cv2
import numpy as np
import torch

import config


def compute_cls_attention_map(model, input_tensor, patch_size=16):
    """
    Runs a forward pass and extracts the [CLS] token's attention over
    patches from the last transformer block (averaged across heads).

    Returns a (H, W) numpy heatmap normalized to [0, 1], resized to the
    input image resolution.
    """
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)

    attn = model.vit_branch.get_last_attention()  # (B, num_heads, N+1, N+1)
    if attn is None:
        return None

    # Average across heads, take CLS token's attention to all patch tokens
    attn = attn.mean(dim=1)          # (B, N+1, N+1)
    cls_attn = attn[0, 0, 1:]        # (N,) - CLS attending to patch tokens

    num_patches = cls_attn.shape[0]
    grid_size = int(num_patches ** 0.5)
    if grid_size * grid_size != num_patches:
        # Non-square patch grid (unexpected backbone); bail out gracefully
        return None

    heatmap = cls_attn.reshape(grid_size, grid_size).cpu().numpy()
    heatmap -= heatmap.min()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    img_size = input_tensor.shape[-1]
    heatmap_resized = cv2.resize(heatmap, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0, 1)
    return heatmap_resized


def overlay_attention(face_rgb_uint8, attn_map, alpha=config.GRADCAM_ALPHA):
    if attn_map is None:
        return face_rgb_uint8  # graceful fallback: show original face

    heatmap = np.uint8(255 * attn_map)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap_color + (1 - alpha) * face_rgb_uint8).astype(np.uint8)
    return overlay


def explain_with_attention(model, input_tensor, face_rgb_uint8):
    """High-level convenience function used by inference.py / app.py."""
    attn_map = compute_cls_attention_map(model, input_tensor)
    overlay = overlay_attention(face_rgb_uint8, attn_map)
    return overlay
