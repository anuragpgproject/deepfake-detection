"""
src/explainability/gradcam.py

Grad-CAM for the CNN branch of the hybrid model. Highlights the spatial
regions of the input image that most influenced the CNN branch's
contribution to the final real/fake decision, e.g. blending boundaries
around the jaw, unnatural skin texture, or warped background regions.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import config


class GradCAM:
    def __init__(self, model, target_layer):
        """
        model: the full HybridDeepfakeDetector (Grad-CAM is computed with
               respect to the final fused logits, but the CAM itself is
               localized using the CNN branch's last conv layer).
        target_layer: an nn.Module (e.g. model.cnn_branch.get_target_layer())
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: (1, 3, H, W) on the correct device.
        class_idx: which class's logit to backprop from (defaults to the
                   predicted class, i.e. the model's own decision).
        Returns a (H, W) heatmap normalized to [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        # Global-average-pool the gradients to get per-channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, class_idx


def overlay_heatmap(face_rgb_uint8, cam, alpha=config.GRADCAM_ALPHA):
    """
    face_rgb_uint8: (H, W, 3) uint8 RGB image (the cropped face).
    cam: (H, W) float heatmap in [0, 1].
    Returns an RGB uint8 image with the heatmap overlaid.
    """
    heatmap = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap_color + (1 - alpha) * face_rgb_uint8).astype(np.uint8)
    return overlay


def explain_with_gradcam(model, input_tensor, face_rgb_uint8):
    """High-level convenience function used by inference.py / app.py."""
    target_layer = model.cnn_branch.get_target_layer()
    cam_engine = GradCAM(model, target_layer)
    cam, class_idx = cam_engine.generate(input_tensor)
    overlay = overlay_heatmap(face_rgb_uint8, cam)
    return overlay, class_idx
