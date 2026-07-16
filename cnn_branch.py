"""
src/models/cnn_branch.py

CNN feature-extraction branch of the hybrid detector. Uses `timm` to load
a pretrained XceptionNet or EfficientNet backbone (ImageNet weights),
stripped of its classification head, so it outputs a pooled feature
vector capturing local spatial artifacts (blending boundaries, texture
inconsistencies, GAN fingerprints, etc.) typical of deepfake generation.
"""

import timm
import torch
import torch.nn as nn

import config


class CNNBranch(nn.Module):
    def __init__(self, backbone_name=config.CNN_BACKBONE, pretrained=True):
        super().__init__()
        # timm model names: "xception", "efficientnet_b0", "efficientnet_b4", ...
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0  # remove classifier head
        )
        self.out_dim = self.backbone.num_features

    def forward(self, x):
        # x: (B, 3, H, W) -> (B, out_dim)
        features = self.backbone(x)
        return features

    def get_target_layer(self):
        """
        Returns the last convolutional layer, used as the target layer for
        Grad-CAM. Layer names differ slightly between backbones, so we
        pick sensible defaults per architecture family.
        """
        name = self.backbone.__class__.__name__.lower()
        if "xception" in name or hasattr(self.backbone, "conv4"):
            return self.backbone.conv4 if hasattr(self.backbone, "conv4") else list(self.backbone.children())[-3]
        if "efficientnet" in name:
            return self.backbone.conv_head
        # generic fallback: last child module with parameters
        for module in reversed(list(self.backbone.modules())):
            if isinstance(module, nn.Conv2d):
                return module
        raise ValueError("Could not locate a convolutional target layer for Grad-CAM")


if __name__ == "__main__":
    model = CNNBranch()
    dummy = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    out = model(dummy)
    print("CNN branch output shape:", out.shape)  # (2, out_dim)
