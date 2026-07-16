"""
src/models/hybrid_model.py

Combines the CNN branch (local artifact detector) and the ViT branch
(global context modeler) via feature-level fusion, followed by a small
MLP classification head that outputs real/fake logits.

Fusion strategy: concatenate pooled CNN + ViT feature vectors, pass
through a gated fusion layer (learns how much to weight each branch per
sample) before the final classifier. This tends to generalize better
across datasets than naive concatenation alone, since the network can
down-weight a branch that is less informative for a given input.
"""

import torch
import torch.nn as nn

import config
from src.models.cnn_branch import CNNBranch
from src.models.vit_branch import ViTBranch


class GatedFusion(nn.Module):
    """Learns a per-sample scalar gate in [0,1] for each branch before
    concatenation, so the classifier can lean on whichever branch is more
    informative for a given image."""

    def __init__(self, cnn_dim, vit_dim):
        super().__init__()
        self.gate_cnn = nn.Sequential(nn.Linear(cnn_dim + vit_dim, cnn_dim), nn.Sigmoid())
        self.gate_vit = nn.Sequential(nn.Linear(cnn_dim + vit_dim, vit_dim), nn.Sigmoid())

    def forward(self, cnn_feat, vit_feat):
        combined = torch.cat([cnn_feat, vit_feat], dim=1)
        g_cnn = self.gate_cnn(combined)
        g_vit = self.gate_vit(combined)
        fused = torch.cat([cnn_feat * g_cnn, vit_feat * g_vit], dim=1)
        return fused


class HybridDeepfakeDetector(nn.Module):
    def __init__(self,
                 cnn_backbone=config.CNN_BACKBONE,
                 vit_backbone=config.VIT_BACKBONE,
                 hidden_dim=config.FUSION_HIDDEN_DIM,
                 num_classes=config.NUM_CLASSES,
                 dropout=config.DROPOUT,
                 pretrained=True):
        super().__init__()

        self.cnn_branch = CNNBranch(cnn_backbone, pretrained=pretrained)
        self.vit_branch = ViTBranch(vit_backbone, pretrained=pretrained)

        self.fusion = GatedFusion(self.cnn_branch.out_dim, self.vit_branch.out_dim)
        fused_dim = self.cnn_branch.out_dim + self.vit_branch.out_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, return_features=False):
        cnn_feat = self.cnn_branch(x)
        vit_feat = self.vit_branch(x)
        fused = self.fusion(cnn_feat, vit_feat)
        logits = self.classifier(fused)

        if return_features:
            return logits, {"cnn_feat": cnn_feat, "vit_feat": vit_feat}
        return logits

    @torch.no_grad()
    def predict(self, x):
        """Convenience method used by inference.py: returns (pred_class,
        confidence) for a batch."""
        self.eval()
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        return pred, conf, probs


def build_model(pretrained=True):
    model = HybridDeepfakeDetector(pretrained=pretrained)
    return model.to(config.DEVICE)


if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(config.DEVICE)
    logits = model(dummy)
    print("Hybrid model output shape:", logits.shape)
    pred, conf, probs = model.predict(dummy)
    print("Predictions:", pred, "Confidence:", conf)
