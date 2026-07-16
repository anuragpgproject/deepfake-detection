"""
src/models/vit_branch.py

Vision Transformer feature-extraction branch. A ViT operates on patch
embeddings with self-attention across the whole image, letting it model
long-range / global contextual relationships (e.g. inconsistent lighting
across the face, mismatched head pose vs. background, asymmetries) that
a convolution's local receptive field can miss.
"""

import timm
import torch
import torch.nn as nn

import config


class ViTBranch(nn.Module):
    def __init__(self, backbone_name=config.VIT_BACKBONE, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        self.out_dim = self.backbone.num_features
        self._attn_weights = None
        self._register_attention_hook()

    def _register_attention_hook(self):
        """
        Hooks the attention softmax of the last transformer block so we can
        retrieve attention maps later for explainability (attention
        rollout / last-layer attention visualization).
        """
        try:
            last_block = self.backbone.blocks[-1]
            last_block.attn.fused_attn = False  # force eager attention so softmax is computed explicitly (timm>=0.9)

            def hook(module, input, output):
                # timm's Attention module recomputes softmax internally;
                # we capture it via a forward hook on the attn_drop layer,
                # which receives the softmax attention probabilities.
                self._attn_weights = output.detach()

            last_block.attn.attn_drop.register_forward_hook(hook)
        except AttributeError:
            # Backbone variant without the expected structure; attention
            # visualization will simply be unavailable for this backbone.
            pass

    def forward(self, x):
        features = self.backbone(x)
        return features

    def get_last_attention(self):
        """Returns the most recently captured attention tensor, shape
        (B, num_heads, N+1, N+1), or None if unavailable."""
        return self._attn_weights


if __name__ == "__main__":
    model = ViTBranch()
    dummy = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    out = model(dummy)
    print("ViT branch output shape:", out.shape)
    attn = model.get_last_attention()
    print("Attention shape:", None if attn is None else attn.shape)
