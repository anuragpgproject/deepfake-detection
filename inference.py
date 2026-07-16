"""
src/inference.py

High-level inference API used by the Streamlit app: takes a raw image
(or a single video frame) and returns:
  - predicted class (Real / Fake)
  - confidence score
  - Grad-CAM overlay (CNN branch explanation)
  - Attention overlay (ViT branch explanation)

Also supports full-video inference by sampling frames, predicting on
each, and aggregating (majority vote + averaged confidence).
"""

import os
import cv2
import numpy as np
import torch

import config
from src.preprocessing import preprocess_single_image, extract_frames_from_video
from src.models.hybrid_model import build_model
from src.explainability.gradcam import explain_with_gradcam
from src.explainability.attention_viz import explain_with_attention


_model_cache = {}


def load_model(checkpoint_path=config.BEST_MODEL_PATH):
    """Loads (and caches) the trained hybrid model for repeated inference
    calls, e.g. across multiple Streamlit interactions."""
    if checkpoint_path in _model_cache:
        return _model_cache[checkpoint_path]

    model = build_model(pretrained=not os.path.exists(checkpoint_path))
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=config.DEVICE)
        model.load_state_dict(state)
        print(f"Loaded trained weights from {checkpoint_path}")
    else:
        print(
            f"WARNING: no checkpoint found at {checkpoint_path}. "
            "Using ImageNet-pretrained backbones with an untrained "
            "classifier head -- predictions will not be meaningful until "
            "you run src/train.py."
        )
    model.eval()
    _model_cache[checkpoint_path] = model
    return model


def predict_image(image_bgr, model=None, checkpoint_path=config.BEST_MODEL_PATH):
    """
    Runs the full pipeline on a single image (BGR, as read by cv2 or a
    Streamlit file upload converted to a numpy array).

    Returns a dict with: label, confidence, probabilities, gradcam_overlay,
    attention_overlay, face_crop.
    """
    if model is None:
        model = load_model(checkpoint_path)

    input_tensor, face_rgb = preprocess_single_image(image_bgr, for_display=True)
    input_tensor = input_tensor.to(config.DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])

    gradcam_overlay, _ = explain_with_gradcam(model, input_tensor, face_rgb)
    attention_overlay = explain_with_attention(model, input_tensor, face_rgb)

    return {
        "label": config.CLASS_NAMES[pred_class],
        "label_idx": pred_class,
        "confidence": round(confidence * 100, config.CONFIDENCE_DECIMALS),
        "probabilities": {
            "Real": round(float(probs[0]) * 100, config.CONFIDENCE_DECIMALS),
            "Fake": round(float(probs[1]) * 100, config.CONFIDENCE_DECIMALS),
        },
        "face_crop": face_rgb,
        "gradcam_overlay": gradcam_overlay,
        "attention_overlay": attention_overlay,
    }


def predict_video(video_path, model=None, checkpoint_path=config.BEST_MODEL_PATH,
                   num_frames=10):
    """
    Samples `num_frames` frames from the video, runs predict_image on each,
    and aggregates results:
      - final label = majority vote across frames
      - confidence  = mean confidence of frames agreeing with the final label
      - representative Grad-CAM / attention overlays taken from the frame
        with the highest confidence (most illustrative example)
    """
    if model is None:
        model = load_model(checkpoint_path)

    frames = extract_frames_from_video(video_path, num_frames=num_frames)
    if not frames:
        raise ValueError("Could not read any frames from the video.")

    frame_results = [predict_image(f, model=model) for f in frames]

    fake_votes = sum(1 for r in frame_results if r["label_idx"] == 1)
    real_votes = len(frame_results) - fake_votes
    final_label_idx = 1 if fake_votes >= real_votes else 0
    final_label = config.CLASS_NAMES[final_label_idx]

    agreeing = [r for r in frame_results if r["label_idx"] == final_label_idx]
    avg_confidence = round(np.mean([r["confidence"] for r in agreeing]), config.CONFIDENCE_DECIMALS)

    best_frame = max(agreeing, key=lambda r: r["confidence"])

    return {
        "label": final_label,
        "label_idx": final_label_idx,
        "confidence": avg_confidence,
        "num_frames_analyzed": len(frame_results),
        "fake_frame_votes": fake_votes,
        "real_frame_votes": real_votes,
        "representative_frame": best_frame,
        "per_frame_results": frame_results,
    }
