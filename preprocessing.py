"""
src/preprocessing.py

Handles all data preparation for the deepfake detection pipeline:
  1. Frame extraction from raw video files (FaceForensics++, Celeb-DF, DFDC).
  2. Face detection & cropping using MTCNN (facenet-pytorch).
  3. Resizing to a standard resolution and normalization.
  4. Train-time augmentation (albumentations) to improve robustness.

Usage (build a frame dataset from raw videos):
    python -m src.preprocessing --input_dir data/raw_videos/real --label real
    python -m src.preprocessing --input_dir data/raw_videos/fake --label fake
"""

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

import torch
from facenet_pytorch import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


# --------------------------------------------------------------------------
# Face detector (singleton) - MTCNN is a light, fast, well-validated
# face detector, appropriate for both images and video frames.
# --------------------------------------------------------------------------
_mtcnn = None


def get_face_detector():
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = MTCNN(
            image_size=config.IMAGE_SIZE,
            margin=int(config.FACE_MARGIN * config.IMAGE_SIZE),
            thresholds=config.MTCNN_THRESHOLDS,
            keep_all=False,
            post_process=False,
            device=config.DEVICE,
        )
    return _mtcnn


def extract_face(image_bgr):
    """
    Detect and crop the largest face in a BGR image (as read by cv2).
    Returns an RGB uint8 numpy array of shape (IMAGE_SIZE, IMAGE_SIZE, 3),
    or None if no face was found.
    """
    detector = get_face_detector()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    face_tensor = detector(image_rgb)  # returns a torch tensor (3,H,W) or None
    if face_tensor is None:
        return None
    face_np = face_tensor.permute(1, 2, 0).byte().cpu().numpy()
    return face_np


def extract_frames_from_video(video_path, num_frames=config.FRAMES_PER_VIDEO):
    """Uniformly sample `num_frames` frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int)
    frames = []
    idx_set = set(indices.tolist())
    current = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if current in idx_set:
            frames.append(frame)
        current += 1
        if current > max(indices):
            break
    cap.release()
    return frames


def process_video_folder(input_dir, output_dir, label_name, limit=None):
    """
    Walk `input_dir` for video files, extract frames, detect faces, and
    save face crops as jpg files into `output_dir`.

    limit: if set, only process the first `limit` videos found. Useful for
    a quick smoke test before committing to a full multi-hour extraction
    run over thousands of videos.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_exts = (".mp4", ".avi", ".mov", ".mkv")
    videos = [f for f in os.listdir(input_dir) if f.lower().endswith(video_exts)]
    if limit is not None:
        videos = videos[:limit]

    print(f"[{label_name}] Found {len(videos)} videos in {input_dir}" +
          (f" (limited to first {limit})" if limit else ""))
    saved = 0
    for vid_name in tqdm(videos, desc=f"Extracting {label_name} faces"):
        vid_path = os.path.join(input_dir, vid_name)
        frames = extract_frames_from_video(vid_path)
        vid_id = os.path.splitext(vid_name)[0]
        for i, frame in enumerate(frames):
            face = extract_face(frame)
            if face is None:
                continue
            out_path = os.path.join(output_dir, f"{vid_id}_{i:03d}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            saved += 1
    print(f"[{label_name}] Saved {saved} face crops to {output_dir}")


def process_image_folder(input_dir, output_dir, label_name, limit=None):
    """Alternative entry point if you already have individual images
    (e.g. DFDC frames already extracted) instead of raw videos."""
    os.makedirs(output_dir, exist_ok=True)
    img_exts = (".jpg", ".jpeg", ".png")
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(img_exts)]
    if limit is not None:
        images = images[:limit]

    saved = 0
    for img_name in tqdm(images, desc=f"Detecting faces in {label_name} images"):
        img_path = os.path.join(input_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        face = extract_face(frame)
        if face is None:
            continue
        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        saved += 1
    print(f"[{label_name}] Saved {saved} face crops to {output_dir}")


# --------------------------------------------------------------------------
# Transforms: normalization (always applied) + augmentation (train only)
# --------------------------------------------------------------------------
def get_train_transforms():
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.2),
        A.GaussNoise(p=0.2),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        A.CoarseDropout(max_holes=1, max_height=32, max_width=32, p=0.2),
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transforms():
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])


def preprocess_single_image(image_bgr, for_display=False):
    """
    Full pipeline for a single incoming image (used by inference.py / the
    Streamlit app): detect face -> resize -> normalize -> tensor.

    Returns (tensor, face_rgb_uint8) where tensor is ready for the model
    and face_rgb_uint8 is the cropped face for display / explainability
    overlays. If no face is detected, falls back to using the whole image.
    """
    face = extract_face(image_bgr)
    if face is None:
        # Fallback: no face detected, use the resized whole image so the
        # app can still give the user a result rather than a hard failure.
        face = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (config.IMAGE_SIZE, config.IMAGE_SIZE))

    transform = get_eval_transforms()
    tensor = transform(image=face)["image"].unsqueeze(0)  # (1,3,H,W)

    if for_display:
        return tensor, face
    return tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract & crop faces for deepfake dataset preparation")
    parser.add_argument("--input_dir", required=True, help="Folder with raw videos or images")
    parser.add_argument("--label", required=True, choices=["real", "fake"])
    parser.add_argument("--mode", default="video", choices=["video", "image"])
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N videos/images (smoke test before a full run)")
    args = parser.parse_args()

    out_dir = config.REAL_FRAMES_DIR if args.label == "real" else config.FAKE_FRAMES_DIR
    if args.mode == "video":
        process_video_folder(args.input_dir, out_dir, args.label, limit=args.limit)
    else:
        process_image_folder(args.input_dir, out_dir, args.label, limit=args.limit)