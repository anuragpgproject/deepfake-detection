"""
config.py
Central configuration for the Hybrid CNN + Vision Transformer Deepfake
Detection System.

Every other module (preprocessing, dataset, models, training, inference,
Streamlit app) imports its settings from here, so this is the single place
to change paths, hyperparameters, or architecture choices.
"""

import os
import torch

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_VIDEO_DIR = os.path.join(DATA_DIR, "raw_videos")      # place FF++/Celeb-DF/DFDC videos here
REAL_FRAMES_DIR = os.path.join(DATA_DIR, "real")          # extracted real face crops
FAKE_FRAMES_DIR = os.path.join(DATA_DIR, "fake")          # extracted fake face crops

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_hybrid_model.pt")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_hybrid_model.pt")

for _d in [DATA_DIR, RAW_VIDEO_DIR, REAL_FRAMES_DIR, FAKE_FRAMES_DIR,
           CHECKPOINT_DIR, OUTPUT_DIR]:
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------
# Datasets supported (folder-based, binary: real / fake)
#   Expected layout after preprocessing:
#     data/real/<video_id>_<frame_idx>.jpg
#     data/fake/<video_id>_<frame_idx>.jpg
#   Works directly with FaceForensics++, Celeb-DF v2, and DFDC once you run
#   src/preprocessing.py on the raw videos of each dataset.
# --------------------------------------------------------------------------
SUPPORTED_DATASETS = ["FaceForensics++", "Celeb-DF", "DFDC"]

# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
IMAGE_SIZE = 224                 # standard input resolution for both CNN & ViT branches
FRAMES_PER_VIDEO = 30            # how many frames to sample per video during extraction
FACE_MARGIN = 0.3                # extra margin around detected face bounding box
MTCNN_THRESHOLDS = [0.6, 0.7, 0.7]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --------------------------------------------------------------------------
# Model architecture
# --------------------------------------------------------------------------
CNN_BACKBONE = "efficientnet_b0"        # options: "xception", "efficientnet_b0", "efficientnet_b4"
VIT_BACKBONE = "vit_small_patch16_224"  # timm model name
FUSION_HIDDEN_DIM = 512
DROPOUT = 0.3
NUM_CLASSES = 2                  # 0 = real, 1 = fake

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
NUM_WORKERS = 0
EARLY_STOP_PATIENCE = 4
SEED = 42
MAX_SAMPLES_PER_CLASS = 15000   # balanced subset cap, for CPU-feasible training time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")

# --------------------------------------------------------------------------
# Inference / App
# --------------------------------------------------------------------------
CONFIDENCE_DECIMALS = 2
GRADCAM_ALPHA = 0.5               # heatmap overlay opacity
CLASS_NAMES = {0: "Real", 1: "Fake"}