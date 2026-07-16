# Hybrid CNN + Vision Transformer Deepfake Detection System

An end-to-end deepfake detection pipeline combining **CNN** (local artifact
detection) and **Vision Transformer** (global context modeling) branches,
with **Grad-CAM** and **attention-map** explainability, and a **Streamlit**
web app for real-time use — built per the project abstract.

## 1. Architecture

```
Input Image/Frame
   │
   ├──► Face Detection & Crop (MTCNN) ──► Resize 224x224, Normalize
   │
   ▼
 ┌────────────────┐      ┌────────────────┐
 │   CNN Branch   │      │   ViT Branch   │
 │ (Xception /    │      │ (ViT-Base/16)  │
 │  EfficientNet) │      │                │
 └───────┬────────┘      └───────┬────────┘
         │  local artifacts       │ global context
         ▼                        ▼
        ┌───────────────────────────┐
        │     Gated Feature Fusion   │
        └─────────────┬─────────────┘
                       ▼
              MLP Classifier Head
                       ▼
            Real / Fake + Confidence
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
     Grad-CAM                 ViT Attention Map
  (CNN explanation)           (ViT explanation)
```

## 2. Project structure

```
deepfake-detection/
├── config.py                    # all paths & hyperparameters
├── requirements.txt
├── app.py                       # Streamlit web app
├── data/
│   ├── raw_videos/              # put downloaded dataset videos here
│   ├── real/                    # extracted real face crops (auto-generated)
│   └── fake/                    # extracted fake face crops (auto-generated)
├── checkpoints/                 # saved model weights
├── outputs/                     # confusion matrix / ROC plots
└── src/
    ├── preprocessing.py         # face extraction, resize, normalize, augment
    ├── dataset.py                # PyTorch Dataset + DataLoader builder
    ├── metrics.py                # Accuracy/Precision/Recall/F1/ROC-AUC/CM
    ├── train.py                  # training loop
    ├── evaluate.py               # standalone evaluation on test split
    ├── inference.py              # single-image / single-video inference API
    ├── models/
    │   ├── cnn_branch.py         # XceptionNet / EfficientNet feature extractor
    │   ├── vit_branch.py         # Vision Transformer feature extractor
    │   └── hybrid_model.py       # gated fusion + classifier head
    └── explainability/
        ├── gradcam.py            # Grad-CAM for CNN branch
        └── attention_viz.py      # attention rollout for ViT branch
```

## 3. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended for training (ViT + CNN
backbones together are compute-heavy). CPU works for inference/demo
purposes but will be slow for training.

## 4. Getting the datasets

This project does **not** bundle any dataset — FaceForensics++, Celeb-DF,
and DFDC all require you to request access / accept license terms
directly from their maintainers:

- **FaceForensics++**: https://github.com/ondyari/FaceForensics
- **Celeb-DF (v2)**: https://github.com/yuezunli/celeb-deepfakeforensics
- **DFDC**: https://ai.meta.com/datasets/dfdc/

Download the raw videos and place them under `data/raw_videos/<real|fake>/`
(or wherever you like — you pass the path explicitly to preprocessing).

## 5. Preprocessing: extract & crop faces

```bash
# Real videos -> data/real/*.jpg
python -m src.preprocessing --input_dir path/to/real_videos --label real --mode video

# Fake videos -> data/fake/*.jpg
python -m src.preprocessing --input_dir path/to/fake_videos --label fake --mode video
```

If a dataset already ships extracted frames/images instead of videos, use
`--mode image` instead.

This samples `FRAMES_PER_VIDEO` frames per video (config.py), detects the
face with MTCNN, crops with a margin, and saves standardized JPGs into
`data/real/` or `data/fake/`.

## 6. Train the model

```bash
python -m src.train
```

This will:
- Build stratified, video-grouped train/val/test splits (to prevent
  frame-leakage across splits from the same source video)
- Train the hybrid CNN+ViT model with early stopping
- Save the best checkpoint to `checkpoints/best_hybrid_model.pt`
- Print final test-set Accuracy, Precision, Recall, F1, ROC-AUC

Key hyperparameters (batch size, learning rate, epochs, backbone choice)
live in `config.py`.

## 7. Evaluate a trained checkpoint

```bash
python -m src.evaluate --checkpoint checkpoints/best_hybrid_model.pt
```

Prints the full metrics suite and saves `outputs/confusion_matrix.png`
and `outputs/roc_curve.png`.

## 8. Run the web app

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).
Upload an image or video; the app shows the predicted label, confidence
score, and Grad-CAM / ViT attention overlays.

> If no checkpoint exists yet, the app still runs using ImageNet-pretrained
> backbones with an **untrained** classifier head, purely so you can verify
> the pipeline end-to-end — predictions won't be meaningful until you train.

## 9. Notes on generalization & extension

- Swap `CNN_BACKBONE` in `config.py` between `"xception"`,
  `"efficientnet_b0"`, `"efficientnet_b4"` to compare local-feature
  extractors.
- Cross-dataset generalization (e.g. train on FF++, test on Celeb-DF) can
  be evaluated by pointing `build_dataloaders()` at different
  `real_dir`/`fake_dir` paths per dataset.
- The gated fusion layer can be swapped for simple concatenation,
  cross-attention fusion, or a learned transformer fusion block if you
  want to experiment further.
- For production/social-media-scale deployment, wrap `src/inference.py`
  behind a REST API (FastAPI) instead of / in addition to the Streamlit
  demo.

## 10. Metrics reported

Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix are
all computed in `src/metrics.py` and used by both `src/train.py` (per
epoch, on the validation set) and `src/evaluate.py` (final test-set
report with saved plots).
