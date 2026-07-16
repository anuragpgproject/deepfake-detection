"""
src/evaluate.py

Standalone evaluation of a trained checkpoint against the test split,
producing the full metrics suite (Accuracy, Precision, Recall, F1,
ROC-AUC, Confusion Matrix) plus saved plots.

Usage:
    python -m src.evaluate --checkpoint checkpoints/best_hybrid_model.pt
"""

import argparse
import torch
import torch.nn as nn

import config
from src.dataset import build_dataloaders
from src.models.hybrid_model import build_model
from src.metrics import compute_all_metrics, plot_confusion_matrix, plot_roc_curve
from src.train import run_eval_epoch


def evaluate(checkpoint_path=config.BEST_MODEL_PATH):
    model = build_model(pretrained=False)
    state = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(state)
    model.eval()

    _, _, test_loader = build_dataloaders()
    criterion = nn.CrossEntropyLoss()

    metrics = run_eval_epoch(model, test_loader, criterion)

    print("=== Evaluation Results ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])
    print("\nClassification Report:")
    print(metrics["classification_report"])

    # Collect predictions again to plot (run_eval_epoch only returns aggregate metrics)
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(config.DEVICE)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    cm_path = plot_confusion_matrix(all_labels, all_preds)
    roc_path = plot_roc_curve(all_labels, all_probs)
    print(f"\nSaved confusion matrix to {cm_path}")
    if roc_path:
        print(f"Saved ROC curve to {roc_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=config.BEST_MODEL_PATH)
    args = parser.parse_args()
    evaluate(args.checkpoint)