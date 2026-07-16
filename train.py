"""
src/train.py

Trains the Hybrid CNN + ViT deepfake detector on the preprocessed
real/fake face-crop dataset (built via src/preprocessing.py from
FaceForensics++, Celeb-DF, and/or DFDC).

Supports fine-grained pause/resume: a checkpoint is saved every
CHECKPOINT_EVERY_N_BATCHES batches (not just once per epoch), so you can
Ctrl+C at almost any point and lose only a few minutes of progress rather
than a partial epoch (which can be ~2 hours on CPU).

Usage:
    python -m src.train              # starts fresh, or auto-resumes if a
                                      # resume checkpoint already exists
    python -m src.train --no_resume  # forces a fresh run, ignoring any
                                      # existing resume checkpoint
"""

import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from src.dataset import build_dataloaders
from src.models.hybrid_model import build_model
from src.metrics import compute_all_metrics

# Full training-state checkpoint (epoch, batch offset, optimizer,
# scheduler, accumulated metrics, best loss), separate from
# BEST_MODEL_PATH/LAST_MODEL_PATH which store only weights for inference.
RESUME_CHECKPOINT_PATH = os.path.join(config.CHECKPOINT_DIR, "training_state.pt")

# How often (in training batches) to save a resumable checkpoint. Lower =
# less progress lost on interruption, but slightly more disk I/O.
CHECKPOINT_EVERY_N_BATCHES = getattr(config, "CHECKPOINT_EVERY_N_BATCHES", 50)


def set_seed(seed=config.SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_eval_epoch(model, loader, criterion):
    """Used for validation and final test evaluation only (no gradient,
    no mid-loop checkpointing needed since these passes are much shorter
    than a full training epoch)."""
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, leave=False, desc="Evaluating"):
            images = images.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_all_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = avg_loss
    return metrics


def save_checkpoint(epoch, batches_done, model, optimizer, scheduler,
                     best_val_loss, epochs_without_improvement, history,
                     partial_train_state=None):
    """
    Saves everything needed to resume training exactly where it left off:
      - which epoch we're in, and how many training batches of that epoch
        are already done (batches_done)
      - optimizer momentum, LR scheduler state
      - early-stopping bookkeeping and full metric history
      - partial_train_state: accumulated loss/labels/preds/probs for the
        *current, in-progress* epoch, so resuming mid-epoch can pick the
        running totals back up instead of restarting the epoch's metrics.
    """
    torch.save({
        "epoch": epoch,
        "batches_done": batches_done,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history,
        "partial_train_state": partial_train_state,
    }, RESUME_CHECKPOINT_PATH)


def train_one_epoch(model, loader, criterion, optimizer, epoch,
                     scheduler, best_val_loss, epochs_without_improvement,
                     history, start_batch=0, partial_state=None):
    """
    Runs (the remainder of) one training epoch, checkpointing every
    CHECKPOINT_EVERY_N_BATCHES batches. If start_batch > 0, the first
    start_batch batches of this epoch are skipped (they were already
    processed before a prior pause) and accumulated metrics resume from
    partial_state.
    """
    model.train()

    if partial_state is not None:
        total_loss = partial_state["total_loss"]
        all_labels = partial_state["all_labels"]
        all_preds = partial_state["all_preds"]
        all_probs = partial_state["all_probs"]
    else:
        total_loss = 0.0
        all_labels, all_preds, all_probs = [], [], []

    total_batches = len(loader)
    pbar = tqdm(enumerate(loader), total=total_batches, initial=start_batch,
                desc=f"Epoch {epoch} training", leave=False)

    for batch_idx, (images, labels, _) in pbar:
        if batch_idx < start_batch:
            continue  # already processed before a prior pause

        images = images.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1].detach()
        preds = logits.argmax(dim=1).detach()

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

        batches_done = batch_idx + 1
        if batches_done % CHECKPOINT_EVERY_N_BATCHES == 0 or batches_done == total_batches:
            save_checkpoint(
                epoch, batches_done, model, optimizer, scheduler,
                best_val_loss, epochs_without_improvement, history,
                partial_train_state={
                    "total_loss": total_loss,
                    "all_labels": all_labels,
                    "all_preds": all_preds,
                    "all_probs": all_probs,
                },
            )

    n_samples = len(all_labels)
    avg_loss = total_loss / n_samples
    metrics = compute_all_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = avg_loss
    return metrics


def train(resume=True):
    set_seed()
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    print(f"Using device: {config.DEVICE}")
    train_loader, val_loader, test_loader = build_dataloaders()

    model = build_model(pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                   weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    start_epoch = 1
    start_batch = 0
    partial_state = None
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    if resume and os.path.exists(RESUME_CHECKPOINT_PATH):
        print(f"Found existing training checkpoint at {RESUME_CHECKPOINT_PATH} -- resuming.")
        ckpt = torch.load(RESUME_CHECKPOINT_PATH, map_location=config.DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        best_val_loss = ckpt["best_val_loss"]
        epochs_without_improvement = ckpt["epochs_without_improvement"]
        history = ckpt["history"]
        start_epoch = ckpt["epoch"]
        start_batch = ckpt["batches_done"]
        partial_state = ckpt["partial_train_state"]

        if start_batch >= len(train_loader):
            # That epoch's training portion was already fully completed;
            # move on to the next epoch fresh.
            start_epoch += 1
            start_batch = 0
            partial_state = None

        if os.path.exists(config.BEST_MODEL_PATH):
            best_state = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)

        print(f"Resuming at epoch {start_epoch}, batch {start_batch}/{len(train_loader)} "
              f"(best_val_loss so far: {best_val_loss:.4f})")

    if start_epoch > config.NUM_EPOCHS:
        print("Training already completed all configured epochs. "
              "Increase NUM_EPOCHS in config.py to train further, or delete "
              f"{RESUME_CHECKPOINT_PATH} to start over.")
        return model, history, None

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        epoch_partial_state = partial_state if epoch == start_epoch else None

        start = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, scheduler,
            best_val_loss, epochs_without_improvement, history,
            start_batch=epoch_start_batch, partial_state=epoch_partial_state,
        )
        val_metrics = run_eval_epoch(model, val_loader, criterion)
        scheduler.step(val_metrics["loss"])

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:02d}/{config.NUM_EPOCHS} ({elapsed:.1f}s) | "
            f"train_loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
            f"f1={val_metrics['f1_score']:.4f} auc={val_metrics['roc_auc']:.4f}"
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        torch.save(model.state_dict(), config.LAST_MODEL_PATH)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, config.BEST_MODEL_PATH)
            epochs_without_improvement = 0
            print(f"  -> New best model saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1

        # Mark this epoch as fully complete (batches_done = full loader
        # length) so a resume after this point correctly starts the next
        # epoch instead of re-running this one.
        save_checkpoint(epoch, len(train_loader), model, optimizer, scheduler,
                         best_val_loss, epochs_without_improvement, history,
                         partial_train_state=None)

        if epochs_without_improvement >= config.EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

    # Final evaluation on the held-out test set using the best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = run_eval_epoch(model, test_loader, criterion)
    print("\n=== Final Test Set Performance ===")
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall   : {test_metrics['recall']:.4f}")
    print(f"F1-score : {test_metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {test_metrics['roc_auc']:.4f}")
    print(test_metrics["classification_report"])

    return model, history, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_resume", action="store_true",
                         help="Ignore any existing checkpoint and start fresh")
    args = parser.parse_args()
    train(resume=not args.no_resume)