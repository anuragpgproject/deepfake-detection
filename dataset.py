"""
src/dataset.py

PyTorch Dataset for the preprocessed real/fake face-crop folders produced
by src/preprocessing.py. Also provides a helper to build stratified
train/val/test DataLoaders.
"""

import os
import glob
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config
from src.preprocessing import get_train_transforms, get_eval_transforms


class DeepfakeFaceDataset(Dataset):
    """
    Expects two folders of already-cropped face images:
        real_dir/*.jpg  -> label 0
        fake_dir/*.jpg  -> label 1
    """

    def __init__(self, real_dir, fake_dir, file_list=None, transform=None):
        self.transform = transform

        if file_list is not None:
            self.samples = file_list
        else:
            real_files = sorted(glob.glob(os.path.join(real_dir, "*.jpg")) +
                                 glob.glob(os.path.join(real_dir, "*.png")))
            fake_files = sorted(glob.glob(os.path.join(fake_dir, "*.jpg")) +
                                 glob.glob(os.path.join(fake_dir, "*.png")))
            self.samples = ([(f, 0) for f in real_files] +
                             [(f, 1) for f in fake_files])

        if len(self.samples) == 0:
            raise RuntimeError(
                "No images found. Run src/preprocessing.py first to "
                "populate data/real and data/fake with face crops."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, torch.tensor(label, dtype=torch.long), path


def build_dataloaders(real_dir=config.REAL_FRAMES_DIR,
                       fake_dir=config.FAKE_FRAMES_DIR,
                       batch_size=config.BATCH_SIZE,
                       val_split=config.VAL_SPLIT,
                       test_split=config.TEST_SPLIT,
                       seed=config.SEED,
                       num_workers=config.NUM_WORKERS,
                       max_samples_per_class=getattr(config, "MAX_SAMPLES_PER_CLASS", None)):
    """
    Splits the available data at the *video level* when possible (grouping
    frames from the same source video together) to avoid leakage between
    train/val/test, then returns three DataLoaders.

    If max_samples_per_class is set, randomly selects whole videos (not
    individual frames, to preserve the leakage-prevention grouping) per
    class until reaching roughly that many frames, before splitting. This
    keeps CPU training time bounded and balances the two classes, since
    Celeb-DF's fake videos outnumber real videos by ~6x.
    """
    random.seed(seed)

    real_files = sorted(glob.glob(os.path.join(real_dir, "*.jpg")) +
                         glob.glob(os.path.join(real_dir, "*.png")))
    fake_files = sorted(glob.glob(os.path.join(fake_dir, "*.jpg")) +
                         glob.glob(os.path.join(fake_dir, "*.png")))

    def group_by_video(files):
        groups = {}
        for f in files:
            base = os.path.basename(f)
            video_id = base.rsplit("_", 1)[0]  # strip trailing _<frame_idx>
            groups.setdefault(video_id, []).append(f)
        return list(groups.values())

    def cap_groups(groups, max_samples):
        """Randomly shuffle video groups and keep taking whole videos
        until the cumulative frame count reaches max_samples."""
        if max_samples is None:
            return groups
        groups = groups[:]  # copy before shuffling
        random.shuffle(groups)
        capped, total = [], 0
        for g in groups:
            if total >= max_samples:
                break
            capped.append(g)
            total += len(g)
        return capped

    def split_groups(groups, label):
        random.shuffle(groups)
        n = len(groups)
        n_test = max(1, int(n * test_split)) if n > 2 else 0
        n_val = max(1, int(n * val_split)) if n > 2 else 0
        test_groups = groups[:n_test]
        val_groups = groups[n_test:n_test + n_val]
        train_groups = groups[n_test + n_val:]

        def flatten(gs):
            return [(f, label) for g in gs for f in g]

        return flatten(train_groups), flatten(val_groups), flatten(test_groups)

    real_groups = group_by_video(real_files)
    fake_groups = group_by_video(fake_files)

    real_groups = cap_groups(real_groups, max_samples_per_class)
    fake_groups = cap_groups(fake_groups, max_samples_per_class)

    train_r, val_r, test_r = split_groups(real_groups, 0)
    train_f, val_f, test_f = split_groups(fake_groups, 1)

    train_samples = train_r + train_f
    val_samples = val_r + val_f
    test_samples = test_r + test_f
    random.shuffle(train_samples)

    train_ds = DeepfakeFaceDataset(None, None, file_list=train_samples,
                                    transform=get_train_transforms())
    val_ds = DeepfakeFaceDataset(None, None, file_list=val_samples,
                                  transform=get_eval_transforms())
    test_ds = DeepfakeFaceDataset(None, None, file_list=test_samples,
                                   transform=get_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} images")
    return train_loader, val_loader, test_loader