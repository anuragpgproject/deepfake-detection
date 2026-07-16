"""
src/dfdc_organize.py

DFDC ships videos and a metadata.json (label: REAL/FAKE) together in each
"part" folder, unlike FaceForensics++/Celeb-DF which are pre-split into
real/fake directories. This script reads metadata.json and copies (or
symlinks) each video into data/raw_videos/real/ or data/raw_videos/fake/
so the rest of the pipeline (src/preprocessing.py) works unchanged.

Usage:
    python -m src.dfdc_organize --dfdc_part_dir /path/to/dfdc_train_part_00
    # repeat once per downloaded part folder

Use --symlink instead of copying if you don't want to duplicate disk space.
"""

import os
import json
import shutil
import argparse

import config


def organize_dfdc_part(part_dir, symlink=False):
    metadata_path = os.path.join(part_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"No metadata.json found in {part_dir}. Make sure this is a "
            "DFDC part folder as downloaded from Kaggle."
        )

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    real_count, fake_count, missing = 0, 0, 0

    for filename, info in metadata.items():
        label = info.get("label", "").upper()
        src_path = os.path.join(part_dir, filename)
        if not os.path.exists(src_path):
            missing += 1
            continue

        dest_dir = config.RAW_VIDEO_DIR
        if label == "REAL":
            dest_dir = os.path.join(dest_dir, "real")
            real_count += 1
        elif label == "FAKE":
            dest_dir = os.path.join(dest_dir, "fake")
            fake_count += 1
        else:
            continue

        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)
        if os.path.exists(dest_path):
            continue

        if symlink:
            os.symlink(os.path.abspath(src_path), dest_path)
        else:
            shutil.copy2(src_path, dest_path)

    print(f"[{os.path.basename(part_dir)}] Real: {real_count} | Fake: {fake_count} | Missing files: {missing}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort DFDC videos into real/fake folders using metadata.json")
    parser.add_argument("--dfdc_part_dir", required=True, help="Path to one downloaded DFDC part folder (contains metadata.json)")
    parser.add_argument("--symlink", action="store_true", help="Symlink instead of copying (saves disk space)")
    args = parser.parse_args()

    organize_dfdc_part(args.dfdc_part_dir, symlink=args.symlink)
