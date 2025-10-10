#!/usr/bin/env python3
"""
verify_split_integrity.py

Verify that the properly split dataset has NO data leakage.
Check that no video IDs appear in multiple splits.
"""

import os
import pickle
from pathlib import Path
from collections import defaultdict

def extract_video_ids_from_split(split_dir):
    """Extract all video IDs from a split directory."""
    video_ids = set()

    for class_dir in Path(split_dir).iterdir():
        if class_dir.is_dir():
            for pkl_file in class_dir.glob("*.pkl"):
                try:
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)

                    video_id = data.get('video_id', pkl_file.stem.replace('aug_00_', '').replace('aug_01_', '').replace('aug_02_', '').replace('original_', ''))
                    video_ids.add(video_id)
                except:
                    continue

    return video_ids

def verify_dataset_integrity(dataset_dir):
    """Verify that there's no video ID overlap between splits."""

    print("VERIFYING DATASET INTEGRITY")
    print("=" * 50)
    print(f"DATASET: {dataset_dir}")
    print()

    # Extract video IDs from each split
    train_ids = extract_video_ids_from_split(Path(dataset_dir) / "train")
    val_ids = extract_video_ids_from_split(Path(dataset_dir) / "val")
    test_ids = extract_video_ids_from_split(Path(dataset_dir) / "test")

    print(f"TRAIN SPLIT: {len(train_ids)} unique video IDs")
    print(f"VAL SPLIT: {len(val_ids)} unique video IDs")
    print(f"TEST SPLIT: {len(test_ids)} unique video IDs")
    print(f"TOTAL: {len(train_ids) + len(val_ids) + len(test_ids)} video IDs")
    print()

    # Check for overlaps
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    print("OVERLAP ANALYSIS:")
    print(f"TRAIN & VAL: {len(train_val_overlap)} overlapping IDs")
    if train_val_overlap:
        print(f"  Overlapping IDs: {sorted(list(train_val_overlap))}")

    print(f"TRAIN & TEST: {len(train_test_overlap)} overlapping IDs")
    if train_test_overlap:
        print(f"  Overlapping IDs: {sorted(list(train_test_overlap))}")

    print(f"VAL & TEST: {len(val_test_overlap)} overlapping IDs")
    if val_test_overlap:
        print(f"  Overlapping IDs: {sorted(list(val_test_overlap))}")
    print()

    # Verify integrity
    if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
        print("INTEGRITY CHECK PASSED!")
        print("No video ID overlap detected - dataset is properly split!")
        print("Ready for training with NO data leakage!")
        return True
    else:
        print("INTEGRITY CHECK FAILED!")
        print("Video ID overlaps detected - data leakage present!")
        return False

if __name__ == "__main__":
    dataset_dir = "C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/wlasl_poses_complete/properly_split_20_class"
    verify_dataset_integrity(dataset_dir)