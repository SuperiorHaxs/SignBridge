#!/usr/bin/env python3
"""
Split pose files into train/val/test sets using conservative approach.

This script:
1. Reads pose files from pose_files/ directory
2. Groups by gloss (class)
3. Splits conservatively to prevent data leakage
4. Creates train/val/test subdirectories with class folders
5. Copies .pose files to appropriate splits

Output: pose_split_20_class/
"""

import os
import sys
import shutil
import json
from pathlib import Path
from collections import defaultdict

# Add project root to path for config import
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import configuration
from config import get_config

# ============================================================================
# CONFIGURATION - Now managed by config system
# ============================================================================
config = get_config()

WLASL_BASE_DIR = str(config.dataset_root)
POSE_FILES_DIR = str(config.pose_files_dir)
METADATA_FILE = str(config.video_to_gloss_mapping)
OUTPUT_BASE_DIR = os.path.join(WLASL_BASE_DIR, "pose_split_20_class")

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 20 classes (same as pickle dataset)
TARGET_CLASSES = [
    'accident', 'apple', 'bath', 'before', 'blue', 'chair', 'clothes',
    'cousin', 'deaf', 'doctor', 'eat', 'enjoy', 'forget', 'give', 'go',
    'graduation', 'halloween', 'help', 'hot', 'hurry'
]


def load_metadata():
    """Load WLASL metadata to map video IDs to glosses."""
    with open(METADATA_FILE, 'r') as f:
        data = json.load(f)

    # Create video_id -> gloss mapping
    # video_to_gloss_mapping.json format: {"video_id": {"gloss": "...", ...}, ...}
    video_to_gloss = {}
    for video_id, info in data.items():
        gloss = info['gloss']
        if gloss in TARGET_CLASSES:
            video_to_gloss[video_id] = gloss

    return video_to_gloss


def get_pose_files_by_class(video_to_gloss):
    """Group pose files by class."""
    files_by_class = defaultdict(list)

    pose_files = sorted([f for f in os.listdir(POSE_FILES_DIR) if f.endswith('.pose')])

    for pose_file in pose_files:
        # Extract video ID from filename (e.g., "00623.pose" -> "00623")
        video_id = pose_file.replace('.pose', '')

        if video_id in video_to_gloss:
            gloss = video_to_gloss[video_id]
            files_by_class[gloss].append(pose_file)

    return files_by_class


def split_files(files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split files into train/val/test."""
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        'train': files[:n_train],
        'val': files[n_train:n_train + n_val],
        'test': files[n_train + n_val:]
    }


def create_split_directories():
    """Create output directory structure."""
    for split in ['train', 'val', 'test']:
        for class_name in TARGET_CLASSES:
            split_dir = os.path.join(OUTPUT_BASE_DIR, split, class_name)
            os.makedirs(split_dir, exist_ok=True)


def copy_files_to_splits(files_by_class):
    """Copy pose files to appropriate split directories."""
    stats = {'train': 0, 'val': 0, 'test': 0}

    for gloss, files in files_by_class.items():
        print(f"\n{gloss}: {len(files)} files")

        # Split files
        splits = split_files(files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

        # Copy to appropriate directories
        for split_name, files_in_split in splits.items():
            print(f"  {split_name}: {len(files_in_split)} files")

            for pose_file in files_in_split:
                src_path = os.path.join(POSE_FILES_DIR, pose_file)
                dst_path = os.path.join(OUTPUT_BASE_DIR, split_name, gloss, pose_file)
                shutil.copy2(src_path, dst_path)
                stats[split_name] += 1

    return stats


def main():
    """Main splitting logic."""
    print("=" * 70)
    print("SPLITTING POSE FILES INTO TRAIN/VAL/TEST")
    print("=" * 70)
    print()
    print(f"Input directory:  {POSE_FILES_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"Target classes: {len(TARGET_CLASSES)}")
    print()

    # Load metadata
    print("Loading metadata...")
    video_to_gloss = load_metadata()
    print(f"Found {len(video_to_gloss)} video-to-gloss mappings")
    print()

    # Group files by class
    print("Grouping pose files by class...")
    files_by_class = get_pose_files_by_class(video_to_gloss)

    print(f"Found {len(files_by_class)} classes with pose files:")
    total_files = sum(len(files) for files in files_by_class.values())
    print(f"Total pose files: {total_files}")
    print()

    # Create directory structure
    print("Creating split directories...")
    create_split_directories()
    print("Directories created")
    print()

    # Copy files
    print("Copying files to splits...")
    stats = copy_files_to_splits(files_by_class)

    print()
    print("=" * 70)
    print("SPLIT COMPLETE!")
    print("=" * 70)
    print()
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print()
    print("Split statistics:")
    print(f"  Train: {stats['train']} files ({stats['train']/total_files*100:.1f}%)")
    print(f"  Val:   {stats['val']} files ({stats['val']/total_files*100:.1f}%)")
    print(f"  Test:  {stats['test']} files ({stats['test']/total_files*100:.1f}%)")
    print()


if __name__ == "__main__":
    main()
