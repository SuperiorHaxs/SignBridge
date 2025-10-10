#!/usr/bin/env python3
"""
Split pose files into train/val/test sets for N classes.

This script:
1. Reads pose files from pose_files/ directory
2. Groups by gloss (class)
3. Splits conservatively to prevent data leakage
4. Creates train/val/test subdirectories with class folders
5. Copies .pose files to appropriate splits

Usage:
    python split_pose_files_50class.py --num-classes 50
    python split_pose_files_50class.py --num-classes 100 --existing-classes-path <path_to_50_class_train>

Output: dataset_splits/{N}_classes/original/pose_split_{N}_class/
"""

import os
import sys
import shutil
import json
import argparse
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

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Base class list (prioritized by frequency and utility)
BASE_CLASS_ORDER = [
    # Top 20 (existing from 20-class set)
    'accident', 'apple', 'bath', 'before', 'blue', 'chair', 'clothes',
    'cousin', 'deaf', 'doctor', 'eat', 'enjoy', 'forget', 'give', 'go',
    'graduation', 'halloween', 'help', 'hot', 'hurry',

    # Next 30 (for 50-class)
    'book', 'drink', 'computer', 'who', 'candy', 'walk', 'thin', 'no',
    'fine', 'year', 'yes', 'table', 'now', 'what', 'finish', 'black',
    'thanksgiving', 'all', 'many', 'like', 'cool', 'orange', 'mother',
    'woman', 'dog', 'hearing', 'tall', 'wrong', 'kiss', 'man',

    # Next 50 (for 100-class) - add more as needed
    'family', 'graduate', 'bed', 'language', 'fish', 'hat', 'bowling',
    'shirt', 'later', 'white', 'study', 'can', 'bird', 'pink', 'want',
    'time', 'dance', 'play', 'color', 'summer', 'winter', 'spring',
    'fall', 'school', 'work', 'home', 'car', 'train', 'airplane',
    'bus', 'bicycle', 'run', 'jump', 'sit', 'stand', 'sleep', 'wake',
    'morning', 'afternoon', 'evening', 'night', 'day', 'week', 'month',
    'yesterday', 'today', 'tomorrow', 'happy', 'sad', 'angry', 'tired'
]


def get_existing_classes(existing_path):
    """Get list of existing classes from a previous split."""
    if not existing_path or not os.path.exists(existing_path):
        return []

    existing_classes = []
    for item in os.listdir(existing_path):
        if os.path.isdir(os.path.join(existing_path, item)):
            existing_classes.append(item)

    return sorted(existing_classes)


def get_top_n_classes(num_classes, existing_classes=None):
    """Get top N classes, prioritizing existing classes."""
    if existing_classes:
        # Start with existing classes
        target_classes = list(existing_classes)
        num_existing = len(existing_classes)

        # Add new classes from BASE_CLASS_ORDER until we reach num_classes
        for gloss in BASE_CLASS_ORDER:
            if gloss not in target_classes:
                target_classes.append(gloss)
                if len(target_classes) >= num_classes:
                    break

        print(f"Building on {num_existing} existing classes, adding {num_classes - num_existing} new classes")
    else:
        # No existing classes, just take first N from BASE_CLASS_ORDER
        target_classes = BASE_CLASS_ORDER[:num_classes]

    return target_classes[:num_classes]


def load_metadata(target_classes):
    """Load WLASL metadata to map video IDs to glosses."""
    with open(METADATA_FILE, 'r') as f:
        data = json.load(f)

    # Create video_id -> gloss mapping
    # video_to_gloss_mapping.json format: {"video_id": {"gloss": "...", ...}, ...}
    video_to_gloss = {}
    for video_id, info in data.items():
        gloss = info['gloss']
        if gloss in target_classes:
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


def create_split_directories(output_dir, target_classes):
    """Create output directory structure."""
    for split in ['train', 'val', 'test']:
        for class_name in target_classes:
            split_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_dir, exist_ok=True)


def copy_files_to_splits(files_by_class, output_dir):
    """Copy pose files to appropriate split directories."""
    stats = {'train': 0, 'val': 0, 'test': 0}

    for gloss, files in sorted(files_by_class.items()):
        print(f"\n{gloss}: {len(files)} files")

        # Split files
        splits = split_files(files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

        # Copy to appropriate directories
        for split_name, files_in_split in splits.items():
            print(f"  {split_name}: {len(files_in_split)} files")

            for pose_file in files_in_split:
                src_path = os.path.join(POSE_FILES_DIR, pose_file)
                dst_path = os.path.join(output_dir, split_name, gloss, pose_file)
                shutil.copy2(src_path, dst_path)
                stats[split_name] += 1

    return stats


def organize_files_by_gloss(output_dir):
    """Organize all pose files by gloss without splitting."""
    print("=" * 70)
    print("ORGANIZING POSE FILES BY GLOSS")
    print("=" * 70)
    print()
    print(f"Input directory:  {POSE_FILES_DIR}")
    print(f"Output directory: {output_dir}")
    print()

    # Load full metadata (no class filtering)
    print("Loading metadata...")
    with open(METADATA_FILE, 'r') as f:
        data = json.load(f)

    video_to_gloss = {}
    for video_id, info in data.items():
        video_to_gloss[video_id] = info['gloss']

    print(f"Found {len(video_to_gloss)} video-to-gloss mappings")
    print()

    # Get all pose files
    pose_files = sorted([f for f in os.listdir(POSE_FILES_DIR) if f.endswith('.pose')])
    print(f"Found {len(pose_files)} pose files")
    print()

    # Group by gloss
    files_by_gloss = defaultdict(list)
    for pose_file in pose_files:
        video_id = pose_file.replace('.pose', '')
        if video_id in video_to_gloss:
            gloss = video_to_gloss[video_id]
            files_by_gloss[gloss].append(pose_file)

    print(f"Organizing into {len(files_by_gloss)} gloss directories...")
    print()

    # Create directories and copy files
    total_copied = 0
    for gloss, files in sorted(files_by_gloss.items()):
        gloss_dir = os.path.join(output_dir, gloss)
        os.makedirs(gloss_dir, exist_ok=True)

        print(f"{gloss}: {len(files)} files")

        for pose_file in files:
            src_path = os.path.join(POSE_FILES_DIR, pose_file)
            dst_path = os.path.join(gloss_dir, pose_file)
            shutil.copy2(src_path, dst_path)
            total_copied += 1

    print()
    print("=" * 70)
    print("ORGANIZATION COMPLETE!")
    print("=" * 70)
    print()
    print(f"Output directory: {output_dir}")
    print(f"Total glosses: {len(files_by_gloss)}")
    print(f"Total files copied: {total_copied}")
    print()


def main():
    """Main splitting logic."""
    parser = argparse.ArgumentParser(description='Split pose files into train/val/test for N classes')
    parser.add_argument('--num-classes', type=int, required=False,
                       help='Number of classes (e.g., 20, 50, 100)')
    parser.add_argument('--existing-classes-path', type=str, default=None,
                       help='Path to existing class split (e.g., path/to/20_class/train) to build upon')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--organize-only', action='store_true',
                       help='Only organize files by gloss without splitting into train/val/test')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Custom output directory (used with --organize-only)')

    args = parser.parse_args()

    # Handle organize-only mode
    if args.organize_only:
        if not args.output_dir:
            print("Error: --output-dir is required when using --organize-only")
            return

        organize_files_by_gloss(args.output_dir)
        return

    # Validate required arguments for split mode
    if not args.num_classes:
        print("Error: --num-classes is required when not using --organize-only")
        return

    # Update ratios if provided
    global TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    TRAIN_RATIO = args.train_ratio
    VAL_RATIO = args.val_ratio
    TEST_RATIO = args.test_ratio

    # Determine output directory
    output_dir = os.path.join(WLASL_BASE_DIR, f"dataset_splits/{args.num_classes}_classes/original/pose_split_{args.num_classes}_class")

    # Get existing classes if path provided
    existing_classes = get_existing_classes(args.existing_classes_path)

    # Get target classes
    target_classes = get_top_n_classes(args.num_classes, existing_classes if existing_classes else None)

    print("=" * 70)
    print(f"SPLITTING POSE FILES INTO TRAIN/VAL/TEST ({args.num_classes} CLASSES)")
    print("=" * 70)
    print()
    print(f"Input directory:  {POSE_FILES_DIR}")
    print(f"Output directory: {output_dir}")
    print(f"Target classes: {len(target_classes)}")
    print()

    if existing_classes:
        num_new = args.num_classes - len(existing_classes)
        print(f"Building on {len(existing_classes)} existing classes:")
        print(f"  Existing: {existing_classes}")
        print(f"  Adding {num_new} new classes: {target_classes[len(existing_classes):]}")
    else:
        print(f"Creating new {args.num_classes}-class split:")
        print(f"  Classes: {target_classes}")

    print()

    # Load metadata
    print("Loading metadata...")
    video_to_gloss = load_metadata(target_classes)
    print(f"Found {len(video_to_gloss)} video-to-gloss mappings for target classes")
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
    create_split_directories(output_dir, target_classes)
    print("Directories created")
    print()

    # Copy files
    print("Copying files to splits...")
    stats = copy_files_to_splits(files_by_class, output_dir)

    print()
    print("=" * 70)
    print("SPLIT COMPLETE!")
    print("=" * 70)
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("Split statistics:")
    print(f"  Train: {stats['train']} files ({stats['train']/total_files*100:.1f}%)")
    print(f"  Val:   {stats['val']} files ({stats['val']/total_files*100:.1f}%)")
    print(f"  Test:  {stats['test']} files ({stats['test']/total_files*100:.1f}%)")
    print()


if __name__ == "__main__":
    main()
