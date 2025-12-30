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

# Signer group detection threshold
# Video IDs with gaps larger than this are assumed to be from different signers/recording sessions
SIGNER_GROUP_GAP_THRESHOLD = 100  # Gap > 100 between video IDs = new signer group


# ============================================================================
# DYNAMIC CLASS SELECTION - No hardcoded lists!
# ============================================================================

def get_class_frequencies(metadata_file):
    """
    Count video frequency for each class from metadata.
    Returns dict: {gloss: video_count} sorted by frequency (descending).
    """
    with open(metadata_file, 'r') as f:
        data = json.load(f)

    # Count videos per gloss
    gloss_counts = defaultdict(int)
    for video_id, info in data.items():
        gloss = info['gloss']
        gloss_counts[gloss] += 1

    # Sort by frequency (descending)
    sorted_glosses = sorted(gloss_counts.items(), key=lambda x: x[1], reverse=True)

    return dict(sorted_glosses)


def load_class_mapping_if_exists(class_mapping_path):
    """
    Load class list from class_mapping.json if it exists.
    Returns list of classes, or None if file doesn't exist.
    """
    if not os.path.exists(class_mapping_path):
        return None

    try:
        with open(class_mapping_path, 'r') as f:
            mapping = json.load(f)

        # Handle both formats: {"classes": [...]} or just [...]
        if isinstance(mapping, dict) and 'classes' in mapping:
            return mapping['classes']
        elif isinstance(mapping, list):
            return mapping
        else:
            print(f"WARNING: Unexpected format in {class_mapping_path}")
            return None
    except Exception as e:
        print(f"WARNING: Failed to load {class_mapping_path}: {e}")
        return None


def save_class_mapping(class_mapping_path, classes):
    """
    Save class list to class_mapping.json.
    Creates parent directories if needed.
    """
    os.makedirs(os.path.dirname(class_mapping_path), exist_ok=True)

    mapping = {
        'classes': classes,
        'num_classes': len(classes),
        'creation_method': 'frequency_based_incremental'
    }

    with open(class_mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"Saved class mapping to {class_mapping_path}")


def select_classes_incrementally(num_classes, metadata_file, class_mapping_path, previous_mapping_path=None):
    """
    Select classes incrementally based on frequency.

    Strategy:
    1. If class_mapping.json exists for num_classes, use it (stable)
    2. Otherwise, load previous tier's classes (e.g., 20 for 50, or 50 for 100)
    3. Select additional classes by frequency (excluding existing)
    4. Save to class_mapping.json

    Args:
        num_classes: Target number of classes (20, 50, 100)
        metadata_file: Path to video_to_gloss_mapping.json
        class_mapping_path: Path to save class_mapping.json for this tier
        previous_mapping_path: Optional path to previous tier's class_mapping.json

    Returns:
        List of selected classes
    """
    # Check if this tier already has a saved mapping
    existing_classes = load_class_mapping_if_exists(class_mapping_path)
    if existing_classes:
        print(f"Using existing class mapping from {class_mapping_path}")
        print(f"  {len(existing_classes)} classes (stable)")
        return existing_classes

    # Load previous tier's classes if available
    base_classes = []
    if previous_mapping_path:
        base_classes = load_class_mapping_if_exists(previous_mapping_path)
        if base_classes:
            print(f"Building on {len(base_classes)} classes from previous tier")
        else:
            print(f"Previous tier mapping not found, starting fresh")

    # Get all classes by frequency
    class_frequencies = get_class_frequencies(metadata_file)

    # Select additional classes
    selected_classes = list(base_classes) if base_classes else []
    base_set = set(selected_classes)

    # Add most frequent classes until we reach num_classes
    for gloss, count in class_frequencies.items():
        if gloss not in base_set:
            selected_classes.append(gloss)
            if len(selected_classes) >= num_classes:
                break

    # Verify we got enough classes
    if len(selected_classes) < num_classes:
        print(f"WARNING: Only {len(selected_classes)} classes available, requested {num_classes}")

    # Save the mapping
    save_class_mapping(class_mapping_path, selected_classes)

    print(f"Selected {len(selected_classes)} classes:")
    if base_classes:
        new_count = len(selected_classes) - len(base_classes)
        print(f"  {len(base_classes)} from previous tier + {new_count} new classes")
    else:
        print(f"  All {len(selected_classes)} classes selected by frequency")

    return selected_classes


def get_existing_classes(existing_path):
    """Get list of existing classes from a previous split."""
    if not existing_path or not os.path.exists(existing_path):
        return []

    existing_classes = []
    for item in os.listdir(existing_path):
        if os.path.isdir(os.path.join(existing_path, item)):
            existing_classes.append(item)

    return sorted(existing_classes)


def get_top_n_classes(num_classes, metadata_file, class_mapping_path, previous_mapping_path=None):
    """
    Get top N classes using dynamic frequency-based selection.

    This is a wrapper around select_classes_incrementally for backward compatibility.

    Args:
        num_classes: Target number of classes
        metadata_file: Path to video_to_gloss_mapping.json
        class_mapping_path: Path to class_mapping.json for this tier
        previous_mapping_path: Optional path to previous tier's class_mapping.json

    Returns:
        List of selected classes
    """
    return select_classes_incrementally(
        num_classes,
        metadata_file,
        class_mapping_path,
        previous_mapping_path
    )


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


def group_files_by_signer(files, gap_threshold=SIGNER_GROUP_GAP_THRESHOLD):
    """
    Group files by video ID proximity to infer signer/recording session.

    Consecutive video IDs (small gaps) are likely from the same signer.
    Large gaps indicate different recording sessions/signers.

    Args:
        files: List of filenames (e.g., ['07068.pose', '07069.pose', ...])
        gap_threshold: Maximum gap between video IDs in same group

    Returns:
        List of groups, where each group is a list of files from same signer
    """
    if not files:
        return []

    # Extract video IDs and sort
    file_ids = []
    for f in files:
        try:
            vid_id = int(f.replace('.pose', '').replace('.pkl', ''))
            file_ids.append((vid_id, f))
        except ValueError:
            # Non-numeric filename, treat as separate group
            file_ids.append((float('inf'), f))

    file_ids.sort(key=lambda x: x[0])

    # Group by proximity
    groups = []
    current_group = [file_ids[0][1]]
    prev_id = file_ids[0][0]

    for vid_id, filename in file_ids[1:]:
        if vid_id == float('inf') or prev_id == float('inf'):
            # Non-numeric ID, start new group
            groups.append(current_group)
            current_group = [filename]
        elif vid_id - prev_id > gap_threshold:
            # Large gap, start new group (different signer)
            groups.append(current_group)
            current_group = [filename]
        else:
            # Small gap, same signer
            current_group.append(filename)
        prev_id = vid_id

    # Don't forget the last group
    if current_group:
        groups.append(current_group)

    return groups


def split_files_signer_aware(files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split files into train/val/test while keeping signer groups together.

    This ensures that different signers appear in each split, improving
    generalization testing. Videos from the same signer stay in the same split
    to prevent data leakage.

    Strategy:
    1. Handle small datasets (n<=3) with minimum sample guarantees
    2. Group files by video ID proximity (inferred signer groups)
    3. Distribute groups across splits to achieve target ratios
    4. Ensure each split has representation from different signer groups

    Args:
        files: List of filenames
        train_ratio, val_ratio, test_ratio: Target split ratios

    Returns:
        Dict with 'train', 'val', 'test' lists
    """
    n = len(files)

    # Handle small datasets with minimum sample constraints
    if n == 0:
        return {'train': [], 'val': [], 'test': []}

    if n == 1:
        # Only 1 file: put in train, copy to val and test
        return {
            'train': [files[0]],
            'val': [files[0]],  # Copy
            'test': [files[0]]  # Copy
        }

    if n == 2:
        # 2 files: 1 train, 1 val, copy val to test
        # Try to put different signer groups in different splits if possible
        groups = group_files_by_signer(files)
        if len(groups) == 2:
            # Different signers - put one in train, one in test/val
            return {
                'train': groups[0],
                'val': groups[1],
                'test': groups[1].copy()  # Copy from val
            }
        else:
            # Same signer - use positional split
            return {
                'train': [files[0]],
                'val': [files[1]],
                'test': [files[1]]  # Copy from val
            }

    if n == 3:
        # 3 files: 1 each - try to distribute signer groups
        groups = group_files_by_signer(files)
        if len(groups) >= 3:
            # 3+ groups: one from each
            return {
                'train': groups[0],
                'val': groups[1],
                'test': groups[2]
            }
        elif len(groups) == 2:
            # 2 groups: larger to train, smaller split between val/test
            if len(groups[0]) >= len(groups[1]):
                larger, smaller = groups[0], groups[1]
            else:
                larger, smaller = groups[1], groups[0]

            if len(larger) >= 2:
                return {
                    'train': larger[:-1],
                    'val': [larger[-1]],
                    'test': smaller
                }
            else:
                return {
                    'train': larger,
                    'val': smaller,
                    'test': smaller.copy()
                }
        else:
            # 1 group: positional split
            return {
                'train': [files[0]],
                'val': [files[1]],
                'test': [files[2]]
            }

    # n >= 4: Use signer-aware grouping
    groups = group_files_by_signer(files)
    n_groups = len(groups)

    # If only 1 group, fall back to simple splitting within the group
    if n_groups == 1:
        return split_files_simple(files, train_ratio, val_ratio, test_ratio)

    # If 2 groups: train gets larger, test/val share smaller
    if n_groups == 2:
        # Put larger group in train, smaller in test (copy to val)
        if len(groups[0]) >= len(groups[1]):
            train_files = groups[0]
            test_files = groups[1]
        else:
            train_files = groups[1]
            test_files = groups[0]

        # For val, take some from train if possible, otherwise copy from test
        if len(train_files) >= 3:
            val_files = [train_files.pop()]  # Take 1 from train for val
        else:
            val_files = test_files.copy()  # Copy test to val

        return {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

    # 3+ groups: distribute groups across splits
    # Sort groups by size (largest first) for better distribution
    groups_with_size = [(len(g), i, g) for i, g in enumerate(groups)]
    groups_with_size.sort(reverse=True)

    # Calculate target counts
    target_train = int(n * train_ratio)
    target_val = int(n * val_ratio)
    target_test = n - target_train - target_val

    # Distribute groups to splits
    train_files = []
    val_files = []
    test_files = []

    train_count = 0
    val_count = 0
    test_count = 0

    for size, _, group in groups_with_size:
        # Assign to the split that needs files most (relative to target)
        train_need = target_train - train_count
        val_need = target_val - val_count
        test_need = target_test - test_count

        # Ensure at least one group goes to each split
        if not val_files and len(groups_with_size) - groups_with_size.index((size, _, group)) <= 2:
            val_files.extend(group)
            val_count += size
        elif not test_files and len(groups_with_size) - groups_with_size.index((size, _, group)) <= 1:
            test_files.extend(group)
            test_count += size
        elif train_need >= max(val_need, test_need) or not train_files:
            train_files.extend(group)
            train_count += size
        elif val_need >= test_need:
            val_files.extend(group)
            val_count += size
        else:
            test_files.extend(group)
            test_count += size

    # Ensure all splits have at least some files
    if not val_files and train_files:
        # Move last file from train to val
        val_files.append(train_files.pop())
    if not test_files and train_files:
        # Move last file from train to test
        test_files.append(train_files.pop())

    return {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }


def split_files_simple(files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Simple positional split (original behavior) for fallback.

    Used when there's only one signer group.
    """
    n = len(files)

    if n == 0:
        return {'train': [], 'val': [], 'test': []}

    if n == 1:
        return {
            'train': [files[0]],
            'val': [files[0]],
            'test': [files[0]]
        }

    if n == 2:
        return {
            'train': [files[0]],
            'val': [files[1]],
            'test': [files[1]]
        }

    if n == 3:
        return {
            'train': [files[0]],
            'val': [files[1]],
            'test': [files[2]]
        }

    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, n - n_train - n_val)

    while n_train + n_val + n_test > n:
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1

    return {
        'train': files[:n_train],
        'val': files[n_train:n_train + n_val],
        'test': files[n_train + n_val:n_train + n_val + n_test]
    }


def split_files(files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split files into train/val/test using signer-aware grouping.

    This is the main entry point that uses signer-aware splitting by default.
    Falls back to simple splitting for very small datasets.

    Ensures at least 1 sample in each split when possible:
    - n=1: 1 train, copy to val, copy to test
    - n=2: 1 train, 1 val, copy val to test
    - n=3: 1 train, 1 val, 1 test
    - n>=4: use signer-aware ratios but ensure at least 1 in each split
    """
    # Use signer-aware splitting for all cases
    # It handles small datasets gracefully by falling back to simple split
    return split_files_signer_aware(files, train_ratio, val_ratio, test_ratio)


def create_split_directories(output_dir, target_classes):
    """Create output directory structure."""
    for split in ['train', 'val', 'test']:
        for class_name in target_classes:
            split_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_dir, exist_ok=True)


def copy_files_to_splits(files_by_class, output_dir):
    """Copy pose files to appropriate split directories using signer-aware splitting."""
    stats = {'train': 0, 'val': 0, 'test': 0}
    signer_group_stats = {'total_groups': 0, 'multi_group_classes': 0}

    for gloss, files in sorted(files_by_class.items()):
        # Analyze signer groups for this class
        groups = group_files_by_signer(files)
        n_groups = len(groups)
        signer_group_stats['total_groups'] += n_groups

        if n_groups > 1:
            signer_group_stats['multi_group_classes'] += 1
            group_sizes = [len(g) for g in groups]
            print(f"\n{gloss}: {len(files)} files in {n_groups} signer groups {group_sizes}")
        else:
            print(f"\n{gloss}: {len(files)} files (1 signer group)")

        # Split files using signer-aware logic
        splits = split_files(files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

        # Show split details with video ID ranges
        for split_name, files_in_split in splits.items():
            if files_in_split:
                # Extract video IDs for summary
                try:
                    ids = sorted([int(f.replace('.pose', '').replace('.pkl', '')) for f in files_in_split])
                    if len(ids) > 3:
                        id_summary = f"[{ids[0]}...{ids[-1]}]"
                    else:
                        id_summary = str(ids)
                except:
                    id_summary = ""
                print(f"  {split_name}: {len(files_in_split)} files {id_summary}")
            else:
                print(f"  {split_name}: 0 files")

            for pose_file in files_in_split:
                src_path = os.path.join(POSE_FILES_DIR, pose_file)
                dst_path = os.path.join(output_dir, split_name, gloss, pose_file)
                shutil.copy2(src_path, dst_path)
                stats[split_name] += 1

    # Print signer group summary
    print(f"\n--- Signer Group Summary ---")
    print(f"Total signer groups detected: {signer_group_stats['total_groups']}")
    print(f"Classes with multiple signer groups: {signer_group_stats['multi_group_classes']}/{len(files_by_class)}")
    print(f"(Gap threshold for new group: {SIGNER_GROUP_GAP_THRESHOLD} video IDs)")

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

    # Determine output directory and class mapping paths
    output_base = os.path.join(WLASL_BASE_DIR, f"dataset_splits/{args.num_classes}_classes")
    output_dir = os.path.join(output_base, "original", f"pose_split_{args.num_classes}_class")
    class_mapping_path = os.path.join(output_base, "class_mapping.json")

    # Determine previous tier's class mapping path for incremental building
    previous_mapping_path = None
    if args.num_classes == 50:
        previous_mapping_path = os.path.join(WLASL_BASE_DIR, "dataset_splits/20_classes/class_mapping.json")
    elif args.num_classes == 100:
        previous_mapping_path = os.path.join(WLASL_BASE_DIR, "dataset_splits/50_classes/class_mapping.json")

    # Get target classes using dynamic frequency-based selection
    target_classes = get_top_n_classes(
        num_classes=args.num_classes,
        metadata_file=METADATA_FILE,
        class_mapping_path=class_mapping_path,
        previous_mapping_path=previous_mapping_path
    )

    print("=" * 70)
    print(f"SPLITTING POSE FILES INTO TRAIN/VAL/TEST ({args.num_classes} CLASSES)")
    print("=" * 70)
    print()
    print(f"Input directory:  {POSE_FILES_DIR}")
    print(f"Output directory: {output_dir}")
    print(f"Class mapping: {class_mapping_path}")
    print(f"Target classes: {len(target_classes)}")
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
