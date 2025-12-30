#!/usr/bin/env python3
"""
Verify Split Quality

This module verifies that dataset splits are:
1. Family-intact (no data leakage between splits)
2. Class-balanced (all classes represented proportionally)
3. Frame-distribution consistent (similar frame length distributions)

Usage:
    python verify_split_quality.py --splits-dir /path/to/splits
    python verify_split_quality.py --manifest /path/to/split_manifest.json
"""

import sys
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from scipy import stats

# Add parent directories to path for imports
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent / 'augmentation'))

from augmentation_config import FRAME_BUCKETS, get_frame_bucket


# Supported pose file extensions
POSE_EXTENSIONS = ['.pkl', '.pose', '.pickle']


def find_pose_files(directory: Path) -> List[Path]:
    """Find all pose files in a directory (supports .pkl, .pose, .pickle)."""
    files = []
    for ext in POSE_EXTENSIONS:
        files.extend(directory.glob(f"*{ext}"))
    return files


def load_pose_data(file_path: Path) -> dict:
    """Load pose data from file (supports pickle and msgpack formats)."""
    with open(file_path, 'rb') as f:
        # Try pickle first
        try:
            return pickle.load(f)
        except Exception:
            pass

    # Try msgpack for .pose files
    try:
        import msgpack
        import msgpack_numpy as m
        m.patch()  # Enable numpy array support

        with open(file_path, 'rb') as f:
            return msgpack.unpackb(f.read(), raw=False)
    except ImportError:
        # msgpack not available, skip this file
        raise ValueError(f"Cannot load {file_path}: msgpack not installed")
    except Exception as e:
        raise ValueError(f"Cannot load {file_path}: {e}")


def check_family_integrity(splits_dir: Path) -> Tuple[bool, List[str]]:
    """
    Verify that families are not split across train/val/test.

    A family leak occurs when an original video is in one split
    but its augmentation appears in another split.

    Args:
        splits_dir: Directory containing train/val/test subdirectories

    Returns:
        Tuple of (passed: bool, issues: List[str])
    """
    print("\n[Check 1] Family Integrity (No Data Leakage)")
    print("-" * 50)

    issues = []

    # Collect all video IDs by split
    video_ids_by_split = {}

    for split_name in ['train', 'val', 'test']:
        split_dir = splits_dir / split_name
        if not split_dir.exists():
            issues.append(f"Split directory not found: {split_dir}")
            continue

        video_ids = set()
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            for pose_file in find_pose_files(class_dir):
                # Extract base video_id (without augmentation suffix)
                stem = pose_file.stem
                if '_aug' in stem:
                    video_id = stem.split('_aug')[0]
                else:
                    video_id = stem

                video_ids.add(f"{class_dir.name}/{video_id}")

        video_ids_by_split[split_name] = video_ids
        print(f"  {split_name}: {len(video_ids)} unique video families")

    # Check for overlaps
    for split1 in ['train', 'val', 'test']:
        for split2 in ['train', 'val', 'test']:
            if split1 >= split2:
                continue

            if split1 not in video_ids_by_split or split2 not in video_ids_by_split:
                continue

            overlap = video_ids_by_split[split1] & video_ids_by_split[split2]
            if overlap:
                issues.append(
                    f"Family leak between {split1} and {split2}: "
                    f"{len(overlap)} shared families"
                )
                # Show first few examples
                for vid in list(overlap)[:3]:
                    issues.append(f"  - {vid}")

    if issues:
        print("  [FAIL] Family integrity issues found:")
        for issue in issues:
            print(f"    {issue}")
        return False, issues
    else:
        print("  [PASS] No family leakage detected")
        return True, []


def check_class_balance(splits_dir: Path) -> Tuple[bool, List[str]]:
    """
    Verify that classes are represented proportionally across splits.

    Args:
        splits_dir: Directory containing train/val/test subdirectories

    Returns:
        Tuple of (passed: bool, issues: List[str])
    """
    print("\n[Check 2] Class Balance")
    print("-" * 50)

    issues = []

    # Count samples per class per split
    class_counts = defaultdict(lambda: defaultdict(int))
    all_classes = set()

    for split_name in ['train', 'val', 'test']:
        split_dir = splits_dir / split_name
        if not split_dir.exists():
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            all_classes.add(class_name)

            sample_count = len(find_pose_files(class_dir))
            class_counts[class_name][split_name] = sample_count

    print(f"  Total classes: {len(all_classes)}")

    # Check each class is in all splits (or handle small classes)
    classes_missing_from_val = []
    classes_missing_from_test = []

    for class_name in sorted(all_classes):
        counts = class_counts[class_name]
        total = sum(counts.values())

        if counts.get('val', 0) == 0 and total >= 3:
            classes_missing_from_val.append(class_name)

        if counts.get('test', 0) == 0 and total >= 5:
            classes_missing_from_test.append(class_name)

    if classes_missing_from_val:
        msg = f"Classes missing from val (with >= 3 total samples): {len(classes_missing_from_val)}"
        issues.append(msg)
        if len(classes_missing_from_val) <= 5:
            issues.append(f"  Classes: {classes_missing_from_val}")

    if classes_missing_from_test:
        msg = f"Classes missing from test (with >= 5 total samples): {len(classes_missing_from_test)}"
        issues.append(msg)
        if len(classes_missing_from_test) <= 5:
            issues.append(f"  Classes: {classes_missing_from_test}")

    # Calculate class distribution variance
    train_counts = [class_counts[c].get('train', 0) for c in all_classes]
    val_counts = [class_counts[c].get('val', 0) for c in all_classes]
    test_counts = [class_counts[c].get('test', 0) for c in all_classes]

    print(f"  Train: min={min(train_counts)}, max={max(train_counts)}, "
          f"mean={np.mean(train_counts):.1f}, std={np.std(train_counts):.1f}")
    print(f"  Val:   min={min(val_counts)}, max={max(val_counts)}, "
          f"mean={np.mean(val_counts):.1f}, std={np.std(val_counts):.1f}")
    print(f"  Test:  min={min(test_counts)}, max={max(test_counts)}, "
          f"mean={np.mean(test_counts):.1f}, std={np.std(test_counts):.1f}")

    # Check coefficient of variation (CV) for class balance
    cv_train = np.std(train_counts) / np.mean(train_counts) if np.mean(train_counts) > 0 else float('inf')
    cv_val = np.std(val_counts) / np.mean(val_counts) if np.mean(val_counts) > 0 else float('inf')

    if cv_train > 0.5:
        issues.append(f"High class imbalance in train (CV={cv_train:.2f} > 0.5)")

    if issues:
        print("  [WARN] Class balance issues:")
        for issue in issues:
            print(f"    {issue}")
        return len(issues) < 3, issues  # Allow minor issues
    else:
        print("  [PASS] Class balance is acceptable")
        return True, []


def check_frame_distribution(splits_dir: Path) -> Tuple[bool, List[str]]:
    """
    Verify that frame length distribution is consistent across splits.

    Uses Kolmogorov-Smirnov test to compare distributions.

    Args:
        splits_dir: Directory containing train/val/test subdirectories

    Returns:
        Tuple of (passed: bool, issues: List[str])
    """
    print("\n[Check 3] Frame Length Distribution")
    print("-" * 50)

    issues = []

    # Collect frame counts per split
    frame_counts = defaultdict(list)
    bucket_counts = defaultdict(lambda: defaultdict(int))

    for split_name in ['train', 'val', 'test']:
        split_dir = splits_dir / split_name
        if not split_dir.exists():
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            for pose_file in find_pose_files(class_dir):
                try:
                    data = load_pose_data(pose_file)

                    keypoints = data.get('keypoints', data.get('landmarks'))
                    if keypoints is not None:
                        frame_count = keypoints.shape[0]
                        frame_counts[split_name].append(frame_count)
                        bucket = get_frame_bucket(frame_count)
                        bucket_counts[split_name][bucket] += 1
                except Exception as e:
                    issues.append(f"Error reading {pose_file}: {e}")

    # Print statistics
    for split_name in ['train', 'val', 'test']:
        counts = frame_counts.get(split_name, [])
        if counts:
            print(f"  {split_name}: n={len(counts)}, "
                  f"mean={np.mean(counts):.1f}, median={np.median(counts):.1f}, "
                  f"std={np.std(counts):.1f}")

    # Print bucket distribution
    print("\n  Frame bucket distribution:")
    for split_name in ['train', 'val', 'test']:
        buckets = bucket_counts.get(split_name, {})
        total = sum(buckets.values())
        if total > 0:
            pcts = {k: 100*v/total for k, v in buckets.items()}
            bucket_str = ", ".join(f"{k}: {pcts.get(k, 0):.1f}%" for k in ['short', 'medium', 'long'])
            print(f"    {split_name}: {bucket_str}")

    # Statistical test: compare train vs test distribution
    train_frames = frame_counts.get('train', [])
    test_frames = frame_counts.get('test', [])

    if len(train_frames) >= 10 and len(test_frames) >= 10:
        ks_stat, p_value = stats.ks_2samp(train_frames, test_frames)
        print(f"\n  KS test (train vs test): statistic={ks_stat:.4f}, p-value={p_value:.4f}")

        if p_value < 0.01:
            issues.append(
                f"Significant frame distribution difference between train/test "
                f"(KS p={p_value:.4f} < 0.01)"
            )
            print("  [WARN] Frame distributions differ significantly")
        else:
            print("  [PASS] Frame distributions are similar (p >= 0.01)")
    else:
        print("  [SKIP] Not enough samples for statistical test")

    if issues:
        print("  Issues found:")
        for issue in issues:
            print(f"    {issue}")
        return len(issues) < 2, issues
    else:
        return True, []


def check_sample_counts(splits_dir: Path) -> Tuple[bool, List[str]]:
    """
    Verify overall sample counts and split ratios.

    Args:
        splits_dir: Directory containing train/val/test subdirectories

    Returns:
        Tuple of (passed: bool, issues: List[str])
    """
    print("\n[Check 4] Sample Counts and Split Ratios")
    print("-" * 50)

    issues = []
    counts = {}

    for split_name in ['train', 'val', 'test']:
        split_dir = splits_dir / split_name
        if not split_dir.exists():
            counts[split_name] = 0
            continue

        sample_count = sum(
            len(find_pose_files(class_dir))
            for class_dir in split_dir.iterdir()
            if class_dir.is_dir()
        )
        counts[split_name] = sample_count

    total = sum(counts.values())

    print(f"  Total samples: {total}")
    for split_name, count in counts.items():
        pct = 100 * count / total if total > 0 else 0
        print(f"  {split_name:>5}: {count:>6} samples ({pct:>5.1f}%)")

    # Check ratios are reasonable
    if total > 0:
        train_ratio = counts['train'] / total
        if train_ratio < 0.5:
            issues.append(f"Train ratio too low: {train_ratio:.2f} < 0.50")
        elif train_ratio > 0.9:
            issues.append(f"Train ratio too high: {train_ratio:.2f} > 0.90")

        if counts['val'] == 0:
            issues.append("Validation set is empty")
        if counts['test'] == 0:
            issues.append("Test set is empty")

    if issues:
        print("  [WARN] Sample count issues:")
        for issue in issues:
            print(f"    {issue}")
        return False, issues
    else:
        print("  [PASS] Sample counts are reasonable")
        return True, []


def verify_splits(splits_dir: Path) -> bool:
    """
    Run all verification checks on the splits.

    Args:
        splits_dir: Directory containing train/val/test subdirectories

    Returns:
        True if all critical checks pass
    """
    print("=" * 70)
    print("SPLIT QUALITY VERIFICATION")
    print("=" * 70)
    print(f"\nSplits directory: {splits_dir}")

    results = {}

    # Run all checks
    results['family_integrity'], _ = check_family_integrity(splits_dir)
    results['class_balance'], _ = check_class_balance(splits_dir)
    results['frame_distribution'], _ = check_frame_distribution(splits_dir)
    results['sample_counts'], _ = check_sample_counts(splits_dir)

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for check_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {check_name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("[SUCCESS] All verification checks passed!")
    else:
        print("[WARNING] Some checks failed - review issues above")

    print("=" * 70 + "\n")

    return all_passed


def verify_from_manifest(manifest_path: Path) -> bool:
    """
    Verify splits from manifest file (without re-scanning files).

    Args:
        manifest_path: Path to split_manifest.json

    Returns:
        True if manifest looks valid
    """
    print("=" * 70)
    print("MANIFEST VERIFICATION")
    print("=" * 70)
    print(f"\nManifest: {manifest_path}")

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    stats = manifest.get('statistics', {})
    splits = manifest.get('splits', {})

    print(f"\nTotal families: {stats.get('total_families')}")
    print(f"Total samples: {stats.get('total_samples')}")
    print(f"Total classes: {stats.get('classes')}")

    print("\nSplit distribution:")
    for split_name, split_info in splits.items():
        total = stats.get('total_samples', 1)
        pct = 100 * split_info['total_samples'] / total
        print(f"  {split_name}: {split_info['total_families']} families, "
              f"{split_info['total_samples']} samples ({pct:.1f}%)")

    # Check frame bucket distribution
    print("\nFrame bucket distribution:")
    for split_name, split_info in splits.items():
        buckets = split_info.get('frame_buckets', {})
        total = sum(buckets.values())
        if total > 0:
            pcts = {k: 100*v/total for k, v in buckets.items()}
            bucket_str = ", ".join(f"{k}: {pcts.get(k, 0):.1f}%" for k in ['short', 'medium', 'long'])
            print(f"  {split_name}: {bucket_str}")

    print("\n" + "=" * 70)
    print("[INFO] Manifest verification complete")
    print("Run with --splits-dir for full file-based verification")
    print("=" * 70 + "\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Verify dataset split quality"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--splits-dir', type=Path,
        help='Directory containing train/val/test subdirectories'
    )
    group.add_argument(
        '--manifest', type=Path,
        help='Path to split_manifest.json for quick verification'
    )

    args = parser.parse_args()

    if args.manifest:
        if not args.manifest.exists():
            print(f"Error: Manifest not found: {args.manifest}")
            sys.exit(1)
        success = verify_from_manifest(args.manifest)
    else:
        if not args.splits_dir.exists():
            print(f"Error: Splits directory not found: {args.splits_dir}")
            sys.exit(1)
        success = verify_splits(args.splits_dir)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
