#!/usr/bin/env python3
"""
Stratified Family-Based Dataset Splitting

This module splits augmented pose data into train/val/test sets while:
1. Keeping families together (original + all its augmentations)
2. Stratifying by class for balanced representation
3. Stratifying by frame-length bucket for distribution consistency

Usage:
    python stratified_family_split.py --input-dir /path/to/augmented --output-dir /path/to/splits
    python stratified_family_split.py --input-dir /path/to/augmented --output-dir /path/to/splits --dry-run
"""

import sys
import json
import pickle
import shutil
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field

# Add parent directories to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(script_dir.parent / 'augmentation'))
sys.path.insert(0, str(project_root))

from augmentation_config import SPLIT_RATIOS, FRAME_BUCKETS, get_frame_bucket


# Supported pose file extensions
POSE_EXTENSIONS = ['.pkl', '.pose', '.pickle']


def find_pose_files(directory: Path) -> List[Path]:
    """Find all pose files in a directory (supports .pkl, .pose, .pickle)."""
    files = []
    for ext in POSE_EXTENSIONS:
        files.extend(directory.glob(f"*{ext}"))
    return files


@dataclass
class PoseFamily:
    """Represents an original video and all its augmentations."""
    video_id: str
    gloss: str
    original_path: Path
    augmented_paths: List[Path] = field(default_factory=list)
    frame_count: int = 0
    frame_bucket: str = 'medium'

    @property
    def total_samples(self) -> int:
        """Total samples in this family (1 original + N augmentations)."""
        return 1 + len(self.augmented_paths)

    @property
    def all_paths(self) -> List[Path]:
        """All paths in this family."""
        return [self.original_path] + self.augmented_paths


def scan_augmented_dataset(input_dir: Path) -> Dict[str, Dict[str, PoseFamily]]:
    """
    Scan augmented dataset and group files into families.

    Args:
        input_dir: Directory containing class folders with pose files

    Returns:
        Dict mapping class_name -> {video_id -> PoseFamily}
    """
    families_by_class = defaultdict(dict)

    # Find all class directories
    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

    print(f"Scanning {len(class_dirs)} class directories...")

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name

        # Group files by original video_id
        files_by_video = defaultdict(list)

        for pose_file in find_pose_files(class_dir):
            # Parse filename to get video_id and augmentation info
            # Expected formats:
            #   - Original: {video_id}.pkl or {video_id}.pose
            #   - Augmented: {video_id}_aug{N}_{technique}.pkl
            stem = pose_file.stem

            if '_aug' in stem:
                # This is an augmented file
                video_id = stem.split('_aug')[0]
                files_by_video[video_id].append(('augmented', pose_file))
            else:
                # This is an original file
                video_id = stem
                files_by_video[video_id].append(('original', pose_file))

        # Create PoseFamily objects
        for video_id, file_list in files_by_video.items():
            original_path = None
            augmented_paths = []

            for file_type, file_path in file_list:
                if file_type == 'original':
                    original_path = file_path
                else:
                    augmented_paths.append(file_path)

            if original_path is None:
                print(f"  Warning: No original file found for {video_id} in {class_name}")
                continue

            # Load original to get frame count
            with open(original_path, 'rb') as f:
                data = pickle.load(f)

            keypoints = data.get('keypoints', data.get('landmarks'))
            if keypoints is not None:
                frame_count = keypoints.shape[0]
            else:
                frame_count = 0

            family = PoseFamily(
                video_id=video_id,
                gloss=class_name,
                original_path=original_path,
                augmented_paths=sorted(augmented_paths),
                frame_count=frame_count,
                frame_bucket=get_frame_bucket(frame_count),
            )

            families_by_class[class_name][video_id] = family

    return dict(families_by_class)


def stratified_split_families(
    families_by_class: Dict[str, Dict[str, PoseFamily]],
    split_ratios: Dict[str, float] = None,
    seed: int = 42,
) -> Dict[str, List[PoseFamily]]:
    """
    Split families into train/val/test sets with stratification.

    Stratifies by:
    1. Class (ensures each class is represented in all splits)
    2. Frame-length bucket (ensures frame distribution is consistent)

    Args:
        families_by_class: Dict mapping class_name -> {video_id -> PoseFamily}
        split_ratios: Dict with train/val/test ratios (default from config)
        seed: Random seed for reproducibility

    Returns:
        Dict mapping split_name -> List[PoseFamily]
    """
    if split_ratios is None:
        split_ratios = SPLIT_RATIOS

    np.random.seed(seed)

    splits = {
        'train': [],
        'val': [],
        'test': [],
    }

    # Process each class
    for class_name in sorted(families_by_class.keys()):
        class_families = list(families_by_class[class_name].values())

        # Group families by frame bucket for stratification
        by_bucket = defaultdict(list)
        for family in class_families:
            by_bucket[family.frame_bucket].append(family)

        # Split each bucket proportionally
        for bucket_name, bucket_families in by_bucket.items():
            # Shuffle for randomness
            np.random.shuffle(bucket_families)

            n_total = len(bucket_families)
            n_train = max(1, int(n_total * split_ratios['train']))
            n_val = max(0, int(n_total * split_ratios['val']))

            # Ensure at least 1 in train if we have any samples
            if n_total == 1:
                # Single sample goes to train
                splits['train'].append(bucket_families[0])
            elif n_total == 2:
                # Two samples: one train, one val (for minimal validation)
                splits['train'].append(bucket_families[0])
                splits['val'].append(bucket_families[1])
            else:
                # Normal split
                splits['train'].extend(bucket_families[:n_train])
                splits['val'].extend(bucket_families[n_train:n_train + n_val])
                splits['test'].extend(bucket_families[n_train + n_val:])

    return splits


def copy_family_to_split(
    family: PoseFamily,
    split_name: str,
    output_dir: Path,
    dry_run: bool = False,
) -> int:
    """
    Copy all files in a family to the appropriate split directory.

    Args:
        family: The PoseFamily to copy
        split_name: 'train', 'val', or 'test'
        output_dir: Base output directory
        dry_run: If True, don't actually copy files

    Returns:
        Number of files copied
    """
    # Create output class directory
    class_output_dir = output_dir / split_name / family.gloss

    if not dry_run:
        class_output_dir.mkdir(parents=True, exist_ok=True)

    files_copied = 0

    for src_path in family.all_paths:
        dst_path = class_output_dir / src_path.name

        if not dry_run:
            shutil.copy2(src_path, dst_path)

        files_copied += 1

    return files_copied


def generate_split_manifest(
    splits: Dict[str, List[PoseFamily]],
    output_dir: Path,
) -> Dict:
    """
    Generate a manifest JSON describing the splits.

    Args:
        splits: Dict mapping split_name -> List[PoseFamily]
        output_dir: Output directory for manifest

    Returns:
        Manifest dictionary
    """
    manifest = {
        'split_ratios': SPLIT_RATIOS,
        'frame_buckets': {k: list(v) for k, v in FRAME_BUCKETS.items()},
        'splits': {},
        'statistics': {},
    }

    for split_name, families in splits.items():
        split_info = {
            'families': [],
            'total_families': len(families),
            'total_samples': sum(f.total_samples for f in families),
            'classes': defaultdict(int),
            'frame_buckets': defaultdict(int),
        }

        for family in families:
            split_info['families'].append({
                'video_id': family.video_id,
                'gloss': family.gloss,
                'original_file': family.original_path.name,
                'augmented_count': len(family.augmented_paths),
                'total_samples': family.total_samples,
                'frame_count': family.frame_count,
                'frame_bucket': family.frame_bucket,
            })

            split_info['classes'][family.gloss] += family.total_samples
            split_info['frame_buckets'][family.frame_bucket] += family.total_samples

        # Convert defaultdicts to regular dicts for JSON
        split_info['classes'] = dict(split_info['classes'])
        split_info['frame_buckets'] = dict(split_info['frame_buckets'])

        manifest['splits'][split_name] = split_info

    # Calculate overall statistics
    total_families = sum(len(f) for f in splits.values())
    total_samples = sum(
        sum(fam.total_samples for fam in families)
        for families in splits.values()
    )

    manifest['statistics'] = {
        'total_families': total_families,
        'total_samples': total_samples,
        'classes': len(set(f.gloss for fams in splits.values() for f in fams)),
        'split_sample_counts': {
            name: sum(f.total_samples for f in families)
            for name, families in splits.items()
        },
    }

    return manifest


def print_split_summary(
    splits: Dict[str, List[PoseFamily]],
    manifest: Dict,
) -> None:
    """Print a summary of the splits."""
    print("\n" + "=" * 70)
    print("STRATIFIED FAMILY-BASED SPLIT SUMMARY")
    print("=" * 70)

    stats = manifest['statistics']
    print(f"\nTotal families: {stats['total_families']}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Total classes: {stats['classes']}")

    print("\nSplit distribution:")
    for split_name in ['train', 'val', 'test']:
        split_info = manifest['splits'][split_name]
        pct = 100 * split_info['total_samples'] / stats['total_samples']
        print(f"  {split_name:>5}: {split_info['total_families']:>4} families, "
              f"{split_info['total_samples']:>6} samples ({pct:>5.1f}%)")

    print("\nFrame bucket distribution per split:")
    for split_name in ['train', 'val', 'test']:
        split_info = manifest['splits'][split_name]
        buckets = split_info['frame_buckets']
        bucket_str = ", ".join(f"{k}: {v}" for k, v in sorted(buckets.items()))
        print(f"  {split_name:>5}: {bucket_str}")

    # Check class coverage
    print("\nClass coverage check:")
    all_classes = set()
    classes_per_split = {}
    for split_name, families in splits.items():
        classes = set(f.gloss for f in families)
        classes_per_split[split_name] = classes
        all_classes.update(classes)

    # Find classes missing from any split
    for split_name, classes in classes_per_split.items():
        missing = all_classes - classes
        if missing:
            print(f"  {split_name}: Missing {len(missing)} classes: {list(missing)[:5]}...")
        else:
            print(f"  {split_name}: All {len(classes)} classes present")

    print("=" * 70 + "\n")


def run_stratified_split(
    input_dir: Path,
    output_dir: Path,
    split_ratios: Dict[str, float] = None,
    seed: int = 42,
    dry_run: bool = False,
) -> Tuple[Dict, Dict]:
    """
    Run the complete stratified family-based splitting process.

    Args:
        input_dir: Directory containing augmented pose data
        output_dir: Output directory for splits
        split_ratios: Custom split ratios (optional)
        seed: Random seed
        dry_run: If True, don't copy files

    Returns:
        Tuple of (splits dict, manifest dict)
    """
    print("=" * 70)
    print("STRATIFIED FAMILY-BASED DATASET SPLITTING")
    print("=" * 70)
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Dry run: {dry_run}")
    print(f"Random seed: {seed}")

    # Step 1: Scan and group into families
    print("\n[Step 1] Scanning dataset and grouping into families...")
    families_by_class = scan_augmented_dataset(input_dir)

    total_families = sum(len(v) for v in families_by_class.values())
    total_classes = len(families_by_class)
    print(f"  Found {total_families} families across {total_classes} classes")

    # Step 2: Perform stratified split
    print("\n[Step 2] Performing stratified split...")
    splits = stratified_split_families(
        families_by_class,
        split_ratios=split_ratios,
        seed=seed,
    )

    for split_name, families in splits.items():
        n_samples = sum(f.total_samples for f in families)
        print(f"  {split_name}: {len(families)} families, {n_samples} samples")

    # Step 3: Copy files to split directories
    print("\n[Step 3] Copying files to split directories...")

    if dry_run:
        print("  (Dry run - no files will be copied)")

    for split_name, families in splits.items():
        files_copied = 0
        for family in families:
            files_copied += copy_family_to_split(
                family, split_name, output_dir, dry_run=dry_run
            )

        action = "would copy" if dry_run else "copied"
        print(f"  {split_name}: {action} {files_copied} files")

    # Step 4: Generate and save manifest
    print("\n[Step 4] Generating manifest...")
    manifest = generate_split_manifest(splits, output_dir)

    manifest_path = output_dir / "split_manifest.json"
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"  Saved manifest to: {manifest_path}")
    else:
        print(f"  Would save manifest to: {manifest_path}")

    # Print summary
    print_split_summary(splits, manifest)

    return splits, manifest


def main():
    parser = argparse.ArgumentParser(
        description="Stratified family-based dataset splitting"
    )
    parser.add_argument(
        '--input-dir', type=Path, required=True,
        help='Directory containing augmented pose data (class folders)'
    )
    parser.add_argument(
        '--output-dir', type=Path, required=True,
        help='Output directory for train/val/test splits'
    )
    parser.add_argument(
        '--train-ratio', type=float, default=0.70,
        help='Training split ratio (default: 0.70)'
    )
    parser.add_argument(
        '--val-ratio', type=float, default=0.15,
        help='Validation split ratio (default: 0.15)'
    )
    parser.add_argument(
        '--test-ratio', type=float, default=0.15,
        help='Test split ratio (default: 0.15)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print what would happen without copying files'
    )

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Warning: Split ratios sum to {total_ratio}, not 1.0")

    split_ratios = {
        'train': args.train_ratio,
        'val': args.val_ratio,
        'test': args.test_ratio,
    }

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Run splitting
    splits, manifest = run_stratified_split(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        split_ratios=split_ratios,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("Dry run complete. Use without --dry-run to perform actual splitting.")
    else:
        print("Splitting complete!")


if __name__ == "__main__":
    main()
