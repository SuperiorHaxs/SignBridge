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
    guarantee_class_coverage: bool = True,
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
        guarantee_class_coverage: If True, ensures every class has at least
            one family in each split (train/val/test) before distributing rest

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

    # Track which classes have coverage in each split
    class_coverage = {
        'train': set(),
        'val': set(),
        'test': set(),
    }

    # Process each class
    for class_name in sorted(families_by_class.keys()):
        class_families = list(families_by_class[class_name].values())

        # Shuffle all families for this class
        np.random.shuffle(class_families)

        n_total = len(class_families)

        if guarantee_class_coverage and n_total >= 3:
            # PHASE 1: Guarantee at least one family per split for this class
            # Take first 3 families and assign one to each split
            splits['train'].append(class_families[0])
            splits['val'].append(class_families[1])
            splits['test'].append(class_families[2])
            class_coverage['train'].add(class_name)
            class_coverage['val'].add(class_name)
            class_coverage['test'].add(class_name)

            # PHASE 2: Distribute remaining families by ratio
            remaining_families = class_families[3:]
            if remaining_families:
                # Group remaining by frame bucket for stratified distribution
                by_bucket = defaultdict(list)
                for family in remaining_families:
                    by_bucket[family.frame_bucket].append(family)

                for bucket_name, bucket_families in by_bucket.items():
                    np.random.shuffle(bucket_families)
                    n_bucket = len(bucket_families)

                    # Calculate proportional split - train gets remainder (most important)
                    n_val = int(n_bucket * split_ratios['val'])
                    n_test = int(n_bucket * split_ratios['test'])
                    n_train = n_bucket - n_val - n_test  # Train gets remainder

                    splits['train'].extend(bucket_families[:n_train])
                    splits['val'].extend(bucket_families[n_train:n_train + n_val])
                    splits['test'].extend(bucket_families[n_train + n_val:])

        elif n_total == 2:
            # Only 2 families: prioritize train and val coverage
            splits['train'].append(class_families[0])
            splits['val'].append(class_families[1])
            class_coverage['train'].add(class_name)
            class_coverage['val'].add(class_name)
            # Note: test will be missing this class

        elif n_total == 1:
            # Only 1 family: must go to train (most important)
            splits['train'].append(class_families[0])
            class_coverage['train'].add(class_name)
            # Note: val and test will be missing this class

        else:
            # No guarantee mode or fallback: use original bucket-based logic
            by_bucket = defaultdict(list)
            for family in class_families:
                by_bucket[family.frame_bucket].append(family)

            for bucket_name, bucket_families in by_bucket.items():
                np.random.shuffle(bucket_families)

                n_bucket = len(bucket_families)

                if n_bucket == 1:
                    splits['train'].append(bucket_families[0])
                elif n_bucket == 2:
                    splits['train'].append(bucket_families[0])
                    splits['val'].append(bucket_families[1])
                else:
                    # Train gets remainder (most important)
                    n_val = max(0, int(n_bucket * split_ratios['val']))
                    n_test = max(0, int(n_bucket * split_ratios['test']))
                    n_train = n_bucket - n_val - n_test

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


def generate_split_manifests(
    splits: Dict[str, List[PoseFamily]],
    output_dir: Path,
    pickle_pool_dir: Path,
) -> Dict[str, Path]:
    """
    Generate manifest files for each split (train/val/test).

    Instead of copying files, manifests contain paths to files in the
    central augmented_pool/pickle/ directory.

    Args:
        splits: Dict mapping split_name -> List[PoseFamily]
        output_dir: Directory to save manifest files
        pickle_pool_dir: Path to augmented_pool/pickle/ (for relative paths)

    Returns:
        Dict mapping split_name -> manifest_path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = {}

    for split_name, families in splits.items():
        manifest = {
            'split': split_name,
            'pickle_pool': str(pickle_pool_dir),
            'total_samples': sum(f.total_samples for f in families),
            'total_families': len(families),
            'classes': {},
        }

        # Group by class
        for family in families:
            gloss = family.gloss
            if gloss not in manifest['classes']:
                manifest['classes'][gloss] = []

            # Store relative paths from pickle_pool
            family_entry = {
                'video_id': family.video_id,
                'frame_bucket': family.frame_bucket,
                'files': [p.name for p in family.all_paths],  # Just filenames
            }
            manifest['classes'][gloss].append(family_entry)

        # Add class count
        manifest['num_classes'] = len(manifest['classes'])

        # Save manifest
        manifest_path = output_dir / f"{split_name}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        manifest_paths[split_name] = manifest_path

    return manifest_paths


def run_stratified_split(
    input_dir: Path,
    output_dir: Path,
    split_ratios: Dict[str, float] = None,
    seed: int = 42,
    dry_run: bool = False,
    use_manifests: bool = True,
    gloss_list: List[str] = None,
) -> Tuple[Dict, Dict]:
    """
    Run the complete stratified family-based splitting process.

    Args:
        input_dir: Directory containing augmented pose data (augmented_pool/pickle/)
        output_dir: Output directory for splits/manifests
        split_ratios: Custom split ratios (optional)
        seed: Random seed
        dry_run: If True, don't create files
        use_manifests: If True, generate manifests instead of copying files
        gloss_list: Optional list of glosses to include (filters to only these classes)

    Returns:
        Tuple of (splits dict, manifest dict)
    """
    print("=" * 70)
    print("STRATIFIED FAMILY-BASED DATASET SPLITTING")
    print("=" * 70)
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {'Manifest-based (no file copying)' if use_manifests else 'Copy files'}")
    print(f"Dry run: {dry_run}")
    print(f"Random seed: {seed}")
    if gloss_list:
        print(f"Filtering to {len(gloss_list)} classes from gloss_list")

    # Step 1: Scan and group into families
    print("\n[Step 1] Scanning dataset and grouping into families...")
    families_by_class = scan_augmented_dataset(input_dir)

    # Filter to only classes in gloss_list if provided
    if gloss_list:
        gloss_set = set(g.lower() for g in gloss_list)
        original_count = len(families_by_class)
        families_by_class = {k: v for k, v in families_by_class.items() if k.lower() in gloss_set}
        filtered_count = len(families_by_class)
        print(f"  Filtered from {original_count} to {filtered_count} classes (gloss_list)")

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

    # Step 3: Generate manifests or copy files
    if use_manifests:
        print("\n[Step 3] Generating split manifests...")
        if not dry_run:
            manifest_paths = generate_split_manifests(splits, output_dir, input_dir)
            for split_name, path in manifest_paths.items():
                print(f"  {split_name}: {path}")
        else:
            print("  (Dry run - no manifests created)")
    else:
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

    # Step 4: Generate and save overall manifest
    print("\n[Step 4] Generating summary manifest...")
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


# =============================================================================
# PHASE 2: TRAIN BALANCING
# =============================================================================

def analyze_train_split(
    splits: Dict[str, List[PoseFamily]],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Analyze the train split to get sample counts and family counts per class.

    Args:
        splits: Dict mapping split_name -> List[PoseFamily]

    Returns:
        Tuple of (train_samples_per_class, train_families_per_class)
    """
    train_families = splits.get('train', [])

    samples_per_class = {}
    families_per_class = {}

    for family in train_families:
        gloss = family.gloss
        if gloss not in samples_per_class:
            samples_per_class[gloss] = 0
            families_per_class[gloss] = 0

        samples_per_class[gloss] += family.total_samples
        families_per_class[gloss] += 1

    return samples_per_class, families_per_class


def balance_train_split(
    splits: Dict[str, List[PoseFamily]],
    pool_dir: Path,
    output_dir: Path,
    target_train_samples: int = 200,
    dry_run: bool = False,
    landmark_config: str = '83pt',
    include_z: bool = True,
) -> Dict[str, Dict]:
    """
    Balance train split by generating additional augmentations for under-represented classes.

    This is Phase 2 of the two-phase approach:
    - Phase 1 created base pool with moderate augmentations
    - Phase 2 (this) generates additional augmentations ONLY for train families
      to ensure each class has exactly target_train_samples in train

    Files are saved to the pool directory (alongside base augmentations) and
    the train manifest is updated to include them.

    Args:
        splits: Dict from stratified_split_families()
        pool_dir: Directory containing the augmented pool (where new files are added)
        output_dir: Directory where manifests are stored
        target_train_samples: Target samples per class in train
        dry_run: If True, calculate but don't generate files
        landmark_config: Landmark extraction configuration
        include_z: Whether to include z-coordinate (default True for 3D poses)

    Returns:
        Balancing summary dict with per-class statistics
    """
    print("\n" + "=" * 70)
    print("PHASE 2: TRAIN SPLIT BALANCING")
    print("=" * 70)
    print(f"\nTarget train samples per class: {target_train_samples}")

    # Import augmentation functions
    sys.path.insert(0, str(script_dir.parent / 'augmentation'))
    from augmentation_config import (
        calculate_train_balancing_augmentations,
        BASE_AUGMENTATIONS_PER_VIDEO,
        MAX_TOTAL_AUGMENTATIONS_PER_VIDEO,
        get_augmentation_sequence,
    )
    from generate_augmented_dataset import (
        generate_augmented_sample,
        load_pose_file_as_keypoints,
    )

    # Analyze current train split
    train_samples, train_families = analyze_train_split(splits)
    print(f"\nCurrent train split analysis:")
    print(f"  Classes: {len(train_samples)}")
    print(f"  Total samples: {sum(train_samples.values())}")
    print(f"  Sample range: {min(train_samples.values())} - {max(train_samples.values())}")

    # Cap classes that exceed the target (subsample to target)
    classes_over_target = [c for c, count in train_samples.items() if count > target_train_samples]
    if classes_over_target:
        print(f"\n  Classes over target ({target_train_samples}): {len(classes_over_target)}")
        print(f"  Will cap these to ~{target_train_samples} samples each")

    # Calculate balancing plan
    balancing_plan = calculate_train_balancing_augmentations(
        train_class_samples=train_samples,
        train_families=train_families,
        target_train_samples=target_train_samples,
        max_total_per_video=MAX_TOTAL_AUGMENTATIONS_PER_VIDEO,
        existing_augs_per_video=BASE_AUGMENTATIONS_PER_VIDEO,
    )

    # Summarize balancing plan
    classes_needing_balance = [c for c, p in balancing_plan.items() if p['samples_needed'] > 0]
    total_additional_samples = sum(
        p['additional_augs_per_video'] * p.get('train_families', 0)
        for p in balancing_plan.values()
    )

    print(f"\nBalancing plan:")
    print(f"  Classes needing balancing: {len(classes_needing_balance)}")
    print(f"  Additional samples to generate: {total_additional_samples}")

    if dry_run:
        print("\n[DRY RUN] Would generate additional augmentations:")
        for class_name in sorted(classes_needing_balance)[:5]:
            plan = balancing_plan[class_name]
            print(f"  {class_name}: {plan['current_samples']} -> {plan['final_samples']} "
                  f"(+{plan['additional_augs_per_video']}/family × {plan['train_families']} families)")
        if len(classes_needing_balance) > 5:
            print(f"  ... and {len(classes_needing_balance) - 5} more classes")
        return {'balancing_plan': balancing_plan, 'new_files_by_class': {}}

    # Track new files generated for manifest update
    new_files_by_class = defaultdict(lambda: defaultdict(list))  # class -> video_id -> [files]

    # Generate additional augmentations for train families
    print("\nGenerating additional augmentations for train...")

    total_generated = 0

    for family in splits['train']:
        gloss = family.gloss
        plan = balancing_plan.get(gloss, {})
        additional_per_video = plan.get('additional_augs_per_video', 0)

        if additional_per_video <= 0:
            continue

        # Load original to generate more augmentations
        try:
            # Determine source format
            original_path = family.original_path
            if original_path.suffix == '.pose':
                keypoints, metadata = load_pose_file_as_keypoints(original_path)
                original_data = {
                    'keypoints': keypoints,
                    'video_id': metadata['video_id'],
                    'gloss': metadata['gloss'],
                }
            else:
                with open(original_path, 'rb') as f:
                    original_data = pickle.load(f)

            # Get existing augmentation count to continue numbering
            existing_aug_count = len(family.augmented_paths)
            start_aug_id = existing_aug_count

            # Generate augmentation sequence
            aug_sequence = get_augmentation_sequence(additional_per_video)

            # Output directory: same pool directory structure (class folder)
            class_output_dir = pool_dir / gloss
            class_output_dir.mkdir(parents=True, exist_ok=True)

            for i, aug_config in enumerate(aug_sequence):
                aug_id = start_aug_id + i
                aug_config['id'] = aug_id

                aug_sample = generate_augmented_sample(
                    original_data,
                    aug_config,
                    landmark_config,
                    include_z=include_z,
                )

                # Save with unique ID (continuing from existing), marked as balance aug
                video_id = original_data.get('video_id', family.video_id)
                aug_filename = f"{video_id}_aug{aug_id:02d}_balance.pkl"
                aug_path = class_output_dir / aug_filename

                with open(aug_path, 'wb') as f:
                    pickle.dump(aug_sample, f)

                # Track for manifest update
                new_files_by_class[gloss][video_id].append(aug_filename)
                total_generated += 1

        except Exception as e:
            print(f"  Warning: Error balancing {family.video_id}: {e}")

    # Update train manifest with new files
    print("\nUpdating train manifest with balancing augmentations...")
    train_manifest_path = output_dir / "train_manifest.json"

    if train_manifest_path.exists():
        with open(train_manifest_path, 'r') as f:
            train_manifest = json.load(f)

        # Add new files to each family in the manifest
        files_added_to_manifest = 0
        for gloss, families in train_manifest.get('classes', {}).items():
            for family_entry in families:
                video_id = family_entry.get('video_id')
                if video_id and gloss in new_files_by_class:
                    new_files = new_files_by_class[gloss].get(video_id, [])
                    if new_files:
                        family_entry['files'].extend(new_files)
                        files_added_to_manifest += len(new_files)

        # Cap classes that exceed target (subsample to target_train_samples)
        import random
        random.seed(42)  # Reproducible subsampling
        files_removed = 0
        for gloss, families in train_manifest.get('classes', {}).items():
            total_class_samples = sum(len(fam['files']) for fam in families)
            if total_class_samples > target_train_samples:
                # Collect originals (first file of each family) and augmented files separately
                originals_by_family = {}  # fam_idx -> original file
                augmented_files = []  # list of (fam_idx, filename)

                for fam_idx, family_entry in enumerate(families):
                    if family_entry['files']:
                        # First file is original - ALWAYS keep it
                        originals_by_family[fam_idx] = family_entry['files'][0]
                        # Rest are augmented
                        for f in family_entry['files'][1:]:
                            augmented_files.append((fam_idx, f))

                num_originals = len(originals_by_family)
                # How many augmented files can we keep to hit target?
                aug_to_keep = max(0, target_train_samples - num_originals)

                # Shuffle and select augmented files to keep
                random.shuffle(augmented_files)
                kept_augmented = augmented_files[:aug_to_keep]

                # Count removed
                files_removed += len(augmented_files) - len(kept_augmented)

                # Rebuild family file lists: original + selected augmented
                for fam_idx, family_entry in enumerate(families):
                    orig = originals_by_family.get(fam_idx)
                    augs = [f for fi, f in kept_augmented if fi == fam_idx]
                    family_entry['files'] = ([orig] if orig else []) + augs

        if files_removed > 0:
            print(f"  Capped overrepresented classes: removed {files_removed} samples")

        # Update total samples count
        train_manifest['total_samples'] = sum(
            len(fam['files'])
            for families in train_manifest['classes'].values()
            for fam in families
        )

        # Save updated manifest
        with open(train_manifest_path, 'w') as f:
            json.dump(train_manifest, f, indent=2)

        print(f"  Added {files_added_to_manifest} files to train manifest")
    else:
        print(f"  Warning: Train manifest not found at {train_manifest_path}")

    # Calculate final statistics (recalculate from actual manifest after capping)
    final_stats = {}
    if train_manifest_path.exists():
        with open(train_manifest_path, 'r') as f:
            final_manifest = json.load(f)
        for gloss, families in final_manifest.get('classes', {}).items():
            actual_samples = sum(len(fam['files']) for fam in families)
            original_count = balancing_plan.get(gloss, {}).get('current_samples', actual_samples)
            final_stats[gloss] = {
                'before': original_count,
                'after': actual_samples,
                'target': target_train_samples,
                'achieved': actual_samples >= target_train_samples,
            }
    else:
        # Fallback to balancing plan
        for class_name, plan in balancing_plan.items():
            final_stats[class_name] = {
                'before': plan['current_samples'],
                'after': min(plan['final_samples'], target_train_samples),
                'target': target_train_samples,
                'achieved': plan['final_samples'] >= target_train_samples,
            }

    achieved_count = sum(1 for s in final_stats.values() if s['achieved'])
    final_samples = [s['after'] for s in final_stats.values()]
    final_min = min(final_samples) if final_samples else 0
    final_max = max(final_samples) if final_samples else 0

    print(f"\nBalancing complete!")
    print(f"  Additional samples generated: {total_generated}")
    print(f"  Classes achieving target: {achieved_count}/{len(final_stats)}")
    print(f"  Final sample range: {final_min} - {final_max}")

    return {
        'balancing_plan': balancing_plan,
        'final_stats': final_stats,
        'total_generated': total_generated,
        'achieved_count': achieved_count,
        'new_files_by_class': dict(new_files_by_class),
    }


def run_two_phase_pipeline(
    input_dir: Path,
    output_dir: Path,
    target_train_samples: int = 200,
    split_ratios: Dict[str, float] = None,
    seed: int = 42,
    dry_run: bool = False,
    landmark_config: str = '83pt',
    include_z: bool = True,
    gloss_list: List[str] = None,
) -> Dict:
    """
    Run complete two-phase pipeline: split then balance train.

    Phase 1: Stratified family-based splitting (manifest-based, no file copying)
    Phase 2: Balance train split by generating additional augmentations in pool
             and updating train manifest

    Args:
        input_dir: Directory containing augmented pose data (the pool)
        output_dir: Output directory for manifests
        target_train_samples: Target samples per class in train
        split_ratios: Custom split ratios
        seed: Random seed
        dry_run: If True, don't generate files
        landmark_config: Landmark extraction config
        include_z: Whether to include z-coordinate (default True for 3D poses)
        gloss_list: Optional list of glosses to include (filters to only these classes)

    Returns:
        Complete pipeline results
    """
    print("=" * 70)
    print("TWO-PHASE BALANCED SPLITTING PIPELINE")
    print("=" * 70)
    print(f"\nPhase 1: Stratified family-based splitting (manifest-based)")
    print(f"Phase 2: Balance train to {target_train_samples} samples/class")
    print(f"         (new files added to pool, manifest updated)")
    if gloss_list:
        print(f"         (filtering to {len(gloss_list)} classes from gloss_list)")

    # Phase 1: Stratified split with manifests (no file copying)
    splits, manifest = run_stratified_split(
        input_dir=input_dir,
        output_dir=output_dir,
        split_ratios=split_ratios,
        seed=seed,
        dry_run=dry_run,
        use_manifests=True,  # Manifest-based, no file copying
        gloss_list=gloss_list,
    )

    # Phase 2: Balance train (generates files in pool, updates train manifest)
    balancing_results = balance_train_split(
        splits=splits,
        pool_dir=input_dir,  # New files go into the pool
        output_dir=output_dir,  # Manifests are here
        target_train_samples=target_train_samples,
        dry_run=dry_run,
        landmark_config=landmark_config,
        include_z=include_z,
    )

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    if not dry_run:
        final_stats = balancing_results.get('final_stats', {})
        if final_stats:
            samples = [s['after'] for s in final_stats.values()]
            print(f"\nFinal train samples per class:")
            print(f"  Min: {min(samples)}")
            print(f"  Max: {max(samples)}")
            print(f"  Target: {target_train_samples}")

    return {
        'splits': splits,
        'manifest': manifest,
        'balancing': balancing_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stratified family-based dataset splitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic splitting (no train balancing)
    python stratified_family_split.py --input-dir /path/to/pool --output-dir /path/to/splits

    # Two-phase pipeline: split + balance train to 200 samples/class
    python stratified_family_split.py --input-dir /path/to/pool --output-dir /path/to/splits --balance-train

    # Custom train target
    python stratified_family_split.py --input-dir /path/to/pool --output-dir /path/to/splits --balance-train --train-target 150

    # Dry run to see plan
    python stratified_family_split.py --input-dir /path/to/pool --output-dir /path/to/splits --balance-train --dry-run
        """
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
        help='Print what would happen without creating files'
    )
    parser.add_argument(
        '--copy-files', action='store_true',
        help='Copy files instead of generating manifests (legacy mode)'
    )
    parser.add_argument(
        '--balance-train', action='store_true',
        help='Run two-phase pipeline: split then balance train to target samples/class'
    )
    parser.add_argument(
        '--train-target', type=int, default=200,
        help='Target samples per class in train split (default: 200, requires --balance-train)'
    )
    parser.add_argument(
        '--landmark-config', type=str, default='83pt',
        help='Landmark config for additional augmentations (default: 83pt)'
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

    if args.balance_train:
        # Two-phase pipeline
        results = run_two_phase_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_train_samples=args.train_target,
            split_ratios=split_ratios,
            seed=args.seed,
            dry_run=args.dry_run,
            landmark_config=args.landmark_config,
        )
        if args.dry_run:
            print("\nDry run complete. Use without --dry-run to execute.")
    else:
        # Standard splitting only
        splits, manifest = run_stratified_split(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            split_ratios=split_ratios,
            seed=args.seed,
            dry_run=args.dry_run,
            use_manifests=not args.copy_files,
        )

        if args.dry_run:
            print("Dry run complete. Use without --dry-run to perform actual splitting.")
        else:
            mode = "file copying" if args.copy_files else "manifest generation"
            print(f"Splitting complete! (Mode: {mode})")


if __name__ == "__main__":
    main()
