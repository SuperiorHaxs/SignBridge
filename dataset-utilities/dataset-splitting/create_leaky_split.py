#!/usr/bin/env python3
"""
Create a deliberately leaky train/val/test split for comparison testing.

Splits individual pickle files randomly across splits WITHOUT keeping
augmentation families together. This mimics the old signer-aware splitter's
behavior where augmented files (e.g., 17013_aug_05.pkl) were treated as
independent samples and could end up in different splits than their source.

Purpose: Compare with stratified family split to measure the accuracy
inflation caused by data leakage from augmented copies in validation.

Usage:
    python create_leaky_split.py --source-manifest datasets/augmented_pool/splits/43_classes
    python create_leaky_split.py --source-manifest datasets/augmented_pool/splits/43_classes --seed 42
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict


def create_leaky_split(source_dir, output_dir=None, seed=42, train_ratio=0.7, val_ratio=0.15):
    """Create a leaky split from an existing family-based manifest."""
    source_dir = Path(source_dir)
    if output_dir is None:
        output_dir = source_dir.parent / f"{source_dir.name}_leaky"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the existing split manifest to get the pickle pool path and all families
    split_manifest_path = source_dir / "split_manifest.json"
    with open(split_manifest_path) as f:
        split_manifest = json.load(f)

    # Collect ALL files across all splits
    all_files_by_class = defaultdict(list)
    for split_name in ('train', 'val', 'test'):
        families = split_manifest['splits'][split_name]['families']
        for fam in families:
            gloss = fam['gloss']
            orig = fam['original_file']
            all_files_by_class[gloss].append(orig)
            # Add augmented files
            base = orig.replace('.pkl', '')
            for i in range(fam['augmented_count']):
                all_files_by_class[gloss].append(f"{base}_aug_{i:02d}.pkl")

    # Now randomly split ALL files per class (leaky - augmented copies scatter)
    random.seed(seed)
    train_manifest = {"classes": {}}
    val_manifest = {"classes": {}}
    test_manifest = {"classes": {}}

    total_train = total_val = total_test = 0

    for gloss in sorted(all_files_by_class.keys()):
        files = all_files_by_class[gloss]
        random.shuffle(files)

        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        # Wrap each file as a single-file "family" for manifest compatibility
        train_manifest["classes"][gloss] = [{"files": [f]} for f in train_files]
        val_manifest["classes"][gloss] = [{"files": [f]} for f in val_files]
        test_manifest["classes"][gloss] = [{"files": [f]} for f in test_files]

        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)

    # Find the pickle pool path from existing manifests
    with open(source_dir / "train_manifest.json") as f:
        orig_train = json.load(f)
    pickle_pool = orig_train.get("pickle_pool", str(source_dir.parent.parent / "pickle"))

    for manifest, name, count in [
        (train_manifest, "train", total_train),
        (val_manifest, "val", total_val),
        (test_manifest, "test", total_test),
    ]:
        manifest["pickle_pool"] = pickle_pool
        manifest["total_samples"] = count
        manifest["description"] = f"LEAKY split ({name}) - augmented files scattered across splits for comparison testing"
        out_path = output_dir / f"{name}_manifest.json"
        with open(out_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    # Count leakage
    train_sources = set()
    val_sources = set()
    for gloss, families in train_manifest["classes"].items():
        for fam in families:
            for f in fam["files"]:
                train_sources.add((gloss, f.split("_aug_")[0].replace(".pkl", "")))
    for gloss, families in val_manifest["classes"].items():
        for fam in families:
            for f in fam["files"]:
                val_sources.add((gloss, f.split("_aug_")[0].replace(".pkl", "")))

    overlap = train_sources & val_sources
    leakage_pct = len(overlap) / len(val_sources) * 100 if val_sources else 0

    print(f"Leaky split created at: {output_dir}")
    print(f"  Train: {total_train} files")
    print(f"  Val:   {total_val} files")
    print(f"  Test:  {total_test} files")
    print(f"  Source video overlap (train & val): {len(overlap)}/{len(val_sources)} ({leakage_pct:.0f}%)")
    print(f"\nTo train: python models/training-scripts/train_asl.py --manifest-dir {output_dir} --num-classes 43")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create deliberately leaky split for comparison')
    parser.add_argument('--source-manifest', type=str, required=True,
                        help='Path to existing family-based split directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: <source>_leaky)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    create_leaky_split(args.source_manifest, args.output_dir, args.seed)
