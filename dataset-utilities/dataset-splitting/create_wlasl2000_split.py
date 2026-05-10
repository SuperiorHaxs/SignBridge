#!/usr/bin/env python3
"""
Create train/val/test splits and manifests for the full WLASL2000 dataset
using original (non-augmented) pickle files only.

Produces manifests compatible with train_asl.py's load_from_manifest().

Strategy:
  - Classes with >=3 samples: stratified random split (70/15/15)
  - Classes with 2 samples: 1 train, 1 val (no test)
  - Classes with 1 sample: 1 train only

Usage:
    python create_wlasl2000_split.py
    python create_wlasl2000_split.py --seed 42 --output-dir path/to/output
"""

import argparse
import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAPPING_PATH = PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "video_to_gloss_mapping.json"
PICKLE_DIR = PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "pickle_files"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "splits" / "2000_classes"


def get_frame_count(pkl_path):
    """Get frame count from a pickle file for bucket assignment."""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        if hasattr(data, 'shape'):
            return data.shape[0]
        elif isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            # Try common keys
            for key in ['poses', 'keypoints', 'data', 'frames']:
                if key in data and hasattr(data[key], 'shape'):
                    return data[key].shape[0]
                elif key in data and isinstance(data[key], list):
                    return len(data[key])
        return 50  # default to medium bucket
    except Exception:
        return 50


def get_frame_bucket(frame_count):
    """Assign frame bucket for manifest compatibility."""
    if frame_count < 50:
        return "short"
    elif frame_count < 80:
        return "medium"
    else:
        return "long"


def create_splits(mapping_path, pickle_dir, output_dir, seed=42):
    """Create stratified train/val/test splits for all 2000 classes."""
    random.seed(seed)

    # Load mapping
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)

    # Group video IDs by gloss (only those with pickle files)
    gloss_to_videos = defaultdict(list)
    for vid_id, info in mapping.items():
        pkl_path = Path(pickle_dir) / f"{vid_id}.pkl"
        if pkl_path.exists():
            gloss_to_videos[info['gloss'].lower()].append(vid_id)

    print(f"Found {len(gloss_to_videos)} glosses with pickle data")
    print(f"Total samples: {sum(len(v) for v in gloss_to_videos.values())}")

    # Split each class
    train_classes = defaultdict(list)
    val_classes = defaultdict(list)
    test_classes = defaultdict(list)

    split_stats = {"3+": 0, "2": 0, "1": 0}

    for gloss, vid_ids in sorted(gloss_to_videos.items()):
        random.shuffle(vid_ids)
        n = len(vid_ids)

        if n >= 3:
            # Standard 70/15/15 split
            n_val = max(1, round(n * 0.15))
            n_test = max(1, round(n * 0.15))
            n_train = n - n_val - n_test

            # Ensure at least 1 in train
            if n_train < 1:
                n_train = 1
                n_val = max(1, (n - 1) // 2)
                n_test = n - 1 - n_val

            train_classes[gloss] = vid_ids[:n_train]
            val_classes[gloss] = vid_ids[n_train:n_train + n_val]
            test_classes[gloss] = vid_ids[n_train + n_val:]
            split_stats["3+"] += 1
        elif n == 2:
            train_classes[gloss] = [vid_ids[0]]
            val_classes[gloss] = [vid_ids[1]]
            # No test sample for 2-sample classes
            split_stats["2"] += 1
        else:  # n == 1
            train_classes[gloss] = [vid_ids[0]]
            split_stats["1"] += 1

    print(f"\nSplit strategy:")
    print(f"  Classes with 3+ samples (full split): {split_stats['3+']}")
    print(f"  Classes with 2 samples (train+val):   {split_stats['2']}")
    print(f"  Classes with 1 sample (train only):   {split_stats['1']}")

    # Build manifests
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create class index mapping (alphabetical order)
    all_glosses = sorted(gloss_to_videos.keys())
    class_mapping = {gloss.upper(): idx for idx, gloss in enumerate(all_glosses)}

    # Save class mapping
    mapping_out = output_dir / "class_index_mapping.json"
    with open(mapping_out, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print(f"\nSaved class mapping: {mapping_out}")

    # Also save as class_mapping.json for compatibility
    class_mapping_compat = {
        "num_classes": len(all_glosses),
        "class_to_index": class_mapping,
        "index_to_class": {v: k for k, v in class_mapping.items()}
    }
    with open(output_dir / "class_mapping.json", 'w') as f:
        json.dump(class_mapping_compat, f, indent=2)

    # Generate manifests for each split
    for split_name, split_classes in [("train", train_classes), ("val", val_classes), ("test", test_classes)]:
        # Use relative path from manifest location to pickle_files dir
        pickle_pool_relative = os.path.relpath(pickle_dir, output_dir)
        manifest = {
            "split": split_name,
            "pickle_pool": pickle_pool_relative.replace("\\", "/"),
            "pool_type": "originals_flat",  # Flag: all pickles in one flat dir, not gloss subdirs
            "total_samples": sum(len(vids) for vids in split_classes.values()),
            "total_classes": len(split_classes),
            "classes": {}
        }

        for gloss in all_glosses:
            vid_ids = split_classes.get(gloss, [])
            if not vid_ids:
                continue

            families = []
            for vid_id in vid_ids:
                pkl_path = Path(pickle_dir) / f"{vid_id}.pkl"
                frame_count = get_frame_count(str(pkl_path))

                families.append({
                    "video_id": vid_id,
                    "frame_bucket": get_frame_bucket(frame_count),
                    "files": [f"{vid_id}.pkl"]
                })

            manifest["classes"][gloss] = families

        manifest_path = output_dir / f"{split_name}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"  {split_name}: {manifest['total_samples']} samples across {manifest['total_classes']} classes -> {manifest_path}")

    # Print summary
    train_total = sum(len(v) for v in train_classes.values())
    val_total = sum(len(v) for v in val_classes.values())
    test_total = sum(len(v) for v in test_classes.values())

    print(f"\n{'='*50}")
    print(f"WLASL2000 Split Summary")
    print(f"{'='*50}")
    print(f"  Classes:  {len(all_glosses)}")
    print(f"  Train:    {train_total} samples ({train_total/11980*100:.1f}%)")
    print(f"  Val:      {val_total} samples ({val_total/11980*100:.1f}%)")
    print(f"  Test:     {test_total} samples ({test_total/11980*100:.1f}%)")
    print(f"  Total:    {train_total + val_total + test_total}")
    print(f"  Output:   {output_dir}")

    # Classes with no val/test coverage
    no_val = len(all_glosses) - len(val_classes)
    no_test = len(all_glosses) - len(test_classes)
    if no_val > 0:
        print(f"\n  Warning: {no_val} classes have no validation sample")
    if no_test > 0:
        print(f"  Warning: {no_test} classes have no test sample")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Create WLASL2000 train/val/test splits")
    parser.add_argument("--mapping", type=str, default=str(MAPPING_PATH),
                        help="Path to video_to_gloss_mapping.json")
    parser.add_argument("--pickle-dir", type=str, default=str(PICKLE_DIR),
                        help="Path to pickle_files directory")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output directory for manifests")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits")
    args = parser.parse_args()

    create_splits(args.mapping, args.pickle_dir, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
