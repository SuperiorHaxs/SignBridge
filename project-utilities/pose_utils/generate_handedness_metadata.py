#!/usr/bin/env python3
"""
generate_handedness_metadata.py — Analyze training pose samples and derive
per-gloss handedness metadata (1-handed vs 2-handed) for runtime inference.

For each gloss:
  - Scan all training pickle files
  - Compute average left-hand and right-hand detection rates across samples
  - Classify: 1-handed (one side always absent), 2-handed, or ambiguous

Output: signbridge-app/handedness_metadata.json
  {
    "HOSPITAL": {"type": "one_handed", "dominant": "right", "l_hand_rate": 0.05, "r_hand_rate": 0.48, "samples": 12},
    "AUTHORITY": {"type": "two_handed", "l_hand_rate": 0.44, "r_hand_rate": 0.53, "samples": 18},
    ...
  }
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "datasets" / "wlasl_poses_complete"
OUTPUT_PATH = PROJECT_ROOT / "applications" / "signbridge-app" / "handedness_metadata.json"

# Thresholds for classification — tuned from inspection of the dataset
ONE_HANDED_SILENT_RATE = 0.10  # if a hand is detected < 10% across a clip, it's "silent"
TWO_HANDED_MIN_RATE    = 0.25  # both hands > 25% => two-handed
MIN_SAMPLES_REQUIRED   = 3     # need at least N training samples to classify


def analyze_pickle(pkl_path: Path):
    """Return (l_hand_detection_rate, r_hand_detection_rate) for one clip."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    kp = data.get('keypoints')
    if kp is None:
        return None
    if hasattr(kp, 'filled'):
        kp = kp.filled(0)
    kp = np.asarray(kp)

    # Training data is (frames, 576, 2-3). Extract hand landmarks at indices 501-542.
    if kp.shape[1] == 576:
        lh = kp[:, 501:522, :2]  # 21 left hand landmarks
        rh = kp[:, 522:543, :2]  # 21 right hand landmarks
    elif kp.shape[1] == 83:
        lh = kp[:, 33:54, :2]
        rh = kp[:, 54:75, :2]
    elif kp.shape[1] == 75:
        lh = kp[:, 33:54, :2]
        rh = kp[:, 54:75, :2]
    else:
        return None

    # Hand "detected" on a frame if at least one landmark is non-zero
    lh_detected = (np.abs(lh).sum(axis=(1, 2)) > 1e-3).mean()
    rh_detected = (np.abs(rh).sum(axis=(1, 2)) > 1e-3).mean()
    return float(lh_detected), float(rh_detected)


def walk_training_samples():
    """
    Yield (gloss, pkl_path) for every training sample we can find.
    Uses gloss-named directories under dataset_splits/.../train/ as source.
    """
    bases = [
        DATA_ROOT / "dataset_splits" / "2000_classes" / "original" / "pickle_split_2000_class" / "train",
        DATA_ROOT / "dataset_splits" / "100_classes" / "original" / "pickle_split_100_class" / "train",
        DATA_ROOT / "dataset_splits" / "50_classes" / "original" / "pickle_from_pose_split_50_class" / "train",
        DATA_ROOT / "dataset_splits" / "20_classes" / "original" / "pickle_from_pose_split_20_class" / "train",
    ]
    seen = set()
    for base in bases:
        if not base.is_dir():
            continue
        for gloss_dir in base.iterdir():
            if not gloss_dir.is_dir():
                continue
            gloss = gloss_dir.name.upper()
            for pkl in gloss_dir.glob("*.pkl"):
                key = (gloss, pkl.stem)
                if key in seen:
                    continue
                seen.add(key)
                yield gloss, pkl


def main():
    per_gloss = defaultdict(lambda: {'l': [], 'r': []})
    count = 0
    for gloss, pkl_path in walk_training_samples():
        try:
            result = analyze_pickle(pkl_path)
        except Exception as e:
            continue
        if result is None:
            continue
        l_rate, r_rate = result
        per_gloss[gloss]['l'].append(l_rate)
        per_gloss[gloss]['r'].append(r_rate)
        count += 1
        if count % 500 == 0:
            print(f"  ...analyzed {count} samples across {len(per_gloss)} glosses", file=sys.stderr)

    print(f"Analyzed {count} samples across {len(per_gloss)} glosses")

    # Aggregate
    metadata = {}
    classification_counts = {'one_handed': 0, 'two_handed': 0, 'ambiguous': 0, 'insufficient_data': 0}
    for gloss, stats in per_gloss.items():
        n = len(stats['l'])
        l_mean = float(np.mean(stats['l']))
        r_mean = float(np.mean(stats['r']))
        if n < MIN_SAMPLES_REQUIRED:
            classification = 'insufficient_data'
            dominant = None
        else:
            l_silent = l_mean < ONE_HANDED_SILENT_RATE
            r_silent = r_mean < ONE_HANDED_SILENT_RATE
            both_active = l_mean >= TWO_HANDED_MIN_RATE and r_mean >= TWO_HANDED_MIN_RATE
            if l_silent and not r_silent:
                classification = 'one_handed'
                dominant = 'right'
            elif r_silent and not l_silent:
                classification = 'one_handed'
                dominant = 'left'
            elif both_active:
                classification = 'two_handed'
                dominant = 'right' if r_mean >= l_mean else 'left'
            else:
                classification = 'ambiguous'
                dominant = 'right' if r_mean >= l_mean else 'left'
        classification_counts[classification] += 1
        metadata[gloss] = {
            'type': classification,
            'dominant': dominant,
            'l_hand_rate': round(l_mean, 3),
            'r_hand_rate': round(r_mean, 3),
            'samples': n,
        }

    # Sort for readability
    metadata = dict(sorted(metadata.items()))

    # Write
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump({
            'thresholds': {
                'silent_rate_below': ONE_HANDED_SILENT_RATE,
                'both_active_above': TWO_HANDED_MIN_RATE,
                'min_samples': MIN_SAMPLES_REQUIRED,
            },
            'classification_counts': classification_counts,
            'glosses': metadata,
        }, f, indent=2)

    print(f"\nWrote {OUTPUT_PATH}")
    print(f"Classification counts: {classification_counts}")
    print("\nSamples:")
    for sample in ['HOSPITAL', 'ABDOMEN', 'AUTHORITY', 'APPOINTMENT', 'HURRY', 'ACCIDENT']:
        if sample in metadata:
            print(f"  {sample}: {metadata[sample]}")


if __name__ == '__main__':
    main()
