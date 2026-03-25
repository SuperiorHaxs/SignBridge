"""
Test if velocity and hand-detection features can separate confused sign pairs.

Loads pose data from the augmented pickle pool and compares feature distributions
between each pair of confused glosses. Reports whether the features show enough
delta to be useful discriminators.

Usage:
    # With a pairs JSON file:
    python test_velocity_for_confusion.py --model-dir <model> --pairs-json <file>

    # Auto-generate pairs from eval_confusion.py output (top N):
    python test_velocity_for_confusion.py --model-dir <model> --manifest-dir <dir> --auto-pairs --top-n 10

    # Pairs JSON format:
    [
        ["waiter", "sunday"],
        ["breakfast", "delicious"],
        ["chicken", "apple"]
    ]
"""

import sys
import os
import json
import pickle
import argparse
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

sys.path.insert(0, str(PROJECT_ROOT / 'models' / 'openhands-modernized' / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'models' / 'training-scripts'))

from openhands_modernized import WLASLPoseProcessor

PICKLE_POOL = PROJECT_ROOT / 'datasets' / 'augmented_pool' / 'pickle'

# Shared processor for spatial feature extraction
_processor = WLASLPoseProcessor()


def extract_hand_velocity(pose_83):
    """Compute per-frame velocity for left and right hands."""
    left_hand = pose_83[:, 33:54, :2]
    right_hand = pose_83[:, 54:75, :2]

    left_centroid = np.mean(left_hand, axis=1)
    right_centroid = np.mean(right_hand, axis=1)

    left_vel = np.linalg.norm(np.diff(left_centroid, axis=0), axis=1)
    right_vel = np.linalg.norm(np.diff(right_centroid, axis=0), axis=1)

    fingertip_offsets = [4, 8, 12, 16, 20]
    left_tips = left_hand[:, fingertip_offsets, :]
    right_tips = right_hand[:, fingertip_offsets, :]
    left_tip_vel = np.mean([np.linalg.norm(np.diff(left_tips[:, i, :], axis=0), axis=1) for i in range(5)], axis=0)
    right_tip_vel = np.mean([np.linalg.norm(np.diff(right_tips[:, i, :], axis=0), axis=1) for i in range(5)], axis=0)

    return {
        'left_centroid_vel_mean': float(np.mean(left_vel)),
        'left_centroid_vel_max': float(np.max(left_vel)),
        'right_centroid_vel_mean': float(np.mean(right_vel)),
        'right_centroid_vel_max': float(np.max(right_vel)),
        'left_tip_vel_mean': float(np.mean(left_tip_vel)),
        'right_tip_vel_mean': float(np.mean(right_tip_vel)),
        'vel_ratio_LR': float(np.mean(left_vel) / max(np.mean(right_vel), 1e-6)),
    }


def extract_hand_detected_flags(pose_83):
    """Detect which hand is active per frame."""
    left_hand = pose_83[:, 33:54, :2]
    right_hand = pose_83[:, 54:75, :2]

    left_present = np.any(np.abs(left_hand) > 1e-4, axis=(1, 2))
    right_present = np.any(np.abs(right_hand) > 1e-4, axis=(1, 2))

    left_centroid = np.mean(left_hand, axis=1)
    right_centroid = np.mean(right_hand, axis=1)

    left_disp = np.zeros(len(pose_83))
    right_disp = np.zeros(len(pose_83))
    left_disp[1:] = np.linalg.norm(np.diff(left_centroid, axis=0), axis=1)
    right_disp[1:] = np.linalg.norm(np.diff(right_centroid, axis=0), axis=1)

    active_thresh = 0.005
    left_active = left_disp > active_thresh
    right_active = right_disp > active_thresh

    left_var = np.mean(np.var(left_hand, axis=0))
    right_var = np.mean(np.var(right_hand, axis=0))

    return {
        'left_present_pct': float(np.mean(left_present) * 100),
        'right_present_pct': float(np.mean(right_present) * 100),
        'left_active_pct': float(np.mean(left_active) * 100),
        'right_active_pct': float(np.mean(right_active) * 100),
        'left_position_var': float(left_var),
        'right_position_var': float(right_var),
        'both_hands_active_pct': float(np.mean(left_active & right_active) * 100),
        'one_hand_only_pct': float(np.mean(left_active ^ right_active) * 100),
        'handedness': 'both' if np.mean(left_active & right_active) > 0.3 else ('left' if np.mean(left_active) > np.mean(right_active) else 'right'),
    }


def extract_spatial_features_named(pose_83):
    """Extract spatial features using the production pipeline and return as named dict."""
    sf = _processor.extract_spatial_features(pose_83)  # (frames, 40)

    # Trajectory features (averaged over frames)
    trajectory = {
        'rh_direction_x': float(np.mean(sf[:, 0])),
        'rh_direction_y': float(np.mean(sf[:, 1])),
        'lh_direction_x': float(np.mean(sf[:, 2])),
        'lh_direction_y': float(np.mean(sf[:, 3])),
        'rh_circularity': float(np.mean(np.abs(sf[:, 4]))),
        'lh_circularity': float(np.mean(np.abs(sf[:, 5]))),
        'rh_linearity': float(np.mean(sf[:, 6])),
        'lh_linearity': float(np.mean(sf[:, 7])),
        'rh_cumul_path': float(sf[-1, 8]) if len(sf) > 0 else 0,
        'lh_cumul_path': float(sf[-1, 9]) if len(sf) > 0 else 0,
    }

    palm = {
        'rh_palm_nx': float(np.mean(sf[:, 10])),
        'rh_palm_ny': float(np.mean(sf[:, 11])),
        'rh_palm_nz': float(np.mean(sf[:, 12])),
        'lh_palm_nx': float(np.mean(sf[:, 13])),
        'lh_palm_ny': float(np.mean(sf[:, 14])),
        'lh_palm_nz': float(np.mean(sf[:, 15])),
    }

    location = {
        'rh_y_rel_nose': float(np.mean(sf[:, 16])),
        'rh_x_rel_nose': float(np.mean(sf[:, 17])),
        'lh_y_rel_nose': float(np.mean(sf[:, 18])),
        'lh_x_rel_nose': float(np.mean(sf[:, 19])),
        'hands_dist': float(np.mean(sf[:, 20])),
        'vertical_zone': float(np.mean(sf[:, 21])),
    }

    handshape_dyn = {
        'rh_openness_delta': float(np.mean(np.abs(sf[:, 22]))),
        'lh_openness_delta': float(np.mean(np.abs(sf[:, 23]))),
        'rh_spread_velocity': float(np.mean(np.abs(sf[:, 24]))),
        'lh_spread_velocity': float(np.mean(np.abs(sf[:, 25]))),
    }

    face_region = {
        'rh_dist_mouth': float(np.mean(sf[:, 26])),
        'rh_dist_chin': float(np.mean(sf[:, 27])),
        'lh_dist_mouth': float(np.mean(sf[:, 28])),
        'lh_dist_chin': float(np.mean(sf[:, 29])),
    }

    contact = {
        'rh_body_contact': float(np.mean(sf[:, 30])),
        'lh_body_contact': float(np.mean(sf[:, 31])),
    }

    repetition = {
        'rh_rep_density': float(np.mean(sf[:, 32])),
        'lh_rep_density': float(np.mean(sf[:, 33])),
    }

    interaction = {
        'hand_hand_contact': float(np.mean(sf[:, 34])),
        'hand_symmetry': float(np.mean(sf[:, 35])),
        'hands_crossed': float(np.mean(sf[:, 36])),
        'hands_vert_diff': float(np.mean(sf[:, 37])),
    }

    rotation = {
        'rh_wrist_rotation': float(np.mean(np.abs(sf[:, 38]))),
        'lh_wrist_rotation': float(np.mean(np.abs(sf[:, 39]))),
    }

    return trajectory, palm, location, handshape_dyn, face_region, contact, repetition, interaction, rotation


def load_pickle_samples(gloss, max_samples=5):
    """Load up to max_samples original (non-augmented) pickle files for a gloss."""
    gloss_dir = PICKLE_POOL / gloss
    if not gloss_dir.exists():
        print(f"  WARNING: No pickle data for '{gloss}'")
        return []

    # Prefer original (non-augmented) samples for cleaner signal
    originals = sorted([f for f in gloss_dir.glob('*.pkl') if '_aug_' not in f.name])
    if not originals:
        originals = sorted(gloss_dir.glob('*.pkl'))

    poses = []
    for pkl_path in originals[:max_samples]:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        keypoints = data['keypoints']  # shape: (frames, 83, 3)
        if len(keypoints) >= 5:  # skip very short clips
            poses.append(keypoints)

    return poses


def average_features(feature_dicts):
    """Average numeric features across multiple samples."""
    if not feature_dicts:
        return {}
    avg = {}
    for key in feature_dicts[0]:
        vals = [d[key] for d in feature_dicts]
        if isinstance(vals[0], str):
            # For categorical: take most common
            from collections import Counter
            avg[key] = Counter(vals).most_common(1)[0][0]
        else:
            avg[key] = float(np.mean(vals))
    return avg


def print_comparison(label_a, features_a, label_b, features_b, feature_type):
    """Pretty-print comparison between two signs."""
    print(f"\n  {feature_type}:")
    keys = [k for k in features_a if not k.endswith('_timeseries')]
    for key in keys:
        va, vb = features_a[key], features_b[key]
        if isinstance(va, str):
            sep = "** YES **" if va != vb else "no"
            print(f"    {key:<28} {va:>12} {vb:>12}  {sep}")
        else:
            delta = abs(va - vb)
            rel = delta / max(abs(va), abs(vb), 1e-6) * 100
            sep = "** YES **" if rel > 30 else ("maybe" if rel > 15 else "no")
            print(f"    {key:<28} {va:>12.4f} {vb:>12.4f}  delta={delta:.4f}  {sep} ({rel:.0f}%)")


def auto_generate_pairs(model_dir, manifest_dir, split, top_n):
    """Run confusion analysis and return top confused pairs."""
    import torch
    from util.openhands_modernized_inference import load_model_from_checkpoint
    from train_asl import WLASLOpenHandsDataset

    model, id_to_gloss, masked_ids = load_model_from_checkpoint(str(model_dir))
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    with open(model_dir / 'class_index_mapping.json') as f:
        id_to_gloss = json.load(f)
    gloss_to_id = {v: int(k) for k, v in id_to_gloss.items()}
    num_classes = len(id_to_gloss)

    manifest_file = manifest_dir / f'{split}_manifest.json'
    with open(manifest_file) as f:
        manifest = json.load(f)

    val_files = []
    val_labels = []
    for gloss, families in manifest['classes'].items():
        for family in families:
            for fname in family['files']:
                fpath = PICKLE_POOL / gloss / fname
                if fpath.exists():
                    val_files.append(str(fpath))
                    val_labels.append(gloss)

    print(f"Auto-generating pairs from {len(val_files)} {split} samples...")
    dataset = WLASLOpenHandsDataset(val_files, val_labels, gloss_to_id, 256, augment=False, use_finger_features=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels_idx = []
    with torch.no_grad():
        for batch in loader:
            pose_sequences = batch['pose_sequence'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            finger_features = batch.get('finger_features')
            if finger_features is not None:
                finger_features = finger_features.to(device)
            logits = model(pose_sequences, attention_masks, finger_features)
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels_idx.extend(labels.cpu().numpy())

    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(all_labels_idx, all_preds):
        confusion[true][pred] += 1

    pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and confusion[i][j] > 0:
                pairs.append((confusion[i][j], id_to_gloss[str(i)], id_to_gloss[str(j)]))
    pairs.sort(reverse=True)

    # Deduplicate: only keep unique gloss pairs (A->B and B->A count as one)
    seen = set()
    unique_pairs = []
    for count, true_g, pred_g in pairs:
        key = tuple(sorted([true_g, pred_g]))
        if key not in seen:
            seen.add(key)
            unique_pairs.append([true_g, pred_g])
        if len(unique_pairs) >= top_n:
            break

    return unique_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Test if velocity/hand-detection features can separate confused sign pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, help="Path to production model directory (needed for --auto-pairs)")
    parser.add_argument("--pairs-json", type=Path, help="JSON file with confusion pairs: [[\"gloss_a\", \"gloss_b\"], ...]")
    parser.add_argument("--manifest-dir", type=Path, help="Path to split manifests (needed for --auto-pairs)")
    parser.add_argument("--auto-pairs", action="store_true", help="Auto-generate pairs from model confusion matrix")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Which split for auto-pairs")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top confused pairs to test")
    parser.add_argument("--samples", type=int, default=5, help="Number of pickle samples to average per gloss")
    args = parser.parse_args()

    # Load or generate pairs
    if args.auto_pairs:
        if not args.model_dir or not args.manifest_dir:
            parser.error("--auto-pairs requires --model-dir and --manifest-dir")
        pairs = auto_generate_pairs(args.model_dir, args.manifest_dir, args.split, args.top_n)
        print(f"\nTop {len(pairs)} confused pairs (auto-generated):")
        for a, b in pairs:
            print(f"  {a} -> {b}")
    elif args.pairs_json:
        with open(args.pairs_json) as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} pairs from {args.pairs_json}")
    else:
        parser.error("Provide either --pairs-json or --auto-pairs")

    print(f"\n{'=' * 70}")
    print("  CAN ALL FEATURES SEPARATE CONFUSED PAIRS?")
    print(f"  Testing: velocity, hand-detection, trajectory, palm, location, handshape")
    print(f"  (>30% delta = YES, 15-30% = maybe, <15% = no)")
    print(f"  Averaging over {args.samples} original samples per gloss")
    print(f"{'=' * 70}")

    summary_yes = 0
    summary_maybe = 0
    summary_no = 0
    pair_results = []

    for true_gloss, pred_gloss in pairs:
        true_poses = load_pickle_samples(true_gloss, args.samples)
        pred_poses = load_pickle_samples(pred_gloss, args.samples)

        if not true_poses or not pred_poses:
            print(f"\n  SKIP: {true_gloss} vs {pred_gloss} (missing data)")
            continue

        print(f"\n{'#' * 70}")
        print(f"  {true_gloss.upper()} (true) vs {pred_gloss.upper()} (predicted)")
        print(f"  Samples: {true_gloss}={len(true_poses)}, {pred_gloss}={len(pred_poses)}")
        print(f"{'#' * 70}")

        # Extract features from all samples and average
        vel_true = average_features([extract_hand_velocity(p) for p in true_poses])
        vel_pred = average_features([extract_hand_velocity(p) for p in pred_poses])
        print_comparison(true_gloss, vel_true, pred_gloss, vel_pred, "VELOCITY (motion_features[0:6])")

        hand_true = average_features([extract_hand_detected_flags(p) for p in true_poses])
        hand_pred = average_features([extract_hand_detected_flags(p) for p in pred_poses])
        print_comparison(true_gloss, hand_true, pred_gloss, hand_pred, "HAND DETECTION (motion_features[6:8])")

        # Extract spatial features (all 9 groups)
        spatial_true = [extract_spatial_features_named(p) for p in true_poses]
        spatial_pred = [extract_spatial_features_named(p) for p in pred_poses]

        spatial_names = [
            "TRAJECTORY (spatial[0:10])",
            "PALM ORIENTATION (spatial[10:16])",
            "HAND LOCATION (spatial[16:22])",
            "HANDSHAPE DYNAMICS (spatial[22:26])",
            "FACE REGION (spatial[26:30])",
            "BODY CONTACT (spatial[30:32])",
            "REPETITION (spatial[32:34])",
            "HAND INTERACTION (spatial[34:38])",
            "WRIST ROTATION (spatial[38:40])",
        ]

        all_feature_pairs = [(vel_true, vel_pred), (hand_true, hand_pred)]
        for idx, name in enumerate(spatial_names):
            grp_true = average_features([s[idx] for s in spatial_true])
            grp_pred = average_features([s[idx] for s in spatial_pred])
            print_comparison(true_gloss, grp_true, pred_gloss, grp_pred, name)
            all_feature_pairs.append((grp_true, grp_pred))

        # Count separable features for this pair
        yes_count = 0
        maybe_count = 0
        no_count = 0
        for features_true, features_pred in all_feature_pairs:
            for key in features_true:
                va, vb = features_true[key], features_pred[key]
                if isinstance(va, str):
                    if va != vb:
                        yes_count += 1
                    else:
                        no_count += 1
                else:
                    delta = abs(va - vb)
                    rel = delta / max(abs(va), abs(vb), 1e-6) * 100
                    if rel > 30:
                        yes_count += 1
                    elif rel > 15:
                        maybe_count += 1
                    else:
                        no_count += 1

        pair_results.append((true_gloss, pred_gloss, yes_count, maybe_count, no_count))
        summary_yes += yes_count
        summary_maybe += maybe_count
        summary_no += no_count

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Pair':<35} {'YES':>5} {'maybe':>6} {'no':>5}  Verdict")
    print(f"  {'-' * 65}")
    for true_g, pred_g, y, m, n in pair_results:
        total = y + m + n
        verdict = "SEPARABLE" if y >= total * 0.4 else ("PARTIAL" if (y + m) >= total * 0.4 else "NOT separable")
        print(f"  {true_g} -> {pred_g:<22} {y:>5} {m:>6} {n:>5}  {verdict}")

    total_features = summary_yes + summary_maybe + summary_no
    print(f"\n  Total features across all pairs: {total_features}")
    print(f"  Separable (>30%): {summary_yes} ({summary_yes/max(total_features,1)*100:.0f}%)")
    print(f"  Maybe (15-30%):   {summary_maybe} ({summary_maybe/max(total_features,1)*100:.0f}%)")
    print(f"  Not (<15%):       {summary_no} ({summary_no/max(total_features,1)*100:.0f}%)")


if __name__ == "__main__":
    main()
