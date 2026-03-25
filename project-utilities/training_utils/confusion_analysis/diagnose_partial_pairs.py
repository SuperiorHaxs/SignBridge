"""
Deep diagnostic for confused pairs that velocity/hand-detection can't fully separate.
Analyzes what ASL parameters actually differ: location, trajectory, handshape, palm orientation.
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
PICKLE_POOL = PROJECT_ROOT / 'datasets' / 'augmented_pool' / 'pickle'


def load_pickle_samples(gloss, max_samples=5):
    gloss_dir = PICKLE_POOL / gloss
    if not gloss_dir.exists():
        return []
    originals = sorted([f for f in gloss_dir.glob('*.pkl') if '_aug_' not in f.name])
    if not originals:
        originals = sorted(gloss_dir.glob('*.pkl'))
    poses = []
    for pkl_path in originals[:max_samples]:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        kp = data['keypoints']
        if len(kp) >= 5:
            poses.append(kp)
    return poses


# ─── Feature extractors ──────────────────────────────────────────────────────

def extract_hand_location_relative_to_body(pose_83):
    """Where are the hands relative to nose, chin, shoulder, chest?"""
    # Key body landmarks
    nose = pose_83[:, 0, :2]           # landmark 0
    left_shoulder = pose_83[:, 11, :2]  # landmark 11
    right_shoulder = pose_83[:, 12, :2] # landmark 12
    chest = (left_shoulder + right_shoulder) / 2

    # Hand centroids
    left_hand = np.mean(pose_83[:, 33:54, :2], axis=1)
    right_hand = np.mean(pose_83[:, 54:75, :2], axis=1)

    # Relative positions (positive y = below in MediaPipe coords)
    rh_rel_nose = np.mean(right_hand - nose, axis=0)
    lh_rel_nose = np.mean(left_hand - nose, axis=0)
    rh_rel_chest = np.mean(right_hand - chest, axis=0)
    lh_rel_chest = np.mean(left_hand - chest, axis=0)

    # Height zone: face (near nose), chest, waist (below chest)
    rh_y_vs_nose = np.mean(right_hand[:, 1] - nose[:, 1])
    lh_y_vs_nose = np.mean(left_hand[:, 1] - nose[:, 1])
    shoulder_to_nose = np.mean(np.abs(chest[:, 1] - nose[:, 1]))

    def zone(y_offset, ref_dist):
        if ref_dist < 1e-4:
            return "face"
        ratio = y_offset / ref_dist
        if ratio < 0.3:
            return "face"
        elif ratio < 1.0:
            return "chest"
        else:
            return "waist"

    return {
        'rh_x_rel_nose': float(rh_rel_nose[0]),
        'rh_y_rel_nose': float(rh_rel_nose[1]),
        'lh_x_rel_nose': float(lh_rel_nose[0]),
        'lh_y_rel_nose': float(lh_rel_nose[1]),
        'rh_x_rel_chest': float(rh_rel_chest[0]),
        'rh_y_rel_chest': float(rh_rel_chest[1]),
        'lh_x_rel_chest': float(lh_rel_chest[0]),
        'lh_y_rel_chest': float(lh_rel_chest[1]),
        'rh_zone': zone(rh_y_vs_nose, shoulder_to_nose),
        'lh_zone': zone(lh_y_vs_nose, shoulder_to_nose),
        'hands_dist_apart': float(np.mean(np.linalg.norm(right_hand - left_hand, axis=1))),
    }


def extract_movement_trajectory(pose_83):
    """Movement direction, shape (linear/circular/repetitive)."""
    right_hand = np.mean(pose_83[:, 54:75, :2], axis=1)
    left_hand = np.mean(pose_83[:, 33:54, :2], axis=1)

    def trajectory_features(hand_pos, name):
        if len(hand_pos) < 3:
            return {}

        # Net displacement vs total path length -> linearity
        displacements = np.diff(hand_pos, axis=0)
        frame_dists = np.linalg.norm(displacements, axis=1)
        total_path = np.sum(frame_dists)
        net_disp = np.linalg.norm(hand_pos[-1] - hand_pos[0])
        linearity = net_disp / max(total_path, 1e-6)

        # Primary movement direction
        net_vec = hand_pos[-1] - hand_pos[0]
        angle = np.degrees(np.arctan2(net_vec[1], net_vec[0]))

        # Vertical vs horizontal dominance
        dx = np.abs(displacements[:, 0])
        dy = np.abs(displacements[:, 1])
        vert_ratio = np.sum(dy) / max(np.sum(dx) + np.sum(dy), 1e-6)

        # Repetitiveness: autocorrelation of y-position
        y_centered = hand_pos[:, 1] - np.mean(hand_pos[:, 1])
        if np.std(y_centered) > 1e-6:
            autocorr = np.correlate(y_centered, y_centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / max(autocorr[0], 1e-6)
            # Find first peak after initial decay
            peaks = []
            for i in range(2, len(autocorr) - 1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.3:
                    peaks.append(i)
                    break
            repetitive = len(peaks) > 0
            rep_period = peaks[0] if peaks else 0
        else:
            repetitive = False
            rep_period = 0

        # Circularity: cross product accumulation
        cross_sum = 0
        for i in range(len(displacements) - 1):
            cross_sum += displacements[i][0] * displacements[i+1][1] - displacements[i][1] * displacements[i+1][0]
        circularity = abs(cross_sum) / max(total_path**2, 1e-6) * len(displacements)

        return {
            f'{name}_linearity': float(linearity),
            f'{name}_direction_deg': float(angle),
            f'{name}_vert_ratio': float(vert_ratio),
            f'{name}_repetitive': repetitive,
            f'{name}_rep_period': int(rep_period),
            f'{name}_circularity': float(circularity),
            f'{name}_total_path': float(total_path),
        }

    feats = {}
    feats.update(trajectory_features(right_hand, 'rh'))
    feats.update(trajectory_features(left_hand, 'lh'))
    return feats


def extract_handshape_features(pose_83):
    """Finger curl, spread, and wrist angle."""
    right_hand = pose_83[:, 54:75, :]  # 21 hand landmarks
    left_hand = pose_83[:, 33:54, :]

    def hand_shape(hand, name):
        if hand.shape[0] < 2:
            return {}
        wrist = hand[:, 0, :2]

        # Finger tip distances from wrist (normalized by hand size)
        tip_indices = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        pip_indices = [3, 6, 10, 14, 18]   # proximal joints

        hand_size = np.mean(np.linalg.norm(hand[:, 9, :2] - wrist, axis=1))  # middle MCP to wrist

        tip_dists = []
        curl_ratios = []
        for tip_i, pip_i in zip(tip_indices, pip_indices):
            tip_dist = np.mean(np.linalg.norm(hand[:, tip_i, :2] - wrist, axis=1))
            pip_dist = np.mean(np.linalg.norm(hand[:, pip_i, :2] - wrist, axis=1))
            tip_dists.append(tip_dist / max(hand_size, 1e-6))
            curl_ratios.append(tip_dist / max(pip_dist, 1e-6))

        # Spread: angles between adjacent fingers
        spreads = []
        for i in range(len(tip_indices) - 1):
            v1 = hand[:, tip_indices[i], :2] - wrist
            v2 = hand[:, tip_indices[i+1], :2] - wrist
            cos = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6)
            spreads.append(float(np.mean(np.degrees(np.arccos(np.clip(cos, -1, 1))))))

        return {
            f'{name}_thumb_ext': float(curl_ratios[0]),
            f'{name}_index_ext': float(curl_ratios[1]),
            f'{name}_middle_ext': float(curl_ratios[2]),
            f'{name}_ring_ext': float(curl_ratios[3]),
            f'{name}_pinky_ext': float(curl_ratios[4]),
            f'{name}_avg_spread': float(np.mean(spreads)),
            f'{name}_hand_openness': float(np.mean(tip_dists)),
        }

    feats = {}
    feats.update(hand_shape(right_hand, 'rh'))
    feats.update(hand_shape(left_hand, 'lh'))
    return feats


def extract_palm_orientation(pose_83):
    """Estimate palm facing direction from hand landmark geometry."""
    right_hand = pose_83[:, 54:75, :]
    left_hand = pose_83[:, 33:54, :]

    def palm_dir(hand, name):
        # Use wrist(0), index_mcp(5), pinky_mcp(17) to define palm plane
        wrist = hand[:, 0, :]
        index_mcp = hand[:, 5, :]
        pinky_mcp = hand[:, 17, :]

        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist

        # Cross product gives normal to palm
        if hand.shape[2] >= 3:
            normal = np.cross(v1, v2)
            normal_norm = np.linalg.norm(normal, axis=1, keepdims=True)
            normal = normal / np.maximum(normal_norm, 1e-6)
            avg_normal = np.mean(normal, axis=0)

            # Classify orientation
            if abs(avg_normal[2]) > abs(avg_normal[0]) and abs(avg_normal[2]) > abs(avg_normal[1]):
                orientation = "forward" if avg_normal[2] > 0 else "backward"
            elif abs(avg_normal[1]) > abs(avg_normal[0]):
                orientation = "down" if avg_normal[1] > 0 else "up"
            else:
                orientation = "right" if avg_normal[0] > 0 else "left"

            return {
                f'{name}_palm_nx': float(avg_normal[0]),
                f'{name}_palm_ny': float(avg_normal[1]),
                f'{name}_palm_nz': float(avg_normal[2]),
                f'{name}_palm_facing': orientation,
            }
        return {}

    feats = {}
    feats.update(palm_dir(right_hand, 'rh'))
    feats.update(palm_dir(left_hand, 'lh'))
    return feats


# ─── Comparison ───────────────────────────────────────────────────────────────

def average_features(feature_dicts):
    if not feature_dicts:
        return {}
    avg = {}
    for key in feature_dicts[0]:
        vals = [d[key] for d in feature_dicts]
        if isinstance(vals[0], (str, bool)):
            from collections import Counter
            avg[key] = Counter(vals).most_common(1)[0][0]
        else:
            avg[key] = float(np.mean(vals))
    return avg


def compare_features(label_a, feats_a, label_b, feats_b, feature_type):
    """Compare and return counts of separable features."""
    print(f"\n  {feature_type}:")
    yes_count = 0
    maybe_count = 0
    no_count = 0
    separable_keys = []

    for key in feats_a:
        va, vb = feats_a[key], feats_b[key]
        if isinstance(va, (str, bool)):
            sep = "** YES **" if va != vb else "no"
            print(f"    {key:<28} {str(va):>12} {str(vb):>12}  {sep}")
            if va != vb:
                yes_count += 1
                separable_keys.append(key)
            else:
                no_count += 1
        else:
            delta = abs(va - vb)
            rel = delta / max(abs(va), abs(vb), 1e-6) * 100
            if rel > 30:
                sep = "** YES **"
                yes_count += 1
                separable_keys.append(key)
            elif rel > 15:
                sep = "maybe"
                maybe_count += 1
            else:
                sep = "no"
                no_count += 1
            print(f"    {key:<28} {va:>12.4f} {vb:>12.4f}  delta={delta:.4f}  {sep} ({rel:.0f}%)")

    return yes_count, maybe_count, no_count, separable_keys


def main():
    partial_pairs = [
        ("waiter", "sunday"),
        ("breakfast", "delicious"),
        ("which", "how"),
        ("fish", "banana"),
        ("why", "drink"),
        ("evening", "chocolate"),
    ]

    feature_extractors = [
        ("HAND LOCATION (relative to body)", extract_hand_location_relative_to_body),
        ("MOVEMENT TRAJECTORY", extract_movement_trajectory),
        ("HANDSHAPE", extract_handshape_features),
        ("PALM ORIENTATION", extract_palm_orientation),
    ]

    print("=" * 70)
    print("  DEEP DIAGNOSTIC: What features CAN separate the partial pairs?")
    print("  Testing: location, trajectory, handshape, palm orientation")
    print("=" * 70)

    all_recommendations = {}

    for true_gloss, pred_gloss in partial_pairs:
        true_poses = load_pickle_samples(true_gloss, 5)
        pred_poses = load_pickle_samples(pred_gloss, 5)

        if not true_poses or not pred_poses:
            continue

        print(f"\n{'#' * 70}")
        print(f"  {true_gloss.upper()} (true) vs {pred_gloss.upper()} (confused as)")
        print(f"{'#' * 70}")

        pair_separable = []

        for feat_name, extractor in feature_extractors:
            feats_true = average_features([extractor(p) for p in true_poses])
            feats_pred = average_features([extractor(p) for p in pred_poses])
            yes, maybe, no, sep_keys = compare_features(
                true_gloss, feats_true, pred_gloss, feats_pred, feat_name)

            if sep_keys:
                pair_separable.append((feat_name, sep_keys))

        print(f"\n  RECOMMENDATION for {true_gloss} vs {pred_gloss}:")
        if pair_separable:
            for feat_type, keys in pair_separable:
                print(f"    {feat_type}: {', '.join(keys)}")
            all_recommendations[f"{true_gloss}->{pred_gloss}"] = pair_separable
        else:
            print(f"    No clear separating features found - may need temporal/sequence modeling")

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"  FEATURE IMPLEMENTATION RECOMMENDATIONS")
    print(f"{'=' * 70}")

    # Count which feature categories help most
    category_counts = {}
    for pair, recs in all_recommendations.items():
        for feat_type, keys in recs:
            category_counts[feat_type] = category_counts.get(feat_type, 0) + len(keys)

    print(f"\n  Feature categories ranked by separating power:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"    {count:>3} separable features  -  {cat}")

    # Specific feature recommendations
    feature_counts = {}
    for pair, recs in all_recommendations.items():
        for feat_type, keys in recs:
            for k in keys:
                feature_counts[k] = feature_counts.get(k, 0) + 1

    print(f"\n  Most impactful individual features (across all pairs):")
    for feat, count in sorted(feature_counts.items(), key=lambda x: -x[1]):
        if count >= 2:
            print(f"    {count} pairs  -  {feat}")


if __name__ == "__main__":
    main()
