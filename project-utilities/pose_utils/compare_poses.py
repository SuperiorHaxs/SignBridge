#!/usr/bin/env python3
"""
compare_poses.py - Compare two pose files to identify deviations

Usage:
    python compare_poses.py <test_pose> <baseline_pose> [--gloss NAME]

Examples:
    python compare_poses.py segment_001.pose test_chair.pose --gloss chair
    python compare_poses.py my_sign.pose dataset/chair/09869.pose
"""

import argparse
import numpy as np
from pathlib import Path


def load_pose_75pt(pose_path):
    """Load pose file and extract 75-point subset (pose + hands, no face)."""
    from pose_format import Pose

    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())

    data = pose.body.data
    fps = pose.body.fps

    if len(data.shape) == 4:
        data = data[:, 0, :, :]  # (frames, keypoints, dims)

    # Extract 75 points: 33 pose + 21 left hand + 21 right hand
    if data.shape[1] == 543 or data.shape[1] == 576:
        pose_75pt = np.concatenate([
            data[:, 0:33, :2],      # Pose landmarks
            data[:, 501:522, :2],   # Left hand
            data[:, 522:543, :2]    # Right hand
        ], axis=1)
    elif data.shape[1] == 75:
        pose_75pt = data[:, :, :2]
    else:
        pose_75pt = data[:, :, :2]

    return pose_75pt, fps


def normalize_pose(pose_data):
    """
    Normalize pose data to be scale and position invariant.
    Centers around shoulder midpoint and scales by shoulder width.
    """
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    normalized = pose_data.copy()

    for frame_idx in range(len(normalized)):
        frame = normalized[frame_idx]

        # Get shoulder positions
        left_shoulder = frame[LEFT_SHOULDER]
        right_shoulder = frame[RIGHT_SHOULDER]

        # Center around shoulder midpoint
        center = (left_shoulder + right_shoulder) / 2

        # Scale by shoulder width (for position invariance)
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        if shoulder_width < 1e-6:
            shoulder_width = 1.0

        # Normalize
        frame_centered = frame - center
        frame_normalized = frame_centered / shoulder_width
        normalized[frame_idx] = frame_normalized

    return normalized


def analyze_pose(pose_data, name, normalized=True):
    """Analyze key characteristics of a pose sequence."""

    # Key landmark indices (in 75-point format)
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HAND_CENTER = 33 + 9   # Middle finger MCP
    RIGHT_HAND_CENTER = 54 + 9
    LEFT_HAND_TIP = 33 + 12     # Middle finger tip
    RIGHT_HAND_TIP = 54 + 12

    frames = len(pose_data)

    # Normalize if requested
    if normalized:
        pose_data = normalize_pose(pose_data)

    # Calculate body center (midpoint of shoulders)
    shoulders_mid = (pose_data[:, LEFT_SHOULDER, :] + pose_data[:, RIGHT_SHOULDER, :]) / 2

    # Hand positions relative to body center
    left_hand_rel = pose_data[:, LEFT_HAND_CENTER, :] - shoulders_mid
    right_hand_rel = pose_data[:, RIGHT_HAND_CENTER, :] - shoulders_mid

    # Wrist positions
    left_wrist_rel = pose_data[:, LEFT_WRIST, :] - shoulders_mid
    right_wrist_rel = pose_data[:, RIGHT_WRIST, :] - shoulders_mid

    # Hand spread (distance between hands)
    hand_distance = np.linalg.norm(
        pose_data[:, LEFT_HAND_CENTER, :] - pose_data[:, RIGHT_HAND_CENTER, :],
        axis=1
    )

    # Hand height relative to face (nose)
    nose_pos = pose_data[:, NOSE, :]
    left_hand_height = pose_data[:, LEFT_HAND_CENTER, 1] - nose_pos[:, 1]
    right_hand_height = pose_data[:, RIGHT_HAND_CENTER, 1] - nose_pos[:, 1]

    # Movement velocity (frame-to-frame)
    if frames > 1:
        left_velocity = np.linalg.norm(np.diff(pose_data[:, LEFT_HAND_CENTER, :], axis=0), axis=1)
        right_velocity = np.linalg.norm(np.diff(pose_data[:, RIGHT_HAND_CENTER, :], axis=0), axis=1)
    else:
        left_velocity = np.array([0])
        right_velocity = np.array([0])

    stats = {
        'frames': frames,
        'left_hand_mean': left_hand_rel.mean(axis=0),
        'right_hand_mean': right_hand_rel.mean(axis=0),
        'left_hand_std': left_hand_rel.std(axis=0),
        'right_hand_std': right_hand_rel.std(axis=0),
        'left_wrist_mean': left_wrist_rel.mean(axis=0),
        'right_wrist_mean': right_wrist_rel.mean(axis=0),
        'hand_spread_mean': hand_distance.mean(),
        'hand_spread_std': hand_distance.std(),
        'hand_spread_max': hand_distance.max(),
        'left_height_mean': left_hand_height.mean(),
        'right_height_mean': right_hand_height.mean(),
        'left_velocity_mean': left_velocity.mean(),
        'right_velocity_mean': right_velocity.mean(),
    }

    return stats


def print_analysis(stats, name):
    """Print analysis results."""
    print(f"\n{'='*60}")
    print(f"Analysis: {name}")
    print(f"{'='*60}")
    print(f"Duration: {stats['frames']} frames")

    print(f"\nHand Positions (normalized, relative to body center):")
    print(f"  Left hand mean:  ({stats['left_hand_mean'][0]:+.3f}, {stats['left_hand_mean'][1]:+.3f})")
    print(f"  Right hand mean: ({stats['right_hand_mean'][0]:+.3f}, {stats['right_hand_mean'][1]:+.3f})")

    print(f"\nHand Movement Range (std dev):")
    print(f"  Left hand:  X={stats['left_hand_std'][0]:.3f}, Y={stats['left_hand_std'][1]:.3f}")
    print(f"  Right hand: X={stats['right_hand_std'][0]:.3f}, Y={stats['right_hand_std'][1]:.3f}")

    print(f"\nHand Spread (distance between hands):")
    print(f"  Mean: {stats['hand_spread_mean']:.3f}")
    print(f"  Std:  {stats['hand_spread_std']:.3f}")
    print(f"  Max:  {stats['hand_spread_max']:.3f}")

    print(f"\nHand Height (relative to nose, + = below face):")
    print(f"  Left:  {stats['left_height_mean']:+.3f}")
    print(f"  Right: {stats['right_height_mean']:+.3f}")

    print(f"\nMovement Velocity (avg frame-to-frame):")
    print(f"  Left:  {stats['left_velocity_mean']:.4f}")
    print(f"  Right: {stats['right_velocity_mean']:.4f}")


def compare_poses(test_stats, baseline_stats):
    """Compare two pose analyses and highlight differences."""
    print(f"\n{'='*60}")
    print("COMPARISON: TEST vs BASELINE")
    print(f"{'='*60}")

    # Calculate differences
    left_pos_diff = test_stats['left_hand_mean'] - baseline_stats['left_hand_mean']
    right_pos_diff = test_stats['right_hand_mean'] - baseline_stats['right_hand_mean']
    spread_diff = test_stats['hand_spread_mean'] - baseline_stats['hand_spread_mean']
    left_height_diff = test_stats['left_height_mean'] - baseline_stats['left_height_mean']
    right_height_diff = test_stats['right_height_mean'] - baseline_stats['right_height_mean']

    # Severity assessment
    def severity(diff, thresholds=(0.1, 0.3, 0.5)):
        abs_diff = abs(diff)
        if abs_diff < thresholds[0]:
            return "OK", ""
        elif abs_diff < thresholds[1]:
            return "MINOR", "~"
        elif abs_diff < thresholds[2]:
            return "MODERATE", "*"
        else:
            return "MAJOR", "**"

    print("\nHand Position Differences:")
    sev, mark = severity(np.linalg.norm(left_pos_diff))
    print(f"  Left hand:  ({left_pos_diff[0]:+.3f}, {left_pos_diff[1]:+.3f}) [{sev}]{mark}")
    sev, mark = severity(np.linalg.norm(right_pos_diff))
    print(f"  Right hand: ({right_pos_diff[0]:+.3f}, {right_pos_diff[1]:+.3f}) [{sev}]{mark}")

    print("\nHand Spread Difference:")
    sev, mark = severity(spread_diff, (0.2, 0.5, 1.0))
    print(f"  {spread_diff:+.3f} [{sev}]{mark}")
    if spread_diff > 0.5:
        print(f"  --> Your hands are much FARTHER APART than baseline")
    elif spread_diff < -0.5:
        print(f"  --> Your hands are much CLOSER TOGETHER than baseline")

    print("\nHand Height Difference (relative to face):")
    sev, mark = severity(left_height_diff)
    dir_left = "LOWER" if left_height_diff > 0 else "HIGHER"
    print(f"  Left:  {left_height_diff:+.3f} [{sev}]{mark}")
    if abs(left_height_diff) > 0.3:
        print(f"  --> Your left hand is {dir_left} than baseline")

    sev, mark = severity(right_height_diff)
    dir_right = "LOWER" if right_height_diff > 0 else "HIGHER"
    print(f"  Right: {right_height_diff:+.3f} [{sev}]{mark}")
    if abs(right_height_diff) > 0.3:
        print(f"  --> Your right hand is {dir_right} than baseline")

    # Overall assessment
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")

    issues = []
    if abs(spread_diff) > 0.5:
        if spread_diff > 0:
            issues.append("- Bring your hands CLOSER together")
        else:
            issues.append("- Spread your hands FARTHER apart")

    if abs(left_height_diff) > 0.3:
        if left_height_diff > 0:
            issues.append("- Raise your LEFT hand higher")
        else:
            issues.append("- Lower your LEFT hand")

    if abs(right_height_diff) > 0.3:
        if right_height_diff > 0:
            issues.append("- Raise your RIGHT hand higher")
        else:
            issues.append("- Lower your RIGHT hand")

    if abs(left_pos_diff[0]) > 0.3:
        if left_pos_diff[0] > 0:
            issues.append("- Move your LEFT hand more to the LEFT")
        else:
            issues.append("- Move your LEFT hand more to the RIGHT")

    if abs(right_pos_diff[0]) > 0.3:
        if right_pos_diff[0] > 0:
            issues.append("- Move your RIGHT hand more to the LEFT")
        else:
            issues.append("- Move your RIGHT hand more to the RIGHT")

    if issues:
        print("To better match the baseline sign:")
        for issue in issues:
            print(issue)
    else:
        print("Your sign appears to match the baseline well!")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two pose files to identify deviations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python compare_poses.py my_segment.pose baseline.pose
    python compare_poses.py my_segment.pose baseline.pose --gloss chair

The tool normalizes both poses to be scale/position invariant before comparing.
        """
    )

    parser.add_argument("test_pose", help="Path to your pose file (the one to test)")
    parser.add_argument("baseline_pose", help="Path to baseline pose file (ground truth)")
    parser.add_argument("--gloss", "-g", help="Name of the sign being compared (for display)")
    parser.add_argument("--raw", action="store_true", help="Use raw coordinates (no normalization)")

    args = parser.parse_args()

    test_path = Path(args.test_pose)
    baseline_path = Path(args.baseline_pose)

    if not test_path.exists():
        print(f"ERROR: Test pose file not found: {test_path}")
        return 1

    if not baseline_path.exists():
        print(f"ERROR: Baseline pose file not found: {baseline_path}")
        return 1

    gloss_name = args.gloss or "sign"
    normalized = not args.raw

    print(f"Comparing pose files for '{gloss_name}'")
    print(f"Test:     {test_path}")
    print(f"Baseline: {baseline_path}")
    print(f"Mode:     {'Normalized' if normalized else 'Raw coordinates'}")

    # Load poses
    test_data, test_fps = load_pose_75pt(test_path)
    baseline_data, baseline_fps = load_pose_75pt(baseline_path)

    # Analyze
    test_stats = analyze_pose(test_data, f"Your '{gloss_name}'", normalized=normalized)
    baseline_stats = analyze_pose(baseline_data, f"Baseline '{gloss_name}'", normalized=normalized)

    # Print individual analyses
    print_analysis(test_stats, f"YOUR POSE ({test_path.name})")
    print_analysis(baseline_stats, f"BASELINE ({baseline_path.name})")

    # Compare
    compare_poses(test_stats, baseline_stats)

    return 0


if __name__ == "__main__":
    exit(main())
