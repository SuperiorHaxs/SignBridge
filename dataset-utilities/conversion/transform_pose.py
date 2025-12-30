#!/usr/bin/env python3
"""
transform_pose.py

Transform a pose file to better match a baseline pose.
Works in NORMALIZED space (same as model) for accurate transformations.

Usage:
    python transform_pose.py <test_pose> <baseline_pose>
    python transform_pose.py <test_pose> <baseline_pose> --blend 0.5

Output is saved to applications/show-and-tell/temp/ directory.
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody

# Add model path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
sys.path.insert(0, str(MODELS_DIR / "openhands-modernized" / "src"))

from openhands_modernized import PoseTransforms

# Landmark indices in 75-point format
LEFT_HAND_START = 33
LEFT_HAND_END = 54
RIGHT_HAND_START = 54
RIGHT_HAND_END = 75

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "applications" / "show-and-tell" / "temp"


def extract_75_points(pose_data):
    """Extract 75-point subset from full MediaPipe pose."""
    if pose_data.shape[1] == 75:
        return pose_data
    elif pose_data.shape[1] in [543, 576]:
        return np.concatenate([
            pose_data[:, 0:33, :],       # Pose landmarks
            pose_data[:, 501:522, :],    # Left hand
            pose_data[:, 522:543, :]     # Right hand
        ], axis=1)
    else:
        raise ValueError(f"Unexpected keypoint count: {pose_data.shape[1]}")


def get_normalization_params(pose_75):
    """Get per-frame normalization parameters (center and std)."""
    frames = pose_75.shape[0]
    centers = np.zeros((frames, 2))
    stds = np.zeros(frames)

    for frame_idx in range(frames):
        frame = pose_75[frame_idx, :, :2]
        valid_mask = (frame != 0).any(axis=1)

        if valid_mask.sum() > 0:
            valid_coords = frame[valid_mask]
            centers[frame_idx] = valid_coords.mean(axis=0)
            stds[frame_idx] = valid_coords.std()
            if stds[frame_idx] == 0:
                stds[frame_idx] = 1.0
        else:
            centers[frame_idx] = [0, 0]
            stds[frame_idx] = 1.0

    return centers, stds


def get_hand_centers_normalized(pose_norm):
    """Get hand centers from normalized pose."""
    left_hand = pose_norm[:, LEFT_HAND_START:LEFT_HAND_END, :2]
    right_hand = pose_norm[:, RIGHT_HAND_START:RIGHT_HAND_END, :2]

    # Get center of each hand (mean of all hand keypoints)
    left_center = np.mean(left_hand, axis=1)
    right_center = np.mean(right_hand, axis=1)

    return left_center, right_center


def get_zero_pattern(pose_75):
    """Get which body parts are mostly zeros in the pose."""
    body_zeros = (pose_75[:, 0:33, :] == 0).sum() / pose_75[:, 0:33, :].size
    left_zeros = (pose_75[:, LEFT_HAND_START:LEFT_HAND_END, :] == 0).sum() / pose_75[:, LEFT_HAND_START:LEFT_HAND_END, :].size
    right_zeros = (pose_75[:, RIGHT_HAND_START:RIGHT_HAND_END, :] == 0).sum() / pose_75[:, RIGHT_HAND_START:RIGHT_HAND_END, :].size
    return {
        'body': body_zeros,
        'left_hand': left_zeros,
        'right_hand': right_zeros
    }


def transform_in_normalized_space(test_norm, baseline_norm, blend=1.0):
    """
    Transform test pose in normalized space to match baseline hand positions.
    """
    transformed = test_norm.copy()

    # Get hand centers in normalized space
    test_left, test_right = get_hand_centers_normalized(test_norm)
    baseline_left, baseline_right = get_hand_centers_normalized(baseline_norm)

    # Calculate mean position differences
    left_diff = np.mean(baseline_left, axis=0) - np.mean(test_left, axis=0)
    right_diff = np.mean(baseline_right, axis=0) - np.mean(test_right, axis=0)

    print(f"\nNormalized space differences:")
    print(f"  Left hand diff:  ({left_diff[0]:+.3f}, {left_diff[1]:+.3f})")
    print(f"  Right hand diff: ({right_diff[0]:+.3f}, {right_diff[1]:+.3f})")

    # Apply blended transformation
    left_adjust = left_diff * blend
    right_adjust = right_diff * blend

    print(f"\nApplying adjustments (blend={blend}):")
    print(f"  Left hand:  ({left_adjust[0]:+.3f}, {left_adjust[1]:+.3f})")
    print(f"  Right hand: ({right_adjust[0]:+.3f}, {right_adjust[1]:+.3f})")

    # Apply to all frames
    for frame in range(transformed.shape[0]):
        # Adjust left hand landmarks
        transformed[frame, LEFT_HAND_START:LEFT_HAND_END, 0] += left_adjust[0]
        transformed[frame, LEFT_HAND_START:LEFT_HAND_END, 1] += left_adjust[1]

        # Adjust right hand landmarks
        transformed[frame, RIGHT_HAND_START:RIGHT_HAND_END, 0] += right_adjust[0]
        transformed[frame, RIGHT_HAND_START:RIGHT_HAND_END, 1] += right_adjust[1]

    return transformed


def apply_zero_pattern(pose_raw, baseline_raw, threshold=0.8):
    """
    Apply the zero pattern from baseline to pose.
    If a body part is mostly zeros in baseline, set it to zeros in pose.
    """
    result = pose_raw.copy()
    baseline_pattern = get_zero_pattern(baseline_raw)

    print(f"\nBaseline zero pattern:")
    print(f"  Body:       {baseline_pattern['body']*100:.1f}% zeros")
    print(f"  Left hand:  {baseline_pattern['left_hand']*100:.1f}% zeros")
    print(f"  Right hand: {baseline_pattern['right_hand']*100:.1f}% zeros")

    applied = []
    if baseline_pattern['left_hand'] > threshold:
        result[:, LEFT_HAND_START:LEFT_HAND_END, :] = 0
        applied.append('left_hand')

    if baseline_pattern['right_hand'] > threshold:
        result[:, RIGHT_HAND_START:RIGHT_HAND_END, :] = 0
        applied.append('right_hand')

    if applied:
        print(f"\nApplying zero pattern to: {', '.join(applied)}")
    else:
        print(f"\nNo zero pattern applied (threshold={threshold*100:.0f}%)")

    return result


def denormalize(pose_norm, centers, stds):
    """Convert normalized pose back to raw coordinates."""
    result = pose_norm.copy()

    for frame_idx in range(result.shape[0]):
        # Reverse: multiply by std, then add center
        result[frame_idx, :, :2] = (pose_norm[frame_idx, :, :2] * stds[frame_idx]) + centers[frame_idx]

    return result


def apply_to_full_pose(full_pose_data, transform_75pt, original_75pt):
    """Apply the 75-point transformation back to full pose data."""
    result = full_pose_data.copy()
    delta = transform_75pt - original_75pt

    num_keypoints = full_pose_data.shape[1]

    if num_keypoints in [543, 576]:
        result[:, 0:33, :] += delta[:, 0:33, :]
        result[:, 501:522, :] += delta[:, 33:54, :]
        result[:, 522:543, :] += delta[:, 54:75, :]
    elif num_keypoints == 75:
        result = transform_75pt

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Transform a test pose to better match a baseline (in normalized space)"
    )
    parser.add_argument("test_pose", help="Test pose file to transform")
    parser.add_argument("baseline_pose", help="Baseline pose file to transform towards")
    parser.add_argument("--blend", "-b", type=float, default=1.0,
                        help="Blend factor 0-1 (default: 1.0)")

    args = parser.parse_args()

    test_path = Path(args.test_pose)
    baseline_path = Path(args.baseline_pose)

    if not test_path.exists():
        print(f"ERROR: Test pose not found: {test_path}")
        sys.exit(1)
    if not baseline_path.exists():
        print(f"ERROR: Baseline pose not found: {baseline_path}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"transformed_{test_path.name}"

    print("=" * 60)
    print("Pose Transformation (Normalized Space)")
    print("=" * 60)
    print(f"Test pose:     {test_path}")
    print(f"Baseline:      {baseline_path}")
    print(f"Output:        {output_path}")
    print(f"Blend factor:  {args.blend}")

    # Load poses
    print("\nLoading pose files...")
    with open(test_path, "rb") as f:
        test_pose = Pose.read(f.read())
    with open(baseline_path, "rb") as f:
        baseline_pose = Pose.read(f.read())

    test_data = test_pose.body.data
    baseline_data = baseline_pose.body.data

    if len(test_data.shape) == 4:
        test_data = test_data[:, 0, :, :]
    if len(baseline_data.shape) == 4:
        baseline_data = baseline_data[:, 0, :, :]

    print(f"  Test shape:     {test_data.shape}")
    print(f"  Baseline shape: {baseline_data.shape}")

    # Extract 75-point subsets
    test_75 = np.array(extract_75_points(test_data)[:, :, :2])  # x,y only
    baseline_75 = np.array(extract_75_points(baseline_data)[:, :, :2])

    # Get normalization parameters for test (we'll need these to denormalize)
    test_centers, test_stds = get_normalization_params(test_75)

    # Normalize both using model's normalization
    transforms = PoseTransforms()

    # Need to add dummy 3rd dimension for normalize function
    test_75_3d = np.zeros((test_75.shape[0], test_75.shape[1], 2))
    test_75_3d[:, :, :2] = test_75
    baseline_75_3d = np.zeros((baseline_75.shape[0], baseline_75.shape[1], 2))
    baseline_75_3d[:, :, :2] = baseline_75

    test_norm = transforms.center_and_scale_normalize(test_75_3d)
    baseline_norm = transforms.center_and_scale_normalize(baseline_75_3d)

    # Show hand positions before transformation
    test_left, test_right = get_hand_centers_normalized(test_norm)
    baseline_left, baseline_right = get_hand_centers_normalized(baseline_norm)

    print("\nTest pose (normalized):")
    print(f"  Left hand mean:  ({np.mean(test_left, axis=0)[0]:+.3f}, {np.mean(test_left, axis=0)[1]:+.3f})")
    print(f"  Right hand mean: ({np.mean(test_right, axis=0)[0]:+.3f}, {np.mean(test_right, axis=0)[1]:+.3f})")

    print("\nBaseline pose (normalized, target):")
    print(f"  Left hand mean:  ({np.mean(baseline_left, axis=0)[0]:+.3f}, {np.mean(baseline_left, axis=0)[1]:+.3f})")
    print(f"  Right hand mean: ({np.mean(baseline_right, axis=0)[0]:+.3f}, {np.mean(baseline_right, axis=0)[1]:+.3f})")

    # Transform in normalized space
    transformed_norm = transform_in_normalized_space(test_norm, baseline_norm, blend=args.blend)

    # Show result
    trans_left, trans_right = get_hand_centers_normalized(transformed_norm)
    print("\nTransformed pose (normalized):")
    print(f"  Left hand mean:  ({np.mean(trans_left, axis=0)[0]:+.3f}, {np.mean(trans_left, axis=0)[1]:+.3f})")
    print(f"  Right hand mean: ({np.mean(trans_right, axis=0)[0]:+.3f}, {np.mean(trans_right, axis=0)[1]:+.3f})")

    # Denormalize back to raw coordinates
    transformed_raw = denormalize(transformed_norm, test_centers, test_stds)

    # Build full pose with 3rd dimension (z/confidence) from original
    original_75_full = np.array(extract_75_points(test_data))
    transformed_75_full = original_75_full.copy()
    transformed_75_full[:, :, :2] = transformed_raw

    # Apply zero pattern from baseline (if left hand not used, set to zeros)
    transformed_75_full = apply_zero_pattern(transformed_75_full, baseline_75_3d)

    # Apply to full pose data
    transformed_full = apply_to_full_pose(np.array(test_data), transformed_75_full, original_75_full)

    # Reshape if original was 4D
    if len(test_pose.body.data.shape) == 4:
        transformed_full = transformed_full[:, np.newaxis, :, :]

    # Create new pose
    new_body = NumPyPoseBody(
        fps=test_pose.body.fps,
        data=transformed_full.astype(np.float32),
        confidence=test_pose.body.confidence
    )

    new_pose = Pose(header=test_pose.header, body=new_body)

    print(f"\nWriting transformed pose to: {output_path}")
    with open(output_path, "wb") as f:
        new_pose.write(f)

    print("\n" + "=" * 60)
    print("TRANSFORMATION COMPLETE")
    print("=" * 60)
    print(f"\nTest prediction:")
    print(f"  python project-utilities/test_pose_prediction.py \"{output_path}\"")

    return output_path


if __name__ == "__main__":
    main()
