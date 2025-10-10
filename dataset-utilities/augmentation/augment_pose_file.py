#!/usr/bin/env python3
"""
Augment original .pose files directly and create augmented .pose files.
Simple approach using geometric transformations only (like create_augmented_visualization.py).
"""

import sys
import os
import numpy as np
from pose_format import Pose

def apply_shear(pose_data, shear_std=0.1):
    """Apply shear transformation."""
    # Generate random shear parameters
    shear_x = np.random.normal(0, shear_std)
    shear_y = np.random.normal(0, shear_std)

    # Apply shear to x, y coordinates
    pose_data[:, :, :, 0] += shear_y * pose_data[:, :, :, 1]  # x += shear_y * y
    pose_data[:, :, :, 1] += shear_x * pose_data[:, :, :, 0]  # y += shear_x * x

    return pose_data


def apply_rotation(pose_data, rotation_std=0.1):
    """Apply rotation transformation."""
    # Generate random rotation angle
    angle_degrees = np.random.normal(0, rotation_std * 50)  # Convert std to degrees range
    angle = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # Get dimensions
    frames, people, keypoints, dims = pose_data.shape

    for frame_idx in range(frames):
        for person_idx in range(people):
            # Get valid keypoints
            xy_coords = pose_data[frame_idx, person_idx, :, :2]
            valid_mask = (xy_coords != 0).any(axis=1)

            if valid_mask.sum() > 0:
                # Find center
                center = xy_coords[valid_mask].mean(axis=0)

                # Center coordinates
                centered_x = pose_data[frame_idx, person_idx, :, 0] - center[0]
                centered_y = pose_data[frame_idx, person_idx, :, 1] - center[1]

                # Apply rotation
                rotated_x = cos_a * centered_x - sin_a * centered_y
                rotated_y = sin_a * centered_x + cos_a * centered_y

                # Move back to original position
                pose_data[frame_idx, person_idx, :, 0] = rotated_x + center[0]
                pose_data[frame_idx, person_idx, :, 1] = rotated_y + center[1]

    return pose_data


def augment_pose_file(input_pose_path, output_pose_path, augmentation_id=0, shear_std=0.1, rotation_std=0.1):
    """
    Create augmented .pose file from original .pose file.

    Args:
        input_pose_path: Path to original .pose file
        output_pose_path: Path to save augmented .pose file
        augmentation_id: Seed for reproducibility
        shear_std: Standard deviation for shear transformation
        rotation_std: Standard deviation for rotation transformation
    """

    print("=" * 70)
    print("AUGMENTING .POSE FILE (Simple Geometric Transforms)")
    print("=" * 70)
    print()
    print(f"Input:  {input_pose_path}")
    print(f"Output: {output_pose_path}")
    print(f"Augmentation ID: {augmentation_id}")
    print()

    # Set random seed for reproducibility
    np.random.seed(augmentation_id)

    # Load original .pose file
    print("Loading original .pose file...")
    with open(input_pose_path, 'rb') as f:
        original_pose = Pose.read(f.read())

    print(f"Original data shape: {original_pose.body.data.shape}")
    print(f"  Frames: {original_pose.body.data.shape[0]}")
    print(f"  People: {original_pose.body.data.shape[1]}")
    print(f"  Keypoints: {original_pose.body.data.shape[2]}")
    print(f"  Dims: {original_pose.body.data.shape[3]}")
    print()

    # Create augmented version - just apply geometric transforms directly
    augmented_data = original_pose.body.data.copy()

    print("Applying geometric transformations...")
    print(f"  - Shear (std={shear_std})")
    print(f"  - Rotation (std={rotation_std})")
    print()

    # Apply transformations directly to pose data
    augmented_data = apply_shear(augmented_data, shear_std=shear_std)
    augmented_data = apply_rotation(augmented_data, rotation_std=rotation_std)

    print("Transformations complete!")
    print()

    # Create new pose with augmented data
    augmented_pose = Pose(
        header=original_pose.header,
        body=original_pose.body.__class__(
            fps=original_pose.body.fps,
            data=augmented_data,
            confidence=original_pose.body.confidence
        )
    )

    # Write to file
    print(f"Writing to: {output_pose_path}")
    with open(output_pose_path, 'wb') as f:
        augmented_pose.write(f)

    file_size = os.path.getsize(output_pose_path)

    print()
    print("=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print()
    print(f"Created: {output_pose_path}")
    print(f"File size: {file_size:,} bytes")
    print()
    print("To visualize:")
    print(f'  visualize_pose.exe -i "{output_pose_path}" -o "{output_pose_path.replace(".pose", ".mp4")}"')
    print()

    return output_pose_path


if __name__ == "__main__":
    # Test with accident sign (video 00623)
    input_pose = "C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/wlasl_poses_complete/pose_files/00623.pose"
    output_pose = "C:/Users/padwe/OneDrive/WLASL-proj/OpenHands-Modernized/experiments/aug_00_00623.pose"

    augment_pose_file(input_pose, output_pose, augmentation_id=0)

    print("Next steps:")
    print("1. Visualize with visualize_pose.exe to verify augmentation")
    print("2. If looks good, use this approach for batch augmentation")
    print()
