#!/usr/bin/env python3
"""
Create augmented .pose file for visualization (without normalization).
Only applies geometric transformations (shear + rotation) to preserve pixel scale.
"""

from pose_format import Pose
import numpy as np

def apply_shear_visualization(pose_data, shear_std=0.1):
    """Apply shear transformation for visualization (deterministic)."""
    shear_x = 0.15  # Fixed shear for consistent visualization
    shear_y = 0.10

    # Apply shear to x, y coordinates
    pose_data[:, :, :, 0] += shear_y * pose_data[:, :, :, 1]  # x += shear_y * y
    pose_data[:, :, :, 1] += shear_x * pose_data[:, :, :, 0]  # y += shear_x * x

    return pose_data

def apply_rotation_visualization(pose_data, angle_degrees=8):
    """Apply rotation transformation for visualization (deterministic)."""
    angle = np.radians(angle_degrees)  # 8 degrees rotation
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # Get center of pose for each frame
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

# Load existing pose file
input_pose = "C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/wlasl_poses_complete/pose_files/05231.pose"

print("=" * 70)
print("CREATING AUGMENTED POSE FOR VISUALIZATION")
print("=" * 70)
print()
print(f"Loading: {input_pose}")

with open(input_pose, 'rb') as f:
    pose = Pose.read(f.read())

print(f"Original data shape: {pose.body.data.shape}")
print(f"  Frames: {pose.body.data.shape[0]}")
print(f"  People: {pose.body.data.shape[1]}")
print(f"  Keypoints: {pose.body.data.shape[2]}")
print(f"  Dims: {pose.body.data.shape[3]}")
print()

# Create augmented version (no normalization - just geometric transforms)
augmented_data = pose.body.data.copy()

print("Applying geometric transformations...")
print("  - Shear transformation (15% x, 10% y)")
print("  - Rotation transformation (8 degrees)")
print()

# Apply transformations
augmented_data = apply_shear_visualization(augmented_data)
augmented_data = apply_rotation_visualization(augmented_data)

print("Transformations complete!")
print()

# Create new pose with augmented data
augmented_pose = Pose(
    header=pose.header,
    body=pose.body.__class__(
        fps=pose.body.fps,
        data=augmented_data,
        confidence=pose.body.confidence
    )
)

# Save augmented version only (original already exists)
output_augmented = "C:/Users/padwe/OneDrive/WLASL-proj/OpenHands-Modernized/augmented_pose_vis.pose"

print(f"Saving augmented to: {output_augmented}")
with open(output_augmented, 'wb') as f:
    augmented_pose.write(f)

print()
print("=" * 70)
print("SUCCESS!")
print("=" * 70)
print()
print("To visualize augmented pose:")
print()
print(f'  visualize_pose.exe -i "{output_augmented}" -o "augmented_vis.mp4"')
print()
print("Compare with original:")
print(f'  visualize_pose.exe -i "{input_pose}" -o "original_vis.mp4"')
print()
print("The augmented version should show:")
print("  - Slight skewing/slanting (shear effect)")
print("  - 8-degree rotation around center")
print()
