#!/usr/bin/env python3
"""
Convert .pose files to pickle format with MediaPipe keypoints.

This script converts pose files (augmented or original) to pickle format
that can be used by the training script.

Supports command-line parameters for flexible input/output paths.
"""

import os
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pose_format import Pose


def pose_to_numpy(pose):
    """Convert pose_format Pose object to numpy arrays"""
    try:
        if pose.body.data is None or len(pose.body.data) == 0:
            return None, None

        pose_data = pose.body.data

        if len(pose_data.shape) == 4:
            keypoints = pose_data[:, 0, :, :2]
            confidences = pose_data[:, 0, :, 2] if pose_data.shape[-1] > 2 else np.ones(pose_data[:, 0, :, :2].shape[:2])
        else:
            keypoints = pose_data[:, :, :2] if pose_data.shape[-1] >= 2 else pose_data
            confidences = pose_data[:, :, 2] if pose_data.shape[-1] > 2 else np.ones(pose_data.shape[:2])

        return keypoints, confidences

    except Exception as e:
        print(f"Error converting pose data: {e}")
        return None, None


def convert_single_pose_file(input_file):
    """
    Convert a single .pose file to pickle format.
    Saves with same name but .pkl extension in same directory.

    Args:
        input_file: Path to .pose file
    """
    print("=" * 70)
    print("CONVERTING SINGLE .POSE FILE TO PICKLE FORMAT")
    print("=" * 70)
    print()
    print(f"Input file: {input_file}")

    # Validate input
    if not os.path.exists(input_file):
        print(f"ERROR: File not found: {input_file}")
        return

    if not input_file.endswith('.pose'):
        print(f"ERROR: Input must be a .pose file")
        return

    try:
        # Load pose file
        with open(input_file, 'rb') as f:
            pose = Pose.read(f.read())

        # Convert to numpy
        keypoints, confidences = pose_to_numpy(pose)

        if keypoints is None:
            print("ERROR: Failed to convert pose data")
            return

        # Extract metadata from filename
        filename = os.path.basename(input_file)
        video_id = filename.replace('.pose', '')

        # Get gloss from parent directory name (if in class folder)
        parent_dir = os.path.basename(os.path.dirname(input_file))
        gloss = parent_dir if parent_dir else 'unknown'

        # Check if augmented
        is_augmented = video_id.startswith('aug_')
        augmentation_id = None
        original_video_id = video_id

        if is_augmented:
            parts = video_id.split('_')
            if len(parts) >= 3:
                try:
                    augmentation_id = int(parts[1])
                    original_video_id = '_'.join(parts[2:])
                except ValueError:
                    pass

        # Create pickle data
        pickle_data = {
            'keypoints': keypoints,
            'confidences': confidences,
            'video_id': original_video_id,
            'gloss': gloss,
            'split': 'train',
            'augmented': is_augmented,
            'augmentation_id': augmentation_id,
            'pose_file': input_file
        }

        # Create output path (same directory, .pkl extension)
        output_file = input_file.replace('.pose', '.pkl')

        # Save pickle file
        with open(output_file, 'wb') as f:
            pickle.dump(pickle_data, f)

        print(f"✓ Converted successfully")
        print(f"Output file: {output_file}")
        print()

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


def convert_pose_directory(input_dir, output_dir):
    """
    Convert all .pose files in a directory structure to pickle format.
    Preserves the directory structure (class folders).

    Args:
        input_dir: Directory containing .pose files (with class subdirectories)
        output_dir: Directory to save pickle files (will create same structure)
    """
    print("=" * 70)
    print("CONVERTING .POSE FILES TO PICKLE FORMAT")
    print("=" * 70)
    print()
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all class directories
    class_dirs = sorted([d for d in os.listdir(input_dir)
                        if os.path.isdir(os.path.join(input_dir, d))])

    if not class_dirs:
        print("No class directories found. Looking for .pose files in input directory...")
        # If no class dirs, treat input_dir as containing pose files directly
        class_dirs = ['.']

    print(f"Found {len(class_dirs)} class directories")
    print()

    total_files = 0
    successful_conversions = 0
    failed_conversions = []

    # Process each class directory
    for class_name in class_dirs:
        input_class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)

        # Create output class directory
        os.makedirs(output_class_dir, exist_ok=True)

        # Get all .pose files in this class
        pose_files = sorted([f for f in os.listdir(input_class_dir)
                            if f.endswith('.pose')])

        if not pose_files:
            continue

        print(f"Processing {class_name}: {len(pose_files)} files")
        total_files += len(pose_files)

        # Convert each pose file
        for pose_file in tqdm(pose_files, desc=f"  {class_name}", ncols=80):
            try:
                input_path = os.path.join(input_class_dir, pose_file)

                # Load pose file
                with open(input_path, 'rb') as f:
                    pose = Pose.read(f.read())

                # Convert to numpy
                keypoints, confidences = pose_to_numpy(pose)

                if keypoints is not None:
                    # Extract metadata from filename and directory
                    video_id = pose_file.replace('.pose', '')
                    gloss = class_name if class_name != '.' else 'unknown'

                    # Check if it's an augmented file
                    is_augmented = video_id.startswith('aug_')
                    augmentation_id = None
                    original_video_id = video_id

                    if is_augmented:
                        # Extract augmentation ID (e.g., "aug_00_00623" -> aug_id=0, video_id="00623")
                        parts = video_id.split('_')
                        if len(parts) >= 3:
                            try:
                                augmentation_id = int(parts[1])
                                original_video_id = '_'.join(parts[2:])
                            except ValueError:
                                pass

                    # Create pickle data
                    pickle_data = {
                        'keypoints': keypoints,
                        'confidences': confidences,
                        'video_id': original_video_id,
                        'gloss': gloss,
                        'split': 'train',  # Default to train
                        'augmented': is_augmented,
                        'augmentation_id': augmentation_id,
                        'pose_file': input_path
                    }

                    # Save pickle file (keep same filename, just change extension)
                    output_filename = pose_file.replace('.pose', '.pkl')
                    output_path = os.path.join(output_class_dir, output_filename)

                    with open(output_path, 'wb') as f:
                        pickle.dump(pickle_data, f)

                    successful_conversions += 1
                else:
                    failed_conversions.append(f"{class_name}/{pose_file} (pose conversion failed)")

            except Exception as e:
                failed_conversions.append(f"{class_name}/{pose_file} (exception: {str(e)})")

        print(f"  Completed: {class_name}")
        print()

    # Report results
    print("=" * 70)
    print("CONVERSION COMPLETE!")
    print("=" * 70)
    print()
    print(f"Total files processed: {total_files}")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {len(failed_conversions)}")

    if successful_conversions > 0:
        success_rate = successful_conversions/(successful_conversions+len(failed_conversions))*100
        print(f"Success rate: {success_rate:.1f}%")

    if len(failed_conversions) > 0:
        print(f"\nFirst 10 failed conversions:")
        for failure in failed_conversions[:10]:
            print(f"  - {failure}")

    print()
    print(f"Output directory: {output_dir}")
    print()


def main():
    """Main conversion logic with command-line arguments."""
    parser = argparse.ArgumentParser(description='Convert .pose files to pickle format')
    parser.add_argument('--input-file', type=str, default=None,
                        help='Single .pose file to convert (outputs to same directory with .pkl extension)')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Input directory containing .pose files with class subdirectories')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for pickle files (required with --input-dir)')

    args = parser.parse_args()

    # Validate arguments
    if args.input_file:
        # Single file mode
        convert_single_pose_file(args.input_file)
    elif args.input_dir and args.output_dir:
        # Directory mode
        convert_pose_directory(args.input_dir, args.output_dir)
    else:
        parser.error("Either --input-file OR (--input-dir AND --output-dir) must be provided")


if __name__ == "__main__":
    main()
