#!/usr/bin/env python3
"""
Generate 75-point augmented dataset for WLASL training.

This script:
1. For pickle input: Loads pickle files (576 keypoints), extracts 75-point subset, applies augmentation
2. For pose input: Loads .pose files, applies geometric transformations directly
3. Saves augmented files in experiments/{N}-classes/ folder

Supports both pickle and pose file augmentation via --input-type parameter.
"""

import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pose_format import Pose

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openhands_modernized import WLASLPoseProcessor, PoseTransforms

# Import pose augmentation functions
dataset_utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dataset_utils_dir)
from augment_pose_file import apply_shear, apply_rotation

# ============================================================================
# CONFIGURATION - All paths and settings
# ============================================================================

# Base directories
WLASL_BASE_DIR = "C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/wlasl_poses_complete"
EXPERIMENTS_BASE_DIR = "C:/Users/padwe/OneDrive/WLASL-proj/OpenHands-Modernized/experiments"

# Will be set based on command-line arguments
INPUT_DIR = None
OUTPUT_DIR = None

# Augmentation settings
NUM_AUGMENTATIONS = 10  # Generate 10 augmented versions per sample
SHEAR_STD = 0.1
ROTATION_STD = 0.1

# Base class order (same as in split_pose_files_nclass.py)
BASE_CLASS_ORDER = [
    # Top 20
    'accident', 'apple', 'bath', 'before', 'blue', 'chair', 'clothes',
    'cousin', 'deaf', 'doctor', 'eat', 'enjoy', 'forget', 'give', 'go',
    'graduation', 'halloween', 'help', 'hot', 'hurry',
    # Next 30 (for 50-class)
    'book', 'drink', 'computer', 'who', 'candy', 'walk', 'thin', 'no',
    'fine', 'year', 'yes', 'table', 'now', 'what', 'finish', 'black',
    'thanksgiving', 'all', 'many', 'like', 'cool', 'orange', 'mother',
    'woman', 'dog', 'hearing', 'tall', 'wrong', 'kiss', 'man',
    # Next 50 (for 100-class)
    'family', 'graduate', 'bed', 'language', 'fish', 'hat', 'bowling',
    'shirt', 'later', 'white', 'study', 'can', 'bird', 'pink', 'want',
    'time', 'dance', 'play', 'color', 'summer', 'winter', 'spring',
    'fall', 'school', 'work', 'home', 'car', 'train', 'airplane',
    'bus', 'bicycle', 'run', 'jump', 'sit', 'stand', 'sleep', 'wake',
    'morning', 'afternoon', 'evening', 'night', 'day', 'week', 'month',
    'yesterday', 'today', 'tomorrow', 'happy', 'sad', 'angry', 'tired'
]


# ============================================================================
# PICKLE FILE FUNCTIONS
# ============================================================================

def load_and_extract_75pt(pickle_path):
    """
    Load pickle file and extract 75-point pose+hands representation.

    Returns:
        pose_data: (frames, 75, 2) - x, y coordinates
        metadata: dict with video info
    """
    processor = WLASLPoseProcessor()

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    # Get keypoints (frames, 576, 2)
    keypoints = data['keypoints']

    # Extract 75-point subset using MediaPipeSubset
    from openhands_modernized import MediaPipeSubset
    pose_75pt = MediaPipeSubset.extract_pose_hands_75(keypoints)

    # Metadata
    metadata = {
        'gloss': data.get('gloss', 'unknown'),
        'video_id': data.get('video_id', 'unknown'),
        'split': data.get('split', 'train'),
        'original_file': pickle_path
    }

    return pose_75pt, metadata


def apply_augmentation_pickle(pose_data, augmentation_id):
    """
    Apply augmentation to 75-point pose data (pickle mode).

    Args:
        pose_data: (frames, 75, 2)
        augmentation_id: int, for seeding

    Returns:
        augmented_data: (frames, 75, 2)
    """
    # Set seed for reproducibility
    np.random.seed(augmentation_id)

    transforms = PoseTransforms()

    # Apply transformations
    augmented = pose_data.copy()
    augmented = transforms.center_and_scale_normalize(augmented)
    augmented = transforms.apply_shear(augmented, shear_std=SHEAR_STD)
    augmented = transforms.apply_rotation(augmented, rotation_std=ROTATION_STD)

    return augmented


def save_augmented_pickle(pose_data, metadata, output_path, augmentation_id):
    """
    Save augmented pose data as pickle file.

    Format matches original pickle structure for compatibility.
    """
    augmented_data = {
        'keypoints': pose_data,  # (frames, 75, 2)
        'gloss': metadata['gloss'],
        'video_id': metadata['video_id'],
        'split': metadata['split'],
        'augmented': True,
        'augmentation_id': augmentation_id,
        'original_file': metadata['original_file'],
        'num_keypoints': 75,
        'keypoint_format': 'pose_hands_75pt'
    }

    with open(output_path, 'wb') as f:
        pickle.dump(augmented_data, f)


# ============================================================================
# POSE FILE FUNCTIONS
# ============================================================================

def load_pose_file(pose_path):
    """
    Load .pose file.

    Returns:
        pose: Pose object
        metadata: dict with video info
    """
    with open(pose_path, 'rb') as f:
        pose = Pose.read(f.read())

    # Extract video ID from filename
    video_id = os.path.basename(pose_path).replace('.pose', '')

    # Get gloss from parent directory name
    gloss = os.path.basename(os.path.dirname(pose_path))

    metadata = {
        'gloss': gloss,
        'video_id': video_id,
        'split': 'train',
        'original_file': pose_path
    }

    return pose, metadata


def apply_augmentation_pose(pose_data, augmentation_id):
    """
    Apply augmentation to pose data (pose mode).
    Uses imported functions from augment_pose_file.py

    Args:
        pose_data: (frames, people, keypoints, dims)
        augmentation_id: int, for seeding

    Returns:
        augmented_data: (frames, people, keypoints, dims)
    """
    # Set seed for reproducibility
    np.random.seed(augmentation_id)

    # Apply geometric transformations (using imported functions)
    augmented = pose_data.copy()
    augmented = apply_shear(augmented, shear_std=SHEAR_STD)
    augmented = apply_rotation(augmented, rotation_std=ROTATION_STD)

    return augmented


def save_augmented_pose(pose_data, original_pose, metadata, output_path, augmentation_id):
    """
    Save augmented pose data as .pose file.
    """
    # Create new pose with augmented data
    augmented_pose = Pose(
        header=original_pose.header,
        body=original_pose.body.__class__(
            fps=original_pose.body.fps,
            data=pose_data,
            confidence=original_pose.body.confidence
        )
    )

    # Write to file
    with open(output_path, 'wb') as f:
        augmented_pose.write(f)


# ============================================================================
# MAIN AUGMENTATION FUNCTION
# ============================================================================

def generate_augmented_dataset(input_type='pickle', num_classes=20, input_dir=None, output_dir=None):
    """
    Generate complete augmented dataset.

    Args:
        input_type: 'pickle' or 'pose'
        num_classes: number of classes (default 20)
        input_dir: custom input directory (overrides default path)
        output_dir: custom output directory (overrides default path)
    """
    # Set paths based on input type
    global INPUT_DIR, OUTPUT_DIR

    if input_dir and output_dir:
        # Use custom directories
        INPUT_DIR = input_dir
        OUTPUT_DIR = output_dir
        file_extension = '.pkl' if input_type == 'pickle' else '.pose'
    else:
        # Use default directory structure
        class_folder = f"{num_classes}-classes"
        experiments_dir = os.path.join(EXPERIMENTS_BASE_DIR, class_folder)
        os.makedirs(experiments_dir, exist_ok=True)

        if input_type == 'pickle':
            INPUT_DIR = os.path.join(WLASL_BASE_DIR, f"conservative_split_{num_classes}_class", "train")
            OUTPUT_DIR = os.path.join(experiments_dir, "augmented_75pt_pickle")
            file_extension = '.pkl'
        else:  # pose
            INPUT_DIR = os.path.join(WLASL_BASE_DIR, f"pose_split_{num_classes}_class", "train")
            OUTPUT_DIR = os.path.join(experiments_dir, "augmented_75pt_pose")
            file_extension = '.pose'

    print("=" * 70)
    print(f"GENERATING 75-POINT AUGMENTED DATASET ({input_type.upper()} MODE)")
    print("=" * 70)
    print()
    print(f"Input type: {input_type}")
    print(f"Number of classes: {num_classes}")
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Augmentations per sample: {NUM_AUGMENTATIONS}")
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all class directories
    all_class_dirs = sorted([d for d in os.listdir(INPUT_DIR)
                            if os.path.isdir(os.path.join(INPUT_DIR, d))])

    # Filter by num_classes if specified
    if num_classes and num_classes > 0:
        # Get target classes from BASE_CLASS_ORDER
        target_classes = BASE_CLASS_ORDER[:num_classes]
        # Filter to only include target classes that exist
        class_dirs = [c for c in target_classes if c in all_class_dirs]
        print(f"Filtering to top {num_classes} classes (found {len(class_dirs)} with data)")
    else:
        class_dirs = all_class_dirs
        print(f"Processing all {len(class_dirs)} classes")

    print(f"Classes to process:")
    for class_name in class_dirs:
        print(f"  - {class_name}")
    print()

    total_files = 0
    total_augmented = 0

    # Process each class
    for class_name in class_dirs:
        print(f"Processing: {class_name}")

        input_class_dir = os.path.join(INPUT_DIR, class_name)
        output_class_dir = os.path.join(OUTPUT_DIR, class_name)

        # Create output class directory
        os.makedirs(output_class_dir, exist_ok=True)

        # Get all files in this class
        input_files = sorted([f for f in os.listdir(input_class_dir)
                              if f.endswith(file_extension)])

        print(f"  Files: {len(input_files)}")

        # Process each file
        for input_file in tqdm(input_files, desc=f"  {class_name}", ncols=80):
            input_path = os.path.join(input_class_dir, input_file)

            # Check if already augmented (check for first augmentation file)
            base_name = input_file.replace(file_extension, '')
            first_aug_filename = f"aug_00_{base_name}{file_extension}"
            if os.path.exists(os.path.join(output_class_dir, first_aug_filename)):
                # Already augmented, skip
                total_files += 1
                total_augmented += NUM_AUGMENTATIONS
                continue

            try:
                if input_type == 'pickle':
                    # Load and extract 75-point data
                    pose_data, metadata = load_and_extract_75pt(input_path)

                    # Generate augmented versions
                    for aug_id in range(NUM_AUGMENTATIONS):
                        # Apply augmentation
                        augmented_pose = apply_augmentation_pickle(pose_data, aug_id)

                        # Create output filename
                        base_name = input_file.replace('.pkl', '')
                        output_filename = f"aug_{aug_id:02d}_{base_name}.pkl"
                        output_path = os.path.join(output_class_dir, output_filename)

                        # Save augmented pickle
                        save_augmented_pickle(augmented_pose, metadata, output_path, aug_id)

                        total_augmented += 1

                else:  # pose
                    # Load pose file
                    original_pose, metadata = load_pose_file(input_path)

                    # Generate augmented versions
                    for aug_id in range(NUM_AUGMENTATIONS):
                        # Apply augmentation
                        augmented_data = apply_augmentation_pose(original_pose.body.data, aug_id)

                        # Create output filename
                        base_name = input_file.replace('.pose', '')
                        output_filename = f"aug_{aug_id:02d}_{base_name}.pose"
                        output_path = os.path.join(output_class_dir, output_filename)

                        # Save augmented pose
                        save_augmented_pose(augmented_data, original_pose, metadata, output_path, aug_id)

                        total_augmented += 1

                total_files += 1

            except Exception as e:
                print(f"\n  ERROR processing {input_file}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"  Completed: {len(input_files)} files -> {len(input_files) * NUM_AUGMENTATIONS} augmented")
        print()

    # Save summary
    summary = {
        'input_type': input_type,
        'num_classes': len(class_dirs),
        'classes': class_dirs,
        'total_original_files': total_files,
        'total_augmented_files': total_augmented,
        'augmentations_per_sample': NUM_AUGMENTATIONS,
        'shear_std': SHEAR_STD,
        'rotation_std': ROTATION_STD
    }

    if input_type == 'pickle':
        summary['num_keypoints'] = 75
        summary['keypoint_format'] = 'pose_hands_75pt (33 pose + 42 hands)'

    summary_path = os.path.join(OUTPUT_DIR, f'augmentation_summary_{input_type}.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)

    print("=" * 70)
    print("GENERATION COMPLETE!")
    print("=" * 70)
    print()
    print(f"Input type: {input_type}")
    print(f"Total classes: {len(class_dirs)}")
    print(f"Total original files: {total_files}")
    print(f"Total augmented files: {total_augmented}")
    print(f"Augmentations per sample: {NUM_AUGMENTATIONS}")
    print()
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Summary file: {summary_path}")
    print()

    if input_type == 'pickle':
        print("Dataset structure:")
        print("  - 75 keypoints (33 pose + 21 left hand + 21 right hand)")
        print("  - Face landmarks EXCLUDED (468 points removed)")
        print("  - Augmentation: shear + rotation applied to 75-point data")
    else:
        print("Dataset structure:")
        print("  - Full pose format (.pose files)")
        print("  - Augmentation: shear + rotation applied directly (no normalization)")
    print()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate augmented dataset for WLASL')
    parser.add_argument('--input-type', choices=['pickle', 'pose'], default='pickle',
                        help='Input file type (pickle or pose)')
    parser.add_argument('--classes', type=int, default=20,
                        help='Number of classes (default: 20)')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Custom input directory (overrides default path)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output directory (overrides default path)')
    args = parser.parse_args()

    generate_augmented_dataset(input_type=args.input_type, num_classes=args.classes,
                              input_dir=args.input_dir, output_dir=args.output_dir)
