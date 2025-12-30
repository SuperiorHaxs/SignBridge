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
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pose_format import Pose

# Add project root to path for config import
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import configuration
from config import get_config

# Add openhands model path from config
config = get_config()
openhands_src = config.openhands_dir / "src"
sys.path.insert(0, str(openhands_src))

# Import the OpenHands modules
from openhands_modernized import WLASLPoseProcessor, PoseTransforms

# Import pose augmentation functions
dataset_utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dataset_utils_dir)
from augment_pose_file import apply_shear, apply_rotation

# ============================================================================
# CONFIGURATION - Now managed by config system
# ============================================================================

# Base directories
WLASL_BASE_DIR = str(config.dataset_root)
EXPERIMENTS_BASE_DIR = str(config.experiments_dir)

# Will be set based on command-line arguments
INPUT_DIR = None
OUTPUT_DIR = None

# ============================================================================
# AUGMENTATION CONFIGURATION - Set count for each strategy (0 = disabled)
# ============================================================================

# Augmentation strategy counts (set to 0 to disable)
AUGMENTATION_CONFIG = {
    # Phase 1: Geometric transformations
    'geometric': 12,           # Shear + Rotation combinations (aug_00 to aug_11) - Increased from 10

    # Phase 2: Horizontal flips
    'horizontal_flip': 6,      # Mirror + variations (aug_12 to aug_17) - Increased from 5

    # Phase 3: Spatial variations
    'spatial_noise': 3,        # Gaussian noise - low/medium/high (aug_18 to aug_20) - Increased from 2
    'translation': 3,          # Position shift - small/medium/large (aug_21 to aug_23) - Increased from 2

    # Phase 4: Scaling
    'scaling': 3,              # Size variations - 0.90x/1.0x/1.10x (aug_24 to aug_26) - Increased from 2

    # Phase 5: Advanced temporal
    'speed_variation': 4,      # Temporal stretch/compress - 0.65x/0.75x/1.25x/1.35x (aug_27 to aug_30) - Increased from 2

    # Phase 6: Occlusion (NEW - simulates missing/blocked keypoints)
    'keypoint_occlusion': 3,   # Random keypoint dropout - 10%/15%/20% (aug_31 to aug_33) - NEW
    'hand_dropout': 2,         # Entire hand dropout - left/right (aug_34 to aug_35) - NEW

    # Phase 7: Combinations
    'combinations': 4,         # Multi-transform combos (aug_36 to aug_39) - Increased from 3
}

# Calculate total augmentations (sum of all non-zero counts)
NUM_AUGMENTATIONS = sum(count for count in AUGMENTATION_CONFIG.values() if count > 0)

# Augmentation parameters (INCREASED to match test distribution gap)
# Analysis showed: test has ~40% larger scale, different framing, 50% longer videos, 50% more missing keypoints
SHEAR_STD = 0.12                    # Increased from 0.08 for more geometric variation
ROTATION_STD = 0.12                 # Increased from 0.08 (~±6° rotation)
NOISE_LOW = 0.010                   # Slight increase for robustness
NOISE_MEDIUM = 0.020                # Increased from 0.015
TRANSLATION_SMALL = 0.10            # Increased from 0.025 - test has different camera framing
TRANSLATION_MEDIUM = 0.20           # Increased from 0.05 - significant position shift
SCALE_SMALL = 0.70                  # Increased range from 0.93 - test subjects are 40% larger
SCALE_LARGE = 1.50                  # Increased range from 1.07 - cover larger scale variation
SPEED_SLOW = 0.60                   # Increased from 0.85 - test videos are 50% longer
SPEED_FAST = 1.60                   # Increased from 1.15 - cover faster signing
SPEED_VERY_SLOW = 0.50              # Increased from 0.75 - extreme temporal variation
SPEED_VERY_FAST = 1.80              # Increased from 1.25 - extreme speed variation

# Occlusion parameters (INCREASED to match test's 32% missing keypoints vs train's 21%)
KEYPOINT_OCCLUSION_LOW = 0.15       # Increased from 0.05 to 15%
KEYPOINT_OCCLUSION_MEDIUM = 0.25    # Increased from 0.08 to 25%
KEYPOINT_OCCLUSION_HIGH = 0.35      # Increased from 0.12 to 35%
HAND_DROPOUT_PROB = 0.15            # Increased from 0.08 to 15%

# ============================================================================
# DYNAMIC CLASS LOADING - No hardcoded lists!
# ============================================================================

def load_class_mapping(num_classes):
    """
    Load class list from class_mapping.json dynamically.

    Args:
        num_classes: Number of classes (e.g., 20, 50, 100)

    Returns:
        List of class names in order, or None if not found
    """
    # Try standard location first: dataset_splits/{N}_classes/{N}_class_mapping.json
    class_mapping_path = os.path.join(
        WLASL_BASE_DIR,
        f"dataset_splits/{num_classes}_classes/{num_classes}_class_mapping.json"
    )

    if os.path.exists(class_mapping_path):
        try:
            with open(class_mapping_path, 'r') as f:
                mapping = json.load(f)

            # Extract classes list from mapping
            if 'classes' in mapping:
                print(f"Loaded {len(mapping['classes'])} classes from {class_mapping_path}")
                return mapping['classes']
            else:
                print(f"WARNING: 'classes' field not found in {class_mapping_path}")
                return None
        except Exception as e:
            print(f"WARNING: Failed to load {class_mapping_path}: {e}")
            return None
    else:
        print(f"WARNING: Class mapping not found at {class_mapping_path}")
        return None


def get_classes_from_directory(input_dir):
    """
    Fallback: Discover classes by scanning the input directory.

    Args:
        input_dir: Path to input directory containing class subdirectories

    Returns:
        Sorted list of class names
    """
    if not os.path.exists(input_dir):
        return []

    classes = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])

    return classes


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


def apply_horizontal_flip(pose_data):
    """
    Apply horizontal flip to pose data, swapping left/right hand keypoints.

    Args:
        pose_data: (frames, 75, 2)

    Returns:
        flipped_data: (frames, 75, 2)
    """
    flipped = pose_data.copy()
    # Flip x-coordinates
    flipped[:, :, 0] = -flipped[:, :, 0]

    # Swap left/right hand keypoints (indices 33-53 left, 54-74 right)
    left_hand = flipped[:, 33:54, :].copy()
    right_hand = flipped[:, 54:75, :].copy()
    flipped[:, 33:54, :] = right_hand
    flipped[:, 54:75, :] = left_hand

    return flipped


def apply_spatial_noise(pose_data, noise_std=0.01):
    """
    Add Gaussian noise to keypoint positions.

    Args:
        pose_data: (frames, 75, 2)
        noise_std: Standard deviation of noise

    Returns:
        noisy_data: (frames, 75, 2)
    """
    noise = np.random.normal(0, noise_std, pose_data.shape)
    # Only add noise to non-zero keypoints
    mask = (pose_data != 0).astype(float)
    return pose_data + (noise * mask)


def apply_translation(pose_data, shift_std=0.05):
    """
    Translate pose by random offset.

    Args:
        pose_data: (frames, 75, 2)
        shift_std: Standard deviation of translation

    Returns:
        translated_data: (frames, 75, 2)
    """
    shift_x = np.random.normal(0, shift_std)
    shift_y = np.random.normal(0, shift_std)
    translated = pose_data.copy()
    translated[:, :, 0] += shift_x
    translated[:, :, 1] += shift_y
    return translated


def apply_scaling(pose_data, scale_factor):
    """
    Scale pose size around center.

    Args:
        pose_data: (frames, 75, 2)
        scale_factor: Scale multiplier (e.g., 0.95 or 1.05)

    Returns:
        scaled_data: (frames, 75, 2)
    """
    scaled = pose_data.copy()
    # Find center of non-zero keypoints
    valid_mask = (pose_data != 0).any(axis=2)
    if valid_mask.any():
        center = pose_data[valid_mask].mean(axis=0)
        # Scale around center
        scaled[:, :, :] = (scaled[:, :, :] - center) * scale_factor + center
    return scaled


def apply_speed_variation(pose_data, speed_factor):
    """
    Apply temporal stretching or compression.

    Args:
        pose_data: (frames, 75, 2)
        speed_factor: Speed multiplier (0.85 = slow, 1.15 = fast)

    Returns:
        resampled_data: (new_frames, 75, 2)
    """
    num_frames = pose_data.shape[0]
    target_frames = int(num_frames * speed_factor)

    if target_frames < 2:
        target_frames = 2

    # Interpolate frames
    indices = np.linspace(0, num_frames - 1, target_frames)
    resampled = np.zeros((target_frames, pose_data.shape[1], pose_data.shape[2]))

    for kp_idx in range(pose_data.shape[1]):
        for coord_idx in range(pose_data.shape[2]):
            resampled[:, kp_idx, coord_idx] = np.interp(
                indices,
                np.arange(num_frames),
                pose_data[:, kp_idx, coord_idx]
            )

    return resampled


def apply_keypoint_occlusion(pose_data, occlusion_prob=0.15):
    """
    Randomly occlude keypoints to simulate tracking errors and occlusions.

    Args:
        pose_data: (frames, 75, 2) - x, y coordinates
        occlusion_prob: Probability of each keypoint being occluded (0-1)

    Returns:
        occluded_data: (frames, 75, 2) with random keypoints set to 0
    """
    occluded = pose_data.copy()
    # Generate random mask: True = keep, False = occlude
    # Shape: (frames, 75)
    mask = np.random.random(pose_data.shape[:2]) > occlusion_prob
    # Apply mask to both x and y coordinates
    occluded = occluded * mask[:, :, np.newaxis]
    return occluded


def apply_hand_dropout(pose_data, dropout_prob=0.15):
    """
    Randomly drop entire hand to simulate severe occlusion (hand behind body, etc.).

    Args:
        pose_data: (frames, 75, 2)
        dropout_prob: Probability of dropping a hand (0-1)

    Returns:
        data_with_dropout: (frames, 75, 2) with one hand potentially set to 0
    """
    dropped = pose_data.copy()

    if np.random.random() < dropout_prob:
        # Randomly choose left hand (33-53) or right hand (54-74)
        if np.random.random() < 0.5:
            # Drop left hand (21 keypoints)
            dropped[:, 33:54, :] = 0
        else:
            # Drop right hand (21 keypoints)
            dropped[:, 54:75, :] = 0

    return dropped


def apply_augmentation_pickle(pose_data, augmentation_id):
    """
    Apply augmentation to 75-point pose data (pickle mode).
    Uses AUGMENTATION_CONFIG to determine which transforms to apply.

    Args:
        pose_data: (frames, 75, 2)
        augmentation_id: int, determines which augmentation to apply

    Returns:
        augmented_data: (frames, 75, 2)
    """
    # Set seed for reproducibility
    np.random.seed(augmentation_id)

    transforms = PoseTransforms()
    augmented = pose_data.copy()

    # Determine augmentation strategy based on ID
    current_id = augmentation_id

    # Phase 1: Geometric
    if AUGMENTATION_CONFIG['geometric'] > 0 and current_id < AUGMENTATION_CONFIG['geometric']:
        augmented = transforms.center_and_scale_normalize(augmented)
        augmented = transforms.apply_shear(augmented, shear_std=SHEAR_STD)
        augmented = transforms.apply_rotation(augmented, rotation_std=ROTATION_STD)
        return augmented

    if AUGMENTATION_CONFIG['geometric'] > 0:
        current_id -= AUGMENTATION_CONFIG['geometric']

    # Phase 2: Horizontal Flip
    if AUGMENTATION_CONFIG['horizontal_flip'] > 0 and current_id < AUGMENTATION_CONFIG['horizontal_flip']:
        augmented = apply_horizontal_flip(augmented)
        if current_id == 1:  # Flip + rotation
            augmented = transforms.apply_rotation(augmented, rotation_std=ROTATION_STD * 0.5)
        elif current_id == 2:  # Flip + shear
            augmented = transforms.apply_shear(augmented, shear_std=SHEAR_STD * 0.5)
        elif current_id == 3:  # Flip + noise
            augmented = apply_spatial_noise(augmented, noise_std=NOISE_LOW)
        elif current_id == 4:  # Flip + translation
            augmented = apply_translation(augmented, shift_std=TRANSLATION_SMALL)
        return augmented

    if AUGMENTATION_CONFIG['horizontal_flip'] > 0:
        current_id -= AUGMENTATION_CONFIG['horizontal_flip']

    # Phase 3: Spatial Noise
    if AUGMENTATION_CONFIG['spatial_noise'] > 0 and current_id < AUGMENTATION_CONFIG['spatial_noise']:
        noise_std = NOISE_LOW if current_id == 0 else NOISE_MEDIUM
        augmented = apply_spatial_noise(augmented, noise_std=noise_std)
        return augmented

    if AUGMENTATION_CONFIG['spatial_noise'] > 0:
        current_id -= AUGMENTATION_CONFIG['spatial_noise']

    # Phase 4: Translation
    if AUGMENTATION_CONFIG['translation'] > 0 and current_id < AUGMENTATION_CONFIG['translation']:
        shift_std = TRANSLATION_SMALL if current_id == 0 else TRANSLATION_MEDIUM
        augmented = apply_translation(augmented, shift_std=shift_std)
        return augmented

    if AUGMENTATION_CONFIG['translation'] > 0:
        current_id -= AUGMENTATION_CONFIG['translation']

    # Phase 5: Scaling
    if AUGMENTATION_CONFIG['scaling'] > 0 and current_id < AUGMENTATION_CONFIG['scaling']:
        scale_factor = SCALE_SMALL if current_id == 0 else SCALE_LARGE
        augmented = apply_scaling(augmented, scale_factor=scale_factor)
        return augmented

    if AUGMENTATION_CONFIG['scaling'] > 0:
        current_id -= AUGMENTATION_CONFIG['scaling']

    # Phase 6: Speed Variation (Updated to 4 variants)
    if AUGMENTATION_CONFIG['speed_variation'] > 0 and current_id < AUGMENTATION_CONFIG['speed_variation']:
        if current_id == 0:
            speed_factor = SPEED_SLOW          # 0.75x
        elif current_id == 1:
            speed_factor = SPEED_FAST          # 1.25x
        elif current_id == 2:
            speed_factor = SPEED_VERY_SLOW     # 0.65x
        else:  # current_id == 3
            speed_factor = SPEED_VERY_FAST     # 1.35x

        augmented = apply_speed_variation(augmented, speed_factor=speed_factor)
        return augmented

    if AUGMENTATION_CONFIG['speed_variation'] > 0:
        current_id -= AUGMENTATION_CONFIG['speed_variation']

    # Phase 7: Keypoint Occlusion (NEW)
    if AUGMENTATION_CONFIG['keypoint_occlusion'] > 0 and current_id < AUGMENTATION_CONFIG['keypoint_occlusion']:
        if current_id == 0:
            occlusion_prob = KEYPOINT_OCCLUSION_LOW       # 10%
        elif current_id == 1:
            occlusion_prob = KEYPOINT_OCCLUSION_MEDIUM    # 15%
        else:  # current_id == 2
            occlusion_prob = KEYPOINT_OCCLUSION_HIGH      # 20%

        augmented = apply_keypoint_occlusion(augmented, occlusion_prob=occlusion_prob)
        return augmented

    if AUGMENTATION_CONFIG['keypoint_occlusion'] > 0:
        current_id -= AUGMENTATION_CONFIG['keypoint_occlusion']

    # Phase 8: Hand Dropout (NEW)
    if AUGMENTATION_CONFIG['hand_dropout'] > 0 and current_id < AUGMENTATION_CONFIG['hand_dropout']:
        # Variant 1: Standard dropout
        # Variant 2: Dropout + slight noise to remaining hand for realism
        augmented = apply_hand_dropout(augmented, dropout_prob=HAND_DROPOUT_PROB)
        if current_id == 1:
            # Add slight noise to make it more realistic
            augmented = apply_spatial_noise(augmented, noise_std=NOISE_LOW)
        return augmented

    if AUGMENTATION_CONFIG['hand_dropout'] > 0:
        current_id -= AUGMENTATION_CONFIG['hand_dropout']

    # Phase 9: Combinations (Updated to 4 variants)
    if AUGMENTATION_CONFIG['combinations'] > 0 and current_id < AUGMENTATION_CONFIG['combinations']:
        if current_id == 0:  # Rotation + Noise
            augmented = transforms.apply_rotation(augmented, rotation_std=ROTATION_STD)
            augmented = apply_spatial_noise(augmented, noise_std=NOISE_LOW)
        elif current_id == 1:  # Shear + Translation
            augmented = transforms.apply_shear(augmented, shear_std=SHEAR_STD)
            augmented = apply_translation(augmented, shift_std=TRANSLATION_SMALL)
        elif current_id == 2:  # Flip + Scaling
            augmented = apply_horizontal_flip(augmented)
            augmented = apply_scaling(augmented, scale_factor=SCALE_LARGE)
        elif current_id == 3:  # Speed + Occlusion (NEW combination)
            augmented = apply_speed_variation(augmented, speed_factor=SPEED_FAST)
            augmented = apply_keypoint_occlusion(augmented, occlusion_prob=KEYPOINT_OCCLUSION_LOW)
        return augmented

    # Fallback: return original (shouldn't reach here)
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


def apply_horizontal_flip_pose(pose_data):
    """
    Apply horizontal flip to pose data (4D format).

    Args:
        pose_data: (frames, people, keypoints, dims)

    Returns:
        flipped_data: (frames, people, keypoints, dims)
    """
    flipped = pose_data.copy()
    # Flip x-coordinates (dim 0)
    flipped[:, :, :, 0] = -flipped[:, :, :, 0]

    # Swap left/right hand keypoints
    # MediaPipe: body (0-32), left hand (33-53), right hand (54-74)
    left_hand = flipped[:, :, 33:54, :].copy()
    right_hand = flipped[:, :, 54:75, :].copy()
    flipped[:, :, 33:54, :] = right_hand
    flipped[:, :, 54:75, :] = left_hand

    return flipped


def apply_spatial_noise_pose(pose_data, noise_std=0.01):
    """
    Add Gaussian noise to pose data (4D format).

    Args:
        pose_data: (frames, people, keypoints, dims)
        noise_std: Standard deviation of noise

    Returns:
        noisy_data: (frames, people, keypoints, dims)
    """
    noise = np.random.normal(0, noise_std, pose_data.shape)
    # Only add noise to non-zero keypoints
    mask = (pose_data != 0).astype(float)
    return pose_data + (noise * mask)


def apply_translation_pose(pose_data, shift_std=0.05):
    """
    Translate pose data (4D format).

    Args:
        pose_data: (frames, people, keypoints, dims)
        shift_std: Standard deviation of translation

    Returns:
        translated_data: (frames, people, keypoints, dims)
    """
    shift_x = np.random.normal(0, shift_std)
    shift_y = np.random.normal(0, shift_std)
    translated = pose_data.copy()
    translated[:, :, :, 0] += shift_x
    translated[:, :, :, 1] += shift_y
    return translated


def apply_scaling_pose(pose_data, scale_factor):
    """
    Scale pose data (4D format).

    Args:
        pose_data: (frames, people, keypoints, dims)
        scale_factor: Scale multiplier

    Returns:
        scaled_data: (frames, people, keypoints, dims)
    """
    scaled = pose_data.copy()
    # Find center of non-zero keypoints
    valid_mask = (pose_data != 0).any(axis=3)
    if valid_mask.any():
        center = pose_data[valid_mask].mean(axis=0)[:2]  # x, y center
        # Scale around center
        scaled[:, :, :, :2] = (scaled[:, :, :, :2] - center) * scale_factor + center
    return scaled


def apply_speed_variation_pose(pose_data, speed_factor):
    """
    Apply temporal stretching/compression to pose data (4D format).

    Args:
        pose_data: (frames, people, keypoints, dims)
        speed_factor: Speed multiplier

    Returns:
        resampled_data: (new_frames, people, keypoints, dims)
    """
    num_frames = pose_data.shape[0]
    target_frames = int(num_frames * speed_factor)

    if target_frames < 2:
        target_frames = 2

    # Interpolate frames
    indices = np.linspace(0, num_frames - 1, target_frames)
    resampled = np.zeros((target_frames, pose_data.shape[1], pose_data.shape[2], pose_data.shape[3]))

    for person_idx in range(pose_data.shape[1]):
        for kp_idx in range(pose_data.shape[2]):
            for dim_idx in range(pose_data.shape[3]):
                resampled[:, person_idx, kp_idx, dim_idx] = np.interp(
                    indices,
                    np.arange(num_frames),
                    pose_data[:, person_idx, kp_idx, dim_idx]
                )

    return resampled


def apply_keypoint_occlusion_pose(pose_data, occlusion_prob=0.15):
    """
    Randomly occlude keypoints in pose data (4D format).

    Args:
        pose_data: (frames, people, keypoints, dims)
        occlusion_prob: Probability of each keypoint being occluded

    Returns:
        occluded_data: (frames, people, keypoints, dims)
    """
    occluded = pose_data.copy()
    # Generate random mask: True = keep, False = occlude
    # Shape: (frames, people, keypoints)
    mask = np.random.random(pose_data.shape[:3]) > occlusion_prob
    # Apply mask to all dimensions
    occluded = occluded * mask[:, :, :, np.newaxis]
    return occluded


def apply_hand_dropout_pose(pose_data, dropout_prob=0.15):
    """
    Randomly drop entire hand in pose data (4D format).

    Args:
        pose_data: (frames, people, keypoints, dims)
        dropout_prob: Probability of dropping a hand

    Returns:
        data_with_dropout: (frames, people, keypoints, dims)
    """
    dropped = pose_data.copy()

    if np.random.random() < dropout_prob:
        # Randomly choose left hand (33-53) or right hand (54-74)
        if np.random.random() < 0.5:
            # Drop left hand (21 keypoints)
            dropped[:, :, 33:54, :] = 0
        else:
            # Drop right hand (21 keypoints)
            dropped[:, :, 54:75, :] = 0

    return dropped


def apply_augmentation_pose(pose_data, augmentation_id):
    """
    Apply augmentation to pose data (pose mode).
    Uses AUGMENTATION_CONFIG to determine which transforms to apply.

    Args:
        pose_data: (frames, people, keypoints, dims)
        augmentation_id: int, determines which augmentation to apply

    Returns:
        augmented_data: (frames, people, keypoints, dims)
    """
    # Set seed for reproducibility
    np.random.seed(augmentation_id)

    augmented = pose_data.copy()
    current_id = augmentation_id

    # Phase 1: Geometric
    if AUGMENTATION_CONFIG['geometric'] > 0 and current_id < AUGMENTATION_CONFIG['geometric']:
        augmented = apply_shear(augmented, shear_std=SHEAR_STD)
        augmented = apply_rotation(augmented, rotation_std=ROTATION_STD)
        return augmented

    if AUGMENTATION_CONFIG['geometric'] > 0:
        current_id -= AUGMENTATION_CONFIG['geometric']

    # Phase 2: Horizontal Flip
    if AUGMENTATION_CONFIG['horizontal_flip'] > 0 and current_id < AUGMENTATION_CONFIG['horizontal_flip']:
        augmented = apply_horizontal_flip_pose(augmented)
        if current_id == 1:  # Flip + rotation
            augmented = apply_rotation(augmented, rotation_std=ROTATION_STD * 0.5)
        elif current_id == 2:  # Flip + shear
            augmented = apply_shear(augmented, shear_std=SHEAR_STD * 0.5)
        elif current_id == 3:  # Flip + noise
            augmented = apply_spatial_noise_pose(augmented, noise_std=NOISE_LOW)
        elif current_id == 4:  # Flip + translation
            augmented = apply_translation_pose(augmented, shift_std=TRANSLATION_SMALL)
        return augmented

    if AUGMENTATION_CONFIG['horizontal_flip'] > 0:
        current_id -= AUGMENTATION_CONFIG['horizontal_flip']

    # Phase 3: Spatial Noise
    if AUGMENTATION_CONFIG['spatial_noise'] > 0 and current_id < AUGMENTATION_CONFIG['spatial_noise']:
        noise_std = NOISE_LOW if current_id == 0 else NOISE_MEDIUM
        augmented = apply_spatial_noise_pose(augmented, noise_std=noise_std)
        return augmented

    if AUGMENTATION_CONFIG['spatial_noise'] > 0:
        current_id -= AUGMENTATION_CONFIG['spatial_noise']

    # Phase 4: Translation
    if AUGMENTATION_CONFIG['translation'] > 0 and current_id < AUGMENTATION_CONFIG['translation']:
        shift_std = TRANSLATION_SMALL if current_id == 0 else TRANSLATION_MEDIUM
        augmented = apply_translation_pose(augmented, shift_std=shift_std)
        return augmented

    if AUGMENTATION_CONFIG['translation'] > 0:
        current_id -= AUGMENTATION_CONFIG['translation']

    # Phase 5: Scaling
    if AUGMENTATION_CONFIG['scaling'] > 0 and current_id < AUGMENTATION_CONFIG['scaling']:
        scale_factor = SCALE_SMALL if current_id == 0 else SCALE_LARGE
        augmented = apply_scaling_pose(augmented, scale_factor=scale_factor)
        return augmented

    if AUGMENTATION_CONFIG['scaling'] > 0:
        current_id -= AUGMENTATION_CONFIG['scaling']

    # Phase 6: Speed Variation (Updated to 4 variants)
    if AUGMENTATION_CONFIG['speed_variation'] > 0 and current_id < AUGMENTATION_CONFIG['speed_variation']:
        if current_id == 0:
            speed_factor = SPEED_SLOW          # 0.75x
        elif current_id == 1:
            speed_factor = SPEED_FAST          # 1.25x
        elif current_id == 2:
            speed_factor = SPEED_VERY_SLOW     # 0.65x
        else:  # current_id == 3
            speed_factor = SPEED_VERY_FAST     # 1.35x

        augmented = apply_speed_variation_pose(augmented, speed_factor=speed_factor)
        return augmented

    if AUGMENTATION_CONFIG['speed_variation'] > 0:
        current_id -= AUGMENTATION_CONFIG['speed_variation']

    # Phase 7: Keypoint Occlusion (NEW)
    if AUGMENTATION_CONFIG['keypoint_occlusion'] > 0 and current_id < AUGMENTATION_CONFIG['keypoint_occlusion']:
        if current_id == 0:
            occlusion_prob = KEYPOINT_OCCLUSION_LOW       # 10%
        elif current_id == 1:
            occlusion_prob = KEYPOINT_OCCLUSION_MEDIUM    # 15%
        else:  # current_id == 2
            occlusion_prob = KEYPOINT_OCCLUSION_HIGH      # 20%

        augmented = apply_keypoint_occlusion_pose(augmented, occlusion_prob=occlusion_prob)
        return augmented

    if AUGMENTATION_CONFIG['keypoint_occlusion'] > 0:
        current_id -= AUGMENTATION_CONFIG['keypoint_occlusion']

    # Phase 8: Hand Dropout (NEW)
    if AUGMENTATION_CONFIG['hand_dropout'] > 0 and current_id < AUGMENTATION_CONFIG['hand_dropout']:
        # Variant 1: Standard dropout
        # Variant 2: Dropout + slight noise to remaining hand for realism
        augmented = apply_hand_dropout_pose(augmented, dropout_prob=HAND_DROPOUT_PROB)
        if current_id == 1:
            # Add slight noise to make it more realistic
            augmented = apply_spatial_noise_pose(augmented, noise_std=NOISE_LOW)
        return augmented

    if AUGMENTATION_CONFIG['hand_dropout'] > 0:
        current_id -= AUGMENTATION_CONFIG['hand_dropout']

    # Phase 9: Combinations (Updated to 4 variants)
    if AUGMENTATION_CONFIG['combinations'] > 0 and current_id < AUGMENTATION_CONFIG['combinations']:
        if current_id == 0:  # Rotation + Noise
            augmented = apply_rotation(augmented, rotation_std=ROTATION_STD)
            augmented = apply_spatial_noise_pose(augmented, noise_std=NOISE_LOW)
        elif current_id == 1:  # Shear + Translation
            augmented = apply_shear(augmented, shear_std=SHEAR_STD)
            augmented = apply_translation_pose(augmented, shift_std=TRANSLATION_SMALL)
        elif current_id == 2:  # Flip + Scaling
            augmented = apply_horizontal_flip_pose(augmented)
            augmented = apply_scaling_pose(augmented, scale_factor=SCALE_LARGE)
        elif current_id == 3:  # Speed + Occlusion (NEW combination)
            augmented = apply_speed_variation_pose(augmented, speed_factor=SPEED_FAST)
            augmented = apply_keypoint_occlusion_pose(augmented, occlusion_prob=KEYPOINT_OCCLUSION_LOW)
        return augmented

    # Fallback: return original
    return augmented


def save_augmented_pose(pose_data, original_pose, metadata, output_path, augmentation_id):
    """
    Save augmented pose data as .pose file.
    """
    # Create confidence mask matching the augmented data shape
    # Shape: (frames, people, keypoints)
    new_frames = pose_data.shape[0]

    # If frames changed (e.g., from speed variation), resize confidence
    if new_frames != original_pose.body.confidence.shape[0]:
        # Create new confidence based on non-zero values in augmented data
        confidence = np.ones((new_frames, pose_data.shape[1], pose_data.shape[2]), dtype=np.float32)
        # Mark zero keypoints as low confidence
        confidence[pose_data[:, :, :, 0] == 0] = 0.0
    else:
        # Same number of frames, can reuse original confidence
        confidence = original_pose.body.confidence

    # Create new pose with augmented data
    augmented_pose = Pose(
        header=original_pose.header,
        body=original_pose.body.__class__(
            fps=original_pose.body.fps,
            data=pose_data,
            confidence=confidence
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
        # Use default augmented_pool structure (MODULAR APPROACH)
        # augmented_pool is a sibling of wlasl_poses_complete
        augmented_pool_dir = Path(WLASL_BASE_DIR).parent / "augmented_pool"

        if input_type == 'pickle':
            # Pickle mode: load from pickle train split, save to augmented_pool/pickle/
            INPUT_DIR = os.path.join(WLASL_BASE_DIR, f"dataset_splits/{num_classes}_classes/original/pickle_from_pose_split_{num_classes}_class/train")
            OUTPUT_DIR = str(augmented_pool_dir / "pickle")
            file_extension = '.pkl'
        else:  # pose (RECOMMENDED - modular approach)
            # Pose mode: load ORIGINAL .pose from train split, save to augmented_pool/pose/
            INPUT_DIR = os.path.join(WLASL_BASE_DIR, f"dataset_splits/{num_classes}_classes/original/pose_split_{num_classes}_class/train")
            OUTPUT_DIR = str(augmented_pool_dir / "pose")
            file_extension = '.pose'

    print("=" * 70)
    print(f"GENERATING 75-POINT AUGMENTED DATASET ({input_type.upper()} MODE)")
    print("=" * 70)
    print()
    print(f"Input type: {input_type}")
    print(f"Number of classes: {num_classes}")
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print(f"AUGMENTATION CONFIG: {NUM_AUGMENTATIONS} total augmentations")
    for strategy, count in AUGMENTATION_CONFIG.items():
        if count > 0:
            print(f"  ✓ {strategy}: {count} variants")
        else:
            print(f"  ✗ {strategy}: disabled (count=0)")
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get target classes dynamically
    if num_classes and num_classes > 0:
        # Try loading from class_mapping.json first
        target_classes = load_class_mapping(num_classes)

        if target_classes is None:
            # Fallback: discover classes from input directory
            print(f"Fallback: discovering classes from input directory...")
            target_classes = get_classes_from_directory(INPUT_DIR)
            print(f"Found {len(target_classes)} classes in directory")

        # Get all class directories that exist in input
        all_class_dirs = sorted([d for d in os.listdir(INPUT_DIR)
                                if os.path.isdir(os.path.join(INPUT_DIR, d))])

        # Filter to only include target classes that exist
        class_dirs = [c for c in target_classes if c in all_class_dirs]
        print(f"Using {len(class_dirs)} classes from mapping (found in input directory)")

        # Warn if some classes are missing
        missing = [c for c in target_classes if c not in all_class_dirs]
        if missing:
            print(f"WARNING: {len(missing)} classes from mapping not found in input directory:")
            for c in missing[:10]:  # Show first 10
                print(f"  - {c}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
    else:
        # Process all classes found in directory
        class_dirs = get_classes_from_directory(INPUT_DIR)
        print(f"Processing all {len(class_dirs)} classes from directory")

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
