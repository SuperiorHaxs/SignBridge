#!/usr/bin/env python3
"""
Generate Balanced Augmented Dataset

This script generates a class-balanced augmented dataset where each class
has approximately the same number of samples after augmentation.

Key features:
1. Calculates per-class augmentation counts based on target (200/class)
2. Uses landmarks-extraction module for flexible keypoint extraction
3. Groups original + augmentations as "families" for proper splitting
4. Outputs to balanced_splits folder

Usage:
    python generate_balanced_dataset.py --classes 125
    python generate_balanced_dataset.py --classes 125 --landmark-config 83pt
    python generate_balanced_dataset.py --classes 125 --dry-run
"""

import os
import sys
import pickle
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

# Add paths for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

# Import project config
from config import get_config
config = get_config()

# Import landmarks extraction module
landmarks_dir = project_root / "dataset-utilities" / "landmarks-extraction"
sys.path.insert(0, str(landmarks_dir))
from landmark_config import LANDMARK_CONFIGS
from extract_landmarks import extract_landmarks, get_landmark_info

# Import augmentation config
from augmentation_config import (
    TARGET_SAMPLES_PER_CLASS,
    TARGET_TRAIN_SAMPLES_PER_CLASS,
    MIN_AUGMENTATIONS_PER_VIDEO,
    MAX_AUGMENTATIONS_PER_VIDEO,
    BASE_AUGMENTATIONS_PER_VIDEO,
    MIN_BASE_AUGMENTATIONS,
    MAX_BASE_AUGMENTATIONS,
    AUGMENTATION_TECHNIQUES,
    FRAME_BUCKETS,
    calculate_augmentation_counts,
    calculate_base_augmentation_counts,
    get_augmentation_sequence,
    get_frame_bucket,
    print_augmentation_plan,
    save_augmentation_plan,
)

# Import existing augmentation functions
from augment_pose_file import apply_shear, apply_rotation


# =============================================================================
# AUGMENTATION FUNCTIONS
# =============================================================================

def apply_horizontal_flip(keypoints: np.ndarray) -> np.ndarray:
    """
    Apply horizontal flip to keypoints.

    Args:
        keypoints: Array of shape (frames, points, 2)

    Returns:
        Flipped keypoints
    """
    flipped = keypoints.copy()
    # Flip x-coordinates around center (0.5 if normalized, or image center)
    flipped[:, :, 0] = -flipped[:, :, 0]
    return flipped


def apply_spatial_noise(keypoints: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    """
    Add Gaussian noise to keypoints.

    Args:
        keypoints: Array of shape (frames, points, 2)
        noise_std: Standard deviation of noise

    Returns:
        Noisy keypoints
    """
    noise = np.random.normal(0, noise_std, keypoints.shape).astype(np.float32)
    return keypoints + noise


def apply_translation(keypoints: np.ndarray, tx: float = 0, ty: float = 0) -> np.ndarray:
    """
    Apply translation to keypoints.

    Args:
        keypoints: Array of shape (frames, points, 2)
        tx, ty: Translation amounts

    Returns:
        Translated keypoints
    """
    translated = keypoints.copy()
    translated[:, :, 0] += tx
    translated[:, :, 1] += ty
    return translated


def apply_scaling(keypoints: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Apply uniform scaling around center.

    Args:
        keypoints: Array of shape (frames, points, 2)
        scale: Scale factor

    Returns:
        Scaled keypoints
    """
    # Find center of mass
    center = np.mean(keypoints, axis=(0, 1))
    # Scale around center
    scaled = (keypoints - center) * scale + center
    return scaled.astype(np.float32)


def apply_speed_variation(keypoints: np.ndarray, speed_factor: float = 1.0) -> np.ndarray:
    """
    Apply temporal speed variation by resampling frames.

    Args:
        keypoints: Array of shape (frames, points, 2)
        speed_factor: Speed multiplier (>1 = faster, <1 = slower)

    Returns:
        Resampled keypoints
    """
    frames, points, coords = keypoints.shape
    new_frames = max(1, int(frames / speed_factor))

    # Create new frame indices
    old_indices = np.linspace(0, frames - 1, new_frames)

    # Interpolate
    resampled = np.zeros((new_frames, points, coords), dtype=np.float32)
    for p in range(points):
        for c in range(coords):
            resampled[:, p, c] = np.interp(
                old_indices,
                np.arange(frames),
                keypoints[:, p, c]
            )

    return resampled


def apply_keypoint_occlusion(
    keypoints: np.ndarray,
    occlusion_rate: float = 0.1
) -> np.ndarray:
    """
    Randomly zero out keypoints to simulate occlusion.

    Args:
        keypoints: Array of shape (frames, points, 2)
        occlusion_rate: Fraction of keypoints to occlude

    Returns:
        Occluded keypoints
    """
    occluded = keypoints.copy()
    frames, points, coords = keypoints.shape

    # Create random mask
    mask = np.random.random((frames, points)) > occlusion_rate
    mask = mask[:, :, np.newaxis]  # Add coord dimension

    # Apply mask (occluded points become 0)
    occluded = occluded * mask

    return occluded.astype(np.float32)


def apply_hand_dropout(keypoints: np.ndarray, drop_left: bool = True) -> np.ndarray:
    """
    Drop all keypoints for one hand.

    Note: Assumes standard 83pt layout:
    - Pose: 0-32 (33 points)
    - Left hand: 33-53 (21 points)
    - Right hand: 54-74 (21 points)
    - Face: 75-82 (8 points)

    Args:
        keypoints: Array of shape (frames, 83, 2)
        drop_left: If True, drop left hand; else drop right hand

    Returns:
        Keypoints with one hand zeroed out
    """
    dropped = keypoints.copy()

    if drop_left:
        dropped[:, 33:54, :] = 0  # Left hand indices
    else:
        dropped[:, 54:75, :] = 0  # Right hand indices

    return dropped


def apply_augmentation(
    keypoints: np.ndarray,
    technique: str,
    params: dict,
    variation_index: int = 0,
) -> np.ndarray:
    """
    Apply a specific augmentation technique to keypoints.

    Args:
        keypoints: Array of shape (frames, points, 2)
        technique: Name of augmentation technique
        params: Parameters for the technique
        variation_index: Index for parameter variation

    Returns:
        Augmented keypoints
    """
    np.random.seed(variation_index)  # For reproducibility

    if technique == 'geometric':
        # Reshape for shear/rotation functions (add batch dim)
        data = keypoints[np.newaxis, :, :, :]  # (1, frames, points, 2)
        data = apply_shear(data, shear_std=params.get('shear_std', 0.1))
        data = apply_rotation(data, rotation_std=params.get('rotation_std', 0.1))
        return data[0]  # Remove batch dim

    elif technique == 'horizontal_flip':
        return apply_horizontal_flip(keypoints)

    elif technique == 'spatial_noise':
        noise_levels = params.get('noise_levels', [0.01, 0.02, 0.03])
        level_idx = variation_index % len(noise_levels)
        return apply_spatial_noise(keypoints, noise_std=noise_levels[level_idx])

    elif technique == 'translation':
        ranges = params.get('translation_ranges', [0.1, 0.2, 0.3])
        range_idx = variation_index % len(ranges)
        tx = np.random.uniform(-ranges[range_idx], ranges[range_idx])
        ty = np.random.uniform(-ranges[range_idx], ranges[range_idx])
        return apply_translation(keypoints, tx, ty)

    elif technique == 'scaling':
        scale_min, scale_max = params.get('scale_range', (0.7, 1.5))
        scale = np.random.uniform(scale_min, scale_max)
        return apply_scaling(keypoints, scale)

    elif technique == 'speed_variation':
        speed_factors = params.get('speed_factors', [0.75, 1.25])
        factor_idx = variation_index % len(speed_factors)
        return apply_speed_variation(keypoints, speed_factors[factor_idx])

    elif technique == 'keypoint_occlusion':
        rates = params.get('occlusion_rates', [0.1, 0.2, 0.3])
        rate_idx = variation_index % len(rates)
        return apply_keypoint_occlusion(keypoints, occlusion_rate=rates[rate_idx])

    elif technique == 'hand_dropout':
        drop_left = (variation_index % 2) == 0
        return apply_hand_dropout(keypoints, drop_left=drop_left)

    elif technique == 'combinations':
        # Apply multiple transforms
        augmented = keypoints.copy()
        combine_techs = params.get('combine_techniques', ['geometric', 'scaling'])
        for sub_tech in combine_techs:
            sub_params = AUGMENTATION_TECHNIQUES.get(sub_tech, {}).get('params', {})
            augmented = apply_augmentation(
                augmented, sub_tech, sub_params, variation_index
            )
        return augmented

    else:
        # Unknown technique, return unchanged
        return keypoints


# =============================================================================
# DATASET SCANNING
# =============================================================================

def scan_pose_files_by_gloss(
    pose_by_gloss_dir: Path,
    class_list: List[str],
    landmark_config: str = '83pt',
) -> Dict[str, List[Dict]]:
    """
    Scan original .pose files from pose_files_by_gloss directory.

    This is the clean-slate approach that reads directly from the source
    pose files, avoiding any contamination from previous splits.

    Args:
        pose_by_gloss_dir: Path to pose_files_by_gloss directory
        class_list: List of class names to include (from class_mapping.json)
        landmark_config: Landmark configuration for extraction

    Returns:
        Dict mapping class name to list of sample info dicts
    """
    dataset_info = defaultdict(list)

    if not pose_by_gloss_dir.exists():
        raise FileNotFoundError(f"pose_files_by_gloss directory not found: {pose_by_gloss_dir}")

    print(f"Scanning {len(class_list)} classes from {pose_by_gloss_dir}")
    print(f"Using landmark config: {landmark_config}")

    # Get landmark indices for extraction
    landmark_info = get_landmark_info(landmark_config)

    for class_name in tqdm(class_list, desc="Scanning pose files"):
        # Try different case variations
        class_dir = None
        for case_variant in [class_name, class_name.lower(), class_name.upper()]:
            potential_dir = pose_by_gloss_dir / case_variant
            if potential_dir.exists():
                class_dir = potential_dir
                break

        if class_dir is None:
            print(f"  Warning: Class directory not found for '{class_name}'")
            continue

        # Scan all .pose files in this class directory
        pose_files = list(class_dir.glob("*.pose"))

        for pose_file in pose_files:
            try:
                # Get video_id from filename
                video_id = pose_file.stem

                # Read pose file to get frame count (without full extraction yet)
                # We'll do full extraction during augmentation
                from pose_format import Pose
                with open(pose_file, 'rb') as f:
                    pose = Pose.read(f.read())

                frames = len(pose.body.data)

                dataset_info[class_name.lower()].append({
                    'path': pose_file,
                    'video_id': video_id,
                    'frames': frames,
                    'bucket': get_frame_bucket(frames),
                    'format': 'pose',  # Mark as pose file for later processing
                })

            except Exception as e:
                print(f"  Warning: Error reading {pose_file}: {e}")

    return dict(dataset_info)


def scan_original_dataset(
    input_dirs: List[Path],
    class_list: Optional[List[str]] = None,
) -> Dict[str, List[Dict]]:
    """
    Scan the original dataset to collect sample information.

    Scans multiple directories (train, val, test) and collects only
    non-augmented original samples.

    Args:
        input_dirs: List of paths to scan (e.g., [train_dir, val_dir, test_dir])
        class_list: Optional list of class names to include

    Returns:
        Dict mapping class name to list of sample info dicts:
        {
            'class_name': [
                {'path': Path, 'video_id': str, 'frames': int, 'bucket': str},
                ...
            ]
        }
    """
    dataset_info = defaultdict(list)
    seen_video_ids = set()  # Track video IDs to avoid duplicates

    # Handle single path input for backwards compatibility
    if isinstance(input_dirs, (str, Path)):
        input_dirs = [Path(input_dirs)]

    for input_dir in input_dirs:
        input_dir = Path(input_dir)
        if not input_dir.exists():
            print(f"Warning: Directory not found: {input_dir}")
            continue

        # Get list of classes to process
        if class_list:
            classes = [c for c in class_list if (input_dir / c.lower()).exists() or (input_dir / c).exists()]
        else:
            classes = sorted([
                d.name for d in input_dir.iterdir()
                if d.is_dir()
            ])

        print(f"Scanning {len(classes)} classes from {input_dir}")

        for class_name in tqdm(classes, desc=f"Scanning {input_dir.name}"):
            class_dir = input_dir / class_name.lower()
            if not class_dir.exists():
                class_dir = input_dir / class_name  # Try original case

            if not class_dir.exists():
                continue

            for pkl_file in class_dir.glob("*.pkl"):
                try:
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)

                    # Skip if already augmented
                    if data.get('augmented', False):
                        continue

                    video_id = data.get('video_id', pkl_file.stem)

                    # Skip if we've already seen this video
                    unique_key = f"{class_name.lower()}/{video_id}"
                    if unique_key in seen_video_ids:
                        continue
                    seen_video_ids.add(unique_key)

                    keypoints = data['keypoints']
                    frames = keypoints.shape[0]

                    dataset_info[class_name.lower()].append({
                        'path': pkl_file,
                        'video_id': video_id,
                        'frames': frames,
                        'bucket': get_frame_bucket(frames),
                    })
                except Exception as e:
                    print(f"Warning: Error reading {pkl_file}: {e}")

    return dict(dataset_info)


# =============================================================================
# AUGMENTATION GENERATION
# =============================================================================

def generate_augmented_sample(
    original_data: dict,
    aug_config: dict,
    landmark_config: str = '83pt',
    include_z: bool = True,
) -> dict:
    """
    Generate a single augmented sample from original data.

    Args:
        original_data: Original pickle data dict
        aug_config: Augmentation configuration
        landmark_config: Landmark extraction config ('75pt', '83pt', etc.)
        include_z: Whether to include z-coordinate (default True for 3D poses)

    Returns:
        Augmented sample dict ready for saving
    """
    # Get keypoints from original data
    full_keypoints = original_data['keypoints']

    # Check if data is already extracted (has expected number of keypoints)
    # or if it's full MediaPipe data that needs extraction
    expected_points = LANDMARK_CONFIGS[landmark_config]['total_points']
    current_points = full_keypoints.shape[1]

    if current_points == expected_points:
        # Data already has the correct number of keypoints, skip extraction
        landmarks = full_keypoints.copy()
    elif current_points > expected_points:
        # Full data, extract landmarks
        landmarks = extract_landmarks(
            full_keypoints,
            config=landmark_config,
            include_z=include_z,
            normalize=False,  # We'll normalize after augmentation
        )
    else:
        raise ValueError(
            f"Data has {current_points} keypoints, but config '{landmark_config}' "
            f"expects {expected_points}. Cannot extract from smaller dataset."
        )

    # Apply augmentation
    technique = aug_config['technique']
    params = aug_config['params']
    variation_idx = aug_config.get('variation_index', 0)

    augmented = apply_augmentation(landmarks, technique, params, variation_idx)

    # Create augmented sample dict
    aug_sample = {
        'keypoints': augmented,
        'video_id': original_data.get('video_id'),
        'gloss': original_data.get('gloss'),
        'augmented': True,
        'augmentation_id': aug_config['id'],
        'augmentation_technique': technique,
        'original_video_id': original_data.get('video_id'),
        'landmark_config': landmark_config,
    }

    # Copy confidence if available
    if 'confidences' in original_data:
        # Confidence needs to match new keypoint count
        # For now, create uniform confidence
        frames, points, _ = augmented.shape
        aug_sample['confidences'] = np.ones((frames, points), dtype=np.float32)

    return aug_sample


def load_pose_file_as_keypoints(pose_path: Path) -> Tuple[np.ndarray, dict]:
    """
    Load a .pose file and extract keypoints as numpy array.

    Pose files from MediaPipe Holistic have 576 points:
    - Pose: 0-32 (33 points)
    - Face: 33-500 (468 points)
    - Left Hand: 501-521 (21 points)
    - Right Hand: 522-542 (21 points)
    - Pose World: 543-575 (33 points)

    Args:
        pose_path: Path to .pose file

    Returns:
        Tuple of (keypoints array with shape (frames, 576, 3), metadata dict)
    """
    from pose_format import Pose

    with open(pose_path, 'rb') as f:
        pose = Pose.read(f.read())

    # Extract data: pose.body.data has shape (frames, people, points, dims)
    # We take first person and get (frames, points, dims)
    data = pose.body.data
    if len(data.shape) == 4:
        keypoints = data[:, 0, :, :]  # Take first person
    else:
        keypoints = data

    # Convert to numpy float32 and handle masked arrays
    if hasattr(keypoints, 'filled'):
        # Handle masked arrays - fill missing values with 0
        keypoints = keypoints.filled(0)
    keypoints = np.asarray(keypoints, dtype=np.float32)

    # Get video_id and gloss from path
    video_id = pose_path.stem
    gloss = pose_path.parent.name

    metadata = {
        'video_id': video_id,
        'gloss': gloss,
        'source_format': 'pose',
        'original_path': str(pose_path),
        'original_shape': keypoints.shape,
    }

    return keypoints, metadata


def generate_family(
    original_path: Path,
    aug_count: int,
    output_dir: Path,
    landmark_config: str = '83pt',
    dry_run: bool = False,
    input_format: str = 'auto',
    include_z: bool = True,
) -> List[Path]:
    """
    Generate a family of samples (original + augmentations) for one video.

    Args:
        original_path: Path to original file (.pkl or .pose)
        aug_count: Number of augmentations to generate
        output_dir: Directory to save augmented files
        landmark_config: Landmark extraction config
        dry_run: If True, don't actually save files
        input_format: 'pkl', 'pose', or 'auto' (detect from extension)
        include_z: Whether to include z-coordinate (default True for 3D poses)

    Returns:
        List of paths to generated files (including processed original)
    """
    generated_files = []

    # Detect format if auto
    if input_format == 'auto':
        input_format = 'pose' if original_path.suffix == '.pose' else 'pkl'

    # Load original data based on format
    if input_format == 'pose':
        raw_keypoints, metadata = load_pose_file_as_keypoints(original_path)
        video_id = metadata['video_id']
        gloss = metadata['gloss']
        # Store raw keypoints for landmark extraction
        original_keypoints = raw_keypoints
    else:
        # Load from pickle
        with open(original_path, 'rb') as f:
            original_data = pickle.load(f)
        video_id = original_data.get('video_id', original_path.stem)
        gloss = original_data.get('gloss', original_path.parent.name)
        original_keypoints = original_data['keypoints']

    # Create output directory for this class
    class_output_dir = output_dir / gloss.lower()
    class_output_dir.mkdir(parents=True, exist_ok=True)

    # Process and save original with new landmark config
    original_landmarks = extract_landmarks(
        original_keypoints,
        config=landmark_config,
        include_z=include_z,
        normalize=True,
    )

    original_processed = {
        'keypoints': original_landmarks,
        'video_id': video_id,
        'gloss': gloss,
        'augmented': False,
        'augmentation_id': None,
        'original_video_id': video_id,
        'landmark_config': landmark_config,
        'original_frames': original_keypoints.shape[0],
    }

    # Save processed original
    original_output_path = class_output_dir / f"{video_id}.pkl"
    if not dry_run:
        with open(original_output_path, 'wb') as f:
            pickle.dump(original_processed, f)
    generated_files.append(original_output_path)

    # Create a data dict for augmentation (compatible with generate_augmented_sample)
    original_data_for_aug = {
        'keypoints': original_keypoints,
        'video_id': video_id,
        'gloss': gloss,
    }

    # Generate augmentations
    aug_sequence = get_augmentation_sequence(aug_count)

    for aug_config in aug_sequence:
        aug_id = aug_config['id']

        try:
            aug_sample = generate_augmented_sample(
                original_data_for_aug,
                aug_config,
                landmark_config,
                include_z=include_z,
            )

            # Save augmented sample
            aug_output_path = class_output_dir / f"{video_id}_aug_{aug_id:02d}.pkl"
            if not dry_run:
                with open(aug_output_path, 'wb') as f:
                    pickle.dump(aug_sample, f)
            generated_files.append(aug_output_path)

        except Exception as e:
            print(f"Warning: Failed to generate aug {aug_id} for {video_id}: {e}")

    return generated_files


# =============================================================================
# MAIN GENERATION PIPELINE
# =============================================================================

def generate_balanced_dataset(
    num_classes: int = None,
    gloss_list: List[str] = None,
    landmark_config: str = '83pt',
    target_per_class: int = TARGET_SAMPLES_PER_CLASS,
    dry_run: bool = False,
    output_base: Optional[Path] = None,
    skip_existing: bool = True,
    use_base_augmentation: bool = True,
    include_z: bool = True,
) -> Dict:
    """
    Generate a balanced augmented dataset (Phase 1 of two-phase approach).

    Outputs to a flat structure: augmented_pool/pickle/{gloss}/*.pkl
    This allows reuse across different class configurations (100, 125, 200, etc.)

    Two-Phase Approach:
        Phase 1 (this function): Generate base pool with fixed augmentations per video
        Phase 2 (balance_train_split): After splitting, augment train to exact target

    Args:
        num_classes: Number of classes (used to load class_mapping if gloss_list not provided)
        gloss_list: Explicit list of glosses to process (takes precedence over num_classes)
        landmark_config: Landmark extraction config ('75pt', '83pt', etc.)
        target_per_class: Target samples per class (only used if use_base_augmentation=False)
        dry_run: If True, only compute plan without generating files
        output_base: Base output directory (default: datasets/augmented_pool/pickle)
        skip_existing: If True, skip glosses that already have augmented data
        use_base_augmentation: If True, use fixed augmentations per video (recommended for two-phase)
        include_z: Whether to include z-coordinate (default True for 3D poses)

    Returns:
        Generation summary dict
    """
    print("=" * 70)
    print("BALANCED DATASET GENERATION (Phase 1: Base Pool)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Landmark config: {landmark_config}")
    if use_base_augmentation:
        print(f"  Mode: Base augmentation ({BASE_AUGMENTATIONS_PER_VIDEO} per video)")
        print(f"  Note: Train balancing (Phase 2) will achieve {TARGET_TRAIN_SAMPLES_PER_CLASS}/class")
    else:
        print(f"  Mode: Target-based ({target_per_class} per class)")
    print(f"  Dry run: {dry_run}")
    print(f"  Skip existing: {skip_existing}")
    print(f"  Include Z: {include_z}")

    # Get landmark info
    landmark_info = get_landmark_info(landmark_config)
    coords = 3 if include_z else 2
    feature_count = landmark_info['total_points'] * coords
    print(f"  Landmarks: {landmark_info['total_points']} points x {coords} coords = {feature_count} features")

    # Use pose_files_by_gloss as the clean-slate source (no contamination from previous splits)
    pose_by_gloss_dir = config.dataset_root / "pose_files_by_gloss"
    if not pose_by_gloss_dir.exists():
        raise ValueError(f"pose_files_by_gloss directory not found: {pose_by_gloss_dir}")
    print(f"  Source (clean): {pose_by_gloss_dir}")

    # Output to flat structure: augmented_pool/pickle/{gloss}/
    if output_base is None:
        output_base = config.dataset_root.parent / "augmented_pool" / "pickle"
    output_dir = output_base  # Flat structure, glosses directly under pickle/
    print(f"  Output: {output_dir}")

    # Determine which glosses to process
    if gloss_list is not None:
        class_list = gloss_list
        print(f"  Glosses provided: {len(class_list)}")
    elif num_classes is not None:
        # Load class list from mapping
        dataset_paths = config.dataset_splits.get(num_classes)
        if not dataset_paths:
            raise ValueError(f"No dataset configuration for {num_classes} classes")
        class_mapping_path = dataset_paths.get('class_mapping')
        if not class_mapping_path or not Path(class_mapping_path).exists():
            raise ValueError(f"class_mapping.json required but not found: {class_mapping_path}")
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
        class_list = class_mapping.get('classes', [])
        print(f"  Classes from {num_classes}-class mapping: {len(class_list)}")
    else:
        raise ValueError("Must provide either num_classes or gloss_list")

    # Check which glosses already have augmented data (if skip_existing)
    glosses_to_process = class_list
    skipped_glosses = []
    if skip_existing:
        glosses_to_process = []
        for gloss in class_list:
            gloss_dir = output_dir / gloss.lower()
            if gloss_dir.exists():
                # Check if it has augmented files (not just original)
                aug_files = list(gloss_dir.glob("*_aug_*.pkl"))
                if aug_files:
                    skipped_glosses.append(gloss)
                    continue
            glosses_to_process.append(gloss)

        if skipped_glosses:
            print(f"  Skipping {len(skipped_glosses)} glosses with existing augmentations")
            print(f"  Processing {len(glosses_to_process)} new glosses")

    if not glosses_to_process:
        print("\n[INFO] All glosses already have augmented data. Nothing to do.")
        return {
            'skipped': skipped_glosses,
            'processed': [],
            'output_dir': output_dir,
        }

    # Scan dataset from clean source (pose_files_by_gloss)
    print("\n" + "-" * 70)
    dataset_info = scan_pose_files_by_gloss(
        pose_by_gloss_dir,
        glosses_to_process,
        landmark_config,
    )
    print(f"Found {sum(len(v) for v in dataset_info.values())} original samples across {len(dataset_info)} classes")

    # Calculate augmentation plan
    class_sample_counts = {k: len(v) for k, v in dataset_info.items()}

    if use_base_augmentation:
        # Phase 1: Fixed augmentations per video for base pool
        augmentation_plan = calculate_base_augmentation_counts(
            class_sample_counts,
            base_per_video=BASE_AUGMENTATIONS_PER_VIDEO,
            min_per_video=MIN_BASE_AUGMENTATIONS,
            max_per_video=MAX_BASE_AUGMENTATIONS,
        )
        print(f"\n[Phase 1] Base pool: {BASE_AUGMENTATIONS_PER_VIDEO} augmentations per video")
    else:
        # Legacy: Try to reach target per class
        augmentation_plan = calculate_augmentation_counts(
            class_sample_counts,
            target_per_class=target_per_class,
        )

    # Print plan summary
    print_augmentation_plan(augmentation_plan)

    # Save plan (per-gloss metadata stored in index)
    plan_path = output_dir / "augmentation_plan.json"
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_augmentation_plan(augmentation_plan, plan_path)

    if dry_run:
        print("\n[DRY RUN] No files generated. Remove --dry-run to generate dataset.")
        return {
            'plan': augmentation_plan,
            'dataset_info': dataset_info,
            'output_dir': output_dir,
            'skipped': skipped_glosses,
        }

    # Generate augmented samples
    print("\n" + "-" * 70)
    print("GENERATING AUGMENTED SAMPLES")
    print("-" * 70)

    total_generated = 0
    family_info = {}  # Track family info for index

    for class_name, samples in tqdm(dataset_info.items(), desc="Processing classes"):
        class_plan = augmentation_plan[class_name]
        aug_per_video = class_plan['augmentations_per_video']

        class_families = []

        for sample_info in samples:
            original_path = sample_info['path']
            video_id = sample_info['video_id']
            input_format = sample_info.get('format', 'auto')

            # Generate family
            family_files = generate_family(
                original_path,
                aug_per_video,
                output_dir,
                landmark_config=landmark_config,
                dry_run=False,
                input_format=input_format,
                include_z=include_z,
            )

            total_generated += len(family_files)

            class_families.append({
                'video_id': video_id,
                'original_path': str(original_path),
                'frame_bucket': sample_info['bucket'],
                'files': [str(f) for f in family_files],
                'file_count': len(family_files),
            })

        family_info[class_name] = class_families

    # Save/update the global pickle index
    index_path = output_dir / "pickle_index.json"
    existing_index = {}
    if index_path.exists():
        with open(index_path, 'r') as f:
            existing_index = json.load(f)

    # Merge new family info into existing index
    existing_index.update(family_info)

    with open(index_path, 'w') as f:
        json.dump(existing_index, f, indent=2)
    print(f"\nUpdated pickle index: {index_path}")
    print(f"  Total glosses in index: {len(existing_index)}")

    # Summary
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total files generated: {total_generated}")
    print(f"Glosses processed: {len(family_info)}")
    if skipped_glosses:
        print(f"Glosses skipped (existing): {len(skipped_glosses)}")
    print(f"Output directory: {output_dir}")
    print(f"Pickle index: {index_path}")
    print("\nNext step: Run stratified_family_split.py to create train/val/test manifests")

    return {
        'total_generated': total_generated,
        'family_info': family_info,
        'output_dir': output_dir,
        'index_path': index_path,
        'skipped': skipped_glosses,
        'processed': list(family_info.keys()),
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate balanced augmented dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 125-class balanced dataset with 83-point landmarks
    python generate_balanced_dataset.py --classes 125

    # Dry run to see augmentation plan
    python generate_balanced_dataset.py --classes 125 --dry-run

    # Use 75-point landmarks (no face)
    python generate_balanced_dataset.py --classes 125 --landmark-config 75pt

    # Custom target samples per class
    python generate_balanced_dataset.py --classes 125 --target 300
        """
    )

    parser.add_argument(
        '--classes', type=int, default=None,
        help='Number of classes (e.g., 20, 50, 100, 125) - loads from class_mapping'
    )
    parser.add_argument(
        '--gloss-file', type=Path, default=None,
        help='JSON file containing list of glosses to process'
    )
    parser.add_argument(
        '--landmark-config', type=str, default='83pt',
        choices=list(LANDMARK_CONFIGS.keys()),
        help='Landmark extraction configuration (default: 83pt)'
    )
    parser.add_argument(
        '--target', type=int, default=TARGET_SAMPLES_PER_CLASS,
        help=f'Target samples per class (default: {TARGET_SAMPLES_PER_CLASS})'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Compute plan without generating files'
    )
    parser.add_argument(
        '--no-skip', action='store_true',
        help='Regenerate even if glosses already have augmented data'
    )
    parser.add_argument(
        '--output', type=Path, default=None,
        help='Output base directory (default: datasets/augmented_pool/pickle/)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.classes is None and args.gloss_file is None:
        parser.error("Must provide either --classes or --gloss-file")

    # Load gloss list from file if provided
    gloss_list = None
    if args.gloss_file:
        with open(args.gloss_file, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            gloss_list = data
        elif 'classes' in data:
            gloss_list = data['classes']
        elif 'gloss_list' in data:
            gloss_list = data['gloss_list']
        else:
            parser.error(f"Cannot parse gloss list from {args.gloss_file}")

    # Run generation
    result = generate_balanced_dataset(
        num_classes=args.classes,
        gloss_list=gloss_list,
        landmark_config=args.landmark_config,
        target_per_class=args.target,
        dry_run=args.dry_run,
        output_base=args.output,
        skip_existing=not args.no_skip,
    )

    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
