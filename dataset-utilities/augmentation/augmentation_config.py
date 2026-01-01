"""
Augmentation Configuration for Balanced Dataset Generation

This module defines:
1. Target samples per class for balanced training
2. Augmentation techniques and their parameters
3. Functions to calculate per-class augmentation counts
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


# =============================================================================
# CLASS BALANCING CONFIGURATION (Two-Phase Approach)
# =============================================================================

# Target number of samples per class IN THE TRAIN SPLIT (not pool)
# The two-phase approach ensures exactly this many samples in train:
#   Phase 1: Generate base augmentations for pool (enough for val/test)
#   Phase 2: After splitting, augment train families to reach this target
TARGET_TRAIN_SAMPLES_PER_CLASS = 50

# Legacy setting (kept for backward compatibility)
TARGET_SAMPLES_PER_CLASS = TARGET_TRAIN_SAMPLES_PER_CLASS

# -----------------------------------------------------------------------------
# Phase 1: Base Pool Generation
# -----------------------------------------------------------------------------
# Generate enough augmentations so val/test have reasonable data
# Val+Test get ~30% of families, each family needs some samples
# Base augmentations: moderate amount, not trying to hit train target yet
BASE_AUGMENTATIONS_PER_VIDEO = 15  # Creates families of size 16 (1 orig + 15 aug)

# Minimum base augmentations (ensures some variety even for large classes)
MIN_BASE_AUGMENTATIONS = 5

# Maximum base augmentations in phase 1 (will add more in phase 2 for train)
MAX_BASE_AUGMENTATIONS = 30

# -----------------------------------------------------------------------------
# Phase 2: Train Balancing
# -----------------------------------------------------------------------------
# After splitting, augment train families to reach TARGET_TRAIN_SAMPLES_PER_CLASS
# This phase generates augmentations ONLY for train split

# Maximum total augmentations per video (base + train balancing combined)
MAX_TOTAL_AUGMENTATIONS_PER_VIDEO = 200

# Legacy settings (kept for backward compatibility)
MIN_AUGMENTATIONS_PER_VIDEO = MIN_BASE_AUGMENTATIONS
MAX_AUGMENTATIONS_PER_VIDEO = MAX_BASE_AUGMENTATIONS

# Split ratios for stratified splitting
SPLIT_RATIOS = {
    'train': 0.70,  # 70% for training
    'val': 0.15,    # 15% for validation
    'test': 0.15,   # 15% for testing
}


# =============================================================================
# AUGMENTATION TECHNIQUES
# =============================================================================

# Each technique has:
#   - 'enabled': Whether to use this technique
#   - 'count': Number of variations to generate
#   - 'params': Parameters for the technique

AUGMENTATION_TECHNIQUES = {
    # Phase 1: Geometric transformations (most important for pose invariance)
    'geometric': {
        'enabled': True,
        'weight': 3,  # Higher weight = more augmentations of this type
        'params': {
            'shear_std': 0.12,      # Standard deviation for shear
            'rotation_std': 0.12,   # Standard deviation for rotation (~±6°)
        }
    },

    # Phase 2: Horizontal flip (important for left/right hand signs)
    'horizontal_flip': {
        'enabled': True,
        'weight': 2,
        'params': {
            'flip_probability': 1.0,  # Always flip when selected
        }
    },

    # Phase 3: Spatial variations
    'spatial_noise': {
        'enabled': True,
        'weight': 1,
        'params': {
            'noise_levels': [0.010, 0.020, 0.030],  # Low, medium, high
        }
    },

    'translation': {
        'enabled': True,
        'weight': 1,
        'params': {
            'translation_ranges': [0.10, 0.20, 0.30],  # Small, medium, large
        }
    },

    # Phase 4: Scaling (important for different camera distances)
    'scaling': {
        'enabled': True,
        'weight': 2,
        'params': {
            'scale_range': (0.70, 1.50),  # Min and max scale factors
        }
    },

    # Phase 5: Temporal variations (important for signing speed differences)
    'speed_variation': {
        'enabled': True,
        'weight': 2,
        'params': {
            'speed_factors': [0.50, 0.60, 0.75, 1.25, 1.50, 1.80],
        }
    },

    # Phase 6: Occlusion simulation (robustness to missing keypoints)
    'keypoint_occlusion': {
        'enabled': True,
        'weight': 1,
        'params': {
            'occlusion_rates': [0.15, 0.25, 0.35],  # 15%, 25%, 35% keypoints
        }
    },

    'hand_dropout': {
        'enabled': True,
        'weight': 1,
        'params': {
            'dropout_probability': 0.15,
        }
    },

    # Phase 7: Combinations (multiple transforms at once)
    'combinations': {
        'enabled': True,
        'weight': 2,
        'params': {
            'combine_techniques': ['geometric', 'scaling', 'spatial_noise'],
        }
    },
}


# =============================================================================
# FRAME LENGTH BUCKETS (for stratified splitting)
# =============================================================================

FRAME_BUCKETS = {
    'short': (0, 50),      # 0-49 frames
    'medium': (50, 80),    # 50-79 frames
    'long': (80, float('inf')),  # 80+ frames
}


def get_frame_bucket(frame_count: int) -> str:
    """
    Determine which frame bucket a video belongs to.

    Args:
        frame_count: Number of frames in the video

    Returns:
        Bucket name ('short', 'medium', or 'long')
    """
    for bucket_name, (min_frames, max_frames) in FRAME_BUCKETS.items():
        if min_frames <= frame_count < max_frames:
            return bucket_name
    return 'long'  # Default to long if somehow out of range


# =============================================================================
# CLASS BALANCING FUNCTIONS
# =============================================================================

def calculate_augmentation_counts(
    class_sample_counts: Dict[str, int],
    target_per_class: int = TARGET_SAMPLES_PER_CLASS,
    min_per_video: int = MIN_AUGMENTATIONS_PER_VIDEO,
    max_per_video: int = MAX_AUGMENTATIONS_PER_VIDEO,
) -> Dict[str, Dict]:
    """
    Calculate how many augmentations each class/video needs for balanced dataset.

    Args:
        class_sample_counts: Dict mapping class name to number of original samples
        target_per_class: Target total samples per class after augmentation
        min_per_video: Minimum augmentations per video
        max_per_video: Maximum augmentations per video

    Returns:
        Dict mapping class name to augmentation plan:
        {
            'class_name': {
                'original_count': int,
                'target_total': int,
                'augmentations_needed': int,
                'augmentations_per_video': int,
                'actual_total': int,
            }
        }
    """
    augmentation_plan = {}

    for class_name, original_count in class_sample_counts.items():
        # Calculate augmentations needed
        augmentations_needed = max(0, target_per_class - original_count)

        # Calculate per-video augmentation count
        if original_count > 0:
            aug_per_video = augmentations_needed // original_count
            # Apply min/max constraints
            aug_per_video = max(min_per_video, min(max_per_video, aug_per_video))
        else:
            aug_per_video = 0

        # Calculate actual total after augmentation
        actual_total = original_count + (original_count * aug_per_video)

        augmentation_plan[class_name] = {
            'original_count': original_count,
            'target_total': target_per_class,
            'augmentations_needed': augmentations_needed,
            'augmentations_per_video': aug_per_video,
            'actual_total': actual_total,
        }

    return augmentation_plan


def calculate_base_augmentation_counts(
    class_sample_counts: Dict[str, int],
    base_per_video: int = BASE_AUGMENTATIONS_PER_VIDEO,
    min_per_video: int = MIN_BASE_AUGMENTATIONS,
    max_per_video: int = MAX_BASE_AUGMENTATIONS,
) -> Dict[str, Dict]:
    """
    Calculate base augmentations for Phase 1 (pool generation).

    In Phase 1, we generate a fixed number of augmentations per video
    to create a pool that can be split. The goal is to have enough
    samples for val/test, not to reach the train target yet.

    Args:
        class_sample_counts: Dict mapping class name to number of original samples
        base_per_video: Base augmentations per video
        min_per_video: Minimum augmentations per video
        max_per_video: Maximum augmentations per video

    Returns:
        Dict mapping class name to augmentation plan
    """
    augmentation_plan = {}

    for class_name, original_count in class_sample_counts.items():
        # Use fixed base augmentations (within min/max bounds)
        aug_per_video = max(min_per_video, min(max_per_video, base_per_video))

        # Calculate pool total
        pool_total = original_count * (1 + aug_per_video)

        augmentation_plan[class_name] = {
            'original_count': original_count,
            'augmentations_per_video': aug_per_video,
            'family_size': 1 + aug_per_video,
            'pool_total': pool_total,
            'phase': 'base_pool',
        }

    return augmentation_plan


def calculate_train_balancing_augmentations(
    train_class_samples: Dict[str, int],
    train_families: Dict[str, int],
    target_train_samples: int = TARGET_TRAIN_SAMPLES_PER_CLASS,
    max_total_per_video: int = MAX_TOTAL_AUGMENTATIONS_PER_VIDEO,
    existing_augs_per_video: int = BASE_AUGMENTATIONS_PER_VIDEO,
) -> Dict[str, Dict]:
    """
    Calculate additional augmentations needed for Phase 2 (train balancing).

    After splitting, some classes may have fewer than target samples in train.
    This calculates how many additional augmentations to generate for train families.

    Args:
        train_class_samples: Dict mapping class name to current train sample count
        train_families: Dict mapping class name to number of families in train
        target_train_samples: Target samples per class in train split
        max_total_per_video: Maximum total augmentations per video
        existing_augs_per_video: Augmentations already generated in Phase 1

    Returns:
        Dict mapping class name to balancing plan
    """
    balancing_plan = {}

    for class_name, current_samples in train_class_samples.items():
        samples_needed = target_train_samples - current_samples
        n_families = train_families.get(class_name, 0)

        if samples_needed <= 0 or n_families == 0:
            # Already at or above target, no balancing needed
            balancing_plan[class_name] = {
                'current_samples': current_samples,
                'target_samples': target_train_samples,
                'samples_needed': 0,
                'additional_augs_per_video': 0,
                'final_samples': current_samples,
                'status': 'already_balanced' if samples_needed <= 0 else 'no_families',
            }
            continue

        # Calculate additional augmentations needed per family (video)
        additional_per_video = samples_needed // n_families

        # Check against max total augmentations
        max_additional = max_total_per_video - existing_augs_per_video - 1  # -1 for original
        additional_per_video = min(additional_per_video, max_additional)
        additional_per_video = max(0, additional_per_video)

        # Calculate final sample count
        final_samples = current_samples + (n_families * additional_per_video)

        balancing_plan[class_name] = {
            'current_samples': current_samples,
            'target_samples': target_train_samples,
            'samples_needed': samples_needed,
            'train_families': n_families,
            'additional_augs_per_video': additional_per_video,
            'final_samples': final_samples,
            'status': 'balanced' if final_samples >= target_train_samples else 'partial',
        }

    return balancing_plan


def get_augmentation_sequence(aug_count: int) -> List[Dict]:
    """
    Generate a sequence of augmentation configurations for a video.

    Args:
        aug_count: Number of augmentations to generate

    Returns:
        List of augmentation configurations, each containing:
        {
            'id': int,
            'technique': str,
            'params': dict,
        }
    """
    augmentations = []

    # Calculate total weight
    total_weight = sum(
        tech['weight'] for tech in AUGMENTATION_TECHNIQUES.values()
        if tech['enabled']
    )

    # Distribute augmentations across techniques based on weight
    aug_id = 0
    for tech_name, tech_config in AUGMENTATION_TECHNIQUES.items():
        if not tech_config['enabled']:
            continue

        # Calculate how many augmentations for this technique
        tech_count = int(aug_count * tech_config['weight'] / total_weight)
        tech_count = max(1, tech_count)  # At least 1 if enabled

        for i in range(tech_count):
            if aug_id >= aug_count:
                break

            augmentations.append({
                'id': aug_id,
                'technique': tech_name,
                'params': tech_config['params'].copy(),
                'variation_index': i,
            })
            aug_id += 1

        if aug_id >= aug_count:
            break

    # Fill remaining slots with geometric (most important)
    while len(augmentations) < aug_count:
        augmentations.append({
            'id': len(augmentations),
            'technique': 'geometric',
            'params': AUGMENTATION_TECHNIQUES['geometric']['params'].copy(),
            'variation_index': len(augmentations),
        })

    return augmentations[:aug_count]


def print_augmentation_plan(augmentation_plan: Dict[str, Dict]) -> None:
    """Print a summary of the augmentation plan.

    Handles both formats:
    - Legacy: 'actual_total' key (target-based approach)
    - New: 'pool_total' key (base augmentation approach)
    """
    print("\n" + "=" * 70)
    print("AUGMENTATION PLAN SUMMARY")
    print("=" * 70)

    # Detect format and get total
    sample_plan = next(iter(augmentation_plan.values()))
    is_base_format = 'pool_total' in sample_plan

    total_original = sum(p['original_count'] for p in augmentation_plan.values())

    if is_base_format:
        total_augmented = sum(p['pool_total'] for p in augmentation_plan.values())
        print(f"\nMode: Base augmentation (Phase 1)")
    else:
        total_augmented = sum(p['actual_total'] for p in augmentation_plan.values())
        print(f"\nMode: Target-based")

    print(f"Classes: {len(augmentation_plan)}")
    print(f"Total original samples: {total_original}")
    print(f"Total after augmentation: {total_augmented}")
    print(f"Augmentation factor: {total_augmented / total_original:.1f}x")

    # Distribution of augmentations per video
    aug_per_video_counts = [p['augmentations_per_video'] for p in augmentation_plan.values()]
    print(f"\nAugmentations per video:")
    print(f"  Min: {min(aug_per_video_counts)}")
    print(f"  Max: {max(aug_per_video_counts)}")
    print(f"  Mean: {sum(aug_per_video_counts) / len(aug_per_video_counts):.1f}")

    # Show some examples
    print(f"\nSample class plans:")
    sorted_plan = sorted(augmentation_plan.items(), key=lambda x: x[1]['original_count'])

    # Show smallest, middle, and largest classes
    indices = [0, len(sorted_plan) // 2, -1]
    for idx in indices:
        class_name, plan = sorted_plan[idx]
        total_key = 'pool_total' if is_base_format else 'actual_total'
        print(f"  {class_name}: {plan['original_count']} original -> "
              f"{plan['augmentations_per_video']} aug/video -> "
              f"{plan[total_key]} total")

    print("=" * 70 + "\n")


def save_augmentation_plan(augmentation_plan: Dict, output_path: Path) -> None:
    """Save augmentation plan to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata
    plan_with_metadata = {
        'config': {
            'target_samples_per_class': TARGET_SAMPLES_PER_CLASS,
            'min_augmentations_per_video': MIN_AUGMENTATIONS_PER_VIDEO,
            'max_augmentations_per_video': MAX_AUGMENTATIONS_PER_VIDEO,
            'split_ratios': SPLIT_RATIOS,
        },
        'classes': augmentation_plan,
    }

    with open(output_path, 'w') as f:
        json.dump(plan_with_metadata, f, indent=2)

    print(f"Saved augmentation plan to: {output_path}")


def load_augmentation_plan(plan_path: Path) -> Tuple[Dict, Dict]:
    """
    Load augmentation plan from JSON file.

    Returns:
        Tuple of (config_dict, class_plans_dict)
    """
    with open(plan_path, 'r') as f:
        data = json.load(f)

    return data.get('config', {}), data.get('classes', {})
