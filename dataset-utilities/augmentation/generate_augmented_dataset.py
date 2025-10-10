#!/usr/bin/env python3
"""
generate_properly_split_dataset.py

Generate augmented dataset with PROPER train/val/test splitting to prevent data leakage.

Key principle: Split by video ID BEFORE augmentation, never mix video IDs across splits.
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from collections import Counter
import argparse
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the augmentation system
# Note: PoseAugmentationGenerator class moved here from generate_augmented_dataset.py
from openhands_modernized import WLASLPoseProcessor

class PoseAugmentationGenerator:
    """Generate multiple augmented versions of pose sequences."""

    def __init__(self):
        self.processor = WLASLPoseProcessor()

    def apply_augmentation(self, pose_sequence: np.ndarray, augmentation_strength: str = 'moderate') -> np.ndarray:
        """Apply comprehensive augmentation to pose sequence."""

        if pose_sequence is None or len(pose_sequence) == 0:
            return pose_sequence

        # Make a copy to avoid modifying the original
        pose_sequence = pose_sequence.copy()

        # CONSERVATIVE augmentation parameters to prevent overfitting artifacts
        if augmentation_strength == 'light':
            noise_range = (0.001, 0.005)  # Much smaller noise
            shift_range = 1              # Minimal temporal shift
            scale_range = (0.98, 1.02)   # Tiny scaling only
            rotation_prob = 0.1          # Rare rotation
            rotation_range = 0.02        # Very small rotation
        elif augmentation_strength == 'moderate':
            noise_range = (0.002, 0.008)  # Reduced noise
            shift_range = 2               # Small temporal shift
            scale_range = (0.96, 1.04)    # Conservative scaling
            rotation_prob = 0.2           # Reduced rotation
            rotation_range = 0.04         # Smaller rotation
        else:  # 'strong' - still much more conservative than before
            noise_range = (0.005, 0.015)  # Still reasonable noise
            shift_range = 3               # Limited shift
            scale_range = (0.94, 1.06)    # Moderate scaling
            rotation_prob = 0.3           # Moderate rotation
            rotation_range = 0.06         # Small rotation

        # 1. Random noise injection
        noise_factor = np.random.uniform(*noise_range)
        noise = np.random.normal(0, noise_factor, pose_sequence.shape)
        pose_sequence = pose_sequence + noise

        # 2. Random temporal shift
        if len(pose_sequence) > 10:
            shift = np.random.randint(-shift_range, shift_range + 1)
            if shift != 0:
                pose_sequence = np.roll(pose_sequence, shift, axis=0)

        # 3. Random scaling
        scale_factor = np.random.uniform(*scale_range)
        pose_sequence = pose_sequence * scale_factor

        # 4. Random rotation (simulate different viewing angles)
        if np.random.random() < rotation_prob:
            angle = np.random.uniform(-rotation_range, rotation_range)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            # Apply rotation to x,y coordinates (assuming they are in pairs)
            for i in range(0, pose_sequence.shape[1], 2):
                if i + 1 < pose_sequence.shape[1]:
                    x, y = pose_sequence[:, i], pose_sequence[:, i + 1]
                    pose_sequence[:, i] = x * cos_a - y * sin_a
                    pose_sequence[:, i + 1] = x * sin_a + y * cos_a

        # 5. CONSERVATIVE time warping (much gentler)
        if len(pose_sequence) > 20 and np.random.random() < 0.15:  # Reduced probability
            seq_len = len(pose_sequence)
            # Generate GENTLE warp factors (much closer to 1.0)
            warp_points = [0, seq_len // 2, seq_len - 1]  # Fewer warp points
            warp_factors = [1.0, np.random.uniform(0.95, 1.05), 1.0]  # Much gentler warping

            # Create time mapping
            old_indices = np.arange(seq_len)
            new_indices = np.interp(old_indices, warp_points,
                                  [sum(warp_factors[:i+1]) * seq_len / sum(warp_factors)
                                   for i in range(len(warp_points))])
            new_indices = np.clip(new_indices, 0, seq_len - 1).astype(int)
            pose_sequence = pose_sequence[new_indices]

        # 6. CONSERVATIVE keypoint dropout (much gentler occlusion)
        if np.random.random() < 0.1:  # Reduced probability
            num_keypoints = pose_sequence.shape[1] // 2
            if num_keypoints > 20:  # Only for datasets with many keypoints
                num_dropout = np.random.randint(1, min(2, num_keypoints // 10))  # Much fewer dropouts
                dropout_keypoints = np.random.choice(num_keypoints, num_dropout, replace=False)
                for kp in dropout_keypoints:
                    # Instead of zeroing, add slight noise to simulate tracking error
                    pose_sequence[:, kp*2:(kp*2)+2] += np.random.normal(0, 0.001, (len(pose_sequence), 2))

        # 7. CONSERVATIVE frame interpolation (very gentle)
        if len(pose_sequence) > 30 and np.random.random() < 0.05:  # Much reduced probability
            num_dropout = 1  # Only 1 frame max
            dropout_frames = np.random.choice(range(2, len(pose_sequence)-2), num_dropout, replace=False)
            for frame_idx in dropout_frames:
                # Smooth interpolation instead of simple average
                pose_sequence[frame_idx] = (pose_sequence[frame_idx-1] + pose_sequence[frame_idx+1]) / 2

        # 8. REMOVED: Sequence reversal causes semantic changes - too risky

        # 9. CONSERVATIVE per-joint scaling (very gentle)
        if np.random.random() < 0.1:  # Much reduced probability
            for i in range(pose_sequence.shape[1]):
                joint_scale = np.random.uniform(0.99, 1.01)  # Tiny scaling only
                pose_sequence[:, i] *= joint_scale

        return pose_sequence

def create_proper_splits(source_dir, target_dir, classes_filter=None,
                        augmentation_per_file=10, augmentation_strength='moderate',
                        test_size=0.2, val_size=0.2):
    """
    Create properly split augmented dataset with NO data leakage.

    Process:
    1. Load all original files
    2. Split by video ID into train/val/test (BEFORE augmentation)
    3. Augment each split separately
    4. Save in organized directory structure
    """

    print("CREATING PROPERLY SPLIT AUGMENTED DATASET")
    print("=" * 70)
    print("CRITICAL: Splitting by video ID BEFORE augmentation to prevent data leakage")
    print(f"SOURCE: {source_dir}")
    print(f"TARGET: {target_dir}")
    print(f"TEST SIZE: {test_size:.1%}")
    print(f"VAL SIZE: {val_size:.1%}")
    print(f"TRAIN SIZE: {1-test_size-val_size:.1%}")
    print()

    # Create output directory
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # Initialize augmentation generator
    augmenter = PoseAugmentationGenerator()

    # Step 1: Load all original files and group by video ID
    print("STEP 1: Loading original files and extracting video IDs...")

    video_files = {}  # video_id -> file_path
    video_labels = {}  # video_id -> label
    all_labels = []

    for file_path in Path(source_dir).rglob("*.pkl"):
        try:
            with open(file_path, 'rb') as f:
                pickle_data = pickle.load(f)

            if isinstance(pickle_data, dict) and 'gloss' in pickle_data:
                gloss_label = pickle_data['gloss'].upper()
                video_id = pickle_data.get('video_id', Path(file_path).stem)

                # Filter by classes if specified
                if classes_filter is None or gloss_label in classes_filter:
                    video_files[video_id] = str(file_path)
                    video_labels[video_id] = gloss_label
                    all_labels.append(gloss_label)

        except Exception as e:
            print(f"WARNING: Could not load {file_path}: {e}")
            continue

    print(f"LOADED: {len(video_files)} unique video IDs")

    # Analyze class distribution
    class_counts = Counter(all_labels)
    print(f"CLASSES: {len(class_counts)} unique classes")
    for cls, count in class_counts.most_common():
        print(f"  {cls}: {count} videos")

    if classes_filter:
        print(f"FILTERED TO: {len(classes_filter)} specified classes")

    # Step 2: Split by video ID (CRITICAL: before augmentation)
    print(f"\nSTEP 2: Splitting by video ID (BEFORE augmentation)...")

    video_ids = list(video_files.keys())
    labels_for_split = [video_labels[vid] for vid in video_ids]

    # First split: train vs temp (val+test)
    train_ids, temp_ids, train_labels, temp_labels = train_test_split(
        video_ids, labels_for_split,
        test_size=test_size + val_size,
        random_state=42,
        stratify=labels_for_split
    )

    # Second split: val vs test
    val_ids, test_ids, val_labels, test_labels = train_test_split(
        temp_ids, temp_labels,
        test_size=test_size / (test_size + val_size),  # Proportion of temp that should be test
        random_state=42,
        stratify=temp_labels
    )

    print(f"SPLIT RESULT:")
    print(f"  TRAIN: {len(train_ids)} videos ({len(train_ids)/len(video_ids):.1%})")
    print(f"  VAL: {len(val_ids)} videos ({len(val_ids)/len(video_ids):.1%})")
    print(f"  TEST: {len(test_ids)} videos ({len(test_ids)/len(video_ids):.1%})")

    # Verify no overlap (CRITICAL CHECK)
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    assert len(train_set & val_set) == 0, f"TRAIN/VAL overlap: {train_set & val_set}"
    assert len(train_set & test_set) == 0, f"TRAIN/TEST overlap: {train_set & test_set}"
    assert len(val_set & test_set) == 0, f"VAL/TEST overlap: {val_set & test_set}"

    print("VERIFIED: No video ID overlap between splits")

    # Step 3: Create split directories and augment each split separately
    print(f"\nSTEP 3: Augmenting each split separately...")

    splits = {
        'train': (train_ids, train_labels),
        'val': (val_ids, val_labels),
        'test': (test_ids, test_labels)
    }

    total_generated = 0
    failed_generations = 0

    for split_name, (split_ids, split_labels) in splits.items():
        print(f"\n--- PROCESSING {split_name.upper()} SPLIT ---")
        print(f"Videos in {split_name}: {len(split_ids)}")

        # Create split directory
        split_dir = Path(target_dir) / split_name
        split_dir.mkdir(exist_ok=True)

        # Process each video in this split
        for video_id in tqdm(split_ids, desc=f"Processing {split_name}"):
            try:
                # Load original file
                source_file = video_files[video_id]
                with open(source_file, 'rb') as f:
                    original_data = pickle.load(f)

                label = video_labels[video_id]

                # Create class subdirectory
                class_dir = split_dir / label
                class_dir.mkdir(exist_ok=True)

                # Extract pose sequence for augmentation
                pose_sequence = augmenter.processor.load_pickle_pose(source_file)
                pose_sequence = augmenter.processor.normalize_pose_sequence(pose_sequence)

                if pose_sequence is None or len(pose_sequence) == 0:
                    print(f"WARNING: Empty pose sequence in {source_file}")
                    continue

                # Copy original file first
                original_filename = Path(source_file).name
                original_target = class_dir / f"original_{original_filename}"
                shutil.copy2(source_file, original_target)

                # Generate augmented versions for this video
                for aug_idx in range(augmentation_per_file):
                    try:
                        # Apply augmentation
                        augmented_sequence = augmenter.apply_augmentation(
                            pose_sequence, augmentation_strength)

                        # Create new pickle data with FIXED keypoints
                        augmented_data = original_data.copy()

                        # Convert augmented sequence back to original keypoints format
                        frames = len(augmented_sequence)
                        num_landmarks = augmented_sequence.shape[1] // 2

                        # Create new keypoints array
                        new_keypoints = original_data['keypoints'].copy()
                        augmented_kpts_reshaped = augmented_sequence.reshape(frames, num_landmarks, 2)

                        # Replace the first num_landmarks with augmented data
                        if hasattr(new_keypoints, 'data'):
                            new_keypoints.data[:frames, :num_landmarks, :] = augmented_kpts_reshaped
                        else:
                            new_keypoints[:frames, :num_landmarks, :] = augmented_kpts_reshaped

                        # Update pickle data
                        augmented_data['keypoints'] = new_keypoints
                        augmented_data['augmented'] = True
                        augmented_data['augmentation_id'] = aug_idx
                        augmented_data['original_file'] = original_filename
                        augmented_data['split'] = split_name  # Mark which split this belongs to

                        # Save augmented file
                        aug_filename = f"aug_{aug_idx:02d}_{original_filename}"
                        aug_path = class_dir / aug_filename

                        with open(aug_path, 'wb') as f:
                            pickle.dump(augmented_data, f)

                        total_generated += 1

                    except Exception as e:
                        print(f"WARNING: Failed augmentation {aug_idx} for {source_file}: {e}")
                        failed_generations += 1

            except Exception as e:
                print(f"ERROR: Failed to process {source_file}: {e}")
                failed_generations += 1

    # Step 4: Create summary
    print(f"\nCREATING PROPERLY SPLIT DATASET COMPLETE!")
    print("=" * 60)
    print(f"ORIGINAL VIDEOS: {len(video_files)}")
    print(f"TOTAL GENERATED: {total_generated}")
    print(f"FAILED GENERATIONS: {failed_generations}")
    print(f"FINAL DATASET STRUCTURE:")
    print(f"  {target_dir}/train/   - Training videos + augmentations")
    print(f"  {target_dir}/val/     - Validation videos + augmentations")
    print(f"  {target_dir}/test/    - Test videos + augmentations")
    print()
    print("CRITICAL SUCCESS: No data leakage - each video ID appears in only ONE split!")

    # Create detailed summary
    summary = {
        'original_videos': len(video_files),
        'total_generated': total_generated,
        'failed_generations': failed_generations,
        'splits': {
            'train': {'video_ids': train_ids, 'count': len(train_ids)},
            'val': {'video_ids': val_ids, 'count': len(val_ids)},
            'test': {'video_ids': test_ids, 'count': len(test_ids)}
        },
        'class_distribution': dict(class_counts),
        'augmentation_per_file': augmentation_per_file,
        'augmentation_strength': augmentation_strength
    }

    summary_path = Path(target_dir) / "split_summary.pkl"
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)

    print(f"SUMMARY: Saved to {summary_path}")
    return summary

def main():
    parser = argparse.ArgumentParser(description='Generate properly split augmented dataset')
    parser.add_argument('--source', default="C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/wlasl_poses_complete/pickle_files",
                       help='Source directory containing original .pkl files')
    parser.add_argument('--target', default="C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/wlasl_poses_complete/properly_split_20_class",
                       help='Target directory for properly split dataset')
    parser.add_argument('--classes', type=int, choices=[20, 50, 100], default=20,
                       help='Filter to top N classes (default: 20)')
    parser.add_argument('--augmentations', type=int, default=10,
                       help='Number of augmented versions per original file (default: 10)')
    parser.add_argument('--strength', choices=['light', 'moderate', 'strong'], default='moderate',
                       help='Augmentation strength (default: moderate)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set proportion (default: 0.2)')
    parser.add_argument('--val-size', type=float, default=0.2,
                       help='Validation set proportion (default: 0.2)')

    args = parser.parse_args()

    # Get top classes
    print(f"FILTERING: Loading top {args.classes} classes...")
    all_labels = []

    for file_path in Path(args.source).rglob("*.pkl"):
        try:
            with open(file_path, 'rb') as f:
                pickle_data = pickle.load(f)
            if isinstance(pickle_data, dict) and 'gloss' in pickle_data:
                all_labels.append(pickle_data['gloss'].upper())
        except:
            continue

    class_counts = Counter(all_labels)
    top_classes = [cls for cls, _ in class_counts.most_common(args.classes)]
    classes_filter = set(top_classes)

    print(f"TOP {args.classes} CLASSES:")
    for i, (cls, count) in enumerate(class_counts.most_common(args.classes)):
        print(f"  {i+1:2d}. {cls}: {count} files")
    print()

    # Generate properly split dataset
    summary = create_proper_splits(
        source_dir=args.source,
        target_dir=args.target,
        classes_filter=classes_filter,
        augmentation_per_file=args.augmentations,
        augmentation_strength=args.strength,
        test_size=args.test_size,
        val_size=args.val_size
    )

    print()
    print("SUCCESS: Properly split augmented dataset created!")
    print("Ready for training with NO data leakage!")

if __name__ == "__main__":
    main()