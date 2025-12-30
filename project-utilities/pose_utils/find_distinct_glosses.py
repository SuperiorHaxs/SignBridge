#!/usr/bin/env python3
"""
find_distinct_glosses.py - Find the most distinct glosses from a pose dataset

This script analyzes pose files for all glosses and selects a subset that are
maximally distinct from each other using a greedy maximum dispersion algorithm.

Usage:
    python find_distinct_glosses.py --input <pose_folder> --output <output_file> --count 100
    python find_distinct_glosses.py --list  # Just list all glosses with sample counts

Examples:
    python find_distinct_glosses.py -i ../../datasets/wlasl_poses_complete/pose_files_by_gloss -o distinct_100.json -n 100
    python find_distinct_glosses.py -i ../../datasets/wlasl_poses_complete/pose_files_by_gloss --list
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import pickle


@dataclass
class GlossFeatures:
    """Features extracted from a gloss's pose files."""
    gloss: str
    sample_count: int
    # Hand positions (normalized, relative to body center)
    left_hand_mean_x: float
    left_hand_mean_y: float
    right_hand_mean_x: float
    right_hand_mean_y: float
    # Movement range (std dev)
    left_hand_std_x: float
    left_hand_std_y: float
    right_hand_std_x: float
    right_hand_std_y: float
    # Hand spread
    hand_spread_mean: float
    hand_spread_std: float
    # Hand height relative to nose
    left_height_mean: float
    right_height_mean: float
    # Movement velocity
    left_velocity_mean: float
    right_velocity_mean: float
    # Duration (normalized)
    duration_mean: float

    def to_vector(self) -> np.ndarray:
        """Convert features to a numpy vector for distance calculations."""
        return np.array([
            self.left_hand_mean_x,
            self.left_hand_mean_y,
            self.right_hand_mean_x,
            self.right_hand_mean_y,
            self.left_hand_std_x,
            self.left_hand_std_y,
            self.right_hand_std_x,
            self.right_hand_std_y,
            self.hand_spread_mean,
            self.hand_spread_std,
            self.left_height_mean,
            self.right_height_mean,
            self.left_velocity_mean,
            self.right_velocity_mean,
            self.duration_mean,
        ])


def load_pose_75pt(pose_path: Path) -> Tuple[np.ndarray, float]:
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

    return np.array(pose_75pt), fps


def normalize_pose(pose_data: np.ndarray) -> np.ndarray:
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

        # Scale by shoulder width
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        if shoulder_width < 1e-6:
            shoulder_width = 1.0

        # Normalize
        frame_centered = frame - center
        frame_normalized = frame_centered / shoulder_width
        normalized[frame_idx] = frame_normalized

    return normalized


def extract_pose_features(pose_data: np.ndarray) -> Dict[str, float]:
    """Extract features from a single pose sequence."""
    # Key landmark indices (in 75-point format)
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HAND_CENTER = 33 + 9   # Middle finger MCP
    RIGHT_HAND_CENTER = 54 + 9

    frames = len(pose_data)

    # Normalize pose
    pose_data = normalize_pose(pose_data)

    # Calculate body center (midpoint of shoulders)
    shoulders_mid = (pose_data[:, LEFT_SHOULDER, :] + pose_data[:, RIGHT_SHOULDER, :]) / 2

    # Hand positions relative to body center
    left_hand_rel = pose_data[:, LEFT_HAND_CENTER, :] - shoulders_mid
    right_hand_rel = pose_data[:, RIGHT_HAND_CENTER, :] - shoulders_mid

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

    return {
        'left_hand_mean': left_hand_rel.mean(axis=0),
        'right_hand_mean': right_hand_rel.mean(axis=0),
        'left_hand_std': left_hand_rel.std(axis=0),
        'right_hand_std': right_hand_rel.std(axis=0),
        'hand_spread_mean': float(hand_distance.mean()),
        'hand_spread_std': float(hand_distance.std()),
        'left_height_mean': float(left_hand_height.mean()),
        'right_height_mean': float(right_hand_height.mean()),
        'left_velocity_mean': float(left_velocity.mean()),
        'right_velocity_mean': float(right_velocity.mean()),
        'duration': frames,
    }


def extract_gloss_features(gloss_dir: Path) -> Optional[GlossFeatures]:
    """Extract aggregated features for a gloss from all its pose files."""
    pose_files = list(gloss_dir.glob("*.pose"))

    if not pose_files:
        return None

    all_features = []

    for pose_file in pose_files:
        try:
            pose_data, fps = load_pose_75pt(pose_file)
            if len(pose_data) < 5:  # Skip very short sequences
                continue
            features = extract_pose_features(pose_data)
            all_features.append(features)
        except Exception as e:
            # Skip problematic files
            continue

    if not all_features:
        return None

    # Aggregate features across all samples (mean)
    n = len(all_features)

    left_hand_means = np.array([f['left_hand_mean'] for f in all_features])
    right_hand_means = np.array([f['right_hand_mean'] for f in all_features])
    left_hand_stds = np.array([f['left_hand_std'] for f in all_features])
    right_hand_stds = np.array([f['right_hand_std'] for f in all_features])

    # Normalize duration to 0-1 range (assuming max ~200 frames)
    durations = np.array([f['duration'] for f in all_features])
    duration_normalized = np.clip(durations / 200.0, 0, 1)

    return GlossFeatures(
        gloss=gloss_dir.name,
        sample_count=n,
        left_hand_mean_x=float(left_hand_means[:, 0].mean()),
        left_hand_mean_y=float(left_hand_means[:, 1].mean()),
        right_hand_mean_x=float(right_hand_means[:, 0].mean()),
        right_hand_mean_y=float(right_hand_means[:, 1].mean()),
        left_hand_std_x=float(left_hand_stds[:, 0].mean()),
        left_hand_std_y=float(left_hand_stds[:, 1].mean()),
        right_hand_std_x=float(right_hand_stds[:, 0].mean()),
        right_hand_std_y=float(right_hand_stds[:, 1].mean()),
        hand_spread_mean=float(np.mean([f['hand_spread_mean'] for f in all_features])),
        hand_spread_std=float(np.mean([f['hand_spread_std'] for f in all_features])),
        left_height_mean=float(np.mean([f['left_height_mean'] for f in all_features])),
        right_height_mean=float(np.mean([f['right_height_mean'] for f in all_features])),
        left_velocity_mean=float(np.mean([f['left_velocity_mean'] for f in all_features])),
        right_velocity_mean=float(np.mean([f['right_velocity_mean'] for f in all_features])),
        duration_mean=float(duration_normalized.mean()),
    )


def compute_distance_matrix(features_list: List[GlossFeatures]) -> np.ndarray:
    """Compute pairwise distance matrix between all glosses."""
    n = len(features_list)

    # Convert to matrix
    vectors = np.array([f.to_vector() for f in features_list])

    # Normalize each feature to have zero mean and unit variance
    # This ensures all features contribute equally to distance
    means = vectors.mean(axis=0)
    stds = vectors.std(axis=0)
    stds[stds < 1e-6] = 1.0  # Avoid division by zero
    vectors_normalized = (vectors - means) / stds

    # Compute Euclidean distance matrix
    from scipy.spatial.distance import cdist
    distance_matrix = cdist(vectors_normalized, vectors_normalized, 'euclidean')

    return distance_matrix


def greedy_max_dispersion(distance_matrix: np.ndarray, k: int,
                          min_samples: int = 0,
                          features_list: List[GlossFeatures] = None) -> List[int]:
    """
    Select k items that maximize the minimum pairwise distance.

    Args:
        distance_matrix: n x n pairwise distance matrix
        k: Number of items to select
        min_samples: Minimum sample count required (0 = no filter)
        features_list: List of GlossFeatures (for filtering by sample count)

    Returns:
        List of indices of selected items
    """
    n = len(distance_matrix)

    # Create mask for eligible glosses (based on min_samples)
    if min_samples > 0 and features_list is not None:
        eligible = set(i for i, f in enumerate(features_list) if f.sample_count >= min_samples)
    else:
        eligible = set(range(n))

    if len(eligible) < k:
        print(f"Warning: Only {len(eligible)} glosses meet minimum sample requirement. Selecting all.")
        return list(eligible)

    # Find the two most distant eligible glosses to start
    max_dist = -1
    start_i, start_j = 0, 1

    eligible_list = list(eligible)
    for i in range(len(eligible_list)):
        for j in range(i + 1, len(eligible_list)):
            idx_i, idx_j = eligible_list[i], eligible_list[j]
            if distance_matrix[idx_i, idx_j] > max_dist:
                max_dist = distance_matrix[idx_i, idx_j]
                start_i, start_j = idx_i, idx_j

    selected = [start_i, start_j]
    remaining = eligible - {start_i, start_j}

    # Greedy selection
    with tqdm(total=k-2, desc="Selecting distinct glosses") as pbar:
        while len(selected) < k and remaining:
            best_candidate = None
            best_min_dist = -1

            for candidate in remaining:
                # Minimum distance from candidate to any selected gloss
                min_dist = min(distance_matrix[candidate, s] for s in selected)

                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_candidate = candidate

            if best_candidate is not None:
                selected.append(best_candidate)
                remaining.remove(best_candidate)

            pbar.update(1)

    return selected


def list_glosses(input_dir: Path):
    """List all glosses with their sample counts."""
    gloss_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    print(f"\nFound {len(gloss_dirs)} glosses in {input_dir}\n")
    print(f"{'Gloss':<30} {'Samples':>8}")
    print("-" * 40)

    total_samples = 0
    for gloss_dir in gloss_dirs:
        pose_count = len(list(gloss_dir.glob("*.pose")))
        total_samples += pose_count
        print(f"{gloss_dir.name:<30} {pose_count:>8}")

    print("-" * 40)
    print(f"{'TOTAL':<30} {total_samples:>8}")
    print(f"\nAverage samples per gloss: {total_samples / len(gloss_dirs):.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Find the most distinct glosses from a pose dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python find_distinct_glosses.py -i ../../datasets/pose_files_by_gloss -o distinct_100.json -n 100
    python find_distinct_glosses.py -i ../../datasets/pose_files_by_gloss --list
    python find_distinct_glosses.py -i ../../datasets/pose_files_by_gloss -n 50 --min-samples 5
        """
    )

    parser.add_argument("--input", "-i", type=Path,
                        default=Path("../../datasets/wlasl_poses_complete/pose_files_by_gloss"),
                        help="Input directory containing gloss subdirectories with pose files")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output JSON file for results (default: distinct_<n>.json)")
    parser.add_argument("--count", "-n", type=int, default=100,
                        help="Number of distinct glosses to select (default: 100)")
    parser.add_argument("--min-samples", type=int, default=3,
                        help="Minimum samples per gloss to be considered (default: 3)")
    parser.add_argument("--list", "-l", action="store_true",
                        help="Just list all glosses with sample counts")
    parser.add_argument("--cache", type=Path, default=None,
                        help="Cache file for extracted features (speeds up reruns)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed progress")

    args = parser.parse_args()

    # Validate input directory
    if not args.input.exists():
        print(f"ERROR: Input directory not found: {args.input}")
        return 1

    # List mode
    if args.list:
        list_glosses(args.input)
        return 0

    # Set default output
    if args.output is None:
        args.output = Path(f"distinct_{args.count}.json")

    print(f"Finding {args.count} most distinct glosses from {args.input}")
    print(f"Minimum samples per gloss: {args.min_samples}")
    print()

    # Step 1: Extract features for all glosses (or load from cache)
    cache_file = args.cache or Path(f".gloss_features_cache.pkl")

    if cache_file.exists():
        print(f"Loading cached features from {cache_file}...")
        with open(cache_file, 'rb') as f:
            features_list = pickle.load(f)
        print(f"Loaded {len(features_list)} gloss features from cache")
    else:
        print("Extracting features from all glosses...")
        gloss_dirs = sorted([d for d in args.input.iterdir() if d.is_dir()])

        features_list = []
        for gloss_dir in tqdm(gloss_dirs, desc="Processing glosses"):
            features = extract_gloss_features(gloss_dir)
            if features is not None:
                features_list.append(features)

        print(f"Extracted features for {len(features_list)} glosses")

        # Save to cache
        print(f"Saving features to cache: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(features_list, f)

    # Filter by minimum samples
    eligible_count = sum(1 for f in features_list if f.sample_count >= args.min_samples)
    print(f"Glosses with >= {args.min_samples} samples: {eligible_count}")

    if eligible_count < args.count:
        print(f"WARNING: Only {eligible_count} glosses meet sample requirement, selecting all")

    # Step 2: Compute distance matrix
    print("\nComputing distance matrix...")
    distance_matrix = compute_distance_matrix(features_list)
    print(f"Distance matrix shape: {distance_matrix.shape}")

    # Step 3: Greedy selection
    print(f"\nSelecting {args.count} most distinct glosses...")
    selected_indices = greedy_max_dispersion(
        distance_matrix,
        args.count,
        min_samples=args.min_samples,
        features_list=features_list
    )

    # Get selected glosses
    selected_features = [features_list[i] for i in selected_indices]

    # Calculate quality metrics
    selected_distances = []
    for i in range(len(selected_indices)):
        for j in range(i + 1, len(selected_indices)):
            selected_distances.append(distance_matrix[selected_indices[i], selected_indices[j]])

    min_pairwise_dist = min(selected_distances)
    avg_pairwise_dist = np.mean(selected_distances)

    # Compare to random selection
    np.random.seed(42)
    random_indices = np.random.choice(
        [i for i, f in enumerate(features_list) if f.sample_count >= args.min_samples],
        size=min(args.count, eligible_count),
        replace=False
    )
    random_distances = []
    for i in range(len(random_indices)):
        for j in range(i + 1, len(random_indices)):
            random_distances.append(distance_matrix[random_indices[i], random_indices[j]])

    random_min_dist = min(random_distances) if random_distances else 0
    random_avg_dist = np.mean(random_distances) if random_distances else 0

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nSelected {len(selected_features)} distinct glosses:")
    print()

    # Sort by name for display
    selected_sorted = sorted(selected_features, key=lambda f: f.gloss)

    print(f"{'#':<4} {'Gloss':<25} {'Samples':>8}")
    print("-" * 40)
    for i, f in enumerate(selected_sorted, 1):
        print(f"{i:<4} {f.gloss:<25} {f.sample_count:>8}")

    print(f"\n{'='*60}")
    print("QUALITY METRICS")
    print(f"{'='*60}")
    print(f"\nSelected glosses:")
    print(f"  Minimum pairwise distance: {min_pairwise_dist:.4f}")
    print(f"  Average pairwise distance: {avg_pairwise_dist:.4f}")
    print(f"\nRandom selection (baseline):")
    print(f"  Minimum pairwise distance: {random_min_dist:.4f}")
    print(f"  Average pairwise distance: {random_avg_dist:.4f}")
    print(f"\nImprovement:")
    print(f"  Min distance: {min_pairwise_dist / random_min_dist:.2f}x better" if random_min_dist > 0 else "  N/A")
    print(f"  Avg distance: {avg_pairwise_dist / random_avg_dist:.2f}x better" if random_avg_dist > 0 else "  N/A")

    # Save results
    results = {
        "config": {
            "input_dir": str(args.input),
            "count": args.count,
            "min_samples": args.min_samples,
            "total_glosses": len(features_list),
            "eligible_glosses": eligible_count,
        },
        "metrics": {
            "min_pairwise_distance": float(min_pairwise_dist),
            "avg_pairwise_distance": float(avg_pairwise_dist),
            "random_min_distance": float(random_min_dist),
            "random_avg_distance": float(random_avg_dist),
        },
        "selected_glosses": [
            {
                "gloss": f.gloss,
                "sample_count": f.sample_count,
                "features": asdict(f),
            }
            for f in selected_sorted
        ],
        "gloss_list": [f.gloss for f in selected_sorted],
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    # Also print just the gloss names for easy copying
    print(f"\n{'='*60}")
    print("GLOSS LIST (copy-paste ready)")
    print(f"{'='*60}")
    print(",".join(f.gloss for f in selected_sorted))

    return 0


if __name__ == "__main__":
    exit(main())
