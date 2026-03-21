"""
Pose Distance Computation

Computes pairwise distances between glosses based on their pose keypoint
representations. Used to select maximally diverse subsets that minimize
confusion pairs.

Strategy:
1. Load all pickle files for each candidate gloss
2. Compute a per-gloss centroid: average keypoints across frames, then across samples
3. Flatten to a single vector per gloss
4. Compute pairwise cosine distance matrix
"""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def load_gloss_centroids(
    glosses: List[str],
    pickle_dir: Path,
    video_to_gloss_path: Path,
    max_samples_per_gloss: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Load pickle files and compute a centroid pose vector for each gloss.

    The centroid is computed by:
    1. For each sample: average keypoints across all frames -> (num_keypoints, coords)
    2. Average across all samples for the gloss -> (num_keypoints, coords)
    3. Flatten -> single vector

    Args:
        glosses: List of gloss names (UPPERCASE)
        pickle_dir: Path to flat pickle directory
        video_to_gloss_path: Path to video_to_gloss_mapping.json
        max_samples_per_gloss: Cap samples per gloss to keep computation fast

    Returns:
        Dict mapping gloss name to flattened centroid vector
    """
    # Build gloss -> pickle file mapping
    with open(video_to_gloss_path, "r") as f:
        video_to_gloss = json.load(f)

    # Group video IDs by gloss
    # Mapping values can be either a string or a dict with a "gloss" key
    gloss_to_videos: Dict[str, List[str]] = defaultdict(list)
    for video_id, entry in video_to_gloss.items():
        if isinstance(entry, dict):
            gloss_name = entry.get("gloss", "")
        else:
            gloss_name = str(entry)
        gloss_upper = gloss_name.upper()
        if gloss_upper in set(glosses):
            gloss_to_videos[gloss_upper].append(video_id)

    centroids = {}
    missing = []

    for gloss in glosses:
        video_ids = gloss_to_videos.get(gloss, [])
        if not video_ids:
            missing.append(gloss)
            continue

        # Cap samples
        video_ids = video_ids[:max_samples_per_gloss]

        sample_means = []
        for vid in video_ids:
            pkl_path = pickle_dir / f"{vid}.pkl"
            if not pkl_path.exists():
                continue
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                kp = data["keypoints"]  # (frames, keypoints, coords)
                # Average across frames
                frame_mean = np.mean(kp, axis=0)  # (keypoints, coords)
                sample_means.append(frame_mean)
            except Exception:
                continue

        if not sample_means:
            missing.append(gloss)
            continue

        # Average across samples, then flatten
        gloss_centroid = np.mean(sample_means, axis=0)  # (keypoints, coords)
        centroids[gloss] = gloss_centroid.flatten().astype(np.float32)

    if missing:
        print(f"  Warning: {len(missing)} glosses had no loadable pickle data: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    return centroids


def compute_distance_matrix(
    glosses: List[str],
    centroids: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise cosine distance matrix for a set of glosses.

    Args:
        glosses: Ordered list of gloss names
        centroids: Dict mapping gloss -> flattened centroid vector

    Returns:
        Tuple of (distance_matrix, valid_glosses) where valid_glosses
        only includes glosses that have centroids
    """
    # Filter to glosses with centroids
    valid = [g for g in glosses if g in centroids]

    n = len(valid)
    if n == 0:
        return np.array([]), []

    # Stack into matrix
    vectors = np.stack([centroids[g] for g in valid])  # (n, dim)

    # Normalize for cosine distance
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized = vectors / norms

    # Cosine similarity -> distance
    similarity = normalized @ normalized.T  # (n, n)
    distance = 1.0 - similarity

    # Ensure diagonal is 0 and matrix is non-negative
    np.fill_diagonal(distance, 0.0)
    distance = np.maximum(distance, 0.0)

    return distance, valid
