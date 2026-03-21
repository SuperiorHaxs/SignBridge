"""
Diverse Subset Selection via Farthest-Point Sampling

Given a candidate pool and a pairwise distance matrix, greedily selects
the most diverse subset of target_count items. This maximizes the minimum
pairwise distance between selected items, reducing confusion pairs.

Algorithm:
1. Start with the most "central" candidate (smallest max-distance to others)
2. Iteratively pick the candidate with the largest minimum-distance to
   any already-selected candidate
3. Repeat until target_count is reached
"""

import numpy as np
from typing import List, Tuple


def farthest_point_sampling(
    distance_matrix: np.ndarray,
    glosses: List[str],
    target_count: int,
    seed_index: int = None,
) -> Tuple[List[str], dict]:
    """
    Select a maximally diverse subset using farthest-point sampling.

    Args:
        distance_matrix: (n, n) pairwise distance matrix
        glosses: List of gloss names corresponding to matrix rows/cols
        target_count: Number of items to select
        seed_index: Optional starting index. If None, uses the most central point.

    Returns:
        Tuple of (selected_glosses, metadata_dict)
    """
    n = len(glosses)
    if target_count >= n:
        # Return all if we don't have enough candidates
        return glosses[:], _compute_metrics(distance_matrix, list(range(n)), glosses)

    if target_count <= 0:
        return [], {}

    # Pick seed: most central point (smallest average distance to all others)
    if seed_index is None:
        avg_distances = np.mean(distance_matrix, axis=1)
        seed_index = int(np.argmin(avg_distances))

    selected_indices = [seed_index]

    # Track min distance from each candidate to the selected set
    min_dist_to_selected = distance_matrix[seed_index].copy()

    for _ in range(target_count - 1):
        # Mask already-selected indices
        min_dist_to_selected[selected_indices] = -1.0

        # Pick the candidate farthest from the current selected set
        next_idx = int(np.argmax(min_dist_to_selected))
        selected_indices.append(next_idx)

        # Update min distances
        new_distances = distance_matrix[next_idx]
        min_dist_to_selected = np.minimum(min_dist_to_selected, new_distances)

    selected_glosses = [glosses[i] for i in selected_indices]
    metrics = _compute_metrics(distance_matrix, selected_indices, glosses)

    return selected_glosses, metrics


def _compute_metrics(
    distance_matrix: np.ndarray,
    selected_indices: List[int],
    glosses: List[str],
) -> dict:
    """Compute diversity metrics for the selected subset."""
    if len(selected_indices) < 2:
        return {"avg_pairwise_distance": 0, "min_pairwise_distance": 0, "max_pairwise_distance": 0}

    idx = np.array(selected_indices)
    sub_matrix = distance_matrix[np.ix_(idx, idx)]

    # Extract upper triangle (no diagonal)
    triu_indices = np.triu_indices_from(sub_matrix, k=1)
    pairwise_dists = sub_matrix[triu_indices]

    # Find the closest pair
    min_idx = int(np.argmin(pairwise_dists))
    row_indices, col_indices = triu_indices
    closest_i = int(row_indices[min_idx])
    closest_j = int(col_indices[min_idx])

    return {
        "avg_pairwise_distance": float(np.mean(pairwise_dists)),
        "min_pairwise_distance": float(np.min(pairwise_dists)),
        "max_pairwise_distance": float(np.max(pairwise_dists)),
        "closest_pair": [glosses[selected_indices[closest_i]], glosses[selected_indices[closest_j]]],
        "closest_pair_distance": float(pairwise_dists[min_idx]),
        "candidate_pool_size": len(glosses),
        "selected_count": len(selected_indices),
    }
