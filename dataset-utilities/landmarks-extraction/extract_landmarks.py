"""
Landmark Extraction Functions

Core functions for extracting landmark subsets from full MediaPipe pose data.
Includes optional finger feature extraction for position-invariant hand shape encoding.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Union, Dict, Tuple, Optional

# Handle imports for both module and standalone usage
try:
    from .landmark_config import (
        LANDMARK_CONFIGS,
        DEFAULT_CONFIG,
        DEFAULT_COORDINATE_MODE,
        COORDINATE_MODES,
        FINGER_FEATURE_CONFIG,
        HAND_POSITIONS,
    )
    from .finger_features import (
        extract_finger_features_from_sequence,
        extract_hand_features,
        get_finger_feature_names,
        FINGER_FEATURE_COUNT,
    )
except ImportError:
    from landmark_config import (
        LANDMARK_CONFIGS,
        DEFAULT_CONFIG,
        DEFAULT_COORDINATE_MODE,
        COORDINATE_MODES,
        FINGER_FEATURE_CONFIG,
        HAND_POSITIONS,
    )
    from finger_features import (
        extract_finger_features_from_sequence,
        extract_hand_features,
        get_finger_feature_names,
        FINGER_FEATURE_COUNT,
    )


def extract_landmarks(
    full_pose_data: np.ndarray,
    config: str = DEFAULT_CONFIG,
    include_z: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    """
    Extract landmark subset from full MediaPipe pose data.

    Args:
        full_pose_data: Full pose array with shape (frames, 576, 2) or (frames, 576, 3)
        config: Configuration name ('75pt', '83pt', '54pt', '62pt')
        include_z: Whether to include z-coordinate (if available)
        normalize: Whether to apply normalization (shoulder-centered, scaled)

    Returns:
        Extracted landmarks with shape (frames, N, 2) or (frames, N, 3)
        where N is the number of points in the selected configuration

    Raises:
        ValueError: If config is unknown or data shape is invalid
    """
    # Validate config
    if config not in LANDMARK_CONFIGS:
        available = list(LANDMARK_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config}'. Available: {available}")

    config_info = LANDMARK_CONFIGS[config]
    indices = config_info['indices']
    expected_points = config_info['total_points']

    # Validate input shape
    if len(full_pose_data.shape) != 3:
        raise ValueError(
            f"Expected 3D array (frames, keypoints, coords), "
            f"got shape {full_pose_data.shape}"
        )

    frames, total_points, coords = full_pose_data.shape

    # Check minimum required keypoints
    max_index = max(indices)
    if total_points <= max_index:
        raise ValueError(
            f"Data has {total_points} keypoints, but config '{config}' "
            f"requires index {max_index}. Expected at least {max_index + 1} keypoints."
        )

    # Handle coordinate dimensions
    if include_z and coords >= 3:
        coord_slice = slice(0, 3)  # x, y, z
        out_coords = 3
    else:
        coord_slice = slice(0, 2)  # x, y only
        out_coords = 2

    # Extract subset
    extracted = full_pose_data[:, indices, coord_slice].copy()

    # Verify output shape
    assert extracted.shape == (frames, expected_points, out_coords), \
        f"Extraction error: expected {(frames, expected_points, out_coords)}, got {extracted.shape}"

    # Apply normalization if requested
    if normalize:
        extracted = normalize_landmarks(extracted, config)

    return extracted.astype(np.float32)


def normalize_landmarks(
    landmarks: np.ndarray,
    config: str = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Normalize landmarks to be shoulder-centered and scaled.

    Normalization steps:
    1. Center on midpoint between shoulders (if available)
    2. Scale based on shoulder width (if available)
    3. Handle missing landmarks gracefully

    Args:
        landmarks: Landmark array with shape (frames, N, 2 or 3)
        config: Configuration name (to know landmark layout)

    Returns:
        Normalized landmarks with same shape
    """
    landmarks = landmarks.copy()
    frames, num_points, coords = landmarks.shape

    # Get shoulder indices within the extracted subset
    # In all configs, pose comes first, and shoulders are at indices 11 and 12
    config_info = LANDMARK_CONFIGS[config]
    indices = config_info['indices']

    # Find where shoulders are in the extracted array
    try:
        left_shoulder_idx = indices.index(11)
        right_shoulder_idx = indices.index(12)
    except ValueError:
        # Shoulders not in this config (e.g., hands-only), skip normalization
        return landmarks

    for frame_idx in range(frames):
        frame = landmarks[frame_idx]

        left_shoulder = frame[left_shoulder_idx]
        right_shoulder = frame[right_shoulder_idx]

        # Check for missing/zero landmarks
        if np.allclose(left_shoulder, 0) or np.allclose(right_shoulder, 0):
            continue

        # Calculate center and scale
        center = (left_shoulder + right_shoulder) / 2
        shoulder_width = np.linalg.norm(right_shoulder[:2] - left_shoulder[:2])

        if shoulder_width < 1e-6:
            continue  # Avoid division by zero

        # Normalize: center and scale
        landmarks[frame_idx] = (frame - center) / shoulder_width

    return landmarks


def extract_landmarks_from_pickle(
    pickle_path: Union[str, Path],
    config: str = DEFAULT_CONFIG,
    include_z: bool = False,
    normalize: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Extract landmarks from a pickle file.

    Args:
        pickle_path: Path to pickle file containing pose data
        config: Configuration name ('75pt', '83pt', etc.)
        include_z: Whether to include z-coordinate
        normalize: Whether to apply normalization

    Returns:
        Tuple of (extracted_landmarks, metadata_dict)

    Raises:
        FileNotFoundError: If pickle file doesn't exist
        KeyError: If pickle doesn't contain 'keypoints' key
    """
    pickle_path = Path(pickle_path)

    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    if 'keypoints' not in data:
        raise KeyError(
            f"Pickle file missing 'keypoints' key. "
            f"Available keys: {list(data.keys())}"
        )

    full_pose_data = data['keypoints']
    extracted = extract_landmarks(
        full_pose_data,
        config=config,
        include_z=include_z,
        normalize=normalize,
    )

    # Return metadata separately
    metadata = {
        'video_id': data.get('video_id'),
        'gloss': data.get('gloss'),
        'split': data.get('split'),
        'augmented': data.get('augmented', False),
        'augmentation_id': data.get('augmentation_id'),
        'original_frames': full_pose_data.shape[0],
        'original_keypoints': full_pose_data.shape[1],
        'extracted_config': config,
        'extracted_points': extracted.shape[1],
    }

    return extracted, metadata


def get_landmark_info(config: str = DEFAULT_CONFIG) -> Dict:
    """
    Get information about a landmark configuration.

    Args:
        config: Configuration name

    Returns:
        Dictionary with configuration details
    """
    if config not in LANDMARK_CONFIGS:
        available = list(LANDMARK_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config}'. Available: {available}")

    config_info = LANDMARK_CONFIGS[config].copy()

    # Add computed info
    config_info['feature_count_xy'] = config_info['total_points'] * 2
    config_info['feature_count_xyz'] = config_info['total_points'] * 3

    return config_info


def flatten_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Flatten landmarks from (frames, points, coords) to (frames, features).

    Args:
        landmarks: Array with shape (frames, N, 2 or 3)

    Returns:
        Flattened array with shape (frames, N * coords)
    """
    frames, points, coords = landmarks.shape
    return landmarks.reshape(frames, points * coords)


def unflatten_landmarks(
    flattened: np.ndarray,
    num_points: int,
    coords: int = 2,
) -> np.ndarray:
    """
    Unflatten landmarks from (frames, features) to (frames, points, coords).

    Args:
        flattened: Array with shape (frames, features)
        num_points: Number of landmark points
        coords: Number of coordinates per point (2 or 3)

    Returns:
        Unflattened array with shape (frames, num_points, coords)
    """
    frames = flattened.shape[0]
    return flattened.reshape(frames, num_points, coords)


def extract_landmarks_with_finger_features(
    full_pose_data: np.ndarray,
    config: str = DEFAULT_CONFIG,
    include_z: bool = False,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract landmarks AND finger features from full MediaPipe pose data.

    This function returns both the standard landmark coordinates and
    derived finger features (extension, spread, distances) that encode
    hand shape in a position-invariant way.

    Args:
        full_pose_data: Full pose array with shape (frames, 576, 2) or (frames, 576, 3)
        config: Configuration name ('75pt', '83pt', '54pt', '62pt')
        include_z: Whether to include z-coordinate (if available)
        normalize: Whether to apply normalization (shoulder-centered, scaled)

    Returns:
        Tuple of:
        - landmarks: (frames, N, 2 or 3) extracted landmark coordinates
        - finger_features: (frames, 30) derived finger features

    Example:
        >>> landmarks, finger_feats = extract_landmarks_with_finger_features(pose_data)
        >>> print(landmarks.shape)  # (100, 83, 2)
        >>> print(finger_feats.shape)  # (100, 30)
    """
    # Extract standard landmarks
    landmarks = extract_landmarks(
        full_pose_data,
        config=config,
        include_z=include_z,
        normalize=normalize,
    )

    # Get hand positions for this config
    if config not in HAND_POSITIONS:
        raise ValueError(f"Finger features not supported for config '{config}'")

    hand_pos = HAND_POSITIONS[config]

    # Extract finger features from the landmarks
    finger_features = extract_finger_features_from_sequence(
        landmarks,
        left_hand_start=hand_pos['left_start'],
        right_hand_start=hand_pos['right_start'],
    )

    return landmarks, finger_features


def extract_combined_features(
    full_pose_data: np.ndarray,
    config: str = DEFAULT_CONFIG,
    include_z: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    """
    Extract landmarks and finger features, returning as a single flattened array.

    Combines landmark coordinates with derived finger features into a single
    feature vector suitable for model input.

    Args:
        full_pose_data: Full pose array with shape (frames, 576, 2) or (frames, 576, 3)
        config: Configuration name ('75pt', '83pt', '54pt', '62pt')
        include_z: Whether to include z-coordinate (if available)
        normalize: Whether to apply normalization

    Returns:
        (frames, landmark_features + 30) array where:
        - First part: flattened landmark coordinates
        - Last 30: finger features

    Example:
        >>> features = extract_combined_features(pose_data, config='83pt')
        >>> # 83 points × 2 coords + 30 finger features = 196 features
        >>> print(features.shape)  # (100, 196)
    """
    landmarks, finger_features = extract_landmarks_with_finger_features(
        full_pose_data,
        config=config,
        include_z=include_z,
        normalize=normalize,
    )

    # Flatten landmarks
    flattened_landmarks = flatten_landmarks(landmarks)

    # Concatenate with finger features
    combined = np.concatenate([flattened_landmarks, finger_features], axis=1)

    return combined.astype(np.float32)


def get_combined_feature_info(config: str = DEFAULT_CONFIG, include_z: bool = False) -> Dict:
    """
    Get information about combined features (landmarks + finger features).

    Args:
        config: Configuration name
        include_z: Whether z-coordinate is included

    Returns:
        Dictionary with feature counts and descriptions
    """
    landmark_info = get_landmark_info(config)
    coords = 3 if include_z else 2

    landmark_features = landmark_info['total_points'] * coords
    finger_features = FINGER_FEATURE_CONFIG['total_features']
    total_features = landmark_features + finger_features

    return {
        'config': config,
        'landmark_points': landmark_info['total_points'],
        'landmark_coords': coords,
        'landmark_features': landmark_features,
        'finger_features': finger_features,
        'total_features': total_features,
        'finger_feature_breakdown': FINGER_FEATURE_CONFIG['breakdown'],
        'description': f'{landmark_features} landmark coords + {finger_features} finger features',
    }
