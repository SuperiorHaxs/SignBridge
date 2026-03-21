"""
Landmarks Extraction Module

Provides standardized landmark extraction from MediaPipe pose data.
Supports multiple configurations (75-point, 83-point, etc.) for
different accuracy/feature trade-offs.

Now includes finger feature extraction for position-invariant hand shape encoding.

Usage:
    from landmarks_extraction import extract_landmarks, LANDMARK_CONFIGS

    # Extract 83-point (pose + hands + minimal face)
    landmarks = extract_landmarks(full_pose_data, config='83pt')

    # Extract 75-point (pose + hands only)
    landmarks = extract_landmarks(full_pose_data, config='75pt')

    # Extract landmarks WITH finger features (recommended for training)
    landmarks, finger_feats = extract_landmarks_with_finger_features(full_pose_data)
    # landmarks: (frames, 83, 2), finger_feats: (frames, 30)

    # Or get combined flattened features
    features = extract_combined_features(full_pose_data, config='83pt')
    # features: (frames, 196) = 83*2 + 30 finger features
"""

from .landmark_config import (
    LANDMARK_CONFIGS,
    POSE_INDICES,
    LEFT_HAND_INDICES,
    RIGHT_HAND_INDICES,
    FACE_MINIMAL_INDICES,
    FACE_LANDMARK_NAMES,
    FINGER_INDICES,
    FINGERTIP_INDICES,
    FINGER_MCP_INDICES,
    FINGER_FEATURE_CONFIG,
    HAND_POSITIONS,
)

from .extract_landmarks import (
    extract_landmarks,
    extract_landmarks_from_pickle,
    get_landmark_info,
    flatten_landmarks,
    unflatten_landmarks,
    extract_landmarks_with_finger_features,
    extract_combined_features,
    get_combined_feature_info,
)

from .finger_features import (
    extract_hand_features,
    extract_finger_features_from_sequence,
    extract_finger_features_from_raw,
    compute_finger_extension,
    compute_finger_spread,
    compute_fingertip_distances,
    compute_hand_openness,
    get_finger_feature_names,
    FINGER_FEATURE_COUNT,
    FINGER_FEATURES_PER_HAND,
)

__all__ = [
    # Landmark configs
    'LANDMARK_CONFIGS',
    'POSE_INDICES',
    'LEFT_HAND_INDICES',
    'RIGHT_HAND_INDICES',
    'FACE_MINIMAL_INDICES',
    'FACE_LANDMARK_NAMES',
    'FINGER_INDICES',
    'FINGERTIP_INDICES',
    'FINGER_MCP_INDICES',
    'FINGER_FEATURE_CONFIG',
    'HAND_POSITIONS',
    # Landmark extraction
    'extract_landmarks',
    'extract_landmarks_from_pickle',
    'get_landmark_info',
    'flatten_landmarks',
    'unflatten_landmarks',
    # Combined extraction (landmarks + finger features)
    'extract_landmarks_with_finger_features',
    'extract_combined_features',
    'get_combined_feature_info',
    # Finger feature extraction
    'extract_hand_features',
    'extract_finger_features_from_sequence',
    'extract_finger_features_from_raw',
    'compute_finger_extension',
    'compute_finger_spread',
    'compute_fingertip_distances',
    'compute_hand_openness',
    'get_finger_feature_names',
    'FINGER_FEATURE_COUNT',
    'FINGER_FEATURES_PER_HAND',
]
