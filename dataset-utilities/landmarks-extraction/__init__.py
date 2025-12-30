"""
Landmarks Extraction Module

Provides standardized landmark extraction from MediaPipe pose data.
Supports multiple configurations (75-point, 83-point, etc.) for
different accuracy/feature trade-offs.

Usage:
    from landmarks_extraction import extract_landmarks, LANDMARK_CONFIGS

    # Extract 83-point (pose + hands + minimal face)
    landmarks = extract_landmarks(full_pose_data, config='83pt')

    # Extract 75-point (pose + hands only)
    landmarks = extract_landmarks(full_pose_data, config='75pt')
"""

from .landmark_config import (
    LANDMARK_CONFIGS,
    POSE_INDICES,
    LEFT_HAND_INDICES,
    RIGHT_HAND_INDICES,
    FACE_MINIMAL_INDICES,
    FACE_LANDMARK_NAMES,
)

from .extract_landmarks import (
    extract_landmarks,
    extract_landmarks_from_pickle,
    get_landmark_info,
)

__all__ = [
    'LANDMARK_CONFIGS',
    'POSE_INDICES',
    'LEFT_HAND_INDICES',
    'RIGHT_HAND_INDICES',
    'FACE_MINIMAL_INDICES',
    'FACE_LANDMARK_NAMES',
    'extract_landmarks',
    'extract_landmarks_from_pickle',
    'get_landmark_info',
]
