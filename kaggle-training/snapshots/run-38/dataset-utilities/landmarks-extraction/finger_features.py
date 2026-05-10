"""
Finger Feature Extraction

Computes derived features from hand landmarks that capture hand shape
information in a position-invariant way.

Features computed per hand (15 features × 2 hands = 30 total):
- Finger extension ratios (5): How extended each finger is (0=bent, 1=straight)
- Finger spread angles (4): Angles between adjacent fingers
- Finger-to-wrist distances (5): Normalized distance from each fingertip to wrist
- Hand openness (1): Overall hand openness score

These features are position/scale invariant and explicitly encode hand shape,
which helps the model distinguish signs that differ only in hand configuration.
"""

import numpy as np
from typing import Dict, Tuple, Optional


# =============================================================================
# HAND LANDMARK INDICES (within 21-point hand)
# =============================================================================

# Hand landmark layout (MediaPipe):
#   0: wrist
#   1-4: thumb (CMC, MCP, IP, TIP)
#   5-8: index (MCP, PIP, DIP, TIP)
#   9-12: middle (MCP, PIP, DIP, TIP)
#   13-16: ring (MCP, PIP, DIP, TIP)
#   17-20: pinky (MCP, PIP, DIP, TIP)

HAND_WRIST = 0

# Finger base joints (MCP - where finger connects to palm)
FINGER_MCP = {
    'thumb': 1,   # CMC for thumb
    'index': 5,
    'middle': 9,
    'ring': 13,
    'pinky': 17,
}

# Finger tips
FINGER_TIP = {
    'thumb': 4,
    'index': 8,
    'middle': 12,
    'ring': 16,
    'pinky': 20,
}

# Finger middle joints (for angle calculation)
FINGER_PIP = {
    'thumb': 2,   # MCP for thumb
    'index': 6,
    'middle': 10,
    'ring': 14,
    'pinky': 18,
}

FINGER_DIP = {
    'thumb': 3,   # IP for thumb
    'index': 7,
    'middle': 11,
    'ring': 15,
    'pinky': 19,
}

FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']


# =============================================================================
# CORE FEATURE EXTRACTION
# =============================================================================

def compute_finger_extension(hand_landmarks: np.ndarray) -> np.ndarray:
    """
    Compute finger extension ratio for each finger.

    Extension ratio = actual_length / max_possible_length
    Where:
    - actual_length = distance from MCP to TIP
    - max_possible_length = sum of segment lengths (MCP->PIP + PIP->DIP + DIP->TIP)

    Returns value between 0 (fully bent) and 1 (fully extended).

    Args:
        hand_landmarks: (21, 2) or (21, 3) array of hand keypoints

    Returns:
        (5,) array of extension ratios for [thumb, index, middle, ring, pinky]
    """
    extensions = np.zeros(5, dtype=np.float32)

    for i, finger in enumerate(FINGER_NAMES):
        mcp = hand_landmarks[FINGER_MCP[finger]]
        pip = hand_landmarks[FINGER_PIP[finger]]
        dip = hand_landmarks[FINGER_DIP[finger]]
        tip = hand_landmarks[FINGER_TIP[finger]]

        # Max length = sum of segments
        seg1 = np.linalg.norm(pip - mcp)
        seg2 = np.linalg.norm(dip - pip)
        seg3 = np.linalg.norm(tip - dip)
        max_length = seg1 + seg2 + seg3

        # Actual length = direct distance from base to tip
        actual_length = np.linalg.norm(tip - mcp)

        # Extension ratio (clamp to [0, 1])
        if max_length > 1e-6:
            extensions[i] = np.clip(actual_length / max_length, 0.0, 1.0)
        else:
            extensions[i] = 0.0

    return extensions


def compute_finger_spread(hand_landmarks: np.ndarray) -> np.ndarray:
    """
    Compute spread angles between adjacent fingers.

    Measures the angle between finger directions (MCP -> TIP vectors).
    Larger angles = fingers more spread apart.

    Args:
        hand_landmarks: (21, 2) or (21, 3) array of hand keypoints

    Returns:
        (4,) array of spread angles in radians for:
        [thumb-index, index-middle, middle-ring, ring-pinky]
    """
    spreads = np.zeros(4, dtype=np.float32)

    # Get finger direction vectors (MCP -> TIP)
    finger_vectors = {}
    for finger in FINGER_NAMES:
        mcp = hand_landmarks[FINGER_MCP[finger]]
        tip = hand_landmarks[FINGER_TIP[finger]]
        vec = tip - mcp
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            finger_vectors[finger] = vec / norm
        else:
            finger_vectors[finger] = np.zeros_like(vec)

    # Compute angles between adjacent fingers
    pairs = [
        ('thumb', 'index'),
        ('index', 'middle'),
        ('middle', 'ring'),
        ('ring', 'pinky'),
    ]

    for i, (f1, f2) in enumerate(pairs):
        v1 = finger_vectors[f1]
        v2 = finger_vectors[f2]

        # Dot product gives cos(angle)
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(dot)

        # Normalize to [0, 1] range (max spread ~pi/2 radians)
        spreads[i] = np.clip(angle / (np.pi / 2), 0.0, 1.0)

    return spreads


def compute_fingertip_distances(hand_landmarks: np.ndarray) -> np.ndarray:
    """
    Compute normalized distances from each fingertip to wrist.

    Distances are normalized by the palm size (wrist to middle MCP distance).

    Args:
        hand_landmarks: (21, 2) or (21, 3) array of hand keypoints

    Returns:
        (5,) array of normalized distances for [thumb, index, middle, ring, pinky]
    """
    distances = np.zeros(5, dtype=np.float32)

    wrist = hand_landmarks[HAND_WRIST]
    middle_mcp = hand_landmarks[FINGER_MCP['middle']]

    # Palm size for normalization
    palm_size = np.linalg.norm(middle_mcp - wrist)
    if palm_size < 1e-6:
        return distances

    for i, finger in enumerate(FINGER_NAMES):
        tip = hand_landmarks[FINGER_TIP[finger]]
        dist = np.linalg.norm(tip - wrist)
        # Normalize by palm size (typical range 1-3x palm size)
        distances[i] = dist / palm_size

    return distances


def compute_hand_openness(hand_landmarks: np.ndarray) -> float:
    """
    Compute overall hand openness score.

    Combines finger extension and spread into a single score.
    0 = closed fist, 1 = fully open hand.

    Args:
        hand_landmarks: (21, 2) or (21, 3) array of hand keypoints

    Returns:
        Scalar openness score in [0, 1]
    """
    # Get extension and spread
    extensions = compute_finger_extension(hand_landmarks)
    spreads = compute_finger_spread(hand_landmarks)

    # Combine: 70% extension, 30% spread
    avg_extension = np.mean(extensions)
    avg_spread = np.mean(spreads)

    openness = 0.7 * avg_extension + 0.3 * avg_spread
    return float(np.clip(openness, 0.0, 1.0))


def extract_hand_features(hand_landmarks: np.ndarray) -> np.ndarray:
    """
    Extract all features for a single hand.

    Args:
        hand_landmarks: (21, 2) or (21, 3) array of hand keypoints

    Returns:
        (15,) array of features:
        - [0:5] finger extension ratios
        - [5:9] finger spread angles
        - [9:14] fingertip-to-wrist distances
        - [14] hand openness score
    """
    features = np.zeros(15, dtype=np.float32)

    # Check for missing hand (all zeros or NaN)
    if np.allclose(hand_landmarks, 0) or np.any(np.isnan(hand_landmarks)):
        return features

    # Extract features
    features[0:5] = compute_finger_extension(hand_landmarks)
    features[5:9] = compute_finger_spread(hand_landmarks)
    features[9:14] = compute_fingertip_distances(hand_landmarks)
    features[14] = compute_hand_openness(hand_landmarks)

    return features


# =============================================================================
# SEQUENCE PROCESSING
# =============================================================================

def extract_finger_features_from_sequence(
    landmarks: np.ndarray,
    left_hand_start: int = 33,
    right_hand_start: int = 54,
) -> np.ndarray:
    """
    Extract finger features from a pose sequence.

    Args:
        landmarks: (frames, keypoints, coords) pose sequence
                   Should be 75pt or 83pt format with hands at standard positions
        left_hand_start: Index where left hand landmarks start (default: 33 for 75pt)
        right_hand_start: Index where right hand landmarks start (default: 54 for 75pt)

    Returns:
        (frames, 30) array of finger features:
        - [0:15] left hand features
        - [15:30] right hand features
    """
    frames = landmarks.shape[0]
    features = np.zeros((frames, 30), dtype=np.float32)

    for frame_idx in range(frames):
        # Extract left hand (21 points)
        left_hand = landmarks[frame_idx, left_hand_start:left_hand_start + 21, :]
        features[frame_idx, 0:15] = extract_hand_features(left_hand)

        # Extract right hand (21 points)
        right_hand = landmarks[frame_idx, right_hand_start:right_hand_start + 21, :]
        features[frame_idx, 15:30] = extract_hand_features(right_hand)

    return features


def extract_finger_features_from_raw(
    full_pose: np.ndarray,
) -> np.ndarray:
    """
    Extract finger features from raw 576-point MediaPipe data.

    Args:
        full_pose: (frames, 576, coords) raw MediaPipe pose

    Returns:
        (frames, 30) array of finger features
    """
    # MediaPipe indices
    LEFT_HAND_START = 501
    RIGHT_HAND_START = 522

    frames = full_pose.shape[0]
    features = np.zeros((frames, 30), dtype=np.float32)

    for frame_idx in range(frames):
        # Extract hands from raw MediaPipe indices
        left_hand = full_pose[frame_idx, LEFT_HAND_START:LEFT_HAND_START + 21, :2]
        right_hand = full_pose[frame_idx, RIGHT_HAND_START:RIGHT_HAND_START + 21, :2]

        features[frame_idx, 0:15] = extract_hand_features(left_hand)
        features[frame_idx, 15:30] = extract_hand_features(right_hand)

    return features


# =============================================================================
# FEATURE NAMES AND METADATA
# =============================================================================

def get_finger_feature_names() -> list:
    """
    Get human-readable names for all 30 finger features.

    Returns:
        List of 30 feature names
    """
    names = []

    for hand in ['left', 'right']:
        # Extension ratios
        for finger in FINGER_NAMES:
            names.append(f'{hand}_{finger}_extension')

        # Spread angles
        pairs = ['thumb_index', 'index_middle', 'middle_ring', 'ring_pinky']
        for pair in pairs:
            names.append(f'{hand}_{pair}_spread')

        # Fingertip distances
        for finger in FINGER_NAMES:
            names.append(f'{hand}_{finger}_distance')

        # Openness
        names.append(f'{hand}_openness')

    return names


FINGER_FEATURE_COUNT = 30
FINGER_FEATURES_PER_HAND = 15

FINGER_FEATURE_INFO = {
    'total_features': FINGER_FEATURE_COUNT,
    'features_per_hand': FINGER_FEATURES_PER_HAND,
    'breakdown': {
        'extension_ratios': 5,  # per hand
        'spread_angles': 4,     # per hand
        'fingertip_distances': 5,  # per hand
        'openness_score': 1,    # per hand
    },
    'description': 'Position-invariant hand shape features for ASL recognition',
}


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    print("Testing finger feature extraction...")
    print("=" * 60)

    # Create synthetic hand data (21 points, 2D)
    # Simulate an open hand with fingers spread
    np.random.seed(42)

    # Open hand configuration
    open_hand = np.array([
        [0.5, 0.8],    # 0: wrist
        [0.3, 0.7],    # 1: thumb CMC
        [0.2, 0.6],    # 2: thumb MCP
        [0.15, 0.5],   # 3: thumb IP
        [0.1, 0.4],    # 4: thumb TIP
        [0.35, 0.5],   # 5: index MCP
        [0.35, 0.35],  # 6: index PIP
        [0.35, 0.2],   # 7: index DIP
        [0.35, 0.1],   # 8: index TIP
        [0.45, 0.48],  # 9: middle MCP
        [0.45, 0.32],  # 10: middle PIP
        [0.45, 0.18],  # 11: middle DIP
        [0.45, 0.05],  # 12: middle TIP
        [0.55, 0.5],   # 13: ring MCP
        [0.55, 0.35],  # 14: ring PIP
        [0.55, 0.22],  # 15: ring DIP
        [0.55, 0.1],   # 16: ring TIP
        [0.65, 0.55],  # 17: pinky MCP
        [0.65, 0.45],  # 18: pinky PIP
        [0.65, 0.35],  # 19: pinky DIP
        [0.65, 0.25],  # 20: pinky TIP
    ], dtype=np.float32)

    # Closed fist configuration (fingers curled)
    closed_fist = np.array([
        [0.5, 0.8],    # 0: wrist
        [0.35, 0.7],   # 1: thumb CMC
        [0.3, 0.65],   # 2: thumb MCP
        [0.35, 0.6],   # 3: thumb IP (curled back)
        [0.4, 0.55],   # 4: thumb TIP (near palm)
        [0.35, 0.55],  # 5: index MCP
        [0.38, 0.5],   # 6: index PIP (bent)
        [0.42, 0.55],  # 7: index DIP (curled)
        [0.4, 0.6],    # 8: index TIP (near palm)
        [0.45, 0.53],  # 9: middle MCP
        [0.48, 0.48],  # 10: middle PIP
        [0.5, 0.53],   # 11: middle DIP
        [0.48, 0.58],  # 12: middle TIP
        [0.55, 0.55],  # 13: ring MCP
        [0.57, 0.5],   # 14: ring PIP
        [0.58, 0.55],  # 15: ring DIP
        [0.56, 0.6],   # 16: ring TIP
        [0.63, 0.58],  # 17: pinky MCP
        [0.65, 0.55],  # 18: pinky PIP
        [0.65, 0.58],  # 19: pinky DIP
        [0.63, 0.62],  # 20: pinky TIP
    ], dtype=np.float32)

    # Test feature extraction
    print("\nOpen hand features:")
    open_features = extract_hand_features(open_hand)
    print(f"  Extension ratios: {open_features[0:5]}")
    print(f"  Spread angles:    {open_features[5:9]}")
    print(f"  Tip distances:    {open_features[9:14]}")
    print(f"  Openness score:   {open_features[14]:.3f}")

    print("\nClosed fist features:")
    closed_features = extract_hand_features(closed_fist)
    print(f"  Extension ratios: {closed_features[0:5]}")
    print(f"  Spread angles:    {closed_features[5:9]}")
    print(f"  Tip distances:    {closed_features[9:14]}")
    print(f"  Openness score:   {closed_features[14]:.3f}")

    # Verify open hand has higher values
    print("\nValidation:")
    ext_diff = np.mean(open_features[0:5]) - np.mean(closed_features[0:5])
    open_diff = open_features[14] - closed_features[14]
    print(f"  Open hand has higher extension: {ext_diff:.3f} (expected > 0)")
    print(f"  Open hand has higher openness:  {open_diff:.3f} (expected > 0)")

    if ext_diff > 0 and open_diff > 0:
        print("\n[SUCCESS] Finger feature extraction working correctly!")
    else:
        print("\n[WARNING] Feature values may need calibration")

    # Test sequence extraction
    print("\nTesting sequence extraction...")
    # Create a mock 75pt sequence (10 frames)
    mock_sequence = np.random.randn(10, 75, 2).astype(np.float32)
    seq_features = extract_finger_features_from_sequence(mock_sequence)
    print(f"  Input shape:  {mock_sequence.shape}")
    print(f"  Output shape: {seq_features.shape}")
    assert seq_features.shape == (10, 30), "Shape mismatch!"
    print("  [OK] Sequence extraction working!")

    # Print feature names
    print("\nFeature names:")
    names = get_finger_feature_names()
    for i, name in enumerate(names):
        print(f"  [{i:2d}] {name}")
