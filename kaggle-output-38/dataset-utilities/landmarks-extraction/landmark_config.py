"""
Landmark Configuration

Defines MediaPipe landmark indices and extraction configurations.

MediaPipe Holistic Layout (576 total points in our pose files):
    Indices 0-32:    Pose (33 points)
    Indices 33-500:  Face (468 points)
    Indices 501-521: Left Hand (21 points)
    Indices 522-542: Right Hand (21 points)
    Indices 543-575: Pose World 3D (33 points) - typically excluded

Reference:
    - MediaPipe Pose: https://google.github.io/mediapipe/solutions/pose.html
    - MediaPipe Hands: https://google.github.io/mediapipe/solutions/hands.html
    - MediaPipe Face: https://google.github.io/mediapipe/solutions/face_mesh.html
"""

# =============================================================================
# POSE LANDMARKS (33 points)
# =============================================================================

POSE_INDICES = list(range(0, 33))

POSE_LANDMARK_NAMES = {
    0: 'nose',
    1: 'left_eye_inner',
    2: 'left_eye',
    3: 'left_eye_outer',
    4: 'right_eye_inner',
    5: 'right_eye',
    6: 'right_eye_outer',
    7: 'left_ear',
    8: 'right_ear',
    9: 'mouth_left',
    10: 'mouth_right',
    11: 'left_shoulder',
    12: 'right_shoulder',
    13: 'left_elbow',
    14: 'right_elbow',
    15: 'left_wrist',
    16: 'right_wrist',
    17: 'left_pinky',
    18: 'right_pinky',
    19: 'left_index',
    20: 'right_index',
    21: 'left_thumb',
    22: 'right_thumb',
    23: 'left_hip',
    24: 'right_hip',
    25: 'left_knee',
    26: 'right_knee',
    27: 'left_ankle',
    28: 'right_ankle',
    29: 'left_heel',
    30: 'right_heel',
    31: 'left_foot_index',
    32: 'right_foot_index',
}

# Upper body subset (excludes face landmarks 0-10 and lower body 23-32)
UPPER_BODY_POSE_INDICES = list(range(11, 23))  # 12 points: shoulders to hand connections


# =============================================================================
# HAND LANDMARKS (21 points each, 42 total)
# =============================================================================

LEFT_HAND_INDICES = list(range(501, 522))   # 21 points
RIGHT_HAND_INDICES = list(range(522, 543))  # 21 points

HAND_LANDMARK_NAMES = {
    0: 'wrist',
    1: 'thumb_cmc',
    2: 'thumb_mcp',
    3: 'thumb_ip',
    4: 'thumb_tip',
    5: 'index_mcp',
    6: 'index_pip',
    7: 'index_dip',
    8: 'index_tip',
    9: 'middle_mcp',
    10: 'middle_pip',
    11: 'middle_dip',
    12: 'middle_tip',
    13: 'ring_mcp',
    14: 'ring_pip',
    15: 'ring_dip',
    16: 'ring_tip',
    17: 'pinky_mcp',
    18: 'pinky_pip',
    19: 'pinky_dip',
    20: 'pinky_tip',
}

# Finger indices grouped by finger (within 21-point hand)
FINGER_INDICES = {
    'thumb': [1, 2, 3, 4],      # CMC, MCP, IP, TIP
    'index': [5, 6, 7, 8],      # MCP, PIP, DIP, TIP
    'middle': [9, 10, 11, 12],  # MCP, PIP, DIP, TIP
    'ring': [13, 14, 15, 16],   # MCP, PIP, DIP, TIP
    'pinky': [17, 18, 19, 20],  # MCP, PIP, DIP, TIP
}

# Fingertip indices (within 21-point hand)
FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

# Finger base (MCP) indices (within 21-point hand)
FINGER_MCP_INDICES = [1, 5, 9, 13, 17]  # thumb CMC, index, middle, ring, pinky


# =============================================================================
# FACE LANDMARKS (Minimal subset - 8 points for ASL disambiguation)
# =============================================================================

# Selected 8 face landmarks for sign language recognition
# These help disambiguate signs that involve face touching or expressions
FACE_MINIMAL_INDICES_RAW = [1, 61, 291, 152, 107, 336, 33, 263]

# Offset by 33 (pose landmarks come first in our data format)
FACE_MINIMAL_INDICES = [idx + 33 for idx in FACE_MINIMAL_INDICES_RAW]

FACE_LANDMARK_NAMES = {
    1: 'nose_tip',           # Center reference point
    61: 'mouth_left',        # Left mouth corner
    291: 'mouth_right',      # Right mouth corner
    152: 'chin',             # Lower face boundary
    107: 'left_eyebrow',     # Left eyebrow center (for expressions)
    336: 'right_eyebrow',    # Right eyebrow center (for expressions)
    33: 'left_eye_outer',    # Left eye outer corner
    263: 'right_eye_outer',  # Right eye outer corner
}

# Full face (468 points) - typically too noisy for our use case
FACE_FULL_INDICES = list(range(33, 501))


# =============================================================================
# EXTRACTION CONFIGURATIONS
# =============================================================================

LANDMARK_CONFIGS = {
    # Current configuration: Pose + Hands (no face)
    '75pt': {
        'name': '75-point (Pose + Hands)',
        'description': 'Full body pose (33) + both hands (42). No face landmarks.',
        'indices': POSE_INDICES + LEFT_HAND_INDICES + RIGHT_HAND_INDICES,
        'total_points': 75,
        'include_face': False,
        'breakdown': {
            'pose': 33,
            'left_hand': 21,
            'right_hand': 21,
            'face': 0,
        }
    },

    # New configuration: Pose + Hands + Minimal Face
    '83pt': {
        'name': '83-point (Pose + Hands + Minimal Face)',
        'description': 'Full body pose (33) + both hands (42) + 8 key face landmarks.',
        'indices': POSE_INDICES + LEFT_HAND_INDICES + RIGHT_HAND_INDICES + FACE_MINIMAL_INDICES,
        'total_points': 83,
        'include_face': True,
        'breakdown': {
            'pose': 33,
            'left_hand': 21,
            'right_hand': 21,
            'face': 8,
        }
    },

    # Upper body only (for cropped videos)
    '54pt': {
        'name': '54-point (Upper Body + Hands)',
        'description': 'Upper body pose (12) + both hands (42). Excludes face and lower body.',
        'indices': UPPER_BODY_POSE_INDICES + LEFT_HAND_INDICES + RIGHT_HAND_INDICES,
        'total_points': 54,
        'include_face': False,
        'breakdown': {
            'pose': 12,
            'left_hand': 21,
            'right_hand': 21,
            'face': 0,
        }
    },

    # Upper body + minimal face
    '62pt': {
        'name': '62-point (Upper Body + Hands + Minimal Face)',
        'description': 'Upper body pose (12) + both hands (42) + 8 key face landmarks.',
        'indices': UPPER_BODY_POSE_INDICES + LEFT_HAND_INDICES + RIGHT_HAND_INDICES + FACE_MINIMAL_INDICES,
        'total_points': 62,
        'include_face': True,
        'breakdown': {
            'pose': 12,
            'left_hand': 21,
            'right_hand': 21,
            'face': 8,
        }
    },
}

# Default configuration
DEFAULT_CONFIG = '83pt'


# =============================================================================
# COORDINATE SETTINGS
# =============================================================================

COORDINATE_MODES = {
    'xy': {
        'dims': 2,
        'description': 'X and Y coordinates only',
    },
    'xyz': {
        'dims': 3,
        'description': 'X, Y, and Z (depth) coordinates',
    },
}

DEFAULT_COORDINATE_MODE = 'xy'


# =============================================================================
# FINGER FEATURE SETTINGS
# =============================================================================

FINGER_FEATURE_CONFIG = {
    'total_features': 30,  # 15 per hand × 2 hands
    'features_per_hand': 15,
    'breakdown': {
        'extension_ratios': 5,      # How extended each finger is (0-1)
        'spread_angles': 4,         # Angles between adjacent fingers
        'fingertip_distances': 5,   # Normalized tip-to-wrist distances
        'openness_score': 1,        # Overall hand openness (0-1)
    },
    'description': 'Position-invariant hand shape features for ASL recognition',
}

# Hand positions within extracted landmarks (for 75pt and 83pt configs)
HAND_POSITIONS = {
    '75pt': {'left_start': 33, 'right_start': 54},
    '83pt': {'left_start': 33, 'right_start': 54},
    '54pt': {'left_start': 12, 'right_start': 33},
    '62pt': {'left_start': 12, 'right_start': 33},
}
