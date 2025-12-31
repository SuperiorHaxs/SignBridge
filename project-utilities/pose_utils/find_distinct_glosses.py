#!/usr/bin/env python3
"""
find_distinct_glosses.py - Find the most distinct glosses from a pose dataset

This script analyzes pose files for all glosses and selects a subset that are
maximally distinct from each other using a greedy maximum dispersion algorithm.

UPDATED v3: Comprehensive feature set matching what the model actually learns.

Features used for distinction (60 dimensions):

BASE FEATURES (30):
- Hand positions XY: 4 (left/right mean x, y)
- Hand positions Z: 4 (left/right mean z, std z) - depth information
- Movement range: 4 (left/right std x, y)
- Hand spread: 2 (mean, std distance between hands)
- Hand height: 2 (left/right relative to nose)
- Velocity: 2 (left/right mean speed)
- Acceleration: 2 (left/right velocity variance - smooth vs jerky)
- Trajectory: 4 (start->end delta x,y per hand - movement direction)
- Symmetry: 2 (hands together/opposite, mirror/parallel movement)
- Body zones: 3 (mean distance to face, chest, side)
- Duration: 1 (normalized sequence length)

FINGER FEATURES (30):
- Finger extension: 10 (5 per hand - how straight each finger is)
- Finger spread: 8 (4 per hand - angles between adjacent fingers)
- Fingertip distances: 10 (5 per hand - tip to wrist distance)
- Hand openness: 2 (1 per hand - overall open/closed)

Usage:
    python find_distinct_glosses.py --input <pose_folder> --output <output_file> --count 100
    python find_distinct_glosses.py --list  # Just list all glosses with sample counts

Examples:
    python find_distinct_glosses.py -i ../../datasets/wlasl_poses_complete/pose_files_by_gloss -o distinct_100.json -n 100
    python find_distinct_glosses.py -i ../../datasets/wlasl_poses_complete/pose_files_by_gloss --list
    python find_distinct_glosses.py -i ../../datasets/wlasl_poses_complete/pose_files_by_gloss -n 125 --finger-weight 2.0
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from tqdm import tqdm
import pickle

# Add landmarks-extraction to path for finger features
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
_landmarks_dir = _project_root / "dataset-utilities" / "landmarks-extraction"
if str(_landmarks_dir) not in sys.path:
    sys.path.insert(0, str(_landmarks_dir))

# Import finger feature extraction
try:
    from finger_features import (
        extract_finger_features_from_sequence,
        FINGER_FEATURE_COUNT,
    )
    FINGER_FEATURES_AVAILABLE = True
except ImportError:
    print("WARNING: finger_features module not found, using basic features only")
    FINGER_FEATURES_AVAILABLE = False
    FINGER_FEATURE_COUNT = 0


@dataclass
class GlossFeatures:
    """Features extracted from a gloss's pose files (60 dimensions total)."""
    gloss: str
    sample_count: int

    # === BASE FEATURES (30) ===

    # Hand positions XY (normalized, relative to body center) - 4 features
    left_hand_mean_x: float
    left_hand_mean_y: float
    right_hand_mean_x: float
    right_hand_mean_y: float

    # Hand positions Z (depth) - 4 features
    left_hand_mean_z: float = 0.0
    left_hand_std_z: float = 0.0
    right_hand_mean_z: float = 0.0
    right_hand_std_z: float = 0.0

    # Movement range XY (std dev) - 4 features
    left_hand_std_x: float = 0.0
    left_hand_std_y: float = 0.0
    right_hand_std_x: float = 0.0
    right_hand_std_y: float = 0.0

    # Hand spread - 2 features
    hand_spread_mean: float = 0.0
    hand_spread_std: float = 0.0

    # Hand height relative to nose - 2 features
    left_height_mean: float = 0.0
    right_height_mean: float = 0.0

    # Movement velocity - 2 features
    left_velocity_mean: float = 0.0
    right_velocity_mean: float = 0.0

    # Acceleration (velocity variance) - 2 features
    left_acceleration: float = 0.0
    right_acceleration: float = 0.0

    # Trajectory direction (start->end delta) - 4 features
    left_trajectory_x: float = 0.0
    left_trajectory_y: float = 0.0
    right_trajectory_x: float = 0.0
    right_trajectory_y: float = 0.0

    # Symmetry features - 2 features
    hands_sync: float = 0.0       # How synchronized hand movements are (0=opposite, 1=together)
    hands_mirror: float = 0.0     # How mirrored the movements are (0=parallel, 1=mirrored)

    # Body contact zones - 3 features
    distance_to_face: float = 0.0
    distance_to_chest: float = 0.0
    distance_to_side: float = 0.0

    # Duration (normalized) - 1 feature
    duration_mean: float = 0.0

    # === FINGER FEATURES (30) ===

    # Finger features - LEFT HAND
    left_thumb_extension: float = 0.0
    left_index_extension: float = 0.0
    left_middle_extension: float = 0.0
    left_ring_extension: float = 0.0
    left_pinky_extension: float = 0.0
    left_thumb_index_spread: float = 0.0
    left_index_middle_spread: float = 0.0
    left_middle_ring_spread: float = 0.0
    left_ring_pinky_spread: float = 0.0
    left_thumb_distance: float = 0.0
    left_index_distance: float = 0.0
    left_middle_distance: float = 0.0
    left_ring_distance: float = 0.0
    left_pinky_distance: float = 0.0
    left_openness: float = 0.0

    # Finger features - RIGHT HAND
    right_thumb_extension: float = 0.0
    right_index_extension: float = 0.0
    right_middle_extension: float = 0.0
    right_ring_extension: float = 0.0
    right_pinky_extension: float = 0.0
    right_thumb_index_spread: float = 0.0
    right_index_middle_spread: float = 0.0
    right_middle_ring_spread: float = 0.0
    right_ring_pinky_spread: float = 0.0
    right_thumb_distance: float = 0.0
    right_index_distance: float = 0.0
    right_middle_distance: float = 0.0
    right_ring_distance: float = 0.0
    right_pinky_distance: float = 0.0
    right_openness: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert features to a numpy vector for distance calculations (60 dims)."""
        # Base features (30)
        base_features = [
            # Hand positions XY (4)
            self.left_hand_mean_x,
            self.left_hand_mean_y,
            self.right_hand_mean_x,
            self.right_hand_mean_y,
            # Hand positions Z / depth (4)
            self.left_hand_mean_z,
            self.left_hand_std_z,
            self.right_hand_mean_z,
            self.right_hand_std_z,
            # Movement range XY (4)
            self.left_hand_std_x,
            self.left_hand_std_y,
            self.right_hand_std_x,
            self.right_hand_std_y,
            # Hand spread (2)
            self.hand_spread_mean,
            self.hand_spread_std,
            # Hand height (2)
            self.left_height_mean,
            self.right_height_mean,
            # Velocity (2)
            self.left_velocity_mean,
            self.right_velocity_mean,
            # Acceleration (2)
            self.left_acceleration,
            self.right_acceleration,
            # Trajectory (4)
            self.left_trajectory_x,
            self.left_trajectory_y,
            self.right_trajectory_x,
            self.right_trajectory_y,
            # Symmetry (2)
            self.hands_sync,
            self.hands_mirror,
            # Body zones (3)
            self.distance_to_face,
            self.distance_to_chest,
            self.distance_to_side,
            # Duration (1)
            self.duration_mean,
        ]

        # Finger features (30) - weighted higher since they're critical for hand shape
        finger_features = [
            # Left hand (15)
            self.left_thumb_extension,
            self.left_index_extension,
            self.left_middle_extension,
            self.left_ring_extension,
            self.left_pinky_extension,
            self.left_thumb_index_spread,
            self.left_index_middle_spread,
            self.left_middle_ring_spread,
            self.left_ring_pinky_spread,
            self.left_thumb_distance,
            self.left_index_distance,
            self.left_middle_distance,
            self.left_ring_distance,
            self.left_pinky_distance,
            self.left_openness,
            # Right hand (15)
            self.right_thumb_extension,
            self.right_index_extension,
            self.right_middle_extension,
            self.right_ring_extension,
            self.right_pinky_extension,
            self.right_thumb_index_spread,
            self.right_index_middle_spread,
            self.right_middle_ring_spread,
            self.right_ring_pinky_spread,
            self.right_thumb_distance,
            self.right_index_distance,
            self.right_middle_distance,
            self.right_ring_distance,
            self.right_pinky_distance,
            self.right_openness,
        ]

        return np.array(base_features + finger_features)


def load_pose_75pt(pose_path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load pose file and extract 75-point subset (pose + hands, no face).

    Returns:
        pose_2d: (frames, 75, 2) - xy coordinates for compatibility
        pose_3d: (frames, 75, 3) - xyz coordinates for depth features
        fps: frames per second
    """
    from pose_format import Pose

    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())

    data = pose.body.data
    fps = pose.body.fps

    if len(data.shape) == 4:
        data = data[:, 0, :, :]  # (frames, keypoints, dims)

    # Ensure we have at least 3 dimensions
    if data.shape[-1] < 3:
        # Pad with zeros for z if missing
        padding = np.zeros((*data.shape[:-1], 3 - data.shape[-1]))
        data = np.concatenate([data, padding], axis=-1)

    # Extract 75 points: 33 pose + 21 left hand + 21 right hand
    if data.shape[1] == 543 or data.shape[1] == 576:
        pose_75pt_3d = np.concatenate([
            data[:, 0:33, :3],      # Pose landmarks
            data[:, 501:522, :3],   # Left hand
            data[:, 522:543, :3]    # Right hand
        ], axis=1)
    elif data.shape[1] == 75:
        pose_75pt_3d = data[:, :, :3]
    else:
        pose_75pt_3d = data[:, :, :3]

    pose_75pt_2d = pose_75pt_3d[:, :, :2]

    return np.array(pose_75pt_2d), np.array(pose_75pt_3d), fps


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


def extract_pose_features(pose_data_2d: np.ndarray, pose_data_3d: np.ndarray) -> Dict[str, float]:
    """
    Extract comprehensive features from a single pose sequence.

    Args:
        pose_data_2d: (frames, 75, 2) - xy coordinates
        pose_data_3d: (frames, 75, 3) - xyz coordinates for depth

    Returns:
        Dictionary of extracted features
    """
    # Key landmark indices (in 75-point format)
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_HAND_CENTER = 33 + 9   # Middle finger MCP (wrist=33, MCP=33+9)
    RIGHT_HAND_CENTER = 54 + 9
    LEFT_WRIST = 33
    RIGHT_WRIST = 54

    frames = len(pose_data_2d)

    # Normalize pose (2D for position features)
    pose_normalized = normalize_pose(pose_data_2d)

    # Calculate body center (midpoint of shoulders)
    shoulders_mid = (pose_normalized[:, LEFT_SHOULDER, :] + pose_normalized[:, RIGHT_SHOULDER, :]) / 2

    # === HAND POSITIONS XY (4 features) ===
    left_hand_rel = pose_normalized[:, LEFT_HAND_CENTER, :] - shoulders_mid
    right_hand_rel = pose_normalized[:, RIGHT_HAND_CENTER, :] - shoulders_mid

    # === HAND POSITIONS Z / DEPTH (4 features) ===
    # Use raw z values (already relative due to camera perspective)
    left_hand_z = pose_data_3d[:, LEFT_HAND_CENTER, 2]
    right_hand_z = pose_data_3d[:, RIGHT_HAND_CENTER, 2]

    # === MOVEMENT RANGE XY (4 features) ===
    # Already captured by std of hand positions

    # === HAND SPREAD (2 features) ===
    hand_distance = np.linalg.norm(
        pose_normalized[:, LEFT_HAND_CENTER, :] - pose_normalized[:, RIGHT_HAND_CENTER, :],
        axis=1
    )

    # === HAND HEIGHT (2 features) ===
    nose_pos = pose_normalized[:, NOSE, :]
    left_hand_height = pose_normalized[:, LEFT_HAND_CENTER, 1] - nose_pos[:, 1]
    right_hand_height = pose_normalized[:, RIGHT_HAND_CENTER, 1] - nose_pos[:, 1]

    # === VELOCITY (2 features) ===
    if frames > 1:
        left_velocity = np.linalg.norm(np.diff(pose_normalized[:, LEFT_HAND_CENTER, :], axis=0), axis=1)
        right_velocity = np.linalg.norm(np.diff(pose_normalized[:, RIGHT_HAND_CENTER, :], axis=0), axis=1)
    else:
        left_velocity = np.array([0])
        right_velocity = np.array([0])

    # === ACCELERATION (2 features) - velocity variance ===
    left_acceleration = float(left_velocity.var()) if len(left_velocity) > 1 else 0.0
    right_acceleration = float(right_velocity.var()) if len(right_velocity) > 1 else 0.0

    # === TRAJECTORY (4 features) - start to end movement direction ===
    # Use first and last 10% of frames to get stable start/end positions
    start_frames = max(1, frames // 10)
    end_frames = max(1, frames // 10)

    left_start = pose_normalized[:start_frames, LEFT_HAND_CENTER, :].mean(axis=0)
    left_end = pose_normalized[-end_frames:, LEFT_HAND_CENTER, :].mean(axis=0)
    right_start = pose_normalized[:start_frames, RIGHT_HAND_CENTER, :].mean(axis=0)
    right_end = pose_normalized[-end_frames:, RIGHT_HAND_CENTER, :].mean(axis=0)

    left_trajectory = left_end - left_start
    right_trajectory = right_end - right_start

    # === SYMMETRY (2 features) ===
    # Sync: correlation of left and right hand movements (1 = moving together, -1 = opposite)
    if frames > 2:
        left_movement = np.diff(pose_normalized[:, LEFT_HAND_CENTER, :], axis=0)
        right_movement = np.diff(pose_normalized[:, RIGHT_HAND_CENTER, :], axis=0)

        # Flatten movements for correlation
        left_flat = left_movement.flatten()
        right_flat = right_movement.flatten()

        if np.std(left_flat) > 1e-6 and np.std(right_flat) > 1e-6:
            hands_sync = float(np.corrcoef(left_flat, right_flat)[0, 1])
            hands_sync = 0.0 if np.isnan(hands_sync) else hands_sync
        else:
            hands_sync = 0.0

        # Mirror: are hands moving in mirrored fashion (x opposite, y same)?
        left_x_movement = np.diff(pose_normalized[:, LEFT_HAND_CENTER, 0])
        right_x_movement = np.diff(pose_normalized[:, RIGHT_HAND_CENTER, 0])

        if np.std(left_x_movement) > 1e-6 and np.std(right_x_movement) > 1e-6:
            x_corr = float(np.corrcoef(left_x_movement, right_x_movement)[0, 1])
            x_corr = 0.0 if np.isnan(x_corr) else x_corr
            hands_mirror = -x_corr  # Negative correlation = mirrored
        else:
            hands_mirror = 0.0
    else:
        hands_sync = 0.0
        hands_mirror = 0.0

    # === BODY CONTACT ZONES (3 features) ===
    # Distance from dominant hand (right) to key body parts
    # Face zone (around nose)
    face_pos = pose_normalized[:, NOSE, :]
    right_hand_pos = pose_normalized[:, RIGHT_HAND_CENTER, :]
    distance_to_face = float(np.linalg.norm(right_hand_pos - face_pos, axis=1).mean())

    # Chest zone (between shoulders and hips)
    chest_pos = (pose_normalized[:, LEFT_SHOULDER, :] + pose_normalized[:, RIGHT_SHOULDER, :] +
                 pose_normalized[:, LEFT_HIP, :] + pose_normalized[:, RIGHT_HIP, :]) / 4
    distance_to_chest = float(np.linalg.norm(right_hand_pos - chest_pos, axis=1).mean())

    # Side zone (lateral distance from body center)
    body_center_x = shoulders_mid[:, 0]
    hand_lateral = np.abs(right_hand_pos[:, 0] - body_center_x)
    distance_to_side = float(hand_lateral.mean())

    result = {
        # Hand positions XY
        'left_hand_mean': left_hand_rel.mean(axis=0),
        'right_hand_mean': right_hand_rel.mean(axis=0),
        'left_hand_std': left_hand_rel.std(axis=0),
        'right_hand_std': right_hand_rel.std(axis=0),
        # Hand positions Z
        'left_hand_mean_z': float(left_hand_z.mean()),
        'left_hand_std_z': float(left_hand_z.std()),
        'right_hand_mean_z': float(right_hand_z.mean()),
        'right_hand_std_z': float(right_hand_z.std()),
        # Hand spread
        'hand_spread_mean': float(hand_distance.mean()),
        'hand_spread_std': float(hand_distance.std()),
        # Hand height
        'left_height_mean': float(left_hand_height.mean()),
        'right_height_mean': float(right_hand_height.mean()),
        # Velocity
        'left_velocity_mean': float(left_velocity.mean()),
        'right_velocity_mean': float(right_velocity.mean()),
        # Acceleration
        'left_acceleration': left_acceleration,
        'right_acceleration': right_acceleration,
        # Trajectory
        'left_trajectory_x': float(left_trajectory[0]),
        'left_trajectory_y': float(left_trajectory[1]),
        'right_trajectory_x': float(right_trajectory[0]),
        'right_trajectory_y': float(right_trajectory[1]),
        # Symmetry
        'hands_sync': hands_sync,
        'hands_mirror': hands_mirror,
        # Body zones
        'distance_to_face': distance_to_face,
        'distance_to_chest': distance_to_chest,
        'distance_to_side': distance_to_side,
        # Duration
        'duration': frames,
    }

    # Extract finger features if available
    if FINGER_FEATURES_AVAILABLE:
        # Use the original (non-normalized) pose for finger features
        # since finger features are already position-invariant
        finger_feats = extract_finger_features_from_sequence(
            pose_data_2d,  # Use original 2D data
            left_hand_start=33,
            right_hand_start=54,
        )

        # Average across all frames
        finger_means = finger_feats.mean(axis=0)

        # Store finger features
        result['finger_features'] = finger_means
    else:
        result['finger_features'] = np.zeros(30)

    return result


def extract_gloss_features(gloss_dir: Path) -> Optional[GlossFeatures]:
    """Extract aggregated features for a gloss from all its pose files (60 dimensions)."""
    pose_files = list(gloss_dir.glob("*.pose"))

    if not pose_files:
        return None

    all_features = []

    for pose_file in pose_files:
        try:
            pose_data_2d, pose_data_3d, fps = load_pose_75pt(pose_file)
            if len(pose_data_2d) < 5:  # Skip very short sequences
                continue
            features = extract_pose_features(pose_data_2d, pose_data_3d)
            all_features.append(features)
        except Exception as e:
            # Skip problematic files
            continue

    if not all_features:
        return None

    # Aggregate features across all samples (mean)
    n = len(all_features)

    # Hand positions XY
    left_hand_means = np.array([f['left_hand_mean'] for f in all_features])
    right_hand_means = np.array([f['right_hand_mean'] for f in all_features])
    left_hand_stds = np.array([f['left_hand_std'] for f in all_features])
    right_hand_stds = np.array([f['right_hand_std'] for f in all_features])

    # Normalize duration to 0-1 range (assuming max ~200 frames)
    durations = np.array([f['duration'] for f in all_features])
    duration_normalized = np.clip(durations / 200.0, 0, 1)

    # Aggregate finger features
    finger_features_all = np.array([f['finger_features'] for f in all_features])
    finger_means = finger_features_all.mean(axis=0)

    return GlossFeatures(
        gloss=gloss_dir.name,
        sample_count=n,

        # === BASE FEATURES (30) ===

        # Hand positions XY (4)
        left_hand_mean_x=float(left_hand_means[:, 0].mean()),
        left_hand_mean_y=float(left_hand_means[:, 1].mean()),
        right_hand_mean_x=float(right_hand_means[:, 0].mean()),
        right_hand_mean_y=float(right_hand_means[:, 1].mean()),

        # Hand positions Z (4)
        left_hand_mean_z=float(np.mean([f['left_hand_mean_z'] for f in all_features])),
        left_hand_std_z=float(np.mean([f['left_hand_std_z'] for f in all_features])),
        right_hand_mean_z=float(np.mean([f['right_hand_mean_z'] for f in all_features])),
        right_hand_std_z=float(np.mean([f['right_hand_std_z'] for f in all_features])),

        # Movement range XY (4)
        left_hand_std_x=float(left_hand_stds[:, 0].mean()),
        left_hand_std_y=float(left_hand_stds[:, 1].mean()),
        right_hand_std_x=float(right_hand_stds[:, 0].mean()),
        right_hand_std_y=float(right_hand_stds[:, 1].mean()),

        # Hand spread (2)
        hand_spread_mean=float(np.mean([f['hand_spread_mean'] for f in all_features])),
        hand_spread_std=float(np.mean([f['hand_spread_std'] for f in all_features])),

        # Hand height (2)
        left_height_mean=float(np.mean([f['left_height_mean'] for f in all_features])),
        right_height_mean=float(np.mean([f['right_height_mean'] for f in all_features])),

        # Velocity (2)
        left_velocity_mean=float(np.mean([f['left_velocity_mean'] for f in all_features])),
        right_velocity_mean=float(np.mean([f['right_velocity_mean'] for f in all_features])),

        # Acceleration (2)
        left_acceleration=float(np.mean([f['left_acceleration'] for f in all_features])),
        right_acceleration=float(np.mean([f['right_acceleration'] for f in all_features])),

        # Trajectory (4)
        left_trajectory_x=float(np.mean([f['left_trajectory_x'] for f in all_features])),
        left_trajectory_y=float(np.mean([f['left_trajectory_y'] for f in all_features])),
        right_trajectory_x=float(np.mean([f['right_trajectory_x'] for f in all_features])),
        right_trajectory_y=float(np.mean([f['right_trajectory_y'] for f in all_features])),

        # Symmetry (2)
        hands_sync=float(np.mean([f['hands_sync'] for f in all_features])),
        hands_mirror=float(np.mean([f['hands_mirror'] for f in all_features])),

        # Body zones (3)
        distance_to_face=float(np.mean([f['distance_to_face'] for f in all_features])),
        distance_to_chest=float(np.mean([f['distance_to_chest'] for f in all_features])),
        distance_to_side=float(np.mean([f['distance_to_side'] for f in all_features])),

        # Duration (1)
        duration_mean=float(duration_normalized.mean()),

        # === FINGER FEATURES (30) ===

        # Left hand finger features (15)
        left_thumb_extension=float(finger_means[0]),
        left_index_extension=float(finger_means[1]),
        left_middle_extension=float(finger_means[2]),
        left_ring_extension=float(finger_means[3]),
        left_pinky_extension=float(finger_means[4]),
        left_thumb_index_spread=float(finger_means[5]),
        left_index_middle_spread=float(finger_means[6]),
        left_middle_ring_spread=float(finger_means[7]),
        left_ring_pinky_spread=float(finger_means[8]),
        left_thumb_distance=float(finger_means[9]),
        left_index_distance=float(finger_means[10]),
        left_middle_distance=float(finger_means[11]),
        left_ring_distance=float(finger_means[12]),
        left_pinky_distance=float(finger_means[13]),
        left_openness=float(finger_means[14]),

        # Right hand finger features (15)
        right_thumb_extension=float(finger_means[15]),
        right_index_extension=float(finger_means[16]),
        right_middle_extension=float(finger_means[17]),
        right_ring_extension=float(finger_means[18]),
        right_pinky_extension=float(finger_means[19]),
        right_thumb_index_spread=float(finger_means[20]),
        right_index_middle_spread=float(finger_means[21]),
        right_middle_ring_spread=float(finger_means[22]),
        right_ring_pinky_spread=float(finger_means[23]),
        right_thumb_distance=float(finger_means[24]),
        right_index_distance=float(finger_means[25]),
        right_middle_distance=float(finger_means[26]),
        right_ring_distance=float(finger_means[27]),
        right_pinky_distance=float(finger_means[28]),
        right_openness=float(finger_means[29]),
    )


def compute_distance_matrix(features_list: List[GlossFeatures],
                           finger_weight: float = 2.0) -> np.ndarray:
    """
    Compute pairwise distance matrix between all glosses.

    Args:
        features_list: List of GlossFeatures
        finger_weight: Weight multiplier for finger features (default 2.0)
                       Higher = finger features matter more for distinctness

    Feature vector layout (60 dimensions):
        [0-29]:  Base features (30) - position, depth, movement, symmetry, etc.
        [30-59]: Finger features (30) - extension, spread, distances, openness
    """
    n = len(features_list)

    # Convert to matrix
    vectors = np.array([f.to_vector() for f in features_list])

    # Normalize each feature to have zero mean and unit variance
    means = vectors.mean(axis=0)
    stds = vectors.std(axis=0)
    stds[stds < 1e-6] = 1.0  # Avoid division by zero
    vectors_normalized = (vectors - means) / stds

    # Apply finger weight (features 30-59 are finger features)
    if finger_weight != 1.0:
        vectors_normalized[:, 30:60] *= finger_weight

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
    python find_distinct_glosses.py -i ../../datasets/pose_files_by_gloss -n 125 --finger-weight 3.0
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
    parser.add_argument("--finger-weight", type=float, default=2.0,
                        help="Weight for finger features in distance calc (default: 2.0, higher = prioritize hand shape)")
    parser.add_argument("--list", "-l", action="store_true",
                        help="Just list all glosses with sample counts")
    parser.add_argument("--cache", type=Path, default=None,
                        help="Cache file for extracted features (speeds up reruns)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore existing cache and re-extract features")
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
    print(f"Finger feature weight: {args.finger_weight}x")
    print(f"Finger features available: {FINGER_FEATURES_AVAILABLE}")
    print()

    # Step 1: Extract features for all glosses (or load from cache)
    # v3 = 60 dimensions (30 base + 30 finger)
    cache_file = args.cache or Path(f".gloss_features_cache_v3.pkl")

    if cache_file.exists() and not args.no_cache:
        print(f"Loading cached features from {cache_file}...")
        with open(cache_file, 'rb') as f:
            features_list = pickle.load(f)
        print(f"Loaded {len(features_list)} gloss features from cache")

        # Check if cache has v3 features (60 dimensions)
        if hasattr(features_list[0], 'left_hand_mean_z') and hasattr(features_list[0], 'hands_sync'):
            print("Cache includes v3 features (60 dimensions)")
        else:
            print("WARNING: Cache is outdated (missing v3 features), re-extracting...")
            args.no_cache = True

    if not cache_file.exists() or args.no_cache:
        print("Extracting features from all glosses (60 dimensions: 30 base + 30 finger)...")
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
    print(f"\nComputing distance matrix (finger weight: {args.finger_weight}x)...")
    distance_matrix = compute_distance_matrix(features_list, finger_weight=args.finger_weight)
    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Feature vector size: {features_list[0].to_vector().shape[0]} dimensions")

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

    print(f"{'#':<4} {'Gloss':<25} {'Samples':>8} {'L-Open':>7} {'R-Open':>7}")
    print("-" * 55)
    for i, f in enumerate(selected_sorted, 1):
        print(f"{i:<4} {f.gloss:<25} {f.sample_count:>8} {f.left_openness:>7.2f} {f.right_openness:>7.2f}")

    print(f"\n{'='*60}")
    print("QUALITY METRICS")
    print(f"{'='*60}")
    print(f"\nFeature dimensions: {features_list[0].to_vector().shape[0]}")
    print(f"  - Base features: 30")
    print(f"      * Position XY: 4 (hand positions)")
    print(f"      * Depth Z: 4 (forward/back movement)")
    print(f"      * Movement range: 4 (std dev XY)")
    print(f"      * Hand spread: 2 (distance between hands)")
    print(f"      * Hand height: 2 (relative to face)")
    print(f"      * Velocity: 2 (movement speed)")
    print(f"      * Acceleration: 2 (velocity variance)")
    print(f"      * Trajectory: 4 (start->end direction)")
    print(f"      * Symmetry: 2 (sync + mirror)")
    print(f"      * Body zones: 3 (face/chest/side distance)")
    print(f"      * Duration: 1")
    print(f"  - Finger features: 30 (extension, spread, distances, openness)")
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
            "finger_weight": args.finger_weight,
            "finger_features_available": FINGER_FEATURES_AVAILABLE,
            "feature_dimensions": features_list[0].to_vector().shape[0],
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
                "left_openness": f.left_openness,
                "right_openness": f.right_openness,
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
