#!/usr/bin/env python3
"""
Prepare Demo Sample

This script creates demo samples either from:
1. A signing video file (--video mode)
2. Glosses from the test dataset (--glosses mode)

Usage (video mode):
    python prepare_demo_sample.py \
        --video path/to/signing_video.mp4 \
        --reference "The woman helps the deaf child" \
        --name "Woman Helps Child" \
        --id woman-helps-child

Usage (glosses mode):
    python prepare_demo_sample.py \
        --glosses "book,read,enjoy" \
        --reference "I enjoy reading books" \
        --name "Reading Books"

    python prepare_demo_sample.py --glosses "drink,water" --reference "Drink water"
    python prepare_demo_sample.py --list  # Show available glosses

Or for interactive mode:
    python prepare_demo_sample.py --interactive
"""
from __future__ import annotations

import os
import sys
import json
import copy
import shutil
import random
import argparse
import subprocess
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project paths
SCRIPT_DIR = Path(__file__).parent
APP_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = APP_DIR.parent.parent
APPLICATIONS_DIR = PROJECT_ROOT / "applications"
PROJECT_UTILITIES_DIR = PROJECT_ROOT / "project-utilities"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"

sys.path.insert(0, str(APPLICATIONS_DIR))
sys.path.insert(0, str(PROJECT_UTILITIES_DIR))
sys.path.insert(0, str(PROJECT_UTILITIES_DIR / "llm_interface"))
sys.path.insert(0, str(PROJECT_UTILITIES_DIR / "synthetic_evaluation"))
# OpenHands paths - src first (for openhands_modernized module), then util (for inference)
OPENHANDS_SRC = MODELS_DIR / "openhands-modernized" / "src"
sys.path.insert(0, str(OPENHANDS_SRC))
sys.path.insert(0, str(OPENHANDS_SRC / "util"))

# Paths to executables
VENV_SCRIPTS = Path(sys.executable).parent
VIDEO_TO_POSE_EXE = VENV_SCRIPTS / "video_to_pose.exe"
VISUALIZE_POSE_EXE = VENV_SCRIPTS / "visualize_pose.exe"

# LLM prompt path
LLM_PROMPT_PATH = PROJECT_UTILITIES_DIR / "llm_interface" / "prompts" / "llm_prompt_topk.txt"

# Import pose_format
try:
    from pose_format import Pose
    POSE_FORMAT_AVAILABLE = True
except ImportError:
    POSE_FORMAT_AVAILABLE = False
    print("WARNING: pose_format library not available")


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"  Running: {description}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False
    print(f"  Done: {description}")
    return True


def get_video_rotation(video_path: str) -> int:
    """
    Get video rotation metadata (for iPhone videos).

    Returns rotation in degrees (0, 90, 180, 270).
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
        cap.release()
        return int(rotation) if rotation else 0
    except Exception:
        return 0


def detect_video_mirroring(video_path: str, rotation: int = None) -> bool:
    """
    Detect if a video is horizontally mirrored (common with iPhone front camera).

    Uses MediaPipe pose detection on sample frames to check if left/right
    shoulders are swapped. In a mirrored video, the detected "left shoulder"
    will have a higher X coordinate than the "right shoulder".

    Args:
        video_path: Path to video file
        rotation: Video rotation in degrees (if known). If None, will be auto-detected.
                  This is important because mirror detection must happen AFTER rotation.

    Returns:
        True if video appears to be mirrored, False otherwise
    """
    try:
        import cv2
        import mediapipe as mp

        # Get rotation if not provided
        if rotation is None:
            rotation = get_video_rotation(video_path)

        # Determine rotation operation (to apply before mirror check)
        rotate_code = None
        if rotation == 90:
            rotate_code = cv2.ROTATE_90_CLOCKWISE
        elif rotation == 180:
            rotate_code = cv2.ROTATE_180
        elif rotation == 270:
            rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames from different parts of the video
        sample_frames = [
            total_frames // 4,
            total_frames // 2,
            3 * total_frames // 4
        ]

        mirrored_votes = 0
        valid_samples = 0

        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Apply rotation FIRST (before checking for mirroring)
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)

            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # MediaPipe indices: 11 = left shoulder, 12 = right shoulder
                left_shoulder_x = landmarks[11].x
                right_shoulder_x = landmarks[12].x

                valid_samples += 1

                # In a selfie/front camera video, the image is mirrored like looking in a mirror
                # Your left shoulder appears on the LEFT side of the image (lower X)
                # In a normal video, your left shoulder appears on the RIGHT (higher X)
                # So mirrored = left_shoulder_x < right_shoulder_x
                if left_shoulder_x < right_shoulder_x:
                    mirrored_votes += 1

        cap.release()
        pose.close()

        # If majority of samples suggest mirroring, return True
        if valid_samples > 0 and mirrored_votes > valid_samples / 2:
            return True

        return False

    except Exception as e:
        print(f"  WARNING: Mirror detection failed: {e}")
        return False


def normalize_video_orientation(input_path: str, output_path: str, check_mirror: bool = True) -> dict:
    """
    Create a properly oriented video from an iPhone video.

    Handles both:
    1. Rotation metadata (landscape videos stored as portrait)
    2. Horizontal mirroring (front camera selfie videos)

    Args:
        input_path: Path to input video
        output_path: Path to save the normalized video
        check_mirror: Whether to check for and fix mirroring

    Returns:
        dict with keys:
            'processed': True if any transformation was applied
            'rotated': True if rotation was applied
            'mirrored': True if mirror flip was applied
    """
    result = {'processed': False, 'rotated': False, 'mirrored': False}

    try:
        import cv2
        import imageio.v3 as iio

        rotation = get_video_rotation(input_path)
        # Pass rotation to mirror detection so it can apply rotation first before checking
        is_mirrored = detect_video_mirroring(input_path, rotation) if check_mirror else False

        if rotation == 0 and not is_mirrored:
            return result  # No transformation needed

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"  ERROR: Failed to open video: {input_path}")
            return result

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Determine rotation operation
        rotate_code = None
        new_width, new_height = width, height

        if rotation == 90:
            rotate_code = cv2.ROTATE_90_CLOCKWISE
            new_width, new_height = height, width
            result['rotated'] = True
        elif rotation == 180:
            rotate_code = cv2.ROTATE_180
            result['rotated'] = True
        elif rotation == 270:
            rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
            new_width, new_height = height, width
            result['rotated'] = True

        if is_mirrored:
            result['mirrored'] = True

        # Log what we're doing
        transforms = []
        if result['rotated']:
            transforms.append(f"rotate {rotation}°")
            print(f"  Detected rotation metadata: {rotation}°")
        if result['mirrored']:
            transforms.append("horizontal flip (mirror correction)")
            print(f"  Detected mirrored video (front camera)")

        print(f"  Applying transforms: {', '.join(transforms)}")
        if result['rotated']:
            print(f"  Dimensions: {width}x{height} -> {new_width}x{new_height}")

        # Read and transform all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply rotation first
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)

            # Apply horizontal flip for mirroring
            if is_mirrored:
                frame = cv2.flip(frame, 1)  # 1 = horizontal flip

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()

        if not frames:
            print("  ERROR: No frames extracted")
            return result

        # Write transformed video
        frames_array = np.stack(frames)
        iio.imwrite(output_path, frames_array, fps=fps, codec='libx264', plugin='pyav')
        print(f"  Created normalized video: {output_path}")

        result['processed'] = True
        return result

    except Exception as e:
        print(f"  ERROR: Video normalization failed: {e}")
        return result


# Keep old function name for backwards compatibility
def normalize_video_rotation(input_path: str, output_path: str) -> bool:
    """Deprecated: Use normalize_video_orientation instead."""
    result = normalize_video_orientation(input_path, output_path, check_mirror=False)
    return result['rotated']


def trim_to_active_window(keypoints, target_frames=50, hand_indices=(54, 75)):
    """
    Trim keypoints to the most active window of target_frames.

    This helps match training data characteristics where segments are
    typically 40-60 frames (1.3-2.0 seconds at 30 FPS).

    Args:
        keypoints: (frames, 75, 2) array
        target_frames: Target number of frames (default 50, ~1.7s at 30 FPS)
        hand_indices: Tuple of (start, end) for the dominant hand

    Returns:
        trimmed_keypoints: Trimmed array
        trim_info: Dict with start_frame and frames_trimmed
    """
    import numpy as np

    if keypoints.shape[0] <= target_frames:
        return keypoints, {'start_frame': 0, 'frames_trimmed': 0}

    # Calculate per-frame movement for the hand
    hand_data = keypoints[:, hand_indices[0]:hand_indices[1], :]

    frame_movements = []
    for i in range(len(hand_data) - 1):
        curr_center = np.mean(hand_data[i], axis=0)
        next_center = np.mean(hand_data[i+1], axis=0)
        movement = np.linalg.norm(next_center - curr_center)
        frame_movements.append(movement)

    frame_movements = np.array(frame_movements)

    # Find the window with highest total movement
    best_start = 0
    best_movement = 0

    for start in range(len(frame_movements) - target_frames + 1):
        window_movement = np.sum(frame_movements[start:start+target_frames])
        if window_movement > best_movement:
            best_movement = window_movement
            best_start = start

    trimmed = keypoints[best_start:best_start+target_frames]
    trim_info = {
        'start_frame': best_start,
        'frames_trimmed': keypoints.shape[0] - target_frames
    }

    return trimmed, trim_info


def trim_to_signing_region(keypoints, min_frames=8):
    """
    Trim segment to only the frames where hands are raised (actively signing).

    Removes the 'hands coming up from rest' and 'hands going down to rest'
    transitions that occur when signers return hands to their sides between signs.

    In MediaPipe coordinates, higher Y = lower position (hands at side).
    We want to keep frames where Y is LOW (hands raised in signing position).

    Algorithm: Find the REST position (hands at sides = highest Y) and select
    frames AFTER the last rest period where hands come up for signing.

    Args:
        keypoints: (frames, 75, 2) array - 33 body + 21 left hand + 21 right hand
        min_frames: Minimum frames to keep (default 8)

    Returns:
        trimmed_keypoints: Trimmed array with only signing frames
        trim_info: Dict with original_frames, trimmed_start, trimmed_end
    """
    import numpy as np

    original_frames = keypoints.shape[0]

    def no_trim_result():
        return keypoints, {
            'original_frames': original_frames,
            'trimmed_start': 0,
            'trimmed_end': original_frames,
            'frames_removed': 0
        }

    # Use both hands to determine signing region
    LEFT_HAND = (33, 54)
    RIGHT_HAND = (54, 75)

    left_hand = keypoints[:, LEFT_HAND[0]:LEFT_HAND[1], :]
    right_hand = keypoints[:, RIGHT_HAND[0]:RIGHT_HAND[1], :]

    # Compute mean Y for each hand, excluding zero keypoints
    hand_y = []
    for i in range(original_frames):
        lh = left_hand[i]
        rh = right_hand[i]
        lh_valid = lh[np.any(lh != 0, axis=1)]
        rh_valid = rh[np.any(rh != 0, axis=1)]
        ly = np.mean(lh_valid[:, 1]) if len(lh_valid) > 0 else np.inf
        ry = np.mean(rh_valid[:, 1]) if len(rh_valid) > 0 else np.inf
        hand_y.append(min(ly, ry))

    hand_y = np.array(hand_y)

    # Handle case where all frames are invalid
    valid_mask = ~np.isinf(hand_y)
    if not np.any(valid_mask):
        return no_trim_result()

    valid_y = hand_y[valid_mask]
    y_min = np.min(valid_y)
    y_max = np.max(valid_y)
    y_range = y_max - y_min

    if y_range < 5.0:  # Less than 5 pixels of movement
        return no_trim_result()

    # Strategy: For signers who return hands to rest between signs, the actual
    # sign often occurs during the "descent" phase - after hands reach peak
    # position and start going back down. We find the frame with minimum Y
    # (hands highest) and use the SECOND HALF of the segment after that point.

    # Create mask for valid frames (hands detected)
    valid_frames = ~np.isinf(hand_y)

    # If no valid frames, return as-is
    if not np.any(valid_frames):
        return no_trim_result()

    # Replace inf with max Y for finding minimum
    hand_y_filled = np.where(np.isinf(hand_y), y_max, hand_y)

    # Strategy: Signs often involve a descent motion (returning hands to rest)
    # as part of the sign itself. We should capture the SECOND HALF of the
    # raised region, which includes this descent phase.
    #
    # Algorithm:
    # 1. Find the raised region (where Y is below threshold)
    # 2. Select frames from ~50% through the segment to ~90%
    # This captures the main signing action and any descent motion.

    # Calculate threshold: Y needs to be below this to count as "raised"
    # Using 40% down from max toward min
    raise_threshold = y_max - (y_range * 0.40)

    # Find first and last frame where Y drops below threshold
    first_raised_frame = None
    last_raised_frame = None
    for i in range(original_frames):
        if hand_y_filled[i] < raise_threshold:
            if first_raised_frame is None:
                first_raised_frame = i
            last_raised_frame = i

    if first_raised_frame is None:
        # No significant raising detected, use second half
        start = int(original_frames * 0.5)
        end = int(original_frames * 0.9)
    else:
        # For many signs, the actual signing motion happens in the SECOND
        # half of the raised region (including descent to rest).
        # Select from ~50% into the segment through ~90%
        start = int(original_frames * 0.5)
        end = int(original_frames * 0.9)

        # But if the raised region is very early, include some of it
        if last_raised_frame < original_frames * 0.5:
            # Raised region is mostly in first half - use it
            start = max(0, first_raised_frame - 2)
            end = min(original_frames, last_raised_frame + 8)

    # Ensure minimum length
    if end - start < min_frames:
        center = (start + end) // 2
        start = max(0, center - min_frames // 2)
        end = min(original_frames, start + min_frames)
        if end - start < min_frames:
            start = max(0, end - min_frames)

    # Final validation
    if start >= end or end - start < min_frames:
        return no_trim_result()

    # Quality check: verify the trimmed region has enough valid frames
    # If we're selecting mostly inf frames, the trimming is wrong - skip it
    valid_in_trim = np.sum(valid_frames[start:end])
    valid_ratio = valid_in_trim / (end - start)
    if valid_ratio < 0.3:  # Less than 30% valid frames
        # Trimmed region has too many invalid frames, use original
        return no_trim_result()

    # Clamp to valid range
    start = max(0, start)
    end = min(original_frames, end)

    trimmed = keypoints[start:end]

    return trimmed, {
        'original_frames': original_frames,
        'trimmed_start': start,
        'trimmed_end': end,
        'frames_removed': original_frames - (end - start)
    }


def mask_stationary_hands(keypoints, movement_ratio_threshold=0.5):
    """
    Detect and mask the less active hand when one hand dominates movement.

    For one-handed signs, the non-signing hand should be masked to match
    training data patterns where undetected hands appear as zeros.

    Args:
        keypoints: (frames, 75, 2) array - 33 body + 21 left hand + 21 right hand
        movement_ratio_threshold: Mask the less active hand if its movement ratio
                                  is below this threshold (default 0.5 = 50%)

    Returns:
        keypoints: Modified array with less active hand masked
        masked_hand: 'left', 'right', or None
    """
    import numpy as np

    # Hand indices in 75-point format
    LEFT_HAND_START, LEFT_HAND_END = 33, 54
    RIGHT_HAND_START, RIGHT_HAND_END = 54, 75

    # Extract hand keypoints
    left_hand = keypoints[:, LEFT_HAND_START:LEFT_HAND_END, :]
    right_hand = keypoints[:, RIGHT_HAND_START:RIGHT_HAND_END, :]

    def calc_movement(hand_data):
        """Calculate total movement for a hand."""
        valid_mask = np.any(hand_data != 0, axis=(1, 2))
        if valid_mask.sum() < 2:
            return 0.0
        centers = np.mean(hand_data, axis=1)  # (frames, 2)
        deltas = np.diff(centers, axis=0)
        movement = np.sum(np.linalg.norm(deltas, axis=1))
        return float(movement)

    left_movement = calc_movement(left_hand)
    right_movement = calc_movement(right_hand)

    masked_hand = None
    max_movement = max(left_movement, right_movement)

    if max_movement > 0:
        left_ratio = left_movement / max_movement
        right_ratio = right_movement / max_movement

        # Mask the less active hand if significantly less active
        if right_ratio < movement_ratio_threshold and left_movement > right_movement:
            # Right hand is less active, mask it
            keypoints[:, RIGHT_HAND_START:RIGHT_HAND_END, :] = 0
            masked_hand = 'right'
        elif left_ratio < movement_ratio_threshold and right_movement > left_movement:
            # Left hand is less active, mask it
            keypoints[:, LEFT_HAND_START:LEFT_HAND_END, :] = 0
            masked_hand = 'left'

    return keypoints, masked_hand


def convert_pose_to_pickle(pose_file, mask_stationary=True, trim_signing_region=True,
                          trim_segments=True, target_frames=50):
    """
    Convert .pose file to pickle format for model inference.

    Args:
        pose_file: Path to .pose file
        mask_stationary: If True, mask hands that aren't moving (default True)
        trim_signing_region: If True, trim to only frames where hands are raised (default True)
        trim_segments: If True, trim to most active window if still too long (default True)
        target_frames: Target frame count when trimming (default 50)
    """
    try:
        from pose_format import Pose
        import numpy as np
        import pickle

        with open(pose_file, "rb") as f:
            buffer = f.read()
            pose = Pose.read(buffer)

        pose_data = pose.body.data

        if len(pose_data.shape) == 4:
            pose_sequence = pose_data[:, 0, :, :]
        else:
            pose_sequence = pose_data

        # Extract 75-point subset (pose + hands, exclude face)
        # MediaPipe formats:
        #   543: 33 pose + 468 face + 21 left hand + 21 right hand
        #   576: 33 pose + 468 face + 21 left hand + 21 right hand + 33 world coords
        if pose_sequence.shape[1] == 543:
            pose_75pt = np.concatenate([
                pose_sequence[:, 0:33, :],      # Pose landmarks
                pose_sequence[:, 501:522, :],   # Left hand landmarks
                pose_sequence[:, 522:543, :]    # Right hand landmarks
            ], axis=1)
        elif pose_sequence.shape[1] == 576:
            # Same as 543 format, just ignore the extra 33 world coords at the end
            pose_75pt = np.concatenate([
                pose_sequence[:, 0:33, :],      # Pose landmarks
                pose_sequence[:, 501:522, :],   # Left hand landmarks
                pose_sequence[:, 522:543, :]    # Right hand landmarks
            ], axis=1)
        elif pose_sequence.shape[1] == 75:
            pose_75pt = pose_sequence
        else:
            print(f"  WARNING: Unknown keypoint count {pose_sequence.shape[1]}, using as-is")
            pose_75pt = pose_sequence

        # Convert to regular numpy array
        keypoints = np.array(pose_75pt[:, :, :2])

        # Step 1: Trim to signing region (remove hands-up/down transitions)
        # This is critical for signers who return hands to sides between signs
        if trim_signing_region:
            keypoints, sign_trim_info = trim_to_signing_region(keypoints)
            if sign_trim_info['frames_removed'] > 0:
                print(f"  Trimmed to signing region: {sign_trim_info['original_frames']} -> {keypoints.shape[0]} frames")

        # Step 2: Mask stationary hands to match training data patterns
        if mask_stationary:
            keypoints, masked_hand = mask_stationary_hands(keypoints)
            if masked_hand:
                print(f"  Masked stationary {masked_hand} hand")

        # Step 3: Trim to most active window if segment is still too long
        # This helps match training data durations (typically 40-60 frames)
        if trim_segments and keypoints.shape[0] > target_frames * 1.5:
            # Determine which hand is dominant for trimming
            LEFT_HAND = (33, 54)
            RIGHT_HAND = (54, 75)

            left_zeros = (keypoints[:, LEFT_HAND[0]:LEFT_HAND[1], :] == 0).sum()
            right_zeros = (keypoints[:, RIGHT_HAND[0]:RIGHT_HAND[1], :] == 0).sum()

            # Use the hand with fewer zeros (more detected)
            hand_indices = RIGHT_HAND if right_zeros < left_zeros else LEFT_HAND

            keypoints, trim_info = trim_to_active_window(keypoints, target_frames, hand_indices)
            if trim_info['frames_trimmed'] > 0:
                print(f"  Trimmed to active window: {keypoints.shape[0]} frames")

        # Create pickle file
        pickle_path = str(pose_file).replace('.pose', '.pkl')

        pickle_data = {
            'keypoints': keypoints,
            'confidences': np.array(pose_75pt[:, :, 2]) if pose_75pt.shape[2] > 2 else np.ones(keypoints.shape[:2]),
            'gloss': 'UNKNOWN'
        }

        with open(pickle_path, 'wb') as f:
            pickle.dump(pickle_data, f)

        return pickle_path

    except Exception as e:
        print(f"  Error converting pose to pickle: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_actual_video_fps(video_path: str) -> dict:
    """
    Get video FPS information.

    Returns:
        dict with keys:
            'metadata_fps': FPS from video metadata
            'actual_fps': Same as metadata_fps (used for consistency)
            'frame_count': Total frames
            'duration_seconds': Video duration (frame_count / fps)
            'fps_ratio': actual_fps / 30 (for scaling frame-based parameters)
    """
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        metadata_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Use metadata FPS - it's reliable for most videos including iPhone
        actual_fps = metadata_fps if metadata_fps > 0 else 30.0
        duration_seconds = frame_count / actual_fps if actual_fps > 0 else 0

        # Calculate scaling ratio (how much to scale frame-based parameters)
        # For 30fps video, ratio = 1.0 (no scaling needed)
        # For 60fps video, ratio = 2.0 (double the frame counts)
        fps_ratio = actual_fps / 30.0 if actual_fps > 0 else 1.0

        return {
            'metadata_fps': metadata_fps,
            'actual_fps': actual_fps,
            'frame_count': frame_count,
            'duration_seconds': duration_seconds,
            'fps_ratio': fps_ratio
        }

    except Exception as e:
        print(f"  WARNING: FPS detection failed: {e}")
        return None


def resample_video_to_30fps(input_path: str, output_path: str, target_fps: int = 30):
    """Resample video to target FPS using imageio with H.264 codec."""
    try:
        import imageio.v3 as iio
        import numpy as np
        import cv2

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"  ERROR: Failed to open video: {input_path}")
            return False

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"  Video: {width}x{height}, {original_fps} FPS, {frame_count} frames")

        if original_fps > target_fps * 1.5:
            frame_skip = max(1, int(original_fps / target_fps))
        else:
            frame_skip = 1

        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            frame_idx += 1

        cap.release()

        if not frames:
            print("  ERROR: No frames extracted")
            return False

        print(f"  Writing {len(frames)} frames at {target_fps} FPS")
        frames_array = np.stack(frames)
        iio.imwrite(output_path, frames_array, fps=target_fps, codec='libx264', plugin='pyav')
        print(f"  Created: {output_path}")
        return True

    except Exception as e:
        print(f"  ERROR: Video resampling failed: {e}")
        return False


# ============================================================================
# GLOSS-BASED SAMPLE CREATION FUNCTIONS
# ============================================================================

def load_settings():
    """Load settings from config/settings.json"""
    settings_path = CONFIG_DIR / "settings.json"
    if settings_path.exists():
        with open(settings_path) as f:
            return json.load(f)
    return {}


def get_dataset_path(num_classes=50):
    """Get the path to the pose test dataset."""
    settings = load_settings()

    # Try settings first - use data_root which points to the dataset base
    if 'data_root' in settings:
        base_path = Path(settings['data_root']) / "dataset_splits"
    else:
        base_path = PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "dataset_splits"

    # Build path to test poses
    test_path = base_path / f"{num_classes}_classes" / "original" / f"pose_split_{num_classes}_class" / "test"

    if not test_path.exists():
        print(f"ERROR: Test dataset not found at: {test_path}")
        return None

    return test_path


def find_pose_file_for_gloss(gloss: str, test_path: Path) -> Path:
    """Find a pose file for the given gloss in the test dataset."""
    gloss_lower = gloss.lower().strip()
    gloss_dir = test_path / gloss_lower

    if not gloss_dir.exists():
        # Try to find case-insensitive match
        for d in test_path.iterdir():
            if d.is_dir() and d.name.lower() == gloss_lower:
                gloss_dir = d
                break
        else:
            return None

    # Get all pose files in the gloss directory
    pose_files = list(gloss_dir.glob("*.pose"))

    if not pose_files:
        return None

    # Return a random pose file
    return random.choice(pose_files)


def load_pose_file(pose_path: Path) -> Pose:
    """Load a pose file."""
    with open(pose_path, "rb") as f:
        return Pose.read(f.read())


def normalize_pose_to_center(pose: Pose, target_center=(256, 256), target_scale=200) -> Pose:
    """
    Normalize a pose to be centered at a target position with consistent scale.
    This ensures poses from different videos align properly when concatenated.

    Args:
        pose: Input Pose object
        target_center: (x, y) center position in output space
        target_scale: Target scale (distance from center to extremities)

    Returns:
        New Pose object with normalized coordinates
    """
    data = pose.body.data.copy()

    # Handle 4D data (frames, people, keypoints, dims)
    if len(data.shape) == 4:
        # Work with first person
        coords = data[:, 0, :, :2]  # (frames, keypoints, 2)
    else:
        coords = data[:, :, :2]  # (frames, keypoints, 2)

    # Find bounding box across all frames (ignoring zero/invalid points)
    valid_mask = np.any(coords != 0, axis=-1)  # (frames, keypoints)

    if not np.any(valid_mask):
        # No valid points, return as-is
        return pose

    # Get valid coordinates
    valid_coords = coords[valid_mask]  # (N, 2)

    # Calculate current bounding box
    min_xy = np.min(valid_coords, axis=0)
    max_xy = np.max(valid_coords, axis=0)
    current_center = (min_xy + max_xy) / 2
    current_size = np.max(max_xy - min_xy)

    if current_size < 1e-6:
        # Degenerate case, return as-is
        return pose

    # Calculate scale factor
    scale_factor = target_scale / (current_size / 2)

    # Apply transformation: center and scale
    # new_coord = (old_coord - current_center) * scale_factor + target_center
    if len(data.shape) == 4:
        # Transform x,y coordinates for all frames and people
        for p in range(data.shape[1]):
            xy = data[:, p, :, :2]
            # Only transform non-zero points
            mask = np.any(xy != 0, axis=-1, keepdims=True)
            transformed = (xy - current_center) * scale_factor + np.array(target_center)
            data[:, p, :, :2] = np.where(mask, transformed, xy)
    else:
        xy = data[:, :, :2]
        mask = np.any(xy != 0, axis=-1, keepdims=True)
        transformed = (xy - current_center) * scale_factor + np.array(target_center)
        data[:, :, :2] = np.where(mask, transformed, xy)

    # Create new pose with normalized data
    normalized_pose = Pose(
        header=copy.deepcopy(pose.header),
        body=type(pose.body)(
            data=data,
            confidence=pose.body.confidence.copy() if pose.body.confidence is not None else None,
            fps=pose.body.fps
        )
    )

    return normalized_pose


def create_rest_pose(reference_pose: Pose, duration_seconds: float = 1.5) -> Pose:
    """
    Create a 'rest' pose segment with hands at rest position.
    Uses the last frame of the reference pose as template.
    """
    fps = reference_pose.body.fps
    num_frames = int(duration_seconds * fps)

    # Get last frame as template (keep all dimensions)
    if len(reference_pose.body.data.shape) == 4:
        last_frame = reference_pose.body.data[-1:, :, :, :]
    else:
        last_frame = reference_pose.body.data[-1:, :, :]

    # Repeat the last frame for the rest duration
    rest_data = np.repeat(last_frame, num_frames, axis=0)

    # Handle confidence
    if reference_pose.body.confidence is not None:
        rest_confidence = np.repeat(reference_pose.body.confidence[-1:], num_frames, axis=0)
    else:
        rest_confidence = None

    # Create rest pose with same structure
    rest_pose = Pose(
        header=copy.deepcopy(reference_pose.header),
        body=type(reference_pose.body)(
            data=rest_data,
            confidence=rest_confidence,
            fps=fps
        )
    )

    return rest_pose


def concatenate_poses(poses: list, gap_seconds: float = 1.5, normalize: bool = True) -> Pose:
    """
    Concatenate multiple pose files with rest gaps between them.

    Args:
        poses: List of Pose objects
        gap_seconds: Seconds of rest between each sign
        normalize: If True, normalize each pose to consistent center/scale before concatenating

    Returns:
        Single concatenated Pose object
    """
    if not poses:
        raise ValueError("No poses to concatenate")

    if len(poses) == 1:
        if normalize:
            return normalize_pose_to_center(poses[0])
        return poses[0]

    # Normalize each pose to consistent coordinate space before concatenating
    if normalize:
        poses = [normalize_pose_to_center(p) for p in poses]

    # Use first pose as reference for structure
    reference = poses[0]
    fps = reference.body.fps

    # Build concatenated data
    all_data = []
    all_confidence = []

    for i, pose in enumerate(poses):
        # Add the sign pose data
        all_data.append(pose.body.data)
        if pose.body.confidence is not None:
            all_confidence.append(pose.body.confidence)

        # Add rest gap after each sign (except last) - skip if gap_seconds is 0
        if i < len(poses) - 1 and gap_seconds > 0:
            rest_pose = create_rest_pose(pose, gap_seconds)
            all_data.append(rest_pose.body.data)
            if rest_pose.body.confidence is not None:
                all_confidence.append(rest_pose.body.confidence)

    # Concatenate along frame axis
    concatenated_data = np.concatenate(all_data, axis=0)
    concatenated_confidence = np.concatenate(all_confidence, axis=0) if all_confidence else None

    # Create final pose
    result = Pose(
        header=copy.deepcopy(reference.header),
        body=type(reference.body)(
            data=concatenated_data,
            confidence=concatenated_confidence,
            fps=fps
        )
    )

    return result


def list_available_glosses(num_classes: int = 50):
    """List all available glosses in the test dataset."""
    test_path = get_dataset_path(num_classes)
    if test_path is None:
        return

    glosses = sorted([d.name for d in test_path.iterdir() if d.is_dir()])

    print(f"\nAvailable glosses in {num_classes}-class dataset ({len(glosses)} total):")
    print("-" * 60)

    # Print in columns
    cols = 5
    for i in range(0, len(glosses), cols):
        row = glosses[i:i+cols]
        print("  " + "  ".join(f"{g:12s}" for g in row))

    print()


def prepare_sample_from_glosses(glosses, reference, name, sample_id, output_dir, description="", gap_seconds=1.5, num_classes=50, model_path=None):
    """Create a demo sample from a list of glosses."""

    if not POSE_FORMAT_AVAILABLE:
        print("ERROR: pose_format library is required for glosses mode")
        return False

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Creating Demo Sample from Glosses")
    print(f"{'='*60}")
    print(f"Glosses: {', '.join(glosses)}")
    print(f"Reference: {reference}")
    print(f"Gap between signs: {gap_seconds} seconds")
    print(f"Output: {output_dir}")
    print()

    # Step 1: Find pose files for each gloss
    print("[1/7] Finding pose files for glosses...")
    test_path = get_dataset_path(num_classes)
    if test_path is None:
        return False

    pose_files = []
    found_glosses = []
    missing_glosses = []

    for gloss in glosses:
        pose_file = find_pose_file_for_gloss(gloss, test_path)
        if pose_file:
            pose_files.append(pose_file)
            found_glosses.append(gloss.upper())
            print(f"  {gloss.upper()}: {pose_file.name}")
        else:
            missing_glosses.append(gloss)
            print(f"  {gloss.upper()}: NOT FOUND")

    if missing_glosses:
        print(f"\nERROR: Could not find pose files for: {', '.join(missing_glosses)}")
        print(f"Use --list to see available glosses")
        return False

    # Step 2: Load and concatenate poses
    print(f"\n[2/7] Loading and concatenating {len(pose_files)} pose files...")
    poses = [load_pose_file(pf) for pf in pose_files]

    for i, (pose, gloss) in enumerate(zip(poses, found_glosses)):
        frames = pose.body.data.shape[0]
        fps = pose.body.fps
        duration = frames / fps
        print(f"  {gloss}: {frames} frames, {fps} FPS, {duration:.2f}s")

    # Create TWO versions:
    # 1. Original coordinates for segmentation/prediction (model expects original scale)
    # 2. Normalized coordinates for visualization (centered display)
    concatenated_pose_original = concatenate_poses(poses, gap_seconds, normalize=False)
    concatenated_pose_normalized = concatenate_poses(poses, gap_seconds, normalize=True)

    total_frames = concatenated_pose_original.body.data.shape[0]
    total_duration = total_frames / concatenated_pose_original.body.fps
    print(f"  Concatenated: {total_frames} frames, {total_duration:.2f}s total")

    # Step 3: Save concatenated pose (original for segmentation/prediction)
    print(f"\n[3/7] Saving concatenated pose file...")
    pose_path = output_dir / "capture.pose"
    with open(pose_path, 'wb') as f:
        concatenated_pose_original.write(f)
    print(f"  Saved: {pose_path}")

    # Also save normalized version for visualization
    pose_path_viz = output_dir / "capture_viz.pose"
    with open(pose_path_viz, 'wb') as f:
        concatenated_pose_normalized.write(f)

    # Step 4: Visualize full pose (use normalized version for proper centering)
    print(f"\n[4/7] Generating pose visualization...")
    pose_video_raw = output_dir / "pose_video_raw.mp4"
    pose_video = output_dir / "pose_video.mp4"

    cmd = [
        str(VISUALIZE_POSE_EXE),
        "-i", str(pose_path_viz),  # Use normalized version
        "-o", str(pose_video_raw),
        "--normalize"
    ]
    if run_command(cmd, "visualize_pose"):
        resample_video_to_30fps(str(pose_video_raw), str(pose_video))
        pose_video_raw.unlink(missing_ok=True)
        pose_path_viz.unlink(missing_ok=True)  # Clean up temp file

    # Step 5: Segment the pose (using motion-based since no video source)
    print(f"\n[5/7] Segmenting pose file...")
    try:
        sys.path.insert(0, str(APP_DIR))
        from motion_based_segmenter import MotionBasedSegmenter

        # min_rest_duration must be less than actual gap frames
        # gap_seconds=1.5 at ~24 FPS = ~36 frames, so use 30 to be safe
        segmenter = MotionBasedSegmenter(
            velocity_threshold=0.02,
            min_sign_duration=10,
            max_sign_duration=120,
            min_rest_duration=30,  # ~1 second - less than 1.5s gap at any FPS
            padding_before=3,
            padding_after=3
        )

        segment_files = segmenter.segment_pose_file(
            str(pose_path),
            str(segments_dir)
        )

        segments_data = []
        for i, seg_file in enumerate(segment_files):
            seg_id = i + 1
            segments_data.append({
                'segment_id': seg_id,
                'pose_file': f"segments/segment_{seg_id:03d}.pose",
                'video_file': f"segments/segment_{seg_id:03d}.mp4",
                'expected_gloss': found_glosses[i] if i < len(found_glosses) else "UNKNOWN"
            })

        print(f"  Detected {len(segments_data)} segments (expected {len(glosses)})")

    except Exception as e:
        print(f"  ERROR: Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 6: Visualize segments and run predictions
    print(f"\n[6/7] Visualizing segments and running predictions...")
    try:
        from openhands_modernized_inference import load_model_from_checkpoint, predict_pose_file

        if model_path:
            checkpoint_path = Path(model_path)
        else:
            checkpoint_path = MODELS_DIR / "openhands-modernized" / "production-models" / f"wlasl_{num_classes}_class_model"
        print(f"  Loading model from: {checkpoint_path}")
        model, tokenizer = load_model_from_checkpoint(str(checkpoint_path))

        for seg in segments_data:
            seg_id = seg['segment_id']
            seg_pose = segments_dir / f"segment_{seg_id:03d}.pose"
            seg_pose_viz = segments_dir / f"segment_{seg_id:03d}_viz.pose"
            seg_video_raw = segments_dir / f"segment_{seg_id:03d}_raw.mp4"
            seg_video = segments_dir / f"segment_{seg_id:03d}.mp4"

            if seg_pose.exists():
                # In glosses mode, the segments come from an already-normalized
                # concatenated pose, so we don't need to re-normalize.
                # Just visualize the segment directly.
                cmd = [
                    str(VISUALIZE_POSE_EXE),
                    "-i", str(seg_pose),
                    "-o", str(seg_video_raw),
                    "--normalize"
                ]
                if run_command(cmd, f"visualize segment {seg_id}"):
                    resample_video_to_30fps(str(seg_video_raw), str(seg_video))
                    seg_video_raw.unlink(missing_ok=True)

                # Predict using original pose (model expects original coordinates)
                pickle_path = convert_pose_to_pickle(str(seg_pose))
                if pickle_path:
                    prediction = predict_pose_file(
                        pickle_path,
                        model=model,
                        tokenizer=tokenizer
                    )
                    seg['top_1'] = prediction['gloss']
                    seg['confidence'] = prediction['confidence']
                    seg['top_k'] = prediction['top_k_predictions'][:5]

                    expected = seg.get('expected_gloss', '?')
                    match = "OK" if seg['top_1'].upper() == expected.upper() else "MISMATCH"
                    print(f"  Segment {seg_id}: {seg['top_1']} ({seg['confidence']:.1%}) [{match}, expected: {expected}]")
                else:
                    seg['top_1'] = "UNKNOWN"
                    seg['confidence'] = 0.0
                    seg['top_k'] = []

    except Exception as e:
        print(f"  ERROR: Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        for seg in segments_data:
            seg['top_1'] = seg.get('expected_gloss', 'UNKNOWN')
            seg['confidence'] = 0.0
            seg['top_k'] = []

    # Build raw sentence
    raw_sentence = " ".join([seg['top_1'] for seg in segments_data])
    print(f"  Raw sentence: {raw_sentence}")

    # Step 7: Run LLM construction and evaluation
    print(f"\n[7/7] Running LLM construction and evaluation...")
    llm_sentence = raw_sentence
    evaluation = {
        "raw": {"bleu": 0.0, "bert": 0.0, "quality": 0.0, "composite": 0.0},
        "llm": {"bleu": 0.0, "bert": 0.0, "quality": 0.0, "composite": 0.0}
    }

    try:
        from llm_factory import create_llm_provider

        if LLM_PROMPT_PATH.exists():
            prompt_template = LLM_PROMPT_PATH.read_text(encoding='utf-8')
        else:
            prompt_template = "Convert these ASL glosses to English: {gloss_details}"

        # Format gloss details
        gloss_details = []
        for i, seg in enumerate(segments_data, 1):
            top_k = seg.get('top_k', [])
            if top_k:
                detail = f"Position {i}:\n"
                for j, p in enumerate(top_k[:3], 1):
                    conf = p.get('confidence', 0) * 100
                    detail += f"  Option {j}: '{p['gloss']}' (confidence: {conf:.1f}%)\n"
                gloss_details.append(detail)

        prompt = prompt_template.format(gloss_details="".join(gloss_details))

        llm = create_llm_provider("googleaistudio")
        response = llm.generate(prompt)

        # Parse response
        response_text = response.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            result = json.loads(response_text)
            llm_sentence = result.get('sentence', raw_sentence)
        except json.JSONDecodeError:
            llm_sentence = response_text

        print(f"  LLM sentence: {llm_sentence}")

    except Exception as e:
        print(f"  WARNING: LLM construction failed: {e}")

    # Calculate evaluation metrics
    try:
        from calculate_sent_bleu import calculate_bleu_score
        evaluation["raw"]["bleu"] = calculate_bleu_score(raw_sentence, reference)
        evaluation["llm"]["bleu"] = calculate_bleu_score(llm_sentence, reference)
        print(f"  BLEU - Raw: {evaluation['raw']['bleu']:.1f}, LLM: {evaluation['llm']['bleu']:.1f}")
    except Exception as e:
        print(f"  WARNING: BLEU calculation failed: {e}")

    try:
        from bert_score import score as bert_score_fn
        _, _, raw_bert = bert_score_fn([raw_sentence], [reference], lang="en", verbose=False)
        _, _, llm_bert = bert_score_fn([llm_sentence], [reference], lang="en", verbose=False)
        evaluation["raw"]["bert"] = float(raw_bert[0]) * 100
        evaluation["llm"]["bert"] = float(llm_bert[0]) * 100
        print(f"  BERT - Raw: {evaluation['raw']['bert']:.1f}, LLM: {evaluation['llm']['bert']:.1f}")
    except Exception as e:
        print(f"  WARNING: BERTScore calculation failed: {e}")

    # Quality score (GPT-2 perplexity)
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        import torch

        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        gpt2_model.eval()

        def get_quality_score(text):
            if not text.strip():
                return 0.0
            inputs = gpt2_tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = gpt2_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
            perplexity = min(np.exp(loss), 1000)
            # Convert perplexity to 0-100 score (lower perplexity = higher score)
            return max(0, 100 * (1 - np.log(perplexity) / np.log(1000)))

        evaluation["raw"]["quality"] = get_quality_score(raw_sentence)
        evaluation["llm"]["quality"] = get_quality_score(llm_sentence)
        print(f"  Quality - Raw: {evaluation['raw']['quality']:.1f}, LLM: {evaluation['llm']['quality']:.1f}")
    except Exception as e:
        print(f"  WARNING: Quality calculation failed: {e}")

    # Gloss Accuracy (% of expected glosses present in generated sentence)
    def calculate_gloss_accuracy(sentence, expected_glosses):
        """Calculate what % of expected glosses appear in the sentence."""
        sentence_lower = sentence.lower()
        found = sum(1 for g in expected_glosses if g.lower() in sentence_lower)
        return (found / len(expected_glosses) * 100) if expected_glosses else 0.0

    evaluation["raw"]["gloss_accuracy"] = calculate_gloss_accuracy(raw_sentence, found_glosses)
    evaluation["llm"]["gloss_accuracy"] = calculate_gloss_accuracy(llm_sentence, found_glosses)
    print(f"  Gloss Accuracy - Raw: {evaluation['raw']['gloss_accuracy']:.1f}%, LLM: {evaluation['llm']['gloss_accuracy']:.1f}%")

    # CTQI v2 (prerequisite chain) - using gloss_accuracy as proxy for both GA and CF1
    for key in ['raw', 'llm']:
        ga = evaluation[key].get('gloss_accuracy', 0.0) / 100.0
        cf1 = ga  # proxy: no separate coverage in this mode
        p = evaluation[key].get('quality', 0.0) / 100.0
        evaluation[key]['composite'] = ga * cf1 * (0.5 + 0.5 * p) * 100.0
    print(f"  CTQI v2 - Raw: {evaluation['raw']['composite']:.1f}, LLM: {evaluation['llm']['composite']:.1f}")

    # Save metadata
    print(f"\nSaving metadata...")
    metadata = {
        "id": sample_id,
        "name": name,
        "description": description or f"Demo sample from glosses: {', '.join(found_glosses)}",
        "reference_sentence": reference,
        "source_glosses": found_glosses,
        "created": datetime.now().strftime("%Y-%m-%d"),
        "precomputed": {
            "pose_file": "capture.pose",
            "pose_video": "pose_video.mp4",
            "segments": segments_data,
            "raw_sentence": raw_sentence,
            "llm_sentence": llm_sentence,
            "evaluation": evaluation
        }
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {metadata_path}")

    # Update samples.json index
    samples_json_path = APP_DIR / "demo-data" / "samples.json"
    samples_json_path.parent.mkdir(parents=True, exist_ok=True)

    if samples_json_path.exists():
        with open(samples_json_path) as f:
            samples_index = json.load(f)
    else:
        samples_index = {"samples": []}

    existing_ids = [s['id'] for s in samples_index['samples']]
    if sample_id not in existing_ids:
        samples_index['samples'].append({
            "id": sample_id,
            "name": name,
            "thumbnail": "pose_video.mp4"
        })
        with open(samples_json_path, 'w') as f:
            json.dump(samples_index, f, indent=2)
        print(f"  Updated: {samples_json_path}")

    print(f"\n{'='*60}")
    print(f"SUCCESS: Demo sample created!")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Glosses: {' '.join(found_glosses)}")
    print(f"Raw sentence: {raw_sentence}")
    print(f"LLM sentence: {llm_sentence}")
    print(f"Reference: {reference}")

    return True


# ============================================================================
# VIDEO-BASED SAMPLE CREATION (original function)
# ============================================================================

def prepare_sample(video_path, reference, name, sample_id, output_dir, description="", num_classes=50,
                   source="iphone", seg_options=None, model_path=None):
    """
    Run full pipeline and save as demo sample.

    Args:
        video_path: Path to input video file
        reference: Reference English sentence
        name: Display name for the sample
        sample_id: Unique identifier for the sample
        output_dir: Directory to save sample files
        description: Optional description
        num_classes: Number of model classes (20, 50, or 100)
        source: Video source type ('webcam' or 'iphone')
                - 'webcam': No rotation/mirroring correction
                - 'iphone': Apply rotation and mirroring correction
        seg_options: Segmentation options dict with keys:
                - 'segtype': 'time' or 'motion'
                - 'sign_duration': Expected sign duration (for time-based)
                - 'pause_duration': Pause between signs (for time-based)
                - 'startup_trim': Seconds to trim from start
                - 'rampdown_trim': Seconds to trim from end
                - 'num_signs': Expected number of signs (for time-based)
    """
    # Set default seg_options
    if seg_options is None:
        seg_options = {
            'segtype': 'motion',
            'sign_duration': 2.5,
            'pause_duration': 1.25,
            'startup_trim': 1.0,
            'rampdown_trim': 1.0,
            'num_signs': None
        }

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}")
        return False

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Preparing demo sample: {name}")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Reference: {reference}")
    print(f"Source: {source}")
    print(f"Segmentation: {seg_options['segtype']}")
    print(f"Output: {output_dir}")
    print()

    # Step 1: Copy original video and detect actual FPS
    print("[1/8] Copying original video...")
    original_video_dest = output_dir / f"original_video{video_path.suffix}"
    shutil.copy2(video_path, original_video_dest)
    print(f"  Copied to: {original_video_dest}")

    # Detect actual FPS (iPhone videos often have wrong metadata)
    fps_info = get_actual_video_fps(str(video_path))
    if fps_info:
        print(f"  FPS Detection:")
        print(f"    Metadata FPS: {fps_info['metadata_fps']:.1f}")
        print(f"    Actual FPS:   {fps_info['actual_fps']:.1f}")
        print(f"    Duration:     {fps_info['duration_seconds']:.2f}s")
        print(f"    Frame count:  {fps_info['frame_count']}")
        print(f"    FPS ratio:    {fps_info['fps_ratio']:.2f}x (for scaling parameters)")
        fps_ratio = fps_info['fps_ratio']
    else:
        print(f"  WARNING: Could not detect actual FPS, using defaults")
        fps_ratio = 1.0

    # Step 1.5: Check and normalize video orientation (for iPhone videos only)
    # Handles both rotation metadata AND front-camera mirroring
    processing_video = video_path  # Default to original
    normalized_video_path = output_dir / "normalized_video.mp4"

    if source == "iphone":
        print(f"\n[1.5/8] Checking video orientation (rotation & mirroring)...")
        orientation_result = normalize_video_orientation(str(video_path), str(normalized_video_path))

        if orientation_result['processed']:
            processing_video = normalized_video_path
            print(f"  Using normalized video for processing")
        else:
            print(f"  No orientation correction needed")
    else:
        print(f"\n[1.5/8] Skipping orientation check (webcam source)")
        print(f"  Webcam videos don't need rotation/mirroring correction")

    # Step 2: Convert video to pose
    print("\n[2/8] Converting video to pose...")
    pose_path = output_dir / "capture.pose"
    cmd = [
        str(VIDEO_TO_POSE_EXE),
        "-i", str(processing_video),
        "-o", str(pose_path),
        "--format", "mediapipe"
    ]
    if not run_command(cmd, "video_to_pose"):
        return False

    # Step 3: Visualize full pose
    print("\n[3/8] Generating pose visualization...")
    pose_video_raw = output_dir / "pose_video_raw.mp4"
    cmd = [
        str(VISUALIZE_POSE_EXE),
        "-i", str(pose_path),
        "-o", str(pose_video_raw),
        "--normalize"
    ]
    if not run_command(cmd, "visualize_pose"):
        return False

    # Resample to 30fps
    pose_video = output_dir / "pose_video.mp4"
    if not resample_video_to_30fps(str(pose_video_raw), str(pose_video)):
        return False

    # Clean up raw video
    pose_video_raw.unlink(missing_ok=True)

    # Step 4: Segment the video/pose
    segtype = seg_options.get('segtype', 'motion')

    if segtype == 'time':
        # Time-based segmentation
        print("\n[4/8] Segmenting using time-based intervals...")
        try:
            # Import the time-based segmenter from project-utilities
            import sys
            sys.path.insert(0, str(PROJECT_UTILITIES_DIR / "segmentation"))
            from webcam_time_segmenter import TimeBasedSegmenter

            segmenter = TimeBasedSegmenter(
                sign_duration=seg_options.get('sign_duration', 2.5),
                pause_duration=seg_options.get('pause_duration', 1.25),
                startup_trim=seg_options.get('startup_trim', 1.0),
                rampdown_trim=seg_options.get('rampdown_trim', 1.0),
                padding_before=0.1,
                padding_after=0.1
            )

            print(f"  Sign duration: {seg_options.get('sign_duration', 2.5)}s")
            print(f"  Pause duration: {seg_options.get('pause_duration', 1.25)}s")
            print(f"  Startup trim: {seg_options.get('startup_trim', 1.0)}s")
            print(f"  Rampdown trim: {seg_options.get('rampdown_trim', 1.0)}s")

            # Segment the pose file
            segment_files = segmenter.segment_pose_file(
                str(pose_path),
                str(segments_dir),
                num_signs=seg_options.get('num_signs'),
                verbose=True
            )

            # Convert to the data structure we need
            segments_data = []
            for i, seg_file in enumerate(segment_files):
                seg_id = i + 1
                segments_data.append({
                    'segment_id': seg_id,
                    'pose_file': f"segments/segment_{seg_id:03d}.pose",
                    'video_file': f"segments/segment_{seg_id:03d}.mp4"
                })

            print(f"  Found {len(segments_data)} segments")

        except Exception as e:
            print(f"  ERROR: Time-based segmentation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    else:
        # Motion-based segmentation (default)
        print("\n[4/8] Segmenting using motion detection...")
        try:
            from motion_based_segmenter import MotionBasedSegmenter

            # Scale frame-based parameters by FPS ratio
            # If video is 47fps instead of expected 30fps, multiply frame counts by 1.57
            # This ensures time-based thresholds remain consistent:
            #   - min_rest_duration=15 at 30fps = 0.5s -> 24 frames at 47fps
            #   - min_sign_duration=10 at 30fps = 0.33s -> 16 frames at 47fps
            #   - max_sign_duration=120 at 30fps = 4s -> 188 frames at 47fps
            scaled_min_sign = int(10 * fps_ratio)
            scaled_max_sign = int(120 * fps_ratio)
            scaled_min_rest = int(15 * fps_ratio)
            scaled_padding_before = int(3 * fps_ratio)
            scaled_padding_after = int(3 * fps_ratio)
            scaled_smoothing = max(3, int(5 * fps_ratio))

            print(f"  Scaled parameters for {fps_ratio:.2f}x FPS ratio:")
            print(f"    min_sign_duration: {scaled_min_sign} frames")
            print(f"    max_sign_duration: {scaled_max_sign} frames")
            print(f"    min_rest_duration: {scaled_min_rest} frames")

            segmenter = MotionBasedSegmenter(
                velocity_threshold=0.02,   # Velocity below this = "at rest"
                min_sign_duration=scaled_min_sign,
                max_sign_duration=scaled_max_sign,
                min_rest_duration=scaled_min_rest,
                padding_before=scaled_padding_before,
                padding_after=scaled_padding_after,
                smoothing_window=scaled_smoothing
            )

            # Segment based on hand velocity from pose data
            segment_files = segmenter.segment_pose_file(
                str(pose_path),
                str(segments_dir)
            )

            # Convert to the data structure we need
            segments_data = []
            for i, seg_file in enumerate(segment_files):
                seg_id = i + 1
                segments_data.append({
                    'segment_id': seg_id,
                    'pose_file': f"segments/segment_{seg_id:03d}.pose",
                    'video_file': f"segments/segment_{seg_id:03d}.mp4"
                })

            print(f"  Found {len(segments_data)} segments")

        except Exception as e:
            print(f"  ERROR: Motion-based segmentation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Step 5: Visualize each segment
    print("\n[5/8] Generating segment visualizations...")
    for seg in segments_data:
        seg_id = seg['segment_id']
        seg_pose = segments_dir / f"segment_{seg_id:03d}.pose"
        seg_video_raw = segments_dir / f"segment_{seg_id:03d}_raw.mp4"
        seg_video = segments_dir / f"segment_{seg_id:03d}.mp4"

        if seg_pose.exists():
            cmd = [
                str(VISUALIZE_POSE_EXE),
                "-i", str(seg_pose),
                "-o", str(seg_video_raw),
                "--normalize"
            ]
            if run_command(cmd, f"visualize segment {seg_id}"):
                resample_video_to_30fps(str(seg_video_raw), str(seg_video))
                seg_video_raw.unlink(missing_ok=True)

    # Step 6: Run model predictions
    print("\n[6/8] Running model predictions...")
    try:
        from openhands_modernized_inference import load_model_from_checkpoint, predict_pose_file

        if model_path:
            checkpoint_path = Path(model_path)
        else:
            checkpoint_path = MODELS_DIR / "openhands-modernized" / "production-models" / f"wlasl_{num_classes}_class_model"
        print(f"  Loading model from: {checkpoint_path}")
        model, tokenizer = load_model_from_checkpoint(str(checkpoint_path))

        # Scale target_frames for trimming based on FPS ratio
        # 50 frames at 30fps = 1.67s, so at 47fps we need 79 frames for same duration
        scaled_target_frames = int(50 * fps_ratio)
        print(f"  Using scaled target_frames: {scaled_target_frames} (for {1.67:.2f}s segments)")

        for seg in segments_data:
            seg_id = seg['segment_id']
            seg_pose = segments_dir / f"segment_{seg_id:03d}.pose"
            if seg_pose.exists():
                # Convert .pose to pickle format with FPS-scaled parameters
                pickle_path = convert_pose_to_pickle(
                    str(seg_pose),
                    mask_stationary=True,
                    trim_segments=True,
                    target_frames=scaled_target_frames
                )
                if pickle_path:
                    prediction = predict_pose_file(
                        pickle_path,
                        model=model,
                        tokenizer=tokenizer
                    )
                    seg['top_1'] = prediction['gloss']
                    seg['confidence'] = prediction['confidence']
                    seg['top_k'] = prediction['top_k_predictions'][:5]
                    print(f"  Segment {seg_id}: {seg['top_1']} ({seg['confidence']:.2%})")
                else:
                    seg['top_1'] = "UNKNOWN"
                    seg['confidence'] = 0.0
                    seg['top_k'] = []

    except Exception as e:
        print(f"  ERROR: Model prediction failed: {e}")
        import traceback
        traceback.print_exc()
        # Continue without predictions
        for seg in segments_data:
            seg['top_1'] = "UNKNOWN"
            seg['confidence'] = 0.0
            seg['top_k'] = []

    # Build raw sentence from predictions
    raw_sentence = " ".join([seg['top_1'] for seg in segments_data])
    print(f"  Raw sentence: {raw_sentence}")

    # Step 7: Run LLM construction
    print("\n[7/8] Constructing sentence with LLM...")
    llm_sentence = ""
    try:
        from llm_factory import create_llm_provider

        # Load prompt template
        if LLM_PROMPT_PATH.exists():
            prompt_template = LLM_PROMPT_PATH.read_text(encoding='utf-8')
        else:
            prompt_template = "Convert these ASL glosses to English: {glosses}"

        # Format top-k data for prompt (same format as glosses mode)
        gloss_details = []
        for i, seg in enumerate(segments_data, 1):
            top_k = seg.get('top_k', [])
            if top_k:
                detail = f"Position {i}:\n"
                for j, p in enumerate(top_k[:3], 1):
                    conf = p.get('confidence', 0) * 100
                    detail += f"  Option {j}: '{p['gloss']}' (confidence: {conf:.1f}%)\n"
                gloss_details.append(detail)

        # Create prompt using the correct placeholder
        prompt = prompt_template.format(gloss_details="".join(gloss_details))

        # Call LLM
        llm = create_llm_provider("googleaistudio")
        response = llm.generate(prompt)

        # Parse JSON response (LLM often wraps in markdown code blocks)
        response_text = response.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            result = json.loads(response_text)
            llm_sentence = result.get('sentence', raw_sentence)
        except json.JSONDecodeError:
            llm_sentence = response_text

        print(f"  LLM sentence: {llm_sentence}")

    except Exception as e:
        print(f"  WARNING: LLM construction failed: {e}")
        llm_sentence = raw_sentence  # Fallback to raw

    # Step 8: Calculate evaluation metrics
    print("\n[8/8] Calculating evaluation metrics...")
    evaluation = {
        "raw": {"bleu": 0.0, "bert": 0.0, "quality": 0.0, "gloss_accuracy": 0.0, "composite": 0.0},
        "llm": {"bleu": 0.0, "bert": 0.0, "quality": 0.0, "gloss_accuracy": 0.0, "composite": 0.0}
    }

    try:
        # BLEU scores - use direct sentence comparison
        from calculate_sent_bleu import calculate_bleu_score
        raw_bleu = calculate_bleu_score(raw_sentence, reference)
        llm_bleu = calculate_bleu_score(llm_sentence, reference)
        evaluation["raw"]["bleu"] = raw_bleu
        evaluation["llm"]["bleu"] = llm_bleu
        print(f"  BLEU - Raw: {raw_bleu:.1f}, LLM: {llm_bleu:.1f}")
    except Exception as e:
        print(f"  WARNING: BLEU calculation failed: {e}")

    try:
        # BERTScore
        from bert_score import score as bert_score_fn
        _, _, raw_bert = bert_score_fn([raw_sentence], [reference], lang="en", verbose=False)
        _, _, llm_bert = bert_score_fn([llm_sentence], [reference], lang="en", verbose=False)
        evaluation["raw"]["bert"] = float(raw_bert[0]) * 100
        evaluation["llm"]["bert"] = float(llm_bert[0]) * 100
        print(f"  BERT - Raw: {evaluation['raw']['bert']:.1f}, LLM: {evaluation['llm']['bert']:.1f}")
    except Exception as e:
        print(f"  WARNING: BERTScore calculation failed: {e}")

    # Quality score (GPT-2 perplexity)
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        import torch

        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        gpt2_model.eval()

        def get_quality_score(text):
            if not text.strip():
                return 0.0
            inputs = gpt2_tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = gpt2_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
            perplexity = min(np.exp(loss), 1000)
            # Convert perplexity to 0-100 score (lower perplexity = higher score)
            return max(0, 100 * (1 - np.log(perplexity) / np.log(1000)))

        evaluation["raw"]["quality"] = get_quality_score(raw_sentence)
        evaluation["llm"]["quality"] = get_quality_score(llm_sentence)
        print(f"  Quality - Raw: {evaluation['raw']['quality']:.1f}, LLM: {evaluation['llm']['quality']:.1f}")
    except Exception as e:
        print(f"  WARNING: Quality calculation failed: {e}")

    # Gloss Accuracy (% of predicted glosses present in reference sentence)
    # For video mode, we check if the predicted glosses appear in the reference
    def calculate_gloss_accuracy(sentence, predicted_glosses):
        """Calculate what % of predicted glosses appear in the sentence."""
        sentence_lower = sentence.lower()
        found = sum(1 for g in predicted_glosses if g.lower() in sentence_lower)
        return (found / len(predicted_glosses) * 100) if predicted_glosses else 0.0

    predicted_glosses = [seg['top_1'] for seg in segments_data if seg.get('top_1') and seg['top_1'] != 'UNKNOWN']
    evaluation["raw"]["gloss_accuracy"] = calculate_gloss_accuracy(raw_sentence, predicted_glosses)
    evaluation["llm"]["gloss_accuracy"] = calculate_gloss_accuracy(llm_sentence, predicted_glosses)
    print(f"  Gloss Accuracy - Raw: {evaluation['raw']['gloss_accuracy']:.1f}%, LLM: {evaluation['llm']['gloss_accuracy']:.1f}%")

    # CTQI v2 (prerequisite chain) - using gloss_accuracy as proxy for both GA and CF1
    for key in ['raw', 'llm']:
        ga = evaluation[key].get('gloss_accuracy', 0.0) / 100.0
        cf1 = ga  # proxy: no separate coverage in this mode
        p = evaluation[key].get('quality', 0.0) / 100.0
        evaluation[key]['composite'] = ga * cf1 * (0.5 + 0.5 * p) * 100.0
    print(f"  CTQI v2 - Raw: {evaluation['raw']['composite']:.1f}, LLM: {evaluation['llm']['composite']:.1f}")

    # Create metadata.json
    print("\nSaving metadata...")
    metadata = {
        "id": sample_id,
        "name": name,
        "description": description or f"Demo sample: {name}",
        "reference_sentence": reference,
        "created": datetime.now().strftime("%Y-%m-%d"),
        "original_video": original_video_dest.name,
        "precomputed": {
            "pose_file": "capture.pose",
            "pose_video": "pose_video.mp4",
            "segments": segments_data,
            "raw_sentence": raw_sentence,
            "llm_sentence": llm_sentence,
            "evaluation": evaluation
        }
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {metadata_path}")

    # Update samples.json index
    samples_json_path = APP_DIR / "demo-data" / "samples.json"
    if samples_json_path.exists():
        with open(samples_json_path) as f:
            samples_index = json.load(f)
    else:
        samples_index = {"samples": []}

    # Check if sample already exists in index
    existing_ids = [s['id'] for s in samples_index['samples']]
    if sample_id not in existing_ids:
        samples_index['samples'].append({
            "id": sample_id,
            "name": name,
            "thumbnail": "pose_video.mp4"
        })
        with open(samples_json_path, 'w') as f:
            json.dump(samples_index, f, indent=2)
        print(f"  Updated: {samples_json_path}")

    # Clean up temporary normalized video if created
    if normalized_video_path.exists():
        normalized_video_path.unlink(missing_ok=True)

    print(f"\n{'='*60}")
    print(f"SUCCESS: Demo sample prepared!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Reference: {reference}")
    print(f"Raw sentence: {raw_sentence}")
    print(f"LLM sentence: {llm_sentence}")

    return True


def prepare_sample_from_session(session_path, reference, name, sample_id, output_dir, description="", num_classes=100, model_path=None):
    """
    Prepare demo sample from an existing Live mode session.

    This uses the pre-segmented pose files from a Live mode session,
    skipping the segmentation step entirely. This gives the best results
    because Live mode's real-time segmentation is more accurate.

    Args:
        session_path: Path to Live mode session folder (e.g., temp/<session-id>)
        reference: Reference English sentence
        name: Display name for the sample
        sample_id: Unique identifier for the sample
        output_dir: Directory to save sample files
        description: Optional description
        num_classes: Number of model classes (20, 50, or 100)
    """
    session_path = Path(session_path)
    output_dir = Path(output_dir)

    if not session_path.exists():
        print(f"ERROR: Session folder not found: {session_path}")
        return False

    # Read session metadata if it exists (contains saved llm_sentence, raw_sentence)
    session_metadata = {}
    session_metadata_path = session_path / "session_metadata.json"
    if session_metadata_path.exists():
        try:
            with open(session_metadata_path, 'r') as f:
                session_metadata = json.load(f)
            print(f"  Found session metadata with keys: {list(session_metadata.keys())}")
        except Exception as e:
            print(f"  Warning: Could not read session metadata: {e}")

    # If output directory already exists, delete it and recreate from scratch
    if output_dir.exists():
        print(f"  Output directory exists, deleting: {output_dir}")
        shutil.rmtree(output_dir)

    # Check for required files
    segments_dir_src = session_path / "segments"
    if not segments_dir_src.exists():
        print(f"ERROR: No segments folder found in session: {segments_dir_src}")
        return False

    # Find segment files (.pose or .pkl from Live Demo mode)
    segment_poses = sorted(segments_dir_src.glob("segment_*.pose"))
    segment_pkls = sorted(segments_dir_src.glob("segment_*.pkl"))
    segment_files_src = segment_poses or segment_pkls
    segment_type = "pose" if segment_poses else "pkl"

    if not segment_files_src:
        print(f"ERROR: No segment files found in: {segments_dir_src}")
        print(f"  Looked for: segment_*.pose and segment_*.pkl")
        return False

    print(f"  Segment type: {segment_type} ({len(segment_files_src)} files)")

    # Find original video and pose
    video_src = None
    for ext in ['.mp4', '.webm', '.mov']:
        candidate = session_path / f"capture{ext}"
        if candidate.exists():
            video_src = candidate
            break

    pose_src = session_path / "capture.pose"

    print(f"\n{'='*60}")
    print(f"Preparing Demo Sample from Live Mode Session")
    print(f"{'='*60}")
    print(f"Session: {session_path}")
    print(f"Reference: {reference}")
    print(f"Segments found: {len(segment_files_src)}")
    print(f"Output: {output_dir}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    # Step 1: Copy files from session
    print("[1/5] Copying files from session...")

    # Copy video if exists
    if video_src and video_src.exists():
        video_dest = output_dir / f"original_video{video_src.suffix}"
        shutil.copy2(video_src, video_dest)
        print(f"  Copied video: {video_dest.name}")

    # NOTE: We do NOT copy capture.pose from session because it was created from
    # a concatenated video (which produces different poses than the originals).
    # Instead, we'll create capture.pose by concatenating the segment pose files
    # after copying them. This ensures the full pose matches the segments exactly.

    # Copy segment files
    segments_data = []
    for i, seg_file in enumerate(segment_files_src):
        seg_id = i + 1
        seg_ext = seg_file.suffix  # .pose or .pkl
        seg_dest = segments_dir / f"segment_{seg_id:03d}{seg_ext}"
        shutil.copy2(seg_file, seg_dest)

        # Also copy video if exists (try multiple extensions)
        video_copied = False
        for vid_ext in ['.mp4', '.webm', '.mov']:
            seg_video_src = seg_file.with_suffix(vid_ext)
            if seg_video_src.exists():
                seg_video_dest = segments_dir / f"segment_{seg_id:03d}{vid_ext}"
                shutil.copy2(seg_video_src, seg_video_dest)
                video_copied = True
                print(f"  Copied segment {seg_id} ({seg_ext} + {vid_ext})")
                break

        if not video_copied:
            print(f"  Copied segment {seg_id} ({seg_ext}, no video)")

        segments_data.append({
            'segment_id': seg_id,
            'segment_file': f"segments/segment_{seg_id:03d}{seg_ext}",
            'segment_type': segment_type,
            'video_file': f"segments/segment_{seg_id:03d}.mp4"
        })

    print(f"  Total: {len(segments_data)} segments")

    # Step 1b: Create capture.pose by concatenating segment pose files
    # This ensures the full pose matches the segments exactly (unlike the session's
    # capture.pose which was created from a re-encoded concatenated video)
    capture_pose = output_dir / "capture.pose"
    if segment_type == "pose":
        print(f"\n  Creating capture.pose from {len(segments_data)} segment poses...")
        try:
            segment_poses = []
            for seg in segments_data:
                seg_pose_path = output_dir / seg['segment_file']
                if seg_pose_path.exists():
                    with open(seg_pose_path, 'rb') as f:
                        pose = Pose.read(f.read())
                    segment_poses.append(pose)

            if segment_poses:
                # Concatenate without gaps (gap_seconds=0) since these are continuous captures
                concatenated = concatenate_poses(segment_poses, gap_seconds=0, normalize=False)
                with open(capture_pose, 'wb') as f:
                    concatenated.write(f)
                total_frames = concatenated.body.data.shape[0]
                print(f"  Created capture.pose: {total_frames} frames ({capture_pose.stat().st_size:,} bytes)")
            else:
                print(f"  WARNING: No segment poses found to concatenate")
        except Exception as e:
            print(f"  ERROR creating capture.pose: {e}")
            import traceback
            traceback.print_exc()

    # Step 2: Generate pose visualization if needed
    print("\n[2/5] Generating visualizations...")

    # First, rename any raw video files (from webcam) to _original
    # so we can create pose visualizations in their place
    for seg in segments_data:
        seg_id = seg['segment_id']
        seg_video = segments_dir / f"segment_{seg_id:03d}.mp4"
        seg_original = segments_dir / f"segment_{seg_id:03d}_original.mp4"
        if seg_video.exists() and not seg_original.exists():
            # This is a raw video, rename it
            seg_video.rename(seg_original)
            print(f"  Renamed raw video: {seg_video.name} -> {seg_original.name}")

    # Generate pose_video from capture.pose (created from concatenated segments above)
    pose_video = output_dir / "pose_video.mp4"

    if capture_pose.exists() and not pose_video.exists():
        print(f"  Creating pose_video from capture.pose...")
        pose_video_raw = output_dir / "pose_video_raw.mp4"
        cmd = [
            str(VISUALIZE_POSE_EXE),
            "-i", str(capture_pose),
            "-o", str(pose_video_raw),
            "--normalize"
        ]
        if run_command(cmd, "visualize_pose"):
            resample_video_to_30fps(str(pose_video_raw), str(pose_video))
            pose_video_raw.unlink(missing_ok=True)
            print(f"  Created: {pose_video.name}")
        else:
            print(f"  WARNING: Failed to visualize capture.pose")
    elif pose_video.exists():
        print("  pose_video.mp4 already exists")
    else:
        print("  No capture.pose available for visualization")

    # Generate segment pose visualizations from .pose files
    for seg in segments_data:
        seg_id = seg['segment_id']
        seg_pose = segments_dir / f"segment_{seg_id:03d}.pose"
        seg_video = segments_dir / f"segment_{seg_id:03d}.mp4"

        if seg_pose.exists():
            # Always create pose visualization (we renamed raw video to _original)
            seg_video_raw = segments_dir / f"segment_{seg_id:03d}_viz_raw.mp4"
            cmd = [
                str(VISUALIZE_POSE_EXE),
                "-i", str(seg_pose),
                "-o", str(seg_video_raw),
                "--normalize"
            ]
            if run_command(cmd, f"visualize segment {seg_id}"):
                resample_video_to_30fps(str(seg_video_raw), str(seg_video))
                seg_video_raw.unlink(missing_ok=True)
                print(f"  Created pose visualization: segment_{seg_id:03d}.mp4")
            else:
                print(f"  WARNING: Failed to visualize segment_{seg_id:03d}.pose")
        else:
            print(f"  No .pose file for segment {seg_id}")

    # Step 3: Run model predictions (skip if already in session metadata)
    print("\n[3/5] Running model predictions...")

    # Check if we have predictions from session metadata
    saved_predictions = session_metadata.get('predictions', [])
    saved_model_classes = session_metadata.get('model_classes', 100)

    # Use session's model classes when we have saved predictions
    # Only re-run predictions if user explicitly specifies --model
    if saved_predictions and saved_model_classes != num_classes and not model_path:
        print(f"  NOTE: Session used {saved_model_classes}-class model (using saved predictions)")
        # Don't clear predictions - use what the session saved

    if saved_predictions and len(saved_predictions) == len(segments_data):
        print(f"  Using saved predictions from session metadata ({len(saved_predictions)} predictions)")
        for seg in segments_data:
            seg_id = seg['segment_id']
            # Find matching prediction
            pred = next((p for p in saved_predictions if p.get('segment_id') == seg_id), None)
            if pred:
                # Handle both formats: 'gloss' (regular mode) and 'top_1' (fast mode)
                seg['top_1'] = pred.get('gloss') or pred.get('top_1', 'UNKNOWN')
                seg['confidence'] = pred.get('confidence', 0.0)
                seg['top_k'] = pred.get('top_k', [])
                print(f"  Segment {seg_id}: {seg['top_1']} ({seg['confidence']:.2%})")
            else:
                seg['top_1'] = "UNKNOWN"
                seg['confidence'] = 0.0
                seg['top_k'] = []
                print(f"  Segment {seg_id}: UNKNOWN (no saved prediction)")
    else:
        # No saved predictions, run model inference
        try:
            from openhands_modernized_inference import load_model_from_checkpoint, predict_pose_file

            if model_path:
                checkpoint_path = Path(model_path)
            else:
                checkpoint_path = MODELS_DIR / "openhands-modernized" / "production-models" / f"wlasl_{num_classes}_class_model"
            print(f"  Loading model from: {checkpoint_path}")
            model, tokenizer = load_model_from_checkpoint(str(checkpoint_path))

            for seg in segments_data:
                seg_id = seg['segment_id']
                seg_type = seg.get('segment_type', 'pose')

                if seg_type == 'pkl':
                    # Already in pickle format from Live Demo mode
                    pickle_path = str(segments_dir / f"segment_{seg_id:03d}.pkl")
                else:
                    # Convert .pose to pickle format
                    seg_pose = segments_dir / f"segment_{seg_id:03d}.pose"
                    if seg_pose.exists():
                        pickle_path = convert_pose_to_pickle(str(seg_pose))
                    else:
                        pickle_path = None

                if pickle_path and Path(pickle_path).exists():
                    prediction = predict_pose_file(
                        pickle_path,
                        model=model,
                        tokenizer=tokenizer
                    )
                    seg['top_1'] = prediction['gloss']
                    seg['confidence'] = prediction['confidence']
                    seg['top_k'] = prediction['top_k_predictions'][:5]
                    print(f"  Segment {seg_id}: {seg['top_1']} ({seg['confidence']:.2%})")
                else:
                    seg['top_1'] = "UNKNOWN"
                    seg['confidence'] = 0.0
                    seg['top_k'] = []
                    print(f"  Segment {seg_id}: UNKNOWN (file not found)")

        except Exception as e:
            print(f"  ERROR: Model prediction failed: {e}")
            import traceback
            traceback.print_exc()
            for seg in segments_data:
                seg['top_1'] = "UNKNOWN"
                seg['confidence'] = 0.0
                seg['top_k'] = []

    # Build raw sentence (use saved if available, otherwise from predictions)
    if session_metadata.get('raw_sentence'):
        raw_sentence = session_metadata['raw_sentence']
        print(f"  Using saved raw sentence: {raw_sentence}")
    else:
        raw_sentence = " ".join([seg['top_1'] for seg in segments_data])
        print(f"  Raw sentence: {raw_sentence}")

    # Step 4: Run LLM construction (skip if already in session metadata)
    print("\n[4/5] Constructing sentence with LLM...")
    llm_sentence = ""

    # Check if we have llm_sentence from session metadata (from Save Session)
    if session_metadata.get('llm_sentence'):
        llm_sentence = session_metadata['llm_sentence']
        print(f"  Using saved LLM sentence from session metadata")
        print(f"  LLM sentence: {llm_sentence}")
    else:
        # No saved sentence, call LLM
        try:
            from llm_factory import create_llm_provider

            if LLM_PROMPT_PATH.exists():
                prompt_template = LLM_PROMPT_PATH.read_text(encoding='utf-8')
            else:
                prompt_template = "Convert these ASL glosses to English: {gloss_details}"

            # Format top-k data for prompt
            gloss_details = []
            for i, seg in enumerate(segments_data, 1):
                top_k = seg.get('top_k', [])
                if top_k:
                    detail = f"Position {i}:\n"
                    for j, p in enumerate(top_k[:3], 1):
                        conf = p.get('confidence', 0) * 100
                        detail += f"  Option {j}: '{p['gloss']}' (confidence: {conf:.1f}%)\n"
                    gloss_details.append(detail)

            prompt = prompt_template.format(gloss_details="".join(gloss_details))

            llm = create_llm_provider("googleaistudio")
            response = llm.generate(prompt)

            # Parse JSON response
            response_text = response.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            try:
                result = json.loads(response_text)
                llm_sentence = result.get('sentence', raw_sentence)
            except json.JSONDecodeError:
                llm_sentence = response_text

            print(f"  LLM sentence: {llm_sentence}")

        except Exception as e:
            print(f"  WARNING: LLM construction failed: {e}")
            llm_sentence = raw_sentence

    # Step 5: Calculate evaluation metrics (using same library as live mode)
    print("\n[5/5] Calculating evaluation metrics...")
    evaluation = {
        "raw": {"bleu": 0.0, "bert": 0.0, "quality": 0.0, "coverage_recall": 0.0, "coverage_precision": 0.0, "coverage_f1": 0.0, "composite": 0.0},
        "llm": {"bleu": 0.0, "bert": 0.0, "quality": 0.0, "coverage_recall": 0.0, "coverage_precision": 0.0, "coverage_f1": 0.0, "composite": 0.0}
    }

    try:
        from evaluation_metrics import (
            calculate_bleu_score,
            calculate_bert_score,
            calculate_quality_score,
            calculate_composite_score_v2_chain,
            calculate_coverage_v2 as calculate_coverage,
            QualityScorer,
            QUALITY_SCORING_AVAILABLE
        )

        # Get quality scorer
        scorer = QualityScorer(verbose=False) if QUALITY_SCORING_AVAILABLE else None

        # Calculate all metrics for raw sentence
        raw_bleu = calculate_bleu_score(raw_sentence, reference) or 0.0
        raw_bert = calculate_bert_score(raw_sentence, reference) or 0.0
        raw_quality = calculate_quality_score(raw_sentence, scorer=scorer) or 0.0
        raw_coverage = calculate_coverage(reference, raw_sentence)

        evaluation["raw"] = {
            'bleu': raw_bleu,
            'bert': raw_bert,
            'quality': raw_quality,
            'coverage_recall': raw_coverage['recall'] or 0.0,
            'coverage_precision': raw_coverage['precision'] or 0.0,
            'coverage_f1': raw_coverage['f1'] or 0.0,
            'missing_words': raw_coverage['missing_words'],
            'hallucinated_words': raw_coverage['hallucinated_words'],
        }

        # Calculate CTQI v2 (prerequisite chain) for raw
        raw_composite = calculate_composite_score_v2_chain(
            gloss_accuracy=raw_coverage['f1'] or 0.0,
            coverage_f1=raw_coverage['f1'] or 0.0,
            plausibility=raw_quality
        ) or 0.0
        evaluation["raw"]["composite"] = raw_composite

        print(f"  Raw - BLEU: {raw_bleu:.1f}, BERT: {raw_bert:.1f}, Quality: {raw_quality:.1f}, Coverage F1: {raw_coverage['f1']:.1f}")

        # Calculate all metrics for LLM sentence
        llm_bleu = calculate_bleu_score(llm_sentence, reference) or 0.0
        llm_bert = calculate_bert_score(llm_sentence, reference) or 0.0
        llm_quality = calculate_quality_score(llm_sentence, scorer=scorer) or 0.0
        llm_coverage = calculate_coverage(reference, llm_sentence)

        evaluation["llm"] = {
            'bleu': llm_bleu,
            'bert': llm_bert,
            'quality': llm_quality,
            'coverage_recall': llm_coverage['recall'] or 0.0,
            'coverage_precision': llm_coverage['precision'] or 0.0,
            'coverage_f1': llm_coverage['f1'] or 0.0,
            'missing_words': llm_coverage['missing_words'],
            'hallucinated_words': llm_coverage['hallucinated_words'],
        }

        # Calculate CTQI v2 (prerequisite chain) for LLM
        llm_composite = calculate_composite_score_v2_chain(
            gloss_accuracy=llm_coverage['f1'] or 0.0,
            coverage_f1=llm_coverage['f1'] or 0.0,
            plausibility=llm_quality
        ) or 0.0
        evaluation["llm"]["composite"] = llm_composite

        print(f"  LLM - BLEU: {llm_bleu:.1f}, BERT: {llm_bert:.1f}, Quality: {llm_quality:.1f}, Coverage F1: {llm_coverage['f1']:.1f}")
        print(f"  Composite - Raw: {raw_composite:.1f}, LLM: {llm_composite:.1f}")

    except Exception as e:
        print(f"  WARNING: Evaluation metrics calculation failed: {e}")
        import traceback
        traceback.print_exc()

    # Save metadata
    print("\nSaving metadata...")
    metadata = {
        "id": sample_id,
        "name": name,
        "description": description or f"Demo sample from Live mode session",
        "reference_sentence": reference,
        "source": "live_session",
        "source_session": str(session_path),
        "created": datetime.now().strftime("%Y-%m-%d"),
        "precomputed": {
            "pose_file": "capture.pose",
            "pose_video": "pose_video.mp4",
            "segments": segments_data,
            "raw_sentence": raw_sentence,
            "llm_sentence": llm_sentence,
            "evaluation": evaluation
        }
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {metadata_path}")

    # Update samples.json index
    samples_json_path = APP_DIR / "demo-data" / "samples.json"
    if samples_json_path.exists():
        with open(samples_json_path) as f:
            samples_index = json.load(f)
    else:
        samples_index = {"samples": []}

    # Find existing entry or create new one
    existing_entry = None
    for s in samples_index['samples']:
        if s['id'] == sample_id:
            existing_entry = s
            break

    if existing_entry:
        # Update existing entry
        existing_entry['name'] = name
        existing_entry['thumbnail'] = "pose_video.mp4"
        print(f"  Updated existing entry in: {samples_json_path}")
    else:
        # Add new entry
        samples_index['samples'].append({
            "id": sample_id,
            "name": name,
            "thumbnail": "pose_video.mp4"
        })
        print(f"  Added new entry to: {samples_json_path}")

    with open(samples_json_path, 'w') as f:
        json.dump(samples_index, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUCCESS: Demo sample prepared from Live mode session!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Reference: {reference}")
    print(f"Raw sentence: {raw_sentence}")
    print(f"LLM sentence: {llm_sentence}")

    return True


def interactive_mode():
    """Interactive mode for preparing samples."""
    print("\n=== Demo Sample Preparation (Interactive Mode) ===\n")

    video_path = input("Video file path: ").strip()
    if not video_path:
        print("ERROR: Video path is required")
        return

    reference = input("Reference sentence: ").strip()
    if not reference:
        print("ERROR: Reference sentence is required")
        return

    name = input("Sample name (e.g., 'Woman Helps Child'): ").strip()
    if not name:
        name = "Untitled Sample"

    # Generate ID from name
    default_id = name.lower().replace(" ", "-").replace("'", "")
    sample_id = input(f"Sample ID [{default_id}]: ").strip() or default_id

    description = input("Description (optional): ").strip()

    default_output = APP_DIR / "demo-data" / "samples" / sample_id
    output_dir = input(f"Output directory [{default_output}]: ").strip() or str(default_output)

    prepare_sample(video_path, reference, name, sample_id, output_dir, description)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare demo sample for ASL Show & Tell",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Video mode (iPhone, motion-based segmentation - default):
    python prepare_demo_sample.py --video video.mp4 --reference "Hello world" --name "Hello" --id hello

  Video mode (webcam, motion-based):
    python prepare_demo_sample.py --video webcam.mp4 --source webcam --reference "Hello" --name "Hello"

  Video mode (iPhone, time-based segmentation):
    python prepare_demo_sample.py --video video.mp4 --source iphone --segtype time \\
        --sign-duration 2.5 --pause-duration 1.25 --reference "Hello" --name "Hello"

  Video mode (webcam, time-based, known number of signs):
    python prepare_demo_sample.py --video webcam.mp4 --source webcam --segtype time \\
        --num-signs 5 --reference "Five signs here" --name "Five Signs"

  Glosses mode:
    python prepare_demo_sample.py --glosses "book,read" --reference "Read a book"
    python prepare_demo_sample.py --glosses "drink,water" --reference "Drink water" --name "Drinking"
    python prepare_demo_sample.py --list  # Show available glosses

  From Live mode session (uses existing segments - best quality):
    python prepare_demo_sample.py --from-session "path/to/temp/session-id" --reference "Hello world" --name "Hello"

Source types:
  --source iphone   Apply rotation/mirroring correction (default)
  --source webcam   No orientation correction needed

Segmentation types:
  --segtype motion  Detect signs by motion/velocity (default)
  --segtype time    Fixed time intervals (2-3s signs, 1-1.5s pauses)
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--video", help="Path to signing video file (video mode)")
    mode_group.add_argument("--glosses", "-g", help="Comma-separated glosses from test dataset (glosses mode)")
    mode_group.add_argument("--from-session", help="Path to Live mode session folder (uses existing segments)")
    mode_group.add_argument("--list", "-l", action="store_true", help="List available glosses and exit")
    mode_group.add_argument("--interactive", action="store_true", help="Run in interactive mode")

    # Common arguments
    parser.add_argument("--reference", "-r", help="Reference English sentence")
    parser.add_argument("--name", "-n", help="Display name for the sample")
    parser.add_argument("--id", help="Sample ID (default: generated from name or glosses)")
    parser.add_argument("--output", "-o", help="Output directory for the sample")
    parser.add_argument("--description", "-d", help="Optional description", default="")

    # Glosses mode specific
    parser.add_argument("--gap", type=float, default=1.5, help="Gap between signs in seconds (default: 1.5)")
    parser.add_argument("--classes", type=int, default=50, choices=[20, 43, 50, 100], help="Number of classes (default: 50)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Custom model directory path (overrides --classes for model selection)")

    # Video source and segmentation options
    parser.add_argument("--source", choices=["webcam", "iphone"], default="iphone",
                        help="Video source type: 'webcam' (no rotation fix) or 'iphone' (fix rotation/mirroring) (default: iphone)")
    parser.add_argument("--segtype", choices=["time", "motion"], default="motion",
                        help="Segmentation type: 'time' (fixed intervals) or 'motion' (motion-based) (default: motion)")

    # Time-based segmentation options (used when --segtype time)
    parser.add_argument("--sign-duration", type=float, default=2.5,
                        help="Expected sign duration in seconds for time-based segmentation (default: 2.5)")
    parser.add_argument("--pause-duration", type=float, default=1.25,
                        help="Expected pause between signs in seconds for time-based segmentation (default: 1.25)")
    parser.add_argument("--startup-trim", type=float, default=1.0,
                        help="Seconds to trim from video start (default: 1.0)")
    parser.add_argument("--rampdown-trim", type=float, default=1.0,
                        help="Seconds to trim from video end (default: 1.0)")
    parser.add_argument("--num-signs", type=int, default=None,
                        help="Expected number of signs (auto-detect if not specified)")

    args = parser.parse_args()

    # List available glosses
    if args.list:
        list_available_glosses(args.classes)
        return

    # Interactive mode
    if args.interactive:
        interactive_mode()
        return

    # Glosses mode
    if args.glosses:
        if not args.reference:
            print("ERROR: --reference is required")
            parser.print_help()
            return

        # Parse glosses
        glosses = [g.strip() for g in args.glosses.split(",")]

        # Generate defaults
        if not args.name:
            args.name = " ".join([g.capitalize() for g in glosses])

        if not args.id:
            args.id = "-".join([g.lower() for g in glosses])

        output_dir = args.output or (APP_DIR / "demo-data" / "samples" / args.id)

        prepare_sample_from_glosses(
            glosses=glosses,
            reference=args.reference,
            name=args.name,
            sample_id=args.id,
            output_dir=output_dir,
            description=args.description,
            gap_seconds=args.gap,
            num_classes=args.classes,
            model_path=args.model
        )
        return

    # Video mode
    if args.video:
        if not args.reference:
            print("ERROR: --reference is required")
            parser.print_help()
            return

        if not args.name:
            print("ERROR: --name is required for video mode")
            parser.print_help()
            return

        if not args.id:
            args.id = args.name.lower().replace(" ", "-").replace("'", "")

        output_dir = args.output or (APP_DIR / "demo-data" / "samples" / args.id)

        # Build segmentation options
        seg_options = {
            'segtype': args.segtype,
            'sign_duration': args.sign_duration,
            'pause_duration': args.pause_duration,
            'startup_trim': args.startup_trim,
            'rampdown_trim': args.rampdown_trim,
            'num_signs': args.num_signs
        }

        prepare_sample(
            video_path=args.video,
            reference=args.reference,
            name=args.name,
            sample_id=args.id,
            output_dir=output_dir,
            description=args.description,
            num_classes=args.classes,
            source=args.source,
            seg_options=seg_options,
            model_path=args.model
        )
        return

    # From-session mode (use existing Live mode segments)
    if args.from_session:
        if not args.reference:
            print("ERROR: --reference is required")
            parser.print_help()
            return

        if not args.name:
            print("ERROR: --name is required")
            parser.print_help()
            return

        if not args.id:
            args.id = args.name.lower().replace(" ", "-").replace("'", "")

        output_dir = args.output or (APP_DIR / "demo-data" / "samples" / args.id)

        prepare_sample_from_session(
            session_path=args.from_session,
            reference=args.reference,
            name=args.name,
            sample_id=args.id,
            output_dir=output_dir,
            description=args.description,
            num_classes=args.classes,
            model_path=args.model
        )
        return

    # No mode selected
    print("Usage:")
    print("  Glosses mode: python prepare_demo_sample.py --glosses 'book,read' --reference 'Read a book'")
    print("  Video mode:   python prepare_demo_sample.py --video video.mp4 --reference 'Hello' --name 'Hello'")
    print("  List glosses: python prepare_demo_sample.py --list")
    print()
    parser.print_help()


if __name__ == "__main__":
    main()
