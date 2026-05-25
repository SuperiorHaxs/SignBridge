#!/usr/bin/env python3
"""
SignBridge App — Standalone Flask app for doctor-patient ASL communication.

Modes:
    --mode demo   Scripted demo with pre-recorded sign videos (default)
    --mode live   Real-time ASL recognition via inference API + camera

Methods:
    --method speak  User communicates via speech (default)
    --method sign   User communicates via ASL signs
"""

import os
import re
import sys
import json
import uuid
import shutil
import threading
import tempfile
import subprocess
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, Response

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
APPLICATIONS_DIR = PROJECT_ROOT / "applications"
PROJECT_UTILITIES_DIR = PROJECT_ROOT / "project-utilities"
MODELS_DIR = PROJECT_ROOT / "models"
APP_DIR = Path(__file__).resolve().parent
PRACTICE_SAMPLES_DIR = APP_DIR / "practice_samples"

# Add paths for imports
sys.path.insert(0, str(PROJECT_UTILITIES_DIR))
sys.path.insert(0, str(PROJECT_UTILITIES_DIR / "llm_interface"))
sys.path.insert(0, str(MODELS_DIR / "openhands-modernized" / "src"))
sys.path.insert(0, str(MODELS_DIR / "openhands-modernized" / "src" / "util"))
sys.path.insert(0, str(APPLICATIONS_DIR / "show-and-tell"))  # for camera_processor
sys.path.insert(0, str(PROJECT_UTILITIES_DIR / "inference_api"))  # for model_registry

from llm_factory import create_llm_provider

# ============================================================================
# FAST PATH — In-memory MediaPipe pose extraction (from show-and-tell)
# ============================================================================
import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("WARNING: MediaPipe not available — using slow video_to_pose path")

_holistic_instance = None
_holistic_lock = threading.Lock()


def _get_holistic():
    """Get or create persistent MediaPipe Holistic instance.

    model_complexity is the single biggest knob for per-frame cost but
    accuracy regresses meaningfully at 0 (subtle finger/hand-shape signs
    drop to low confidence). Default is 1 (the previous working value);
    expose as an env var so future experimentation is one restart away.
    """
    global _holistic_instance
    if not MEDIAPIPE_AVAILABLE:
        return None
    with _holistic_lock:
        if _holistic_instance is None:
            complexity = int(os.environ.get('SIGNBRIDGE_HOLISTIC_COMPLEXITY', '1'))
            print(f"[MediaPipe] Initializing Holistic model (complexity={complexity})...")
            mp_holistic = mp.solutions.holistic
            _holistic_instance = mp_holistic.Holistic(
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
                model_complexity=complexity,
            )
            print("[MediaPipe] Holistic model ready")
        return _holistic_instance


def _reset_holistic():
    """Reset MediaPipe instance to clear timestamp state."""
    global _holistic_instance
    with _holistic_lock:
        if _holistic_instance is not None:
            try:
                _holistic_instance.close()
            except Exception:
                pass
            _holistic_instance = None


FACE_INDICES = [1, 61, 291, 152, 107, 336, 33, 263]  # 8 minimal face landmarks


def _decode_video_bytes(video_bytes):
    """Decode video bytes to list of BGR frames via OpenCV."""
    tmp = tempfile.mktemp(suffix='.webm')
    try:
        with open(tmp, 'wb') as f:
            f.write(video_bytes)
        cap = cv2.VideoCapture(tmp)
        if not cap.isOpened():
            return []
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _extract_poses(frames, target_frames=15, _retry=False):
    """Extract 83-point poses from frames using MediaPipe Holistic in-memory."""
    holistic = _get_holistic()
    if holistic is None:
        return None

    # Downsample if too many frames
    if len(frames) > target_frames:
        step = len(frames) / target_frames
        frames = [frames[int(i * step)] for i in range(target_frames)]

    pose_sequence = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = holistic.process(rgb)
        except Exception as e:
            if 'timestamp' in str(e).lower() and not _retry:
                _reset_holistic()
                return _extract_poses(frames, target_frames, _retry=True)
            raise

        landmarks = []
        # Body (33)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
        else:
            landmarks.extend([[0, 0, 0, 0]] * 33)
        # Left hand (21)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z, 1.0])
        else:
            landmarks.extend([[0, 0, 0, 0]] * 21)
        # Right hand (21)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z, 1.0])
        else:
            landmarks.extend([[0, 0, 0, 0]] * 21)
        # Face (8)
        if results.face_landmarks:
            for idx in FACE_INDICES:
                lm = results.face_landmarks.landmark[idx]
                landmarks.append([lm.x, lm.y, lm.z, getattr(lm, 'visibility', 1.0)])
        else:
            landmarks.extend([[0, 0, 0, 0]] * 8)

        pose_sequence.append(landmarks)

    return np.array(pose_sequence) if pose_sequence else None


def _poses_to_pickle(pose_array):
    """Convert (N, 83, 4) pose array to pickle dict for model inference."""
    if pose_array is None or len(pose_array) == 0:
        return None
    return {
        'keypoints': pose_array[:, :, :3],
        'confidences': pose_array[:, :, 3],
        'gloss': 'UNKNOWN',
    }


def _reject_outlier_landmarks(pose_array, threshold=0.12):
    """
    Replace outlier landmarks (MediaPipe "hallucinations") with linearly interpolated
    values. An outlier is a frame where a specific landmark jumped farther than
    `threshold` in normalized coords from both neighbors (physically implausible).

    Args:
        pose_array: (N, 83, 4)
        threshold: normalized-coord distance considered a spike (default 12% of frame)
    """
    if pose_array is None or len(pose_array) < 3:
        return pose_array
    p = pose_array.copy()
    n = p.shape[0]
    xy = p[:, :, :2]

    for j in range(p.shape[1]):
        # Walk frames; if frame i is far from BOTH i-1 and i+1, it's an outlier
        for i in range(1, n - 1):
            a = xy[i - 1, j]
            b = xy[i, j]
            c = xy[i + 1, j]
            # Skip if either neighbor is missing
            if (abs(a[0]) + abs(a[1]) < 1e-6) or (abs(c[0]) + abs(c[1]) < 1e-6):
                continue
            if np.linalg.norm(b - a) > threshold and np.linalg.norm(b - c) > threshold:
                # Also check the neighbors aren't far from each other (sanity)
                if np.linalg.norm(c - a) < threshold * 1.5:
                    # Replace with midpoint
                    p[i, j, :2] = (a + c) / 2
    return p


def _temporal_smooth_pose(pose_array, window=3):
    """
    Reduce frame-to-frame jitter by averaging each landmark over a small
    temporal window. Only averages across VALID frames (zero-valued landmarks
    are skipped in the average so they don't drag values toward origin).

    Args:
        pose_array: (N, 83, 4)
        window: moving-average window size (odd number, default 3)

    Returns:
        (N, 83, 4) smoothed pose
    """
    if pose_array is None or len(pose_array) < 2 or window < 2:
        return pose_array
    half = window // 2
    n = len(pose_array)
    smoothed = pose_array.copy()

    xy = pose_array[:, :, :2]
    valid = (np.abs(xy[:, :, 0]) > 1e-6) | (np.abs(xy[:, :, 1]) > 1e-6)  # (N, 83)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_valid = valid[lo:hi]  # (w, 83)
        counts = window_valid.sum(axis=0).astype(np.float32)  # (83,)
        mask = counts > 0
        if not mask.any():
            continue
        window_xy = xy[lo:hi] * window_valid[:, :, None]  # zero-out invalid
        avg = window_xy.sum(axis=0) / np.maximum(counts[:, None], 1)
        smoothed[i, mask, :2] = avg[mask]
    return smoothed


def _drop_sparse_hand_detections(pose_array, min_detection_fraction=0.5):
    """
    If MediaPipe Hands detected a hand in fewer than `min_detection_fraction` of
    the frames, treat all those detections as noise and zero them out. Prevents
    brief false positives from being smeared into a phantom drifting hand by the
    interior-gap interpolation in `_smooth_missing_landmarks`.

    Hand landmark ranges in our 83-point layout:
        left hand  = indices 33..53
        right hand = indices 54..74
    A hand is "detected" in a frame iff its wrist landmark (33 or 54) is non-zero.
    """
    if pose_array is None or len(pose_array) < 3:
        return pose_array
    p = pose_array.copy()
    n = len(p)
    for wrist_idx, hand_start, hand_end in [(33, 33, 54), (54, 54, 75)]:
        wrist = p[:, wrist_idx, :2]
        valid = (np.abs(wrist).sum(axis=1) > 1e-6)
        if valid.sum() / max(n, 1) < min_detection_fraction:
            p[:, hand_start:hand_end, :] = 0
    return p


def _one_euro_smooth_pose(pose_array, fps=15.0, mincutoff=1.0, beta=0.5, dcutoff=1.0):
    """
    One-Euro filter — adaptive low-pass, heavy smoothing when still and light when
    fast. Preserves sign motion while killing static jitter. Reference:
    https://gery.casiez.net/1euro/

    Operates on (x, y) of each landmark; resets state when a landmark is missing
    (zero) so we don't blend across detection dropouts.
    """
    if pose_array is None or len(pose_array) < 2:
        return pose_array
    import math
    p = pose_array.copy()
    n, j_count = p.shape[0], p.shape[1]
    if p.shape[2] < 2:
        return p
    dt = 1.0 / max(fps, 1.0)

    def _alpha(cutoff):
        r = 2.0 * math.pi * cutoff * dt
        return r / (r + 1.0)

    a_d = _alpha(dcutoff)

    # Per-landmark missing mask (zero in both x and y)
    missing = (np.abs(p[:, :, 0]) < 1e-6) & (np.abs(p[:, :, 1]) < 1e-6)

    for j in range(j_count):
        for axis in (0, 1):
            prev_x = None
            prev_dx = 0.0
            for i in range(n):
                if missing[i, j]:
                    prev_x = None
                    prev_dx = 0.0
                    continue
                v = float(p[i, j, axis])
                if prev_x is None:
                    prev_x = v
                    prev_dx = 0.0
                    continue
                dx = (v - prev_x) / dt
                dx_hat = a_d * dx + (1 - a_d) * prev_dx
                cutoff = mincutoff + beta * abs(dx_hat)
                a = _alpha(cutoff)
                v_hat = a * v + (1 - a) * prev_x
                p[i, j, axis] = v_hat
                prev_x = v_hat
                prev_dx = dx_hat
    return p


def _smooth_missing_landmarks(pose_array):
    """
    Fill in missing landmarks (zero entries) via linear interpolation from
    neighboring frames. Reduces flickering in visualization and gives the model
    continuous trajectories instead of sudden (0,0) dropouts.

    Args:
        pose_array: (N, 83, 4) — last dim is [x, y, z, visibility]

    Returns:
        (N, 83, 4) with missing landmarks interpolated. Landmarks that are
        missing in ALL frames are left as zero.
    """
    if pose_array is None or len(pose_array) < 2:
        return pose_array

    p = pose_array.copy()
    n = len(p)
    # A landmark is "missing" when both x and y are exactly zero
    missing = (np.abs(p[:, :, 0]) < 1e-6) & (np.abs(p[:, :, 1]) < 1e-6)

    for j in range(p.shape[1]):
        col_missing = missing[:, j]
        if not col_missing.any():
            continue
        if col_missing.all():
            continue  # landmark never detected — leave as zero

        valid_idx = np.where(~col_missing)[0]
        missing_idx = np.where(col_missing)[0]
        for i in missing_idx:
            # Only interpolate INTERIOR gaps. Leading or trailing missing frames
            # mean MediaPipe didn't detect the landmark before signing started or
            # after signing ended (e.g. user dropped their hand and tracking lost
            # it). Extrapolating from the last valid position would render a
            # phantom "stuck" hand at the held-up location, so we leave those as
            # zero and let the renderer skip them.
            before = valid_idx[valid_idx < i]
            after = valid_idx[valid_idx > i]
            if len(before) == 0 or len(after) == 0:
                continue  # leading or trailing — leave as zero
            # Linear interpolate between bracketing valid frames
            b, a = before[-1], after[0]
            t = (i - b) / (a - b)
            p[i, j] = (1 - t) * p[b, j] + t * p[a, j]

    return p


# Per-gloss handedness metadata (loaded once)
_HANDEDNESS_METADATA = None


def _load_handedness_metadata():
    global _HANDEDNESS_METADATA
    if _HANDEDNESS_METADATA is not None:
        return _HANDEDNESS_METADATA
    path = Path(__file__).parent / "handedness_metadata.json"
    if path.exists():
        with open(path, 'r') as f:
            data = json.load(f)
        _HANDEDNESS_METADATA = data.get('glosses', {})
        print(f"[Handedness] Loaded metadata for {len(_HANDEDNESS_METADATA)} glosses")
    else:
        _HANDEDNESS_METADATA = {}
        print(f"[Handedness] Metadata file not found at {path}")
    return _HANDEDNESS_METADATA


def _smart_mask_for_prediction(pose_array, predicted_gloss):
    """
    Given a predicted gloss, check if it's a 1-handed sign. If yes, mask the
    NON-dominant (idle) hand of the pose and return masked pose for re-inference.
    Returns None if no masking needed.
    """
    if not predicted_gloss:
        return None
    metadata = _load_handedness_metadata()
    entry = metadata.get(predicted_gloss.upper())
    if not entry or entry.get('type') != 'one_handed':
        return None

    dominant = entry.get('dominant')
    if dominant not in ('right', 'left'):
        return None

    masked = pose_array.copy()
    if dominant == 'right':
        # Mask left hand landmarks (33-53) and left wrist (15)
        masked[:, 33:54, :] = 0
        masked[:, 15, :] = 0
    else:
        # Mask right hand landmarks (54-74) and right wrist (16)
        masked[:, 54:75, :] = 0
        masked[:, 16, :] = 0
    return masked


def _canonicalize_pose(pose_array):
    """
    Stage 3: Canonical pose normalization.

    Transforms ANY incoming pose sequence (iPhone portrait, laptop landscape,
    ESP32-CAM on a lanyard looking up, etc.) into a device-agnostic canonical
    form that matches what the model saw in training:

      1. Translate: origin at shoulder midpoint (sternum)
      2. Scale:     shoulder distance = 1.0 (person-invariant)
      3. Rotate:    shoulders horizontal (fixes tilted cameras / lanyard view)
      4. Y-flip:    so positive Y is up (matches typical pose conventions)

    The model's internal preprocessing also normalizes by shoulder distance,
    but our explicit canonicalization also handles rotation which the model
    does NOT compensate for on its own.

    Args:
        pose_array: (N, 83, 4) — [x, y, z, visibility]

    Returns:
        (pose_array_canonical, info_dict)
    """
    if pose_array is None or len(pose_array) < 2:
        return pose_array, {}

    p = pose_array.copy().astype(np.float32)
    n = p.shape[0]

    # Get per-frame shoulder positions (landmarks 11, 12)
    ls = p[:, 11, :2]
    rs = p[:, 12, :2]

    # Valid frames = shoulders detected (non-zero)
    ls_valid = np.abs(ls).sum(axis=1) > 1e-4
    rs_valid = np.abs(rs).sum(axis=1) > 1e-4
    valid = ls_valid & rs_valid
    if not valid.any():
        return pose_array, {'reason': 'no_shoulders_detected'}

    # Use MEDIAN across valid frames for stable anchor (shoulders don't move during signing)
    ls_med = np.median(ls[valid], axis=0)
    rs_med = np.median(rs[valid], axis=0)
    shoulder_mid = (ls_med + rs_med) / 2
    shoulder_vec = rs_med - ls_med
    shoulder_dist = float(np.linalg.norm(shoulder_vec))
    if shoulder_dist < 1e-4:
        return pose_array, {'reason': 'degenerate_shoulders'}

    # Rotation: angle of shoulder line vs horizontal
    # In image coords, x increases right, y increases DOWN
    # shoulder_vec = (rs_med - ls_med) should be horizontal (right-pointing) if upright
    angle = float(np.arctan2(shoulder_vec[1], shoulder_vec[0]))
    # Build rotation matrix to rotate BY -angle (so shoulders become horizontal)
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    # Apply to all landmarks (x, y only — z is left alone)
    # Step 1: translate so shoulder midpoint is origin
    p_xy = p[:, :, :2] - shoulder_mid  # (N, 83, 2)
    # Step 2: scale by shoulder distance
    p_xy = p_xy / shoulder_dist
    # Step 3: rotate so shoulders are horizontal
    # Reshape for matrix multiply: (N*83, 2) @ R.T
    orig_shape = p_xy.shape
    p_xy_flat = p_xy.reshape(-1, 2) @ R.T
    p_xy = p_xy_flat.reshape(orig_shape)
    # Step 4: preserve zero-landmark semantics — if original landmark was (0,0), keep it zero
    original_zero = (np.abs(pose_array[:, :, 0]) < 1e-6) & (np.abs(pose_array[:, :, 1]) < 1e-6)
    p_xy[original_zero] = 0.0

    p[:, :, :2] = p_xy

    info = {
        'reason': 'canonicalized',
        'shoulder_dist': round(shoulder_dist, 4),
        'rotation_deg': round(float(np.degrees(angle)), 2),
        'shoulder_midpoint': [round(float(shoulder_mid[0]), 4), round(float(shoulder_mid[1]), 4)],
    }
    return p, info


def _mask_idle_hand(pose_array, path_threshold=0.15):
    """
    Zero out landmarks of a hand that barely moves during the sequence.
    Rationale: for single-handed signs, training data typically has the
    non-signing hand OUT OF FRAME. When MediaPipe detects a hand that's
    just hanging at the user's side, the model interprets the static hand
    as part of a two-handed sign. Zeroing it simulates "not in frame".

    Empirical: active signing wrist path > 0.5 (in normalized coords);
    idle/jitter paths are typically 0.05-0.15. Threshold 0.15 catches idle hands.

    Args:
        pose_array: (N, 83, 4)
        path_threshold: min wrist path length (normalized units) for the hand
                       to be considered "active". Below this, hand is masked.

    Returns:
        (pose_array_masked, info_dict)
    """
    if pose_array is None or len(pose_array) < 2:
        return pose_array, {}

    p = pose_array.copy()
    lw = p[:, 15, :2]  # left wrist
    rw = p[:, 16, :2]  # right wrist

    # Only consider frames where the wrist is actually detected
    lw_valid = np.abs(lw).sum(axis=1) > 1e-4
    rw_valid = np.abs(rw).sum(axis=1) > 1e-4

    lw_path = 0.0
    if lw_valid.sum() > 1:
        lw_seq = lw[lw_valid]
        lw_path = float(np.sum(np.linalg.norm(np.diff(lw_seq, axis=0), axis=1)))
    rw_path = 0.0
    if rw_valid.sum() > 1:
        rw_seq = rw[rw_valid]
        rw_path = float(np.sum(np.linalg.norm(np.diff(rw_seq, axis=0), axis=1)))

    info = {'left_path': round(lw_path, 3), 'right_path': round(rw_path, 3), 'masked': []}

    # Mask the left hand if it's essentially idle AND the right hand is clearly active
    if lw_path < path_threshold and rw_path > path_threshold * 2:
        p[:, 33:54, :] = 0  # left hand landmarks
        p[:, 15, :] = 0      # left wrist in body
        info['masked'].append('left')

    if rw_path < path_threshold and lw_path > path_threshold * 2:
        p[:, 54:75, :] = 0
        p[:, 16, :] = 0
        info['masked'].append('right')

    return p, info


def _trim_to_motion_window(pose_array, velocity_threshold=0.015, smoothing=3, pad_frames=2):
    """
    Trim to the active signing window, excluding the trailing "hand drop" motion.

    Strategy:
      1. Find frames with meaningful velocity
      2. Identify the motion peak
      3. From the peak, trim trailing frames that look like a "drop" (monotonic
         downward wrist motion) — we don't want HOSPITAL's hand-returning-to-side
         confused with ABDOMEN's torso touch.
    """
    if pose_array is None or len(pose_array) < 4:
        return pose_array, 0, len(pose_array) if pose_array is not None else 0, {'reason': 'too_short'}

    n = len(pose_array)
    lw = pose_array[:, 15, :2]
    rw = pose_array[:, 16, :2]

    lv = np.zeros(n)
    rv = np.zeros(n)
    lv[1:] = np.linalg.norm(np.diff(lw, axis=0), axis=1)
    rv[1:] = np.linalg.norm(np.diff(rw, axis=0), axis=1)
    vel = np.maximum(lv, rv)

    missing = (np.abs(lw).sum(axis=1) < 1e-4) & (np.abs(rw).sum(axis=1) < 1e-4)
    vel[missing] = 0

    if smoothing > 1:
        kernel = np.ones(smoothing) / smoothing
        vel = np.convolve(vel, kernel, mode='same')

    active = vel > velocity_threshold
    if not active.any():
        return pose_array, 0, n, {'reason': 'no_motion_detected', 'max_vel': float(vel.max())}

    active_indices = np.where(active)[0]
    start = max(0, int(active_indices[0]) - pad_frames)
    raw_end = int(active_indices[-1]) + 1

    # ── Exclude trailing "hand drop" using wrist HEIGHT rather than velocity ──
    # Signs happen at/above shoulder level. When the wrist drops well below the
    # shoulder line, it's going to rest, not signing. Find the last frame where
    # the active wrist was still above the "signing zone" threshold.
    peak_idx = int(np.argmax(vel))

    # Use whichever wrist has more total motion as the "active" wrist
    rw_range = (rw[:, 1].max() - rw[:, 1].min()) if len(rw) else 0
    lw_range = (lw[:, 1].max() - lw[:, 1].min()) if len(lw) else 0
    if rw_range > lw_range:
        active_wrist_y = rw[:, 1]
        active_wrist_xy = rw
    else:
        active_wrist_y = lw[:, 1]
        active_wrist_xy = lw

    # Shoulder y: use median across frames (shoulders are mostly static)
    shoulder_y_series = (pose_array[:, 11, 1] + pose_array[:, 12, 1]) / 2
    valid_shoulder = shoulder_y_series > 1e-4
    shoulder_y = float(np.median(shoulder_y_series[valid_shoulder])) if valid_shoulder.any() else 0.5

    # Signing zone: wrist is within ~0.25 normalized-units below shoulder
    # (chest + upper torso). Below that is drop territory.
    signing_ceiling_y = shoulder_y + 0.25

    # Walk from peak_idx forward. Find the first sustained run of frames where
    # the wrist is below signing zone — that's the drop.
    drop_detected_at = raw_end
    below_streak = 0
    STREAK_REQUIRED = 3
    for i in range(max(peak_idx, 1), n):
        # Skip frames where the active wrist landmark is missing
        if abs(active_wrist_xy[i, 0]) + abs(active_wrist_xy[i, 1]) < 1e-4:
            below_streak = 0
            continue
        if active_wrist_y[i] > signing_ceiling_y:
            below_streak += 1
            if below_streak >= STREAK_REQUIRED:
                drop_detected_at = i - below_streak + 1
                break
        else:
            below_streak = 0

    end = min(n, min(raw_end, drop_detected_at) + pad_frames)

    if end - start < 8:
        return pose_array, 0, n, {'reason': 'window_too_small', 'raw_start': int(active_indices[0]), 'raw_end': raw_end}

    trimmed = pose_array[start:end]
    info = {
        'reason': 'trimmed',
        'original_frames': n,
        'trimmed_frames': len(trimmed),
        'start_idx': start,
        'end_idx': end,
        'peak_idx': peak_idx,
        'drop_detected_at': drop_detected_at if drop_detected_at < raw_end else None,
        'shoulder_y': round(shoulder_y, 3),
        'signing_ceiling_y': round(signing_ceiling_y, 3),
        'max_velocity': float(vel.max()),
        'mean_active_velocity': float(vel[active].mean()),
    }
    return trimmed, start, end, info


# ============================================================================
# DIRECT IN-PROCESS INFERENCE (avoids HTTP round-trip to inference API)
# ============================================================================
_direct_model_cache = {}  # domain -> (model, id_to_gloss, masked_class_ids)
_direct_model_lock = threading.Lock()


def _get_direct_model(domain):
    """Load and cache model for direct in-process inference."""
    with _direct_model_lock:
        if domain not in _direct_model_cache:
            from model_registry import ModelRegistry
            registry = ModelRegistry()
            model, id_to_gloss, masked_class_ids = registry.get_model(domain)
            _direct_model_cache[domain] = (model, id_to_gloss, masked_class_ids)
            print(f"[DirectInfer] Cached model for domain '{domain}'")
        return _direct_model_cache[domain]


# Per-sign confidence floor. Predictions below this get marked
# confident=False and `status='unclear'`; the client drops them from
# the LLM-bound buffer so weak guesses don't poison the caption.
# (This is a *gloss-level* gate. The user-facing "CTQI hard floor"
# in Settings is a separate sentence-level gate that hides the whole
# caption as "[unclear]" -- see CTQI_HARD_FLOOR in app.js.)
PREDICTION_CONFIDENCE_THRESHOLD = float(os.environ.get("SIGNBRIDGE_MIN_CONFIDENCE", "0.35"))


def _apply_confidence_gate(prediction):
    """
    Stage 5: Mark prediction as unclear if top-1 confidence is too low.
    Mutates the prediction dict, adding a 'confident' flag and a 'status' message.
    """
    conf = prediction.get('confidence', 0)
    prediction['confident'] = conf >= PREDICTION_CONFIDENCE_THRESHOLD
    if not prediction['confident']:
        prediction['status'] = 'unclear'
        prediction['status_detail'] = (
            f"Top prediction '{prediction.get('gloss','?')}' at {conf*100:.0f}% "
            f"(below {PREDICTION_CONFIDENCE_THRESHOLD*100:.0f}% threshold). "
            f"Please sign again."
        )
    else:
        prediction['status'] = 'ok'
    return prediction


def _detect_clip_handedness(pose_array):
    """
    Classify whether this clip shows 1-handed or 2-handed signing, based on
    relative motion of the two wrists. Returns ('one_handed' | 'two_handed' | 'ambiguous', dominant_side)
    """
    if pose_array is None or len(pose_array) < 3:
        return 'ambiguous', None
    lw = pose_array[:, 15, :2]
    rw = pose_array[:, 16, :2]

    def path(arr):
        valid = np.abs(arr).sum(axis=1) > 1e-4
        if valid.sum() < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(arr[valid], axis=0), axis=1)))

    l_path = path(lw)
    r_path = path(rw)
    # If one path is very small and the other is substantial, it's 1-handed
    if max(l_path, r_path) < 0.05:
        return 'ambiguous', None
    ratio = min(l_path, r_path) / max(l_path, r_path)
    if ratio < 0.25:  # one hand has <25% the motion of the other
        dominant = 'right' if r_path > l_path else 'left'
        return 'one_handed', dominant
    return 'two_handed', 'right' if r_path >= l_path else 'left'


def _predict_with_handedness_aware_refinement(pose_array, domain):
    """
    Stage 4: Two-pass inference with handedness-aware refinement.

    Only refine when BOTH conditions hold:
      - The clip itself looks 1-handed (one hand barely moves)
      - One of the top-3 predictions is a 1-handed sign per metadata

    This prevents spurious mask-and-re-predict changing a low-confidence
    wrong answer into a high-confidence different wrong answer.
    """
    pred1 = _predict_from_poses(pose_array, domain=domain)
    clip_handedness, clip_dominant = _detect_clip_handedness(pose_array)

    if clip_handedness != 'one_handed':
        return pred1, {'refined': False, 'clip_handedness': clip_handedness}

    # Find a 1-handed candidate in top-3 matching the dominant side
    metadata = _load_handedness_metadata()
    top_k = pred1.get('top_k_predictions', [])[:3]
    candidate = None
    for item in top_k:
        g = (item.get('gloss') or '').upper()
        entry = metadata.get(g)
        if entry and entry.get('type') == 'one_handed' and entry.get('dominant') == clip_dominant:
            candidate = g
            break

    if candidate is None:
        return pred1, {'refined': False, 'clip_handedness': 'one_handed', 'reason': 'no_matching_one_handed_candidate_in_top3'}

    masked = _smart_mask_for_prediction(pose_array, candidate)
    if masked is None:
        return pred1, {'refined': False, 'reason': 'masking_skipped'}

    pred2 = _predict_from_poses(masked, domain=domain)
    if pred2.get('confidence', 0) > pred1.get('confidence', 0):
        return pred2, {
            'refined': True,
            'clip_handedness': 'one_handed',
            'candidate': candidate,
            'original_top': pred1.get('gloss'),
            'original_conf': round(pred1.get('confidence', 0), 3),
            'refined_top': pred2.get('gloss'),
            'refined_conf': round(pred2.get('confidence', 0), 3),
        }
    return pred1, {
        'refined': False,
        'clip_handedness': 'one_handed',
        'candidate': candidate,
        'reason': f"mask lowered confidence ({pred2.get('confidence', 0):.2f} vs {pred1.get('confidence', 0):.2f})",
    }


def _predict_from_poses(pose_array, domain="doctor_visit"):
    """
    Run inference directly from pose array, bypassing pickle file I/O
    and HTTP round-trip to the inference API.
    """
    import time as _t
    import torch
    from openhands_modernized import WLASLPoseProcessor

    _t0 = _t.time()
    model, id_to_gloss, masked_class_ids = _get_direct_model(domain)
    processor = WLASLPoseProcessor()
    _t1 = _t.time()

    # Pose array is (N, 83, 4) with x, y, z, visibility
    # Model expects (N, 83, 3) — just xyz
    pose_sequence = pose_array[:, :, :3].copy()

    # Normalize
    pose_sequence = processor.preprocess_pose_sequence(pose_sequence, augment=False)

    # Extract features
    finger_features_tensor = None
    if hasattr(model, 'config') and model.config.use_finger_features:
        finger_features = processor.extract_finger_features(pose_sequence)
        max_len = 256
        if len(finger_features) > max_len:
            finger_features = finger_features[:max_len]
        else:
            pad = np.zeros((max_len - len(finger_features), 30), dtype=np.float32)
            finger_features = np.concatenate([finger_features, pad], axis=0)
        finger_features_tensor = torch.tensor(finger_features, dtype=torch.float32).unsqueeze(0)

    motion_features_tensor = None
    if hasattr(model, 'config') and getattr(model.config, 'use_motion_features', False):
        motion_features = processor.extract_motion_features(pose_sequence)
        max_len = 256
        if len(motion_features) > max_len:
            motion_features = motion_features[:max_len]
        else:
            pad = np.zeros((max_len - len(motion_features), 8), dtype=np.float32)
            motion_features = np.concatenate([motion_features, pad], axis=0)
        motion_features_tensor = torch.tensor(motion_features, dtype=torch.float32).unsqueeze(0)

    spatial_features_tensor = None
    if hasattr(model, 'config') and getattr(model.config, 'use_spatial_features', False):
        spatial_features = processor.extract_spatial_features(pose_sequence)
        max_len = 256
        spatial_dim = getattr(model.config, 'spatial_features', 40)
        if len(spatial_features) > max_len:
            spatial_features = spatial_features[:max_len]
        else:
            pad = np.zeros((max_len - len(spatial_features), spatial_dim), dtype=np.float32)
            spatial_features = np.concatenate([spatial_features, pad], axis=0)
        spatial_features_tensor = torch.tensor(spatial_features, dtype=torch.float32).unsqueeze(0)

    pose_sequence, attention_mask = processor.pad_or_truncate_sequence(pose_sequence, max_length=256)

    _t2 = _t.time()

    pose_tensor = torch.tensor(pose_sequence, dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.tensor(attention_mask, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(pose_tensor, mask_tensor,
                       finger_features=finger_features_tensor,
                       motion_features=motion_features_tensor,
                       spatial_features=spatial_features_tensor)

        if masked_class_ids:
            for cid in masked_class_ids:
                logits[:, cid] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        confidence, pred_id = torch.max(probs, dim=-1)

        top_probs, top_ids = torch.topk(probs, k=min(5, probs.shape[-1]), dim=-1)
        top_k = []
        for p, idx in zip(top_probs[0], top_ids[0]):
            gloss = id_to_gloss.get(str(idx.item()), f"UNK_{idx.item()}")
            top_k.append({'gloss': gloss, 'confidence': p.item()})

    _t3 = _t.time()
    print(f"[DirectInfer] model_load={_t1-_t0:.3f}s features={_t2-_t1:.3f}s forward={_t3-_t2:.3f}s total={_t3-_t0:.3f}s")

    return {
        'gloss': id_to_gloss.get(str(pred_id.item()), f"UNK_{pred_id.item()}"),
        'confidence': confidence.item(),
        'top_k_predictions': top_k,
    }

# ============================================================================
# APP SETUP
# ============================================================================
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# WebSocket support for streaming-live mode (step 1: transport only)
from flask_sock import Sock
sock = Sock(app)
# Allow large binary frames (320x240 JPEG @ q=0.6 is well under 50KB; 256KB is plenty)
app.config['SOCK_SERVER_OPTIONS'] = {'max_message_size': 262144}

# pose-format CLI binaries (same venv as Python). Used by the optional
# /api/practice-pose-video endpoint to render the user's clip into a
# pose-only video for the sign bank toggle.
_VENV_SCRIPTS = Path(sys.executable).parent
VIDEO_TO_POSE_EXE = _VENV_SCRIPTS / "video_to_pose.exe"
VISUALIZE_POSE_EXE = _VENV_SCRIPTS / "visualize_pose.exe"


def _find_ffmpeg():
    """Locate ffmpeg — PATH first, then imageio_ffmpeg bundle."""
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None




# Mode: "demo" (scripted) or "live" (real-time inference)
APP_MODE = os.environ.get("SIGNBRIDGE_MODE", "demo")
APP_METHOD = os.environ.get("SIGNBRIDGE_METHOD", "speak")

# Data paths
DEMO_DATA_DIR = Path(__file__).parent / "demo-data"
SIGN_BANK_DIR = APPLICATIONS_DIR / "show-and-tell" / "sign-bank"
SAMPLES_DIR = APPLICATIONS_DIR / "show-and-tell" / "demo-data" / "samples"

# Inference API URL (for live mode)
INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "http://localhost:3006")

# LLM prompt template for live mode
LLM_PROMPT_PATH = PROJECT_UTILITIES_DIR / "llm_interface" / "prompts" / "llm_prompt_topk_self_critique.txt"

# Configurable API base URL (for when API is separated)
API_BASE_URL = os.environ.get("SIGNBRIDGE_API_URL", None)

# ============================================================================
# LIVE MODE — Inference API + Camera Processor
# ============================================================================
_camera_processor = None


def _get_camera_processor():
    """Lazy-load camera processor for live mode."""
    global _camera_processor
    if _camera_processor is not None:
        return _camera_processor

    from camera_processor import CameraProcessor

    processor = CameraProcessor()

    # Set predict function to route through inference API
    def predict_via_api(pickle_path, domain="doctor_visit"):
        import requests
        payload = {"pickle_path": str(pickle_path), "domain": domain}
        print(f"[SignBridge] predict_via_api -> {INFERENCE_API_URL}/predict domain={domain}")
        resp = requests.post(
            f"{INFERENCE_API_URL}/predict",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        print(f"[SignBridge] predict_via_api result: domain={result.get('domain')}, gloss={result.get('gloss')}")
        return result

    processor.set_predict_fn(predict_via_api)
    _camera_processor = processor
    return processor


def _load_llm_prompt():
    """Load the closed-captions LLM prompt template."""
    if LLM_PROMPT_PATH.exists():
        return LLM_PROMPT_PATH.read_text(encoding='utf-8')
    return None


def _domain_context(domain):
    """
    Resolve a domain key to its human-readable label, description, and the
    full word list the corresponding model can predict. Used to inject
    domain-aware context into the LLM prompt so it interprets glosses with
    the right semantic prior.
    """
    registry = _load_registry()
    entry = registry.get("domains", {}).get(domain, {}) if registry else {}
    label = entry.get("label") or domain or "general"
    description = entry.get("description") or ""

    # Vocabulary = classes the model can actually predict for this domain.
    # class_index_mapping.json lists ALL classes the model was trained on;
    # masked_classes.json lists IDs that are masked out at inference time
    # (the model literally cannot output them for this domain). Including
    # masked classes in the prompt would mislead the LLM into expecting
    # words that can never appear in top-K.
    vocab = []
    model_dir = entry.get("model_dir") or (registry.get("fallback_model") if registry else None)
    if model_dir:
        model_path = MODELS_DIR / "openhands-modernized" / "production-models" / model_dir
        class_file = model_path / "class_index_mapping.json"
        mask_file  = model_path / "masked_classes.json"
        if class_file.exists():
            try:
                with open(class_file, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                masked_ids = set()
                if mask_file.exists():
                    try:
                        with open(mask_file, 'r', encoding='utf-8') as mf:
                            raw_mask = json.load(mf)
                        # masked_classes.json is typically a list of int IDs;
                        # tolerate both list-of-int and list-of-str forms.
                        masked_ids = {str(x) for x in raw_mask}
                    except Exception as e:
                        print(f"[Domain] Failed to load masked classes for {domain}: {e}")
                # mapping is {"0": "ABDOMEN", "1": "ACCIDENT", ...}
                vocab = sorted({
                    v for k, v in mapping.items()
                    if isinstance(v, str) and str(k) not in masked_ids
                })
                if masked_ids:
                    print(f"[Domain] {domain}: {len(vocab)} active vocab "
                          f"(after masking {len(masked_ids)} classes)")
            except Exception as e:
                print(f"[Domain] Failed to load vocab for {domain}: {e}")
    return {"label": label, "description": description, "vocabulary": vocab}


def _build_domain_section(domain):
    """Render the {domain_section} block for the LLM prompt, or '' if unknown."""
    ctx = _domain_context(domain)
    if not ctx["label"]:
        return ""
    parts = [
        "═══════════════════════════════════════",
        "DOMAIN CONTEXT",
        "═══════════════════════════════════════",
        f"Domain: {ctx['label']}",
    ]
    if ctx["description"]:
        parts.append(f"Description: {ctx['description']}")
    if ctx["vocabulary"]:
        parts.append(
            "Vocabulary the signer is likely to use (the model can only predict "
            "from this list — out-of-domain words will not appear):"
        )
        # Wrap to ~10 words per line for readability.
        words = ctx["vocabulary"]
        for i in range(0, len(words), 10):
            parts.append("  " + ", ".join(words[i:i + 10]))
    parts.append(
        "Interpret the signed predictions as a conversation in this domain. "
        "Use it as semantic prior when choosing among top-K alternates."
    )
    return "\n".join(parts)

# ============================================================================
# ROUTES — Pages
# ============================================================================

@app.route("/")
def index():
    return render_template("index.html", mode=APP_MODE, method=APP_METHOD)


# ============================================================================
# STREAMING LIVE MODE — Step 2: motion gate + segmentation
# ----------------------------------------------------------------------------
# Frames arrive as JPEG bytes over /ws/live-stream. A per-connection
# StreamingSession runs the closed-captions motion detector + segmentation
# state machine and emits motion / segment_start / segment_end events. No
# pose extraction or model inference yet — that's wired in step 3.
#
# The standalone /ws-test page subscribes to those events and renders a
# motion bar, signing indicator, and a segment log so the algorithm can be
# verified before any model is in the loop.
# ============================================================================

from streaming_session import StreamingSession


@app.route("/ws-test")
def ws_test():
    """Standalone page for verifying the streaming-live segmentation."""
    return render_template("ws_test.html")


@sock.route("/ws/practice-stream")
def ws_practice_stream(ws):
    """
    Practice WebSocket — same StreamingSession + motion gate as /ws/live-stream
    but tailored for Sign Bank practice:
      - Start message includes `target_gloss` (what the user is trying to sign).
      - When a segment closes, runs pose+inference and emits a `practice_result`
        event with top-K and a target-match flag.
      - Phase 2 will add pose comparison vs training reference + captured-video
        playback (saved server-side, served via /api/practice-clips/<id>).
    """
    import time as _time
    import threading as _threading
    from concurrent.futures import ThreadPoolExecutor

    print("[WS] /ws/practice-stream connected")
    send_lock = _threading.Lock()

    def send(payload):
        with send_lock:
            try:
                ws.send(json.dumps(payload))
            except Exception as e:
                print(f"[WS-Practice] send failed: {e}")

    session = StreamingSession(send_json=send)
    state = {"domain": "emergency", "target": ""}
    inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ws-practice")

    def run_practice(segment_id, frames):
        try:
            send({"type": "inference_start", "segment_id": segment_id, "frames": len(frames)})
            t0 = _time.time()
            target = min(len(frames), 90)
            full_pose = _extract_poses(frames, target_frames=target)
            if full_pose is None or len(full_pose) == 0:
                send({"type": "practice_error", "segment_id": segment_id, "msg": "no pose detected"})
                return
            full_pose = _reject_outlier_landmarks(full_pose)
            full_pose = _smooth_missing_landmarks(full_pose)
            trimmed, ts_idx, te_idx, trim_info = _trim_to_motion_window(full_pose)
            t1 = _time.time()

            # Same loosened gate as live: drop only no_motion_detected.
            if trim_info.get('reason') == 'no_motion_detected':
                send({"type": "practice_error", "segment_id": segment_id,
                      "msg": "no hand motion in segment"})
                return
            if trimmed is None or len(trimmed) == 0:
                send({"type": "practice_error", "segment_id": segment_id,
                      "msg": "trim left no frames"})
                return

            prediction, refine_info = _predict_with_handedness_aware_refinement(
                trimmed, domain=state["domain"]
            )
            prediction = _apply_confidence_gate(prediction)
            t2 = _time.time()

            target = state["target"].strip().upper()
            predicted = (prediction.get("gloss") or "").strip().upper()
            top_k = prediction.get("top_k_predictions", [])[:5]
            top_k_upper = [(p.get("gloss") or "").strip().upper() for p in top_k]
            match_top1 = bool(target) and (predicted == target)
            match_top_k = bool(target) and (target in top_k_upper)

            print(f"[WS-Practice] segment {segment_id}: {len(frames)}fr -> "
                  f"trim={len(trimmed)} ({trim_info.get('reason','?')}) | "
                  f"target={target} predicted={predicted} ({prediction.get('confidence',0)*100:.0f}%) "
                  f"top1_match={match_top1} topk_match={match_top_k} | "
                  f"pose={t1-t0:.2f}s infer={t2-t1:.2f}s")

            send({
                "type": "practice_result",
                "segment_id": segment_id,
                "gloss": prediction.get("gloss", "?"),
                "confidence": prediction.get("confidence", 0),
                "confident": prediction.get("confident", True),
                "status": prediction.get("status", "ok"),
                "top_k": top_k,
                "target": target,
                "match_top1": match_top1,
                "match_top_k": match_top_k,
                "frames_in": len(frames),
                "frames_after_trim": len(trimmed),
                "trim_reason": trim_info.get("reason", "?"),
                "pose_ms": int((t1 - t0) * 1000),
                "infer_ms": int((t2 - t1) * 1000),
                "total_ms": int((t2 - t0) * 1000),
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            send({"type": "practice_error", "segment_id": segment_id, "msg": str(e)})

    session.dispatch_segment = lambda seg_id, frames: inference_executor.submit(
        run_practice, seg_id, frames
    )

    bytes_total = 0
    window_start = _time.time()
    window_count = 0
    window_bytes = 0

    try:
        while True:
            msg = ws.receive(timeout=30)
            if msg is None:
                break
            now = _time.time()
            if isinstance(msg, (bytes, bytearray)):
                bytes_total += len(msg)
                window_count += 1
                window_bytes += len(msg)
                session.on_frame(msg)
                if now - window_start >= 1.0:
                    fps = window_count / (now - window_start)
                    kbps = (window_bytes / 1024.0) / (now - window_start)
                    send({"type": "stats", "recv_fps": round(fps, 1), "recv_kbps": round(kbps, 1),
                          "total_frames": session.frame_count, "total_bytes": bytes_total})
                    window_start = now; window_count = 0; window_bytes = 0
            else:
                try:
                    payload = json.loads(msg)
                except (ValueError, TypeError):
                    payload = {"raw": str(msg)}
                kind = payload.get("type") if isinstance(payload, dict) else None
                print(f"[WS-Practice] control: {payload}")
                if kind in ("start", "tune") and isinstance(payload, dict):
                    overrides = {k: payload.get(k) for k in (
                        "motion_threshold", "motion_threshold_continue",
                        "cooldown_frames", "min_sign_frames",
                        "sign_debounce_s", "post_segment_quiet_s",
                        "zone_top", "zone_bottom",
                        "require_hand_present", "fps",
                    )}
                    session.reconfigure(**overrides)
                    if payload.get("domain"): state["domain"] = str(payload["domain"])
                    if payload.get("target_gloss") is not None:
                        state["target"] = str(payload["target_gloss"])
                    if kind == "start":
                        session.reset()
                elif kind == "reset":
                    session.reset()
                elif kind == "ping":
                    # Client keep-alive (sent during pause). Acknowledge so the
                    # 30s receive timeout doesn't close a paused-but-alive WS.
                    send({"type": "pong"})
                    continue
                elif kind == "stop":
                    send({"type": "ack-control", "echo": payload})
                    break
                send({"type": "ack-control", "echo": payload, "config": session.cfg, "target": state["target"]})
    except Exception as e:
        print(f"[WS-Practice] connection ended: {type(e).__name__}: {e}")
    finally:
        inference_executor.shutdown(wait=False, cancel_futures=True)
        print(f"[WS-Practice] disconnected after {session.frame_count} frames, "
              f"{bytes_total/1024:.0f} KB, {session.segment_id} segments")


@sock.route("/ws/live-stream")
def ws_live_stream(ws):
    """
    Streaming-live WebSocket. Step 3: motion gate + segmentation + inference.

    Protocol:
      Client -> server (binary): one JPEG frame per message
      Client -> server (text):   {"type": "start", "fps": 15, ...tuning..., "domain": "..."}
                                 {"type": "tune", ...same keys...}   live re-tuning
                                 {"type": "reset"}                   clear state
                                 {"type": "stop"}                    close connection

      Server -> client (text):   {"type": "motion",          score, active, signing, frame, segment_frames, threshold}
                                 {"type": "segment_start",   segment_id, frame, ts}
                                 {"type": "segment_end",     segment_id, frames, duration_ms,
                                                             accepted, rejected_reason?}
                                 {"type": "segment_suppressed", reason, ...}
                                 {"type": "inference_start", segment_id, frames}
                                 {"type": "gloss",           segment_id, gloss, confidence, top_k,
                                                             confident, status, pose_ms, infer_ms, total_ms,
                                                             frames_in, frames_after_trim}
                                 {"type": "gloss_error",     segment_id, msg}
                                 {"type": "stats",           recv_fps, recv_kbps, total_frames, total_bytes}
                                 {"type": "ack-control",     echo}
                                 {"type": "error",           where, msg}
    """
    import time as _time
    import threading as _threading
    from concurrent.futures import ThreadPoolExecutor

    print("[WS] /ws/live-stream connected")
    send_lock = _threading.Lock()

    def send(payload):
        # Single ws.send call site — protects against concurrent sends from the
        # inference worker thread vs the main receive loop.
        with send_lock:
            try:
                ws.send(json.dumps(payload))
            except Exception as e:
                print(f"[WS] send failed: {e}")

    session = StreamingSession(send_json=send)

    # Per-session inference executor. max_workers=1 keeps inference strictly
    # sequential within a session (the model is GIL-bound and one segment at a
    # time matches signing cadence). The segmentation loop never blocks because
    # dispatch_segment just submits and returns.
    domain_holder = {"domain": "emergency"}
    inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ws-infer")

    def run_inference(segment_id, frames):
        """Pose-extract + infer for one segment. Runs on the executor thread."""
        try:
            send({
                "type": "inference_start",
                "segment_id": segment_id,
                "frames": len(frames),
            })
            t0 = _time.time()

            # Keep variable-length sequence — no fixed 25-frame downsample.
            # Cap at 90 (6 seconds at 15fps) just so a runaway segment can't
            # blow up MediaPipe; real signs are well under this.
            target = min(len(frames), 90)
            full_pose = _extract_poses(frames, target_frames=target)
            if full_pose is None or len(full_pose) == 0:
                send({"type": "gloss_error", "segment_id": segment_id,
                      "msg": "no pose detected"})
                return

            full_pose = _reject_outlier_landmarks(full_pose)
            full_pose = _smooth_missing_landmarks(full_pose)
            trimmed, ts_idx, te_idx, trim_info = _trim_to_motion_window(full_pose)
            t1 = _time.time()

            if trimmed is None or len(trimmed) == 0:
                send({"type": "gloss_error", "segment_id": segment_id,
                      "msg": "motion-window trim left no frames"})
                return

            # Reject only the truly unambiguous failure case from the trim:
            # `no_motion_detected` means the pose data has zero motion above
            # the velocity threshold -- there is literally no hand movement
            # in the segment, so the input is just incidental pixel noise.
            #
            # We *do not* reject `window_too_small` or `too_short` -- those
            # legitimately fire on real but quick signs (e.g. BATHROOM's
            # brief B-handshape shake), and rejecting them was suppressing
            # legitimate fast signs. The 35% confidence gate further down
            # is the safety net for any noise that slips through here.
            _trim_reason = trim_info.get('reason')
            if _trim_reason == 'no_motion_detected':
                print(f"[WS] segment {segment_id}: rejected (trim reason={_trim_reason}, "
                      f"{len(full_pose)} input frames -- no hand motion in pose data)")
                send({
                    "type": "gloss_error",
                    "segment_id": segment_id,
                    "msg": f"no hand motion in segment (trim={_trim_reason})",
                })
                return

            prediction, refine_info = _predict_with_handedness_aware_refinement(
                trimmed, domain=domain_holder["domain"]
            )
            prediction = _apply_confidence_gate(prediction)
            t2 = _time.time()

            print(f"[WS] segment {segment_id}: {len(frames)}fr -> {len(full_pose)}poses "
                  f"-> trim[{ts_idx}:{te_idx}]={len(trimmed)} ({trim_info.get('reason','?')}) | "
                  f"pose={t1-t0:.2f}s infer={t2-t1:.2f}s | "
                  f"{prediction.get('gloss','?')} ({prediction.get('confidence',0)*100:.0f}%)")

            send({
                "type": "gloss",
                "segment_id": segment_id,
                "gloss": prediction.get("gloss", "?"),
                "confidence": prediction.get("confidence", 0),
                "confident": prediction.get("confident", True),
                "status": prediction.get("status", "ok"),
                "top_k": prediction.get("top_k_predictions", [])[:5],
                "frames_in": len(frames),
                "frames_after_trim": len(trimmed),
                "trim_reason": trim_info.get("reason", "?"),
                "pose_ms": int((t1 - t0) * 1000),
                "infer_ms": int((t2 - t1) * 1000),
                "total_ms": int((t2 - t0) * 1000),
                "refined": refine_info.get("refined", False),
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            send({"type": "gloss_error", "segment_id": segment_id, "msg": str(e)})

    # Hook the segmenter to the inference executor. The session calls this from
    # the same thread that handles incoming frames; we want it to return fast,
    # so just submit() and let the executor run inference asynchronously.
    session.dispatch_segment = lambda seg_id, frames: inference_executor.submit(
        run_inference, seg_id, frames
    )

    bytes_total = 0
    window_start = _time.time()
    window_count = 0
    window_bytes = 0

    try:
        while True:
            msg = ws.receive(timeout=30)
            if msg is None:
                break

            now = _time.time()

            if isinstance(msg, (bytes, bytearray)):
                bytes_total += len(msg)
                window_count += 1
                window_bytes += len(msg)
                session.on_frame(msg)

                if now - window_start >= 1.0:
                    fps = window_count / (now - window_start)
                    kbps = (window_bytes / 1024.0) / (now - window_start)
                    send({
                        "type": "stats",
                        "recv_fps": round(fps, 1),
                        "recv_kbps": round(kbps, 1),
                        "total_frames": session.frame_count,
                        "total_bytes": bytes_total,
                    })
                    # Stats are still sent to the client every second
                    # for the diagnostic panel; the server-side print
                    # was just noisy in the terminal. Re-enable by
                    # un-commenting if you need to debug throughput.
                    # print(f"[WS] {fps:.1f} fps, {kbps:.1f} KB/s "
                    #       f"(total {session.frame_count} frames, "
                    #       f"{bytes_total/1024:.0f} KB, segments {session.segment_id})")
                    window_start = now
                    window_count = 0
                    window_bytes = 0
            else:
                try:
                    payload = json.loads(msg)
                except (ValueError, TypeError):
                    payload = {"raw": str(msg)}
                kind = payload.get("type") if isinstance(payload, dict) else None
                print(f"[WS] control: {payload}")

                if kind in ("start", "tune") and isinstance(payload, dict):
                    overrides = {k: payload.get(k) for k in (
                        "motion_threshold", "motion_threshold_continue",
                        "cooldown_frames", "min_sign_frames",
                        "sign_debounce_s", "post_segment_quiet_s",
                        "zone_top", "zone_bottom",
                        "require_hand_present", "fps",
                    )}
                    session.reconfigure(**overrides)
                    if "domain" in payload and payload["domain"]:
                        domain_holder["domain"] = str(payload["domain"])
                    if kind == "start":
                        session.reset()
                elif kind == "reset":
                    session.reset()
                elif kind == "ping":
                    # Client keep-alive (sent during pause). Acknowledge so the
                    # 30s receive timeout doesn't close a paused-but-alive WS.
                    send({"type": "pong"})
                    continue
                elif kind == "stop":
                    send({"type": "ack-control", "echo": payload})
                    break

                send({"type": "ack-control", "echo": payload, "config": session.cfg})

    except Exception as e:
        print(f"[WS] connection ended: {type(e).__name__}: {e}")
    finally:
        inference_executor.shutdown(wait=False, cancel_futures=True)
        print(f"[WS] disconnected after {session.frame_count} frames, "
              f"{bytes_total/1024:.0f} KB total, {session.segment_id} segments")


# ============================================================================
# REGISTRY — Single source of truth for domains, models, and status
# ============================================================================
REGISTRY_FILE = MODELS_DIR / "openhands-modernized" / "production-models" / "registry.json"


def _load_registry():
    """Load registry.json and return the parsed dict."""
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, 'r') as f:
            return json.load(f)
    return {"domains": {}, "fallback_model": None}


def _get_model_dir_for_domain(registry, domain):
    """Resolve model directory name for a domain, falling back if needed."""
    domains = registry.get("domains", {})
    entry = domains.get(domain, {})
    model_dir = entry.get("model_dir")
    if model_dir:
        return model_dir
    return registry.get("fallback_model")


# ============================================================================
# ROUTES — API
# ============================================================================

@app.route("/api/registry")
def get_registry():
    """Return the full domain registry — single source of truth for the frontend."""
    import json as _json
    return app.response_class(
        _json.dumps(_load_registry(), sort_keys=False),
        mimetype='application/json'
    )


@app.route("/api/conversation")
def get_conversation():
    """Return the conversation script."""
    conv_path = DEMO_DATA_DIR / "conversation.json"
    with open(conv_path) as f:
        return jsonify(json.load(f))


@app.route("/api/sign-bank")
def get_sign_bank():
    """Return sign bank videos grouped by domain vocabulary (from registry)."""
    # Collect all sign bank files
    all_signs = {}
    if SIGN_BANK_DIR.exists():
        for f in sorted(SIGN_BANK_DIR.iterdir()):
            if f.suffix.lower() in ('.mp4', '.webm', '.mov'):
                all_signs[f.stem.lower()] = {
                    "gloss": f.stem,
                    "video_url": f"/sign-bank/{f.name}",
                }

    registry = _load_registry()
    domains = registry.get("domains", {})

    # Build domain -> glosses mapping (only for domains with a model)
    domain_vocabs = {}
    for domain_key, entry in domains.items():
        model_dir_name = entry.get("model_dir")
        if not model_dir_name:
            continue
        class_file = MODELS_DIR / "openhands-modernized" / "production-models" / model_dir_name / "class_index_mapping.json"
        if not class_file.exists():
            continue
        with open(class_file, 'r') as f:
            mapping = json.load(f)
        domain_vocabs[domain_key] = set(v.lower() for v in mapping.values())

    # Also check fallback model
    fallback = registry.get("fallback_model")
    if fallback:
        fb_class_file = MODELS_DIR / "openhands-modernized" / "production-models" / fallback / "class_index_mapping.json"
        if fb_class_file.exists():
            with open(fb_class_file, 'r') as f:
                mapping = json.load(f)
            domain_vocabs["_fallback"] = set(v.lower() for v in mapping.values())

    # Group signs: assign each to the first domain whose vocab contains it
    # Priority: ready domains first (in registry order), then fallback
    priority = [k for k, v in domains.items() if v.get("model_dir")]
    if "_fallback" in domain_vocabs:
        priority.append("_fallback")

    assigned = set()
    groups = []

    for domain_key in priority:
        if domain_key not in domain_vocabs:
            continue
        vocab = domain_vocabs[domain_key]
        group_signs = []
        for gloss_lower, sign_data in sorted(all_signs.items()):
            if gloss_lower in vocab and gloss_lower not in assigned:
                group_signs.append(sign_data)
                assigned.add(gloss_lower)
        if group_signs:
            if domain_key == "_fallback":
                label = "General"
            else:
                label = domains[domain_key].get("label", domain_key.replace("_", " ").title())
            groups.append({
                "domain": domain_key if domain_key != "_fallback" else "generic",
                "label": label,
                "signs": group_signs,
            })

    # Catch any signs not in any vocabulary
    uncategorized = [s for g, s in sorted(all_signs.items()) if g not in assigned]
    if uncategorized:
        groups.append({
            "domain": "other",
            "label": "Other",
            "signs": uncategorized,
        })

    total = sum(len(g["signs"]) for g in groups)
    return jsonify({"count": total, "groups": groups})


@app.route("/api/vocabulary/<domain>")
def get_vocabulary(domain):
    """Return the list of sign glosses available for a domain."""
    registry = _load_registry()
    model_dir_name = _get_model_dir_for_domain(registry, domain)
    if not model_dir_name:
        return jsonify({"error": f"Unknown domain: {domain}"}), 404

    class_file = MODELS_DIR / "openhands-modernized" / "production-models" / model_dir_name / "class_index_mapping.json"
    if not class_file.exists():
        return jsonify({"error": "Class mapping not found"}), 404

    with open(class_file, 'r') as f:
        mapping = json.load(f)

    # Filter out masked classes if mask file exists
    masked_ids = set()
    mask_file = MODELS_DIR / "openhands-modernized" / "production-models" / model_dir_name / "masked_classes.json"
    if mask_file.exists():
        with open(mask_file, 'r') as f:
            masked_ids = set(json.load(f).get("masked_class_ids", []))

    glosses = sorted(v for k, v in mapping.items() if int(k) not in masked_ids)
    return jsonify({"domain": domain, "count": len(glosses), "glosses": glosses})


@app.route("/api/construct-sentence", methods=["POST"])
def construct_sentence():
    """Use LLM to construct a natural sentence from ASL glosses."""
    glosses = []
    try:
        data = request.get_json()
        glosses = data.get("glosses", [])
        conversation_history = data.get("conversation_history", [])

        if not glosses:
            return jsonify({"success": False, "error": "No glosses provided"}), 400

        raw_glosses = " ".join(glosses)

        # Build conversation context
        context_lines = []
        for msg in conversation_history:
            speaker = msg.get("speaker", "unknown").capitalize()
            text = msg.get("text", "")
            context_lines.append(f"{speaker}: {text}")
        context_str = "\n".join(context_lines) if context_lines else "No prior conversation."

        prompt = f"""You are translating ASL glosses into natural English for a Deaf patient communicating with a doctor during a medical visit.

Conversation so far:
{context_str}

The patient just signed these ASL glosses: {raw_glosses}

Convert these glosses into a natural, grammatically correct English sentence that makes sense in the context of this doctor visit conversation.

Rules:
- Use all the glosses provided, in order
- Add appropriate filler words (the, a, is, are, etc.) for natural English
- The sentence should fit naturally as the patient's response to the doctor's last statement
- Keep it simple and conversational
- Output ONLY the English sentence, nothing else

English sentence:"""

        llm = create_llm_provider(
            provider="googleaistudio",
            model_name="gemini-2.0-flash",
            max_tokens=100,
            timeout=15,
        )
        response = llm.generate(prompt)

        generated = response.strip()
        if generated.startswith('"') and generated.endswith('"'):
            generated = generated[1:-1]

        print(f"[DoctorDemo LLM] Glosses: {glosses} -> {generated}")

        return jsonify({"success": True, "glosses": glosses, "sentence": generated})

    except Exception as e:
        print(f"[DoctorDemo LLM] Error: {e}")
        return jsonify({"success": False, "error": str(e), "fallback": " ".join(glosses)}), 500


@app.route("/api/english-to-glosses", methods=["POST"])
def english_to_glosses():
    """Convert an English sentence to ASL glosses using the domain vocabulary."""
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        domain = data.get("domain", "doctor_visit")
        conversation_history = data.get("conversation_history", [])

        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400

        # Build union of all available vocabularies so the LLM has the widest
        # possible gloss set.  Also track which domain each gloss belongs to
        # (prefer the selected domain, then others, then fallback).
        registry = _load_registry()
        domains_cfg = registry.get("domains", {})
        fallback = registry.get("fallback_model")

        all_glosses = set()          # union across all models
        gloss_to_domain = {}         # gloss -> best domain key for inference

        # Collect glosses: selected domain first, then other ready domains, then fallback
        model_dirs_to_scan = []
        selected_model = _get_model_dir_for_domain(registry, domain)
        if selected_model:
            model_dirs_to_scan.append((domain, selected_model))
        for dk, entry in domains_cfg.items():
            md = entry.get("model_dir")
            if md and dk != domain:
                model_dirs_to_scan.append((dk, md))
        if fallback and not any(md == fallback for _, md in model_dirs_to_scan):
            model_dirs_to_scan.append(("generic", fallback))

        for dk, md in model_dirs_to_scan:
            class_file = MODELS_DIR / "openhands-modernized" / "production-models" / md / "class_index_mapping.json"
            if not class_file.exists():
                continue
            with open(class_file, 'r') as f:
                mapping = json.load(f)
            for v in mapping.values():
                g = v.lower()
                all_glosses.add(g)
                if g not in gloss_to_domain:
                    gloss_to_domain[g] = dk  # first domain wins (selected > others > fallback)

        available_glosses = sorted(all_glosses)
        vocab_list = ", ".join(available_glosses) if available_glosses else "(no vocabulary loaded)"

        # Build conversation context
        context_lines = []
        for msg in conversation_history:
            speaker = msg.get("speaker", "unknown").capitalize()
            ctx_text = msg.get("text", "")
            context_lines.append(f"{speaker}: {ctx_text}")
        context_str = "\n".join(context_lines) if context_lines else "No prior conversation."

        prompt = f"""You are converting English into ASL (American Sign Language) glosses for a Deaf person to sign.

Available ASL signs in the vocabulary:
{vocab_list}

Conversation so far:
{context_str}

The hearing person just said: "{text}"

Convert this into a sequence of ASL glosses that a Deaf person could sign to communicate the same meaning. Rules:
- ONLY use glosses from the available vocabulary list above
- ASL drops articles (a, the), prepositions, and conjugations — keep it minimal
- Order glosses in ASL grammar (topic-comment structure when appropriate)
- Output ONLY the glosses separated by spaces, nothing else
- If you cannot express the sentence with available glosses, use the closest available ones

ASL glosses:"""

        llm = create_llm_provider(
            provider="googleaistudio",
            model_name="gemini-2.0-flash",
            max_tokens=80,
            timeout=15,
        )
        response = llm.generate(prompt)

        glosses_raw = response.strip().upper()
        # Clean up: remove quotes, punctuation, keep only words
        glosses = [g for g in re.split(r'[\s,]+', glosses_raw) if g.isalpha()]

        # Filter to only glosses actually in vocabulary
        vocab_set = set(g.upper() for g in available_glosses)
        valid_glosses = [g for g in glosses if g in vocab_set]

        # Check which glosses have sign bank videos
        sign_videos = {}
        if SIGN_BANK_DIR.exists():
            for g in valid_glosses:
                video_path = SIGN_BANK_DIR / f"{g.lower()}.mp4"
                if video_path.exists():
                    sign_videos[g] = f"/sign-bank/{g.lower()}.mp4"

        print(f"[English->ASL] \"{text}\" -> {valid_glosses} (videos: {list(sign_videos.keys())})")

        return jsonify({
            "success": True,
            "text": text,
            "glosses": valid_glosses,
            "sign_videos": sign_videos,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[English->ASL] Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# ROUTES — Live Mode API
# ============================================================================

@app.route("/api/warm-model", methods=["POST"])
def warm_model():
    """Pre-load the inference model and MediaPipe Holistic for the given
    domain so the first real segment doesn't pay the cold-start cost
    (model loading + MediaPipe initialization typically takes 4-8s).
    Fire-and-forget from the client when the user enters the conversation
    window for a scenario."""
    try:
        data = request.get_json(silent=True) or {}
        domain = data.get("domain") or "emergency"
        if MEDIAPIPE_AVAILABLE:
            _get_holistic()
        _get_direct_model(domain)
        return jsonify({"success": True, "domain": domain})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/process-sign", methods=["POST"])
def process_sign():
    """
    Process a video blob of a single sign (live mode).
    Uses fast in-memory MediaPipe path (~1s) with fallback to video_to_pose (~8s).
    """
    import time as _time

    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video provided"}), 400

    video_file = request.files["video"]
    video_bytes = video_file.read()
    domain = request.form.get("domain", "doctor_visit")

    if len(video_bytes) == 0:
        return jsonify({"success": False, "error": "Empty video"}), 400

    t0 = _time.time()

    try:
        if MEDIAPIPE_AVAILABLE:
            # ── FAST PATH: in-memory MediaPipe ──
            frames = _decode_video_bytes(video_bytes)
            if not frames:
                return jsonify({"success": False, "error": "Could not decode video"}), 400

            t1 = _time.time()

            # Extract poses at higher resolution so motion-window trimming has detail
            raw_target = min(len(frames), 45)
            full_pose = _extract_poses(frames, target_frames=raw_target)
            if full_pose is None or len(full_pose) == 0:
                return jsonify({"success": False, "error": "No pose detected in video"}), 400

            # Reject hallucinated outlier positions + fill missing landmarks
            full_pose = _reject_outlier_landmarks(full_pose)
            full_pose = _smooth_missing_landmarks(full_pose)

            # Stage 3: canonical pose normalization — disabled by default.
            # The model's internal preprocessor already does shoulder-scale normalization.
            # Applying our own transform pushes inputs out of the training distribution.
            # Only enable for heavily tilted cameras (e.g., ESP32 on lanyard).
            if os.environ.get('SIGNBRIDGE_CANONICALIZE') == '1':
                full_pose, canon_info = _canonicalize_pose(full_pose)

            # Trim rest/preparation frames
            trimmed, ts_idx, te_idx, trim_info = _trim_to_motion_window(full_pose)

            # Downsample trimmed window to 25 frames (more temporal context than 15)
            MODEL_FRAMES = 25
            if len(trimmed) > MODEL_FRAMES:
                step = len(trimmed) / MODEL_FRAMES
                sample_indices = [int(i * step) for i in range(MODEL_FRAMES)]
                pose_array = trimmed[sample_indices]
            else:
                pose_array = trimmed

            t2 = _time.time()

            prediction, refine_info = _predict_with_handedness_aware_refinement(pose_array, domain=domain)
            prediction = _apply_confidence_gate(prediction)

            t3 = _time.time()

            refine_note = f" [refined: {refine_info.get('original_top')}→{refine_info.get('refined_top')}]" if refine_info.get('refined') else ''
            print(f"[FastPath] {len(frames)}fr -> {len(full_pose)}poses, "
                  f"trim[{ts_idx}:{te_idx}]={len(trimmed)} ({trim_info.get('reason','?')}) -> model={len(pose_array)} | "
                  f"decode={t1-t0:.2f}s pose={t2-t1:.2f}s infer={t3-t2:.2f}s total={t3-t0:.2f}s | "
                  f"{prediction.get('gloss','?')} ({prediction.get('confidence',0)*100:.0f}%){refine_note}")

            return jsonify({
                "success": True,
                "gloss": prediction.get("gloss", "?"),
                "confidence": prediction.get("confidence", 0),
                "top_k": prediction.get("top_k_predictions", [])[:5],
                "confident": prediction.get('confident', True),
                "status": prediction.get('status', 'ok'),
                "status_detail": prediction.get('status_detail'),
            })

        else:
            # ── SLOW PATH: video_to_pose subprocess fallback ──
            print(f"[SlowPath] /api/process-sign domain={domain}")
            processor = _get_camera_processor()
            prediction = processor.process_video_bytes(video_bytes, video_format="webm", domain=domain)

            if prediction is None:
                return jsonify({"success": False, "error": "Could not process video"}), 400

            print(f"[SlowPath] total={_time.time()-t0:.2f}s | {prediction['gloss']} ({prediction['confidence']*100:.0f}%)")

            return jsonify({
                "success": True,
                "gloss": prediction["gloss"],
                "confidence": prediction["confidence"],
                "top_k": prediction.get("top_k_predictions", [])[:5],
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# Signer 12 mapping: gloss (lowercase) -> video_id (zero-padded str)
# Lazy-loaded on first use
_SIGNER_12_MAP = None


def _load_signer_12_map():
    global _SIGNER_12_MAP
    if _SIGNER_12_MAP is not None:
        return _SIGNER_12_MAP
    path = PROJECT_ROOT / "dataset-utilities" / "categorization" / "signer_12_all_signs.json"
    if path.exists():
        with open(path, 'r') as f:
            data = json.load(f)
        _SIGNER_12_MAP = {k.lower(): v for k, v in data.get('signs', {}).items()}
        print(f"[Signer12] Loaded {len(_SIGNER_12_MAP)} signs from signer 12")
    else:
        _SIGNER_12_MAP = {}
        print("[Signer12] signer_12_all_signs.json not found")
    return _SIGNER_12_MAP


def _compare_poses_programmatic(user_pose, ref_pose):
    """
    Compare user's pose vs reference training sample using the project's
    existing pose_utils/compare_poses analyzer. Returns diagnostic metrics.

    Input: (N, 83, 2+) arrays. The analyzer expects (N, 75, 2) format
    (33 body + 21 left hand + 21 right hand — face is at the end of 83-pt).
    """
    import numpy as _np
    import sys as _sys
    _pu_path = str(PROJECT_UTILITIES_DIR / "pose_utils")
    if _pu_path not in _sys.path:
        _sys.path.insert(0, _pu_path)
    from compare_poses import analyze_pose as _cp_analyze

    # Drop visibility & face (keep first 75 points = body+hands)
    user_75 = _np.asarray(user_pose[:, :75, :2], dtype=_np.float64)
    ref_75 = _np.asarray(ref_pose[:, :75, :2], dtype=_np.float64)

    try:
        u_stats = _cp_analyze(user_75, "user", normalized=True)
        r_stats = _cp_analyze(ref_75, "reference", normalized=True)
    except Exception as e:
        return {"error": f"Analysis failed: {e}"}

    # Compute differences (same logic as compare_poses.compare_poses)
    left_pos_diff = _np.array(u_stats['left_hand_mean']) - _np.array(r_stats['left_hand_mean'])
    right_pos_diff = _np.array(u_stats['right_hand_mean']) - _np.array(r_stats['right_hand_mean'])
    spread_diff = float(u_stats['hand_spread_mean'] - r_stats['hand_spread_mean'])
    left_height_diff = float(u_stats['left_height_mean'] - r_stats['left_height_mean'])
    right_height_diff = float(u_stats['right_height_mean'] - r_stats['right_height_mean'])
    vel_diff_l = float(u_stats['left_velocity_mean'] - r_stats['left_velocity_mean'])
    vel_diff_r = float(u_stats['right_velocity_mean'] - r_stats['right_velocity_mean'])

    def _sev(val, thresholds=(0.1, 0.3, 0.5)):
        a = abs(val)
        if a < thresholds[0]: return 'ok'
        if a < thresholds[1]: return 'info'
        if a < thresholds[2]: return 'warn'
        return 'error'

    issues = []
    severity = []

    # Hand position (combined magnitude)
    left_mag = float(_np.linalg.norm(left_pos_diff))
    right_mag = float(_np.linalg.norm(right_pos_diff))
    if left_mag >= 0.3:
        dir_x = 'LEFT' if left_pos_diff[0] < 0 else 'RIGHT'
        dir_y = 'UP' if left_pos_diff[1] < 0 else 'DOWN'
        issues.append(f"Left hand position differs {left_mag:.2f} shoulder-widths ({dir_x}/{dir_y} of reference)")
        severity.append(_sev(left_mag))
    if right_mag >= 0.3:
        dir_x = 'LEFT' if right_pos_diff[0] < 0 else 'RIGHT'
        dir_y = 'UP' if right_pos_diff[1] < 0 else 'DOWN'
        issues.append(f"Right hand position differs {right_mag:.2f} shoulder-widths ({dir_x}/{dir_y} of reference)")
        severity.append(_sev(right_mag))

    # Hand spread
    if abs(spread_diff) > 0.3:
        direction = 'FARTHER APART' if spread_diff > 0 else 'CLOSER TOGETHER'
        issues.append(f"Your hands are {direction} than reference by {abs(spread_diff):.2f} shoulder-widths")
        severity.append(_sev(spread_diff, (0.2, 0.5, 1.0)))

    # Hand height relative to face
    if abs(left_height_diff) >= 0.3:
        direction = 'LOWER' if left_height_diff > 0 else 'HIGHER'
        issues.append(f"Left hand {abs(left_height_diff):.2f} shoulder-widths {direction} than reference (relative to face)")
        severity.append(_sev(left_height_diff))
    if abs(right_height_diff) >= 0.3:
        direction = 'LOWER' if right_height_diff > 0 else 'HIGHER'
        issues.append(f"Right hand {abs(right_height_diff):.2f} shoulder-widths {direction} than reference (relative to face)")
        severity.append(_sev(right_height_diff))

    # Velocity mismatch — indicates pacing/speed differences
    if abs(vel_diff_l) > 0.02:
        direction = 'faster' if vel_diff_l > 0 else 'slower'
        issues.append(f"Left hand moving {direction} than reference (velocity diff {vel_diff_l:+.3f})")
        severity.append('info')
    if abs(vel_diff_r) > 0.02:
        direction = 'faster' if vel_diff_r > 0 else 'slower'
        issues.append(f"Right hand moving {direction} than reference (velocity diff {vel_diff_r:+.3f})")
        severity.append('info')

    # Duration
    duration_ratio = len(user_pose) / max(len(ref_pose), 1)
    if abs(duration_ratio - 1) > 0.5:
        if duration_ratio < 0.5:
            issues.append(f"Your sign is {duration_ratio:.1f}x reference duration \u2014 signed too fast or capture cut short")
        else:
            issues.append(f"Your sign is {duration_ratio:.1f}x reference duration")
        severity.append('info')

    if not issues:
        issues.append("Pose structure closely matches reference. If model still predicts wrong, the mismatch is in handshape/finger details the model learned.")
        severity.append('ok')

    return {
        "left_hand_position_diff": {"x": float(left_pos_diff[0]), "y": float(left_pos_diff[1]), "magnitude": left_mag},
        "right_hand_position_diff": {"x": float(right_pos_diff[0]), "y": float(right_pos_diff[1]), "magnitude": right_mag},
        "hand_spread_diff": spread_diff,
        "left_hand_height_diff": left_height_diff,
        "right_hand_height_diff": right_height_diff,
        "left_velocity_diff": vel_diff_l,
        "right_velocity_diff": vel_diff_r,
        "duration_ratio": round(duration_ratio, 3),
        "user_stats": {k: (v.tolist() if hasattr(v, 'tolist') else float(v)) for k, v in u_stats.items() if k != 'frames'},
        "reference_stats": {k: (v.tolist() if hasattr(v, 'tolist') else float(v)) for k, v in r_stats.items() if k != 'frames'},
        "issues": issues,
        "severity": severity,
    }


@app.route("/api/practice-diagnose", methods=["POST"])
def practice_diagnose():
    """
    Diagnostic endpoint for practice mode — runs inference AND returns:
      - User's extracted pose (for visualization)
      - A reference training-sample pose for the target gloss (if available)
      - Top-K predictions
    Allows visual comparison of user's signing vs. what the model was trained on.
    """
    import pickle as _pickle
    import random as _random
    import numpy as _np

    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video provided"}), 400

    video_file = request.files["video"]
    video_bytes = video_file.read()
    domain = request.form.get("domain", "doctor_visit")
    target_gloss = request.form.get("target", "").strip().upper()

    if len(video_bytes) == 0:
        return jsonify({"success": False, "error": "Empty video"}), 400

    try:
        if not MEDIAPIPE_AVAILABLE:
            return jsonify({"success": False, "error": "Diagnostic needs MediaPipe fast path"}), 500

        frames = _decode_video_bytes(video_bytes)
        if not frames:
            return jsonify({"success": False, "error": "Could not decode video"}), 400

        # Extract poses at higher resolution for visualization (up to 45 frames).
        full_target = min(len(frames), 45)
        full_user_pose = _extract_poses(frames, target_frames=full_target)
        if full_user_pose is None or len(full_user_pose) == 0:
            return jsonify({"success": False, "error": "No pose detected in your video"}), 400

        # Reject outlier positions (MediaPipe hallucinations), drop sparse
        # false-positive hand detections, then fill remaining interior gaps.
        full_user_pose = _reject_outlier_landmarks(full_user_pose)
        full_user_pose = _drop_sparse_hand_detections(full_user_pose, min_detection_fraction=0.5)
        full_user_pose = _smooth_missing_landmarks(full_user_pose)

        # Stage 3: canonical pose normalization — opt-in for tilted cameras only.
        if os.environ.get('SIGNBRIDGE_CANONICALIZE') == '1':
            full_user_pose, canon_info = _canonicalize_pose(full_user_pose)
            print(f"[Diagnose] Canonicalized: {canon_info}")

        # Auto-trim to the active signing window
        trimmed_pose, trim_start, trim_end, trim_info = _trim_to_motion_window(full_user_pose)
        print(f"[Diagnose] Motion trim: {trim_info}")

        # Downsample trimmed window to 25 frames for model (increased from 15 for more context)
        MODEL_FRAMES = 25
        if len(trimmed_pose) > MODEL_FRAMES:
            step = len(trimmed_pose) / MODEL_FRAMES
            model_indices = [int(i * step) for i in range(MODEL_FRAMES)]
            user_pose = trimmed_pose[model_indices]
        else:
            user_pose = trimmed_pose

        prediction, refine_info = _predict_with_handedness_aware_refinement(user_pose, domain=domain)
        prediction = _apply_confidence_gate(prediction)
        print(f"[Diagnose] Handedness refine: {refine_info}")
        print(f"[Diagnose] Confidence gate: confident={prediction.get('confident')} conf={prediction.get('confidence',0):.2f}")

        # Find a reference training sample for the target gloss
        # Prefer Signer 12 (canonical reference signer across the dataset)
        reference_pose = None
        reference_source = None
        pickle_candidates = []

        if target_gloss:
            signer12_map = _load_signer_12_map()
            video_id = signer12_map.get(target_gloss.lower())
            if video_id:
                s12_path = PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "pickle_files" / f"{video_id}.pkl"
                if s12_path.exists():
                    pickle_candidates.append(s12_path)

            # Fallback: any training sample from gloss-split directories
            candidates = [
                PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "dataset_splits" / "2000_classes" / "original" / "pickle_split_2000_class" / "train" / target_gloss,
                PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "dataset_splits" / "2000_classes" / "original" / "pickle_split_2000_class" / "train" / target_gloss.lower(),
                PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "dataset_splits" / "100_classes" / "original" / "pickle_split_100_class" / "train" / target_gloss.lower(),
                PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "dataset_splits" / "100_classes" / "original" / "pickle_split_100_class" / "train" / target_gloss,
                PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "dataset_splits" / "20_classes" / "original" / "pickle_from_pose_split_20_class" / "train" / target_gloss.lower(),
            ]
            for cand_dir in candidates:
                if cand_dir.is_dir():
                    pkls = sorted(cand_dir.glob("*.pkl"))
                    if pkls:
                        pickle_candidates.append(pkls[len(pkls) // 2])

            for pkl_path in pickle_candidates:
                try:
                    with open(pkl_path, 'rb') as f:
                        ref_data = _pickle.load(f)
                    kp = ref_data.get('keypoints')
                    if kp is None:
                        continue

                    if hasattr(kp, 'filled'):
                        kp = kp.filled(0.0)
                    kp = _np.asarray(kp, dtype=_np.float32)
                    kp = _np.nan_to_num(kp, nan=0.0, posinf=0.0, neginf=0.0)

                    # Training data is (N, 576, 2-3). Extract 83-point subset:
                    # 33 body + 21 left hand + 21 right hand + 8 face
                    if kp.shape[1] == 576:
                        face_indices = [33 + i for i in [1, 61, 291, 152, 107, 336, 33, 263]]
                        indices = list(range(33)) + list(range(501, 522)) + list(range(522, 543)) + face_indices
                        subset = kp[:, indices, :]
                    else:
                        subset = kp

                    if subset.shape[-1] == 2:
                        zeros = _np.zeros((subset.shape[0], subset.shape[1], 1), dtype=_np.float32)
                        subset = _np.concatenate([subset, zeros], axis=-1)

                    reference_pose = subset.tolist()
                    reference_source = str(pkl_path.name)
                    print(f"[Diagnose] Reference loaded: {pkl_path.name} ({len(reference_pose)} frames) for {target_gloss}")
                    break
                except Exception as e:
                    print(f"[Diagnose] Failed to load reference {pkl_path}: {e}")

        # For the playback, trim to the active signing window with a small pad on
        # each side. Otherwise post-sign frames where the user is just holding
        # their hands up (or where MediaPipe is still tracking a slowly-dropping
        # hand) make the playback look like the sign "remains" after it ended.
        DISPLAY_PRE_ROLL = 3
        DISPLAY_POST_ROLL = 3
        n_full = len(full_user_pose)
        if 0 <= trim_start < trim_end <= n_full and (trim_end - trim_start) < n_full:
            ds = max(0, trim_start - DISPLAY_PRE_ROLL)
            de = min(n_full, trim_end + DISPLAY_POST_ROLL)
            display_source = full_user_pose[ds:de]
        else:
            display_source = full_user_pose

        # One Euro filter adapts smoothing to motion velocity (heavy when still,
        # light when fast) so static jitter goes away without smearing sign motion.
        display_pose = _reject_outlier_landmarks(display_source)
        display_pose = _one_euro_smooth_pose(display_pose, fps=15.0)
        # Send 4-channel (x, y, z, visibility) so the client can hide low-confidence bones.
        user_pose_xyzv = display_pose[:, :, :4].tolist()

        # Also smooth the reference pose for the visualization (training data has noise too)
        if reference_pose:
            ref_arr = _np.array(reference_pose, dtype=_np.float32)
            if ref_arr.ndim == 3 and ref_arr.shape[-1] >= 2:
                if ref_arr.shape[-1] == 2:
                    pad = _np.ones((ref_arr.shape[0], ref_arr.shape[1], 2), dtype=_np.float32)
                    ref_for_smooth = _np.concatenate([ref_arr, pad], axis=-1)
                else:
                    pad = _np.ones((ref_arr.shape[0], ref_arr.shape[1], 1), dtype=_np.float32)
                    ref_for_smooth = _np.concatenate([ref_arr, pad], axis=-1)
                smoothed_ref = _reject_outlier_landmarks(ref_for_smooth, threshold=80)  # threshold in pixel space for ref
                smoothed_ref = _temporal_smooth_pose(smoothed_ref, window=5)
                reference_pose = smoothed_ref[:, :, :3].tolist()

        # Debug: log coordinate ranges for both poses to diagnose display issues
        _user_np = full_user_pose[:, :, :2]
        _user_nz = _user_np[_user_np.sum(axis=-1) > 0]
        if len(_user_nz) > 0:
            print(f"[Diagnose] User pose: display={full_user_pose.shape}, model={user_pose.shape}, "
                  f"x=[{_user_nz[:, 0].min():.3f}, {_user_nz[:, 0].max():.3f}], "
                  f"y=[{_user_nz[:, 1].min():.3f}, {_user_nz[:, 1].max():.3f}]")
            ls = full_user_pose[0, 11, :2]
            print(f"[Diagnose] User frame 0 L-shoulder (idx 11): x={ls[0]:.3f}, y={ls[1]:.3f}")
        if reference_pose:
            ref_np = _np.array(reference_pose)
            ref_nz = ref_np[ref_np.sum(axis=-1) > 0][:, :2] if ref_np.shape[-1] >= 2 else None
            if ref_nz is not None and len(ref_nz) > 0:
                print(f"[Diagnose] Ref pose: shape={ref_np.shape}, "
                      f"x=[{ref_nz[:, 0].min():.3f}, {ref_nz[:, 0].max():.3f}], "
                      f"y=[{ref_nz[:, 1].min():.3f}, {ref_nz[:, 1].max():.3f}]")
                ls = ref_np[0, 11, :2]
                print(f"[Diagnose] Ref frame 0 L-shoulder (idx 11): x={ls[0]:.3f}, y={ls[1]:.3f}")

        # Run programmatic pose comparison if we have a reference
        comparison = None
        if reference_pose:
            try:
                comparison = _compare_poses_programmatic(full_user_pose, _np.array(reference_pose))
                if comparison and 'issues' in comparison:
                    print(f"[Diagnose] Comparison issues ({len(comparison['issues'])}):")
                    for sev, msg in zip(comparison.get('severity', []), comparison['issues']):
                        print(f"  [{sev}] {msg}")
            except Exception as e:
                print(f"[Diagnose] Comparison failed: {e}")
                comparison = {"error": str(e)}

        # Persist the diagnostic session to disk for offline analysis
        sample_ts = None
        try:
            samples_dir = PRACTICE_SAMPLES_DIR
            samples_dir.mkdir(exist_ok=True)
            import time as _t
            ts = int(_t.time() * 1000)
            sample_ts = ts
            safe_gloss = target_gloss.replace(' ', '_') if target_gloss else 'UNKNOWN'
            predicted = prediction.get('gloss', '?').replace(' ', '_')
            sample_name = f"{ts}_{safe_gloss}_pred-{predicted}"
            sample_path = samples_dir / f"{sample_name}.pkl"

            with open(sample_path, 'wb') as f:
                _pickle.dump({
                    'timestamp_ms': ts,
                    'target_gloss': target_gloss,
                    'domain': domain,
                    'prediction': prediction,
                    'user_pose_full': full_user_pose,           # (N, 83, 4) display resolution
                    'user_pose_trimmed': trimmed_pose,          # after motion-window trim
                    'user_pose_model': user_pose,               # (15, 83, 4) model input
                    'motion_trim_info': trim_info,
                    'reference_pose': _np.array(reference_pose) if reference_pose else None,
                    'reference_source': reference_source,
                    'comparison': comparison,
                }, f)
            print(f"[Diagnose] Saved sample: {sample_path.relative_to(PROJECT_ROOT).as_posix()}")

            # Also persist the original recording bytes so the pose-video toggle
            # can re-render them on demand without a re-upload from the client.
            video_path_out = samples_dir / f"{sample_name}.webm"
            with open(video_path_out, 'wb') as vf:
                vf.write(video_bytes)
        except Exception as e:
            print(f"[Diagnose] Failed to save sample: {e}")

        # Cap diagnostic history at the most recent N sessions (across all file
        # types: .pkl, .webm, .pose, .pose.mp4). Each session shares a unix-ms
        # timestamp prefix; we keep the newest MAX_PRACTICE_SAMPLES timestamps
        # and delete every file belonging to older ones.
        try:
            samples_dir = PRACTICE_SAMPLES_DIR
            if samples_dir.is_dir():
                MAX_PRACTICE_SAMPLES = 10
                timestamps = sorted({
                    int(p.name.split('_', 1)[0])
                    for p in samples_dir.iterdir()
                    if p.is_file() and p.name and p.name[0].isdigit() and '_' in p.name
                }, reverse=True)
                stale = set(timestamps[MAX_PRACTICE_SAMPLES:])
                if stale:
                    removed = 0
                    for p in samples_dir.iterdir():
                        if not p.is_file():
                            continue
                        try:
                            ts_prefix = int(p.name.split('_', 1)[0])
                        except (ValueError, IndexError):
                            continue
                        if ts_prefix in stale:
                            p.unlink(missing_ok=True)
                            removed += 1
                    if removed:
                        print(f"[Diagnose] Pruned {removed} files from {len(stale)} old practice sessions")
        except Exception as e:
            print(f"[Diagnose] Sample cleanup failed: {e}")

        return jsonify({
            "success": True,
            "gloss": prediction.get("gloss", "?"),
            "confidence": prediction.get("confidence", 0),
            "top_k": prediction.get("top_k_predictions", [])[:5],
            "user_pose": user_pose_xyzv,
            "user_frames": len(user_pose_xyzv),
            "reference_pose": reference_pose,
            "reference_source": reference_source,
            "reference_frames": len(reference_pose) if reference_pose else 0,
            "target": target_gloss,
            "comparison": comparison,
            "sample_id": sample_ts,
            "motion_trim": {
                "start": trim_start,
                "end": trim_end,
                "info": trim_info,
            },
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/practice-pose-video/<int:sample_id>", methods=["GET"])
def practice_pose_video(sample_id):
    """
    Render a saved practice clip as a pose-only stick-figure video using the
    pose-format library's CLI binaries (same approach as the show-and-tell app).

    Pipeline: original .webm -> .pose (video_to_pose.exe)
              .pose          -> raw .mp4 at ~1000fps (visualize_pose.exe)
              raw .mp4       -> 30fps H.264 .mp4 (ffmpeg)

    The first call for a given sample renders the cached .pose.mp4; subsequent
    calls return the cache directly.
    """
    samples_dir = PRACTICE_SAMPLES_DIR
    if not samples_dir.is_dir():
        return jsonify({"success": False, "error": "No practice samples directory"}), 404

    matches = list(samples_dir.glob(f"{sample_id}_*.webm")) + list(samples_dir.glob(f"{sample_id}_*.mp4"))
    if not matches:
        return jsonify({"success": False, "error": f"Sample {sample_id} not found"}), 404
    src_video = matches[0]

    # Cache hit?
    cached = src_video.with_name(src_video.stem + ".pose.mp4")
    if cached.exists() and cached.stat().st_size > 0:
        return send_file(str(cached), mimetype="video/mp4", conditional=True)

    # Sanity-check binaries
    if not VIDEO_TO_POSE_EXE.exists():
        return jsonify({"success": False, "error": f"video_to_pose.exe not found at {VIDEO_TO_POSE_EXE}"}), 500
    if not VISUALIZE_POSE_EXE.exists():
        return jsonify({"success": False, "error": f"visualize_pose.exe not found at {VISUALIZE_POSE_EXE}"}), 500
    ffmpeg_path = _find_ffmpeg()
    if not ffmpeg_path:
        return jsonify({"success": False, "error": "ffmpeg not available"}), 500

    pose_path = src_video.with_name(src_video.stem + ".pose")
    raw_mp4 = src_video.with_name(src_video.stem + ".pose.raw.mp4")

    try:
        # Step 1: video_to_pose
        r = subprocess.run(
            [str(VIDEO_TO_POSE_EXE), "-i", str(src_video), "-o", str(pose_path), "--format", "mediapipe"],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0 or not pose_path.exists():
            return jsonify({"success": False, "error": f"video_to_pose failed: {r.stderr[:300]}"}), 500

        # Step 2: visualize_pose
        r = subprocess.run(
            [str(VISUALIZE_POSE_EXE), "-i", str(pose_path), "-o", str(raw_mp4), "--normalize"],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0 or not raw_mp4.exists():
            return jsonify({"success": False, "error": f"visualize_pose failed: {r.stderr[:300]}"}), 500

        # Step 3: resample to 30fps H.264 so browsers can play it
        r = subprocess.run(
            [
                ffmpeg_path, "-y", "-i", str(raw_mp4),
                "-vf", "fps=30",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(cached),
            ],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0 or not cached.exists():
            return jsonify({"success": False, "error": f"ffmpeg resample failed: {r.stderr[:300]}"}), 500

    finally:
        # Clean up intermediates regardless of outcome
        try: pose_path.unlink(missing_ok=True)
        except Exception: pass
        try: raw_mp4.unlink(missing_ok=True)
        except Exception: pass

    return send_file(str(cached), mimetype="video/mp4", conditional=True)


@app.route("/api/process-signing-clip", methods=["POST"])
def process_signing_clip():
    """
    Process a continuous signing clip (e.g. 5-15 seconds) and return a list of
    gloss predictions with frame timestamps.

    Fast path: HybridSegmenter for motion-based boundaries (pixel diff, ~cheap),
    then per-segment MediaPipe Holistic + direct in-process inference. No
    subprocess, no HTTP round-trip, no temp files beyond the MP4 conversion
    ffmpeg needs.

    Response:
        { "success": True,
          "glosses": [ {gloss, confidence, top_k, start_frame, end_frame, confident, status}, ... ],
          "segment_count": N,
          "total_time": sec,
          "fps": float
        }
    """
    import time as _time
    from pathlib import Path as _Path

    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video provided"}), 400

    video_file = request.files["video"]
    video_bytes = video_file.read()
    domain = request.form.get("domain", "doctor_visit")
    if len(video_bytes) == 0:
        return jsonify({"success": False, "error": "Empty video"}), 400

    print(f"[Clip] domain={domain} size={len(video_bytes)//1024}KB")

    if not MEDIAPIPE_AVAILABLE:
        return jsonify({"success": False, "error": "Fast path requires MediaPipe"}), 500

    t0 = _time.time()
    tmp_dir = _Path(tempfile.mkdtemp(prefix="signbridge_clip_"))
    try:
        webm_path = tmp_dir / "clip.webm"
        mp4_path = tmp_dir / "clip.mp4"
        with open(webm_path, 'wb') as f:
            f.write(video_bytes)

        processor = _get_camera_processor()
        if processor._convert_webm_to_mp4(str(webm_path), str(mp4_path)):
            video_path = str(mp4_path)
        else:
            video_path = str(webm_path)

        t_ffmpeg = _time.time()

        # Step 1: HybridSegmenter on the video (pixel-based motion, cheap)
        sys.path.insert(0, str(PROJECT_UTILITIES_DIR / "segmentation"))
        from hybrid_segmenter import HybridSegmenter
        segmenter = HybridSegmenter()
        segments = segmenter.detect_segments_from_video(video_path, verbose=False)

        t_segment = _time.time()
        print(f"[Clip] ffmpeg={t_ffmpeg-t0:.1f}s segment={t_segment-t_ffmpeg:.1f}s -> {len(segments)} segments")

        if not segments:
            return jsonify({"success": True, "glosses": [], "segment_count": 0, "message": "No signs detected"})

        # Step 2: Read all frames once, get fps
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        all_frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            all_frames.append(frame)
        cap.release()

        if not all_frames:
            return jsonify({"success": False, "error": "Could not read video frames"}), 500

        t_decode = _time.time()
        print(f"[Clip] decoded {len(all_frames)} frames in {t_decode-t_segment:.1f}s (fps={fps:.1f})")

        # Step 3: Per-segment pose extraction + inference (using the same pipeline as /api/process-sign)
        predictions = []
        t_inf_start = _time.time()
        for i, (start, end) in enumerate(segments):
            seg_frames = all_frames[start:end + 1]
            if not seg_frames:
                continue

            # Downsample input to ~20 frames for pose extraction (keeps inference fast)
            target = min(len(seg_frames), 20)
            pose = _extract_poses(seg_frames, target_frames=target)
            if pose is None or len(pose) == 0:
                continue

            # Apply the same cleaning pipeline as the live single-sign path
            pose = _reject_outlier_landmarks(pose)
            pose = _smooth_missing_landmarks(pose)
            trimmed, _, _, _ = _trim_to_motion_window(pose)

            MODEL_FRAMES = 15
            if len(trimmed) > MODEL_FRAMES:
                step = len(trimmed) / MODEL_FRAMES
                idx = [int(j * step) for j in range(MODEL_FRAMES)]
                model_in = trimmed[idx]
            else:
                model_in = trimmed

            pred, refine = _predict_with_handedness_aware_refinement(model_in, domain=domain)
            pred = _apply_confidence_gate(pred)
            predictions.append({
                "gloss": pred.get("gloss", "?"),
                "confidence": pred.get("confidence", 0),
                "top_k": pred.get("top_k_predictions", [])[:3],
                "start_frame": start,
                "end_frame": end,
                "start_sec": round(start / fps, 2),
                "end_sec": round(end / fps, 2),
                "confident": pred.get("confident", True),
                "status": pred.get("status", "ok"),
                "refined": refine.get("refined", False),
            })
            print(f"  Seg {i+1}/{len(segments)} [{start}:{end}]: {pred.get('gloss','?')} ({pred.get('confidence',0)*100:.0f}%) "
                  f"{'[REF]' if refine.get('refined') else ''} {'(unclear)' if not pred.get('confident', True) else ''}")

        t_done = _time.time()
        print(f"[Clip] inference {len(predictions)} segments: {t_done-t_inf_start:.1f}s | total: {t_done-t0:.1f}s")

        return jsonify({
            "success": True,
            "glosses": predictions,
            "segment_count": len(predictions),
            "total_time": round(t_done - t0, 1),
            "fps": round(fps, 1),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)


@app.route("/api/construct-sentence-live", methods=["POST"])
def construct_sentence_live():
    """
    Construct a sentence from detected glosses using the closed-captions
    LLM prompt template (live mode).
    """
    try:
        data = request.get_json()
        gloss_predictions = data.get("gloss_predictions", [])
        conversation_history = data.get("conversation_history", [])
        # Running-caption mode: the client supplies its prior caption for this
        # utterance and ONLY the new glosses since that caption was generated.
        # The LLM extends rather than rebuilds. Empty/missing means "first call
        # of a new utterance, build from scratch".
        running_caption = (data.get("running_caption") or "").strip()
        # Regenerate hint: when the user clicked the Regenerate (🔄) icon on a
        # low-quality caption, the client sends the previous output here so the
        # LLM is told to try a different interpretation. Triggers the FRESH path
        # (running_caption is sent empty alongside) -- the previous attempt is
        # the *rejected* answer, not the prior caption to extend.
        previous_attempt = (data.get("previous_attempt") or "").strip()
        # Domain key (e.g. "emergency", "doctor_visit"). Fed to the prompt
        # so the LLM has the right semantic prior + vocabulary scope.
        domain = (data.get("domain") or "emergency").strip()

        if not gloss_predictions:
            return jsonify({"success": False, "error": "No gloss predictions"}), 400

        # Build context section
        context_lines = []
        for msg in conversation_history:
            speaker = msg.get("speaker", "unknown").capitalize()
            text = msg.get("text", "")
            context_lines.append(f"{speaker}: {text}")
        context_str = "\n".join(context_lines) if context_lines else "No prior conversation."

        # Build gloss details with top-k predictions
        gloss_details_lines = []
        for i, pred in enumerate(gloss_predictions):
            gloss_details_lines.append(f"Position {i + 1}:")
            top_k = pred.get("top_k", [])
            if top_k:
                for j, option in enumerate(top_k[:3]):
                    conf = option.get("confidence", 0) * 100
                    gloss_details_lines.append(
                        f"  Option {j + 1}: '{option['gloss']}' (confidence: {conf:.1f}%)"
                    )
            else:
                conf = pred.get("confidence", 0) * 100
                gloss_details_lines.append(
                    f"  Option 1: '{pred['gloss']}' (confidence: {conf:.1f}%)"
                )
        gloss_details = "\n".join(gloss_details_lines)
        if running_caption:
            print(f"[Live LLM] EXTEND mode. Prior caption: \"{running_caption}\"")
            print(f"[Live LLM] NEW glosses ({len(gloss_predictions)}):\n{gloss_details}")
        else:
            print(f"[Live LLM] FRESH start. Glosses ({len(gloss_predictions)}):\n{gloss_details}")

        # Build the running-caption section (extend mode marker).
        if running_caption:
            running_section = (
                f"Running caption (you generated this earlier for the current "
                f"utterance — extend or refine it with the new signs below):\n"
                f"    \"{running_caption}\""
            )
        else:
            running_section = ""

        # Build the regenerate section. When set, the user rejected our prior
        # attempt; we instruct the LLM to produce a meaningfully different
        # interpretation rather than recreating the same output.
        if previous_attempt:
            regenerate_section = (
                "═══════════════════════════════════════\n"
                "REGENERATE MODE — IMPORTANT\n"
                "═══════════════════════════════════════\n"
                f"Your previous attempt for these signs was:\n"
                f"    \"{previous_attempt}\"\n"
                "The user rejected it as low quality. You MUST produce a "
                "MEANINGFULLY DIFFERENT sentence — not paraphrasing, not "
                "synonyms of the same words.\n"
                "\n"
                "ACTIVELY EXPLORE THE TOP-2 AND TOP-3 ALTERNATES:\n"
                "- For each position, the Option 2 and Option 3 entries below "
                "  are real possibilities — the model wasn't sure. Try them.\n"
                "- The Option 1 word does NOT need to be in your output. If it "
                "  doesn't fit, swap it for an Option 2 or Option 3 word that "
                "  forms a more sensible sentence with the other positions.\n"
                "- It is OK to use a 0.5%-confidence alternate if it produces "
                "  a coherent sentence; the previous run already failed with "
                "  the top-1 picks.\n"
                "\n"
                "ALSO TRY A DIFFERENT SYNTACTIC STRUCTURE:\n"
                "- If your previous attempt was a declarative, try imperative.\n"
                "- If your previous attempt used the topic as subject, try the\n"
                "  comment as subject (or vice versa).\n"
                "- Consider a fragment / interjection if the data warrants it.\n"
                "\n"
                "DO NOT repeat the previous wording or its near-synonyms. If "
                "no different coherent sentence is achievable, output a shorter "
                "fragment that uses fewer (but coherent) words and set "
                "low_confidence: true."
            )
        else:
            regenerate_section = ""

        # Domain context section (label + description + vocabulary).
        domain_section = _build_domain_section(domain)

        # Try loading the prompt template
        prompt_template = _load_llm_prompt()
        if prompt_template:
            context_section = f"Conversation context:\n{context_str}"
            prompt = prompt_template.replace("{context_section}", context_section)
            prompt = prompt.replace("{running_caption_section}", running_section)
            prompt = prompt.replace("{regenerate_hint_section}", regenerate_section)
            prompt = prompt.replace("{domain_section}", domain_section)
            prompt = prompt.replace("{gloss_details}", gloss_details)
        else:
            # Fallback to inline prompt
            raw_glosses = " ".join(p["gloss"] for p in gloss_predictions)
            prompt = f"""You are translating ASL glosses into natural English for a Deaf patient communicating with a doctor.

Conversation so far:
{context_str}

The patient just signed these ASL glosses: {raw_glosses}

Convert these glosses into a natural, grammatically correct English sentence.
Output ONLY the English sentence, nothing else.

English sentence:"""

        llm = create_llm_provider(
            provider="googleaistudio",
            model_name="gemini-2.0-flash",
            max_tokens=150,
            timeout=15,
        )
        response = llm.generate(prompt)

        # Parse response — prompt template expects JSON with "sentence" key
        generated = response.strip()
        # Strip markdown code fences if present (```json ... ```)
        if generated.startswith('```'):
            lines = generated.split('\n')
            # Remove first line (```json) and last line (```)
            lines = [l for l in lines if not l.strip().startswith('```')]
            generated = '\n'.join(lines).strip()

        sentence = generated  # fallback
        plausibility_overall = None
        parsed_json = None
        try:
            # Fix double braces that LLM sometimes outputs from template patterns
            fixed = generated.replace('{{', '{').replace('}}', '}')
            parsed_json = json.loads(fixed)
            sentence = parsed_json.get("sentence", parsed_json.get("revised_sentence", generated))

            # Extract the LLM's self-critique plausibility (3 sub-scores).
            # Geometric mean matches the prompt's stated "overall" semantics
            # and is more conservative than arithmetic mean for live CTQI.
            pl = parsed_json.get("plausibility")
            if isinstance(pl, dict):
                g = float(pl.get("grammatical", 0) or 0)
                s = float(pl.get("semantic", 0) or 0)
                n = float(pl.get("naturalness", 0) or 0)
                if g > 0 and s > 0 and n > 0:
                    plausibility_overall = (g * s * n) ** (1.0 / 3.0)
        except (json.JSONDecodeError, TypeError):
            # Plain text response
            if sentence.startswith('"') and sentence.endswith('"'):
                sentence = sentence[1:-1]
            # Try to extract sentence from malformed JSON-like output
            import re
            m = re.search(r'"sentence"\s*:\s*"([^"]+)"', generated)
            if m:
                sentence = m.group(1)

        if plausibility_overall is not None:
            print(f"[Live LLM] Glosses: {[p['gloss'] for p in gloss_predictions]} -> {sentence}  [P={plausibility_overall:.1f}]")
        else:
            # Help diagnose missing plausibility — log what we got back from the LLM
            # so we can see whether the schema was violated.
            if isinstance(parsed_json, dict):
                print(f"[Live LLM] Glosses: {[p['gloss'] for p in gloss_predictions]} -> {sentence}  [P=missing; JSON keys={list(parsed_json.keys())}]")
            else:
                print(f"[Live LLM] Glosses: {[p['gloss'] for p in gloss_predictions]} -> {sentence}  [P=missing; non-JSON response]")
                print(f"[Live LLM] Raw response (first 300 chars): {generated[:300]!r}")

        return jsonify({
            "success": True,
            "sentence": sentence,
            # Plausibility (0-100) from the LLM's self-critique. Client uses
            # this with avg gloss confidence to compute a live CTQI estimate.
            "plausibility": plausibility_overall,
        })

    except Exception as e:
        import traceback
        print(f"[Live LLM] ERROR: {e}")
        traceback.print_exc()
        glosses = [p.get("gloss", "") for p in data.get("gloss_predictions", [])]
        return jsonify({"success": False, "error": str(e), "fallback": " ".join(glosses)}), 500


# ============================================================================
# ROUTES — Caption Video (upload → pipeline → download)
# ============================================================================

# In-memory job store: { job_id: { status, progress, message, output_path, error } }
CAPTION_JOBS = {}
CAPTION_JOBS_DIR = Path(tempfile.gettempdir()) / "signbridge_caption_jobs"
CAPTION_JOBS_DIR.mkdir(exist_ok=True)

CLOSED_CAPTIONS_DIR = Path(__file__).parent.parent / "closed-captions"


def _run_caption_job(job_id: str, input_path: str):
    """Run the caption_video pipeline in a background thread."""
    job = CAPTION_JOBS[job_id]
    output_path = str(CAPTION_JOBS_DIR / job_id / "captioned.mp4")
    Path(output_path).parent.mkdir(exist_ok=True)

    try:
        # Import caption_video inline so it only loads heavy libs when needed
        sys.path.insert(0, str(CLOSED_CAPTIONS_DIR))
        from caption_video import caption_video as _caption_video

        # Monkey-patch progress updates into the print output by wrapping
        import builtins
        original_print = builtins.print

        def progress_print(*args, **kwargs):
            msg = " ".join(str(a) for a in args)
            original_print(*args, **kwargs)
            # Map pipeline step messages to progress %
            if "[1/5]" in msg:
                job.update(progress=10, message="Converting video to pose...")
            elif "[2/5]" in msg:
                job.update(progress=30, message="Segmenting signs...")
            elif "[3/5]" in msg:
                job.update(progress=55, message="Running model inference...")
            elif "[4/5]" in msg:
                job.update(progress=75, message="Building captions with LLM...")
            elif "[5/5]" in msg:
                job.update(progress=90, message="Burning captions onto video...")

        builtins.print = progress_print
        try:
            _caption_video(input_path, output_path)
        finally:
            builtins.print = original_print

        job.update(status="done", progress=100, message="Done!", output_path=output_path)

    except Exception as e:
        import traceback
        traceback.print_exc()
        job.update(status="error", message=str(e))
    finally:
        # Clean up the uploaded input
        try:
            os.remove(input_path)
        except Exception:
            pass


class _Job(dict):
    """Simple thread-safe job state dict."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lock = threading.Lock()

    def update(self, **kwargs):
        with self._lock:
            super().update(kwargs)


@app.route("/api/caption-video", methods=["POST"])
def caption_video_upload():
    """Accept video upload, start background captioning job, return job_id."""
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video file provided"}), 400

    video_file = request.files["video"]
    if not video_file.filename:
        return jsonify({"success": False, "error": "Empty filename"}), 400

    job_id = str(uuid.uuid4())

    # Save upload to temp file (keep extension)
    ext = Path(video_file.filename).suffix or ".mp4"
    input_path = str(CAPTION_JOBS_DIR / f"{job_id}_input{ext}")
    video_file.save(input_path)

    # Create job and start thread
    job = _Job(status="running", progress=0, message="Starting pipeline...",
               output_path=None, error=None)
    CAPTION_JOBS[job_id] = job

    t = threading.Thread(target=_run_caption_job, args=(job_id, input_path), daemon=True)
    t.start()

    return jsonify({"success": True, "job_id": job_id})


@app.route("/api/caption-video/<job_id>", methods=["GET"])
def caption_video_status(job_id):
    """Poll job status."""
    job = CAPTION_JOBS.get(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404
    return jsonify({"success": True, **{k: v for k, v in job.items() if k != "_lock"}})


@app.route("/api/caption-video/<job_id>/download", methods=["GET"])
def caption_video_download(job_id):
    """Download the captioned video once the job is done."""
    job = CAPTION_JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job.get("status") != "done":
        return jsonify({"error": "Job not complete"}), 400
    output_path = job.get("output_path")
    if not output_path or not Path(output_path).exists():
        return jsonify({"error": "Output file missing"}), 500
    return send_file(output_path, mimetype="video/mp4",
                     as_attachment=True, download_name="captioned.mp4")


# ============================================================================
# ROUTES — External Camera Proxy (MJPEG)
# ============================================================================

@app.route("/api/camera-proxy")
def camera_proxy():
    """
    Proxy an MJPEG stream from an external camera (ESP32-CAM, DroidCam, etc.)
    through the Flask server. This solves HTTPS mixed-content issues since the
    browser loads from same-origin instead of directly from an HTTP camera.

    Usage: /api/camera-proxy?url=http://192.168.1.100:81/stream
    """
    import ipaddress
    import requests as http_requests

    url = request.args.get("url", "").strip()
    if not url:
        return jsonify({"error": "Missing 'url' parameter"}), 400

    # Security: only allow local network addresses
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        hostname = parsed.hostname
        if hostname:
            addr = ipaddress.ip_address(hostname)
            if not addr.is_private and not addr.is_loopback:
                return jsonify({"error": "Only local network cameras allowed"}), 403
    except (ValueError, TypeError):
        # Hostname might be a name like 'esp32-cam.local' — allow mDNS names
        if not (hostname and (hostname.endswith('.local') or hostname.startswith('192.168.')
                or hostname.startswith('10.') or hostname.startswith('172.'))):
            return jsonify({"error": "Only local network cameras allowed"}), 403

    # Open the upstream GET first so we can copy its real Content-Type (which
    # carries the multipart boundary param). HEAD doesn't work on esp-http-server
    # since the firmware only registers GET handlers -- it returns 404+text/html,
    # which would then poison the proxy response and break the browser <img>.
    try:
        upstream = http_requests.get(url, stream=True, timeout=10)
    except http_requests.RequestException as e:
        print(f"[CameraProxy] Upstream connect failed: {e}")
        return jsonify({"error": f"Upstream camera unreachable: {e}"}), 502

    content_type = upstream.headers.get('Content-Type',
                                        'multipart/x-mixed-replace; boundary=frame')

    def stream_mjpeg():
        try:
            for chunk in upstream.iter_content(chunk_size=4096):
                if not chunk:
                    break
                yield chunk
        except (http_requests.RequestException, GeneratorExit) as e:
            print(f"[CameraProxy] Stream ended: {e}")
        except Exception as e:
            print(f"[CameraProxy] Unexpected error: {e}")
        finally:
            try:
                upstream.close()
            except Exception:
                pass

    resp = Response(stream_mjpeg(), content_type=content_type)
    # The <img crossOrigin='anonymous'> in the page forces CORS checks even on
    # same-origin loads; make the response explicitly cross-origin-safe.
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Cache-Control'] = 'no-store'
    return resp


# ============================================================================
# ROUTES — Static file serving
# ============================================================================

@app.route("/cert")
def download_cert():
    """Download the SSL certificate for mobile trust."""
    cert_path = Path(__file__).parent / "cert.pem"
    if cert_path.exists():
        return send_from_directory(Path(__file__).parent, "cert.pem",
                                  mimetype="application/x-pem-file",
                                  as_attachment=True,
                                  download_name="signbridge.pem")
    return "No certificate found", 404


@app.route("/sign-bank/<path:filename>")
def serve_sign_bank(filename):
    """Serve sign-bank video files."""
    return send_from_directory(SIGN_BANK_DIR, filename)


@app.route("/samples/<sample_id>/original_video.mp4")
def serve_sample_video(sample_id):
    """Serve full-sentence original video from demo-data samples."""
    sample_dir = SAMPLES_DIR / sample_id
    video_path = sample_dir / "original_video.mp4"
    if video_path.exists():
        return send_from_directory(str(sample_dir), "original_video.mp4")
    print(f"[Samples] Not found: {video_path}")
    return "Sample video not found", 404


@app.route("/api/samples/<sample_id>")
def get_sample_metadata(sample_id):
    """Return metadata for a breakdown sample."""
    metadata_path = SAMPLES_DIR / sample_id / "metadata.json"
    if not metadata_path.exists():
        return jsonify({"error": "Sample not found"}), 404
    with open(metadata_path) as f:
        return jsonify(json.load(f))


@app.route("/demo-data/samples/<path:filepath>")
def serve_sample_file(filepath):
    """Serve media files from demo-data samples directory."""
    return send_from_directory(SAMPLES_DIR, filepath)


# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SignBridge App")
    parser.add_argument("--mode", choices=["demo", "live"], default="demo",
                        help="demo = scripted sign videos, live = real-time inference via API")
    parser.add_argument("--method", choices=["sign", "speak"], default="speak",
                        help="sign = user communicates in ASL, speak = user communicates via speech")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    APP_MODE = args.mode
    APP_METHOD = args.method

    print(f"\n  SignBridge App")
    print(f"  Mode:      {APP_MODE}")
    print(f"  Method:    {APP_METHOD}")
    print(f"  Sign bank: {SIGN_BANK_DIR}")
    if APP_MODE == "live":
        print(f"  Inference: {INFERENCE_API_URL}")
        print(f"  LLM prompt: {LLM_PROMPT_PATH}")
    print(f"  API base:  {API_BASE_URL or 'local'}\n")

    import ssl
    cert_dir = Path(__file__).parent
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"

    if cert_file.exists() and key_file.exists():
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(cert_file), str(key_file))
        print("  Running with HTTPS (speech recognition enabled)\n")
        app.run(host="0.0.0.0", port=args.port, debug=True, ssl_context=context)
    else:
        print("  WARNING: No SSL certs found — speech recognition won't work on mobile\n")
        app.run(host="0.0.0.0", port=args.port, debug=True)
