#!/usr/bin/env python3
"""
Compare two ASL video processing pipelines:
1. Pose-First: Video → Full Pose → Segment Poses → Trim → Predict
2. Video-First: Video → Segment Videos → MediaPipe Holistic (pixel coords) → Predict
"""

import sys
import os
import cv2
import numpy as np
import tempfile
import pickle

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(APP_DIR))

sys.path.insert(0, APP_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'openhands-modernized', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'openhands-modernized', 'src', 'util'))

from pose_format import Pose
from openhands_modernized_inference import predict_pose_file, load_model_from_checkpoint

# Video file to test
VIDEO_PATH = r"C:\Users\ashwi\Downloads\IMG_1481.MOV"
EXPECTED_SIGNS = ["CHAIR", "BLACK", "TABLE"]

# Model paths
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'openhands-modernized', 'production-models', 'wlasl_50_class_model')

print("=" * 70)
print("ASL PIPELINE COMPARISON TEST")
print("=" * 70)
print(f"Video: {VIDEO_PATH}")
print(f"Expected signs: {EXPECTED_SIGNS}")
print()

# Load the model once
print("Loading model...")
model, label_map = load_model_from_checkpoint(MODEL_DIR)
print(f"Model loaded with {len(label_map)} classes")
print()

# Get video info
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print(f"Video info: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")
print(f"Duration: {total_frames/fps:.2f} seconds")
print()

# Helper function to convert pose to pickle
def pose_to_pickle(pose_data, output_path):
    """Convert pose data (frames, points, 2) to pickle format."""
    # Ensure we have the right shape
    if len(pose_data.shape) == 4:
        pose_data = pose_data[:, 0, :, :]  # Remove person dimension

    # Extract 75 points if needed
    if pose_data.shape[1] == 543:
        pose_75pt = np.concatenate([
            pose_data[:, 0:33, :],      # Pose landmarks
            pose_data[:, 501:522, :],   # Left hand landmarks
            pose_data[:, 522:543, :]    # Right hand landmarks
        ], axis=1)
    elif pose_data.shape[1] == 576:
        pose_75pt = np.concatenate([
            pose_data[:, 0:33, :],
            pose_data[:, 501:522, :],
            pose_data[:, 522:543, :]
        ], axis=1)
    elif pose_data.shape[1] == 75:
        pose_75pt = pose_data
    else:
        pose_75pt = pose_data

    keypoints = np.array(pose_75pt[:, :, :2])

    pickle_data = {
        'keypoints': keypoints,
        'confidences': np.ones(keypoints.shape[:2]),
        'gloss': 'UNKNOWN'
    }

    with open(output_path, 'wb') as f:
        pickle.dump(pickle_data, f)

    return output_path

# ============================================================================
# APPROACH 1: Pose-First Pipeline
# ============================================================================
print("=" * 70)
print("APPROACH 1: POSE-FIRST PIPELINE")
print("Video -> Full Pose -> Segment Poses -> Trim -> Predict")
print("=" * 70)

# Step 1: Convert full video to pose
print("\n[Step 1] Converting full video to pose file...")
pose_output = os.path.join(tempfile.gettempdir(), "pipeline_test_full.pose")

import subprocess
video_to_pose_cmd = [
    r"C:\Users\ashwi\Projects\WLASL-proj\asl-v1\venv\Scripts\video_to_pose.exe",
    "--format", "mediapipe",
    "-i", VIDEO_PATH,
    "-o", pose_output
]

result = subprocess.run(video_to_pose_cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"Error: {result.stderr}")
    sys.exit(1)

# Load the pose
with open(pose_output, "rb") as f:
    full_pose = Pose.read(f.read())

print(f"Full pose: {full_pose.body.data.shape} (frames, people, points, dims)")

# Step 2: Segment using motion detection on pose data
print("\n[Step 2] Segmenting pose using motion detection...")

def calculate_pose_motion(pose_data):
    """Calculate motion between consecutive frames."""
    if len(pose_data) < 2:
        return np.array([0])

    motion = []
    for i in range(1, len(pose_data)):
        diff = np.abs(pose_data[i] - pose_data[i-1])
        mask = (pose_data[i] != 0) & (pose_data[i-1] != 0)
        if np.any(mask):
            motion.append(np.mean(diff[mask]))
        else:
            motion.append(0)
    return np.array([0] + motion)

def segment_pose_by_motion(pose, min_frames=15, cooldown=10, motion_threshold_factor=1.5):
    """Segment pose based on motion detection."""
    data = pose.body.data[:, 0, :, :]  # (frames, points, dims)
    motion = calculate_pose_motion(data)

    threshold = np.mean(motion) * motion_threshold_factor

    segments = []
    in_sign = False
    start_frame = 0
    cooldown_counter = 0

    for i, m in enumerate(motion):
        if not in_sign:
            if m > threshold:
                in_sign = True
                start_frame = i
                cooldown_counter = 0
        else:
            if m < threshold:
                cooldown_counter += 1
                if cooldown_counter >= cooldown:
                    end_frame = i - cooldown
                    if end_frame - start_frame >= min_frames:
                        segments.append((start_frame, end_frame))
                    in_sign = False
                    cooldown_counter = 0
            else:
                cooldown_counter = 0

    if in_sign and len(motion) - start_frame >= min_frames:
        segments.append((start_frame, len(motion)))

    return segments, motion, threshold

segments_pose, motion, threshold = segment_pose_by_motion(full_pose)
print(f"Found {len(segments_pose)} segments: {segments_pose}")

# Step 3: Predict each segment
print("\n[Step 3] Predicting each segment...")

pose_first_results = []
for i, (start, end) in enumerate(segments_pose):
    # Extract segment data directly from pose body data
    segment_data = full_pose.body.data[start:end]

    # Save as pickle
    segment_file = os.path.join(tempfile.gettempdir(), f"pose_first_seg_{i}.pkl")
    pose_to_pickle(segment_data, segment_file)

    result = predict_pose_file(segment_file, model, label_map)
    predictions = result['top_k_predictions']
    pose_first_results.append({
        "segment": i + 1,
        "frames": f"{start}-{end}",
        "num_frames": end - start,
        "predictions": predictions
    })

    top_pred = predictions[0]
    print(f"  Segment {i+1} (frames {start}-{end}): "
          f"{top_pred['gloss']} ({top_pred['confidence']*100:.1f}%)")

# ============================================================================
# APPROACH 2: Video-First Pipeline with MediaPipe Holistic
# ============================================================================
print("\n" + "=" * 70)
print("APPROACH 2: VIDEO-FIRST PIPELINE (MediaPipe Holistic + Pixel Coords)")
print("Video -> Motion Segment -> MediaPipe Holistic -> Predict")
print("=" * 70)

import mediapipe as mp

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Step 1: Segment video using motion detection
print("\n[Step 1] Segmenting video using motion detection...")

def segment_video_by_motion(video_path, motion_threshold_scale=0.5, min_frames=15, cooldown=12):
    """Segment video based on pixel-level motion detection."""
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale threshold based on resolution
    base_threshold = 2000000
    base_pixels = 1920 * 1080
    actual_pixels = width * height
    scaled_threshold = base_threshold * (actual_pixels / base_pixels) * motion_threshold_scale

    prev_gray = None
    segments = []
    motion_scores = []

    in_sign = False
    start_frame = 0
    cooldown_counter = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is not None:
            frame_delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            motion_score = np.sum(thresh)
        else:
            motion_score = 0

        motion_scores.append(motion_score)
        is_active = motion_score > scaled_threshold

        if not in_sign:
            if is_active:
                in_sign = True
                start_frame = frame_idx
                cooldown_counter = 0
        else:
            if not is_active:
                cooldown_counter += 1
                if cooldown_counter >= cooldown:
                    end_frame = frame_idx - cooldown
                    if end_frame - start_frame >= min_frames:
                        segments.append((start_frame, end_frame))
                    in_sign = False
                    cooldown_counter = 0
            else:
                cooldown_counter = 0

        prev_gray = gray
        frame_idx += 1

    if in_sign and frame_idx - start_frame >= min_frames:
        segments.append((start_frame, frame_idx))

    cap.release()
    return segments, motion_scores, scaled_threshold

segments_video, motion_scores, vid_threshold = segment_video_by_motion(VIDEO_PATH)
print(f"Found {len(segments_video)} segments: {segments_video}")

# Step 2: Process each segment with MediaPipe Holistic (pixel coordinates)
print("\n[Step 2] Processing segments with MediaPipe Holistic (pixel coords)...")

def process_video_segment_mediapipe(video_path, start_frame, end_frame):
    """Extract pose from video segment using MediaPipe Holistic with pixel coordinates."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_frames = []

    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        # Build pose array: 33 pose + 21 left hand + 21 right hand = 75 points
        frame_landmarks = []

        # Pose landmarks (33 points) - convert to pixel coordinates
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                frame_landmarks.append([lm.x * width, lm.y * height])
        else:
            frame_landmarks.extend([[0, 0]] * 33)

        # Left hand (21 points)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                frame_landmarks.append([lm.x * width, lm.y * height])
        else:
            frame_landmarks.extend([[0, 0]] * 21)

        # Right hand (21 points)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                frame_landmarks.append([lm.x * width, lm.y * height])
        else:
            frame_landmarks.extend([[0, 0]] * 21)

        all_frames.append(frame_landmarks)

    cap.release()
    return np.array(all_frames), width, height

video_first_results = []
for i, (start, end) in enumerate(segments_video):
    landmarks, w, h = process_video_segment_mediapipe(VIDEO_PATH, start, end)

    if len(landmarks) < 5:
        print(f"  Segment {i+1}: Too few frames ({len(landmarks)}), skipping")
        continue

    # Save as pickle directly (already 75 points with pixel coords)
    segment_file = os.path.join(tempfile.gettempdir(), f"video_first_seg_{i}.pkl")

    pickle_data = {
        'keypoints': landmarks,  # (frames, 75, 2)
        'confidences': np.ones(landmarks.shape[:2]),
        'gloss': 'UNKNOWN'
    }

    with open(segment_file, 'wb') as f:
        pickle.dump(pickle_data, f)

    result = predict_pose_file(segment_file, model, label_map)
    predictions = result['top_k_predictions']
    video_first_results.append({
        "segment": i + 1,
        "frames": f"{start}-{end}",
        "num_frames": len(landmarks),
        "predictions": predictions
    })

    top_pred = predictions[0]
    print(f"  Segment {i+1} (frames {start}-{end}, {len(landmarks)} frames): "
          f"{top_pred['gloss']} ({top_pred['confidence']*100:.1f}%)")

holistic.close()

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print(f"\nExpected signs: {EXPECTED_SIGNS}")

print("\n--- POSE-FIRST PIPELINE ---")
for r in pose_first_results:
    top = r["predictions"][0]
    print(f"  Segment {r['segment']} ({r['frames']}): {top['gloss'].upper()} ({top['confidence']*100:.1f}%)")
    for p in r["predictions"][:3]:
        marker = "*" if p["gloss"].upper() in EXPECTED_SIGNS else " "
        print(f"    {marker} {p['gloss']}: {p['confidence']*100:.1f}%")

print("\n--- VIDEO-FIRST PIPELINE (MediaPipe Holistic) ---")
for r in video_first_results:
    top = r["predictions"][0]
    print(f"  Segment {r['segment']} ({r['frames']}): {top['gloss'].upper()} ({top['confidence']*100:.1f}%)")
    for p in r["predictions"][:3]:
        marker = "*" if p["gloss"].upper() in EXPECTED_SIGNS else " "
        print(f"    {marker} {p['gloss']}: {p['confidence']*100:.1f}%")

# Check which expected signs were found
print("\n--- DETECTION SUMMARY ---")
for sign in EXPECTED_SIGNS:
    # Check pose-first
    pose_found = False
    pose_conf = 0
    for r in pose_first_results:
        for p in r["predictions"][:5]:
            if p["gloss"].upper() == sign:
                if p["confidence"] > pose_conf:
                    pose_conf = p["confidence"]
                    if p == r["predictions"][0]:
                        pose_found = True

    # Check video-first
    video_found = False
    video_conf = 0
    for r in video_first_results:
        for p in r["predictions"][:5]:
            if p["gloss"].upper() == sign:
                if p["confidence"] > video_conf:
                    video_conf = p["confidence"]
                    if p == r["predictions"][0]:
                        video_found = True

    pose_status = f"[OK] #1 ({pose_conf*100:.1f}%)" if pose_found else (f"[..] top-5 ({pose_conf*100:.1f}%)" if pose_conf > 0 else "[X] Not found")
    video_status = f"[OK] #1 ({video_conf*100:.1f}%)" if video_found else (f"[..] top-5 ({video_conf*100:.1f}%)" if video_conf > 0 else "[X] Not found")

    print(f"  {sign}:")
    print(f"    Pose-First:  {pose_status}")
    print(f"    Video-First: {video_status}")

print("\nDone!")
