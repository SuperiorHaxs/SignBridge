#!/usr/bin/env python3
"""
Scan video for signs by testing different frame windows.
Since the user signed: record -> CHAIR -> pause -> BLACK -> pause -> TABLE -> stop
We'll scan through the video in windows to find where each sign is.
"""

import sys
import os
import cv2
import numpy as np
import tempfile
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # misc -> project-utilities -> project root

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'openhands-modernized', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'openhands-modernized', 'src', 'util'))

from openhands_modernized_inference import predict_pose_file, load_model_from_checkpoint
import mediapipe as mp

VIDEO_PATH = r"C:\Users\ashwi\Downloads\IMG_1481.MOV"
TARGET_SIGNS = ["CHAIR", "BLACK", "TABLE"]
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'openhands-modernized', 'production-models', 'wlasl_50_class_model')

print("=" * 70)
print("SCANNING VIDEO FOR TARGET SIGNS")
print("=" * 70)
print(f"Video: {VIDEO_PATH}")
print(f"Looking for: {TARGET_SIGNS}")
print()

# Load model
print("Loading model...")
model, label_map = load_model_from_checkpoint(MODEL_DIR)
print(f"Model loaded")
print()

# Get video info
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print(f"Video: {total_frames} frames, {fps:.1f} FPS, {width}x{height}, {total_frames/fps:.1f}s duration")
print()

# Setup MediaPipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_pose_segment(video_path, start_frame, end_frame):
    """Extract pose using MediaPipe Holistic with pixel coordinates."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_frames = []

    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        frame_landmarks = []

        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                frame_landmarks.append([lm.x * w, lm.y * h])
        else:
            frame_landmarks.extend([[0, 0]] * 33)

        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                frame_landmarks.append([lm.x * w, lm.y * h])
        else:
            frame_landmarks.extend([[0, 0]] * 21)

        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                frame_landmarks.append([lm.x * w, lm.y * h])
        else:
            frame_landmarks.extend([[0, 0]] * 21)

        all_frames.append(frame_landmarks)

    cap.release()
    return np.array(all_frames)

def predict_segment(landmarks):
    """Predict from landmarks array."""
    segment_file = os.path.join(tempfile.gettempdir(), "scan_segment.pkl")

    pickle_data = {
        'keypoints': landmarks,
        'confidences': np.ones(landmarks.shape[:2]),
        'gloss': 'UNKNOWN'
    }

    with open(segment_file, 'wb') as f:
        pickle.dump(pickle_data, f)

    result = predict_pose_file(segment_file, model, label_map)
    return result['top_k_predictions']

# Divide video into rough thirds (skipping first 2 sec and last 2 sec for recording motion)
# Video is 18.3 seconds, so:
# - Skip first ~60 frames (2 sec) - recording start
# - Skip last ~60 frames (2 sec) - recording stop
# - Remaining ~430 frames for 3 signs

start_offset = 60  # Skip first 2 seconds
end_offset = total_frames - 60  # Skip last 2 seconds
usable_frames = end_offset - start_offset

print(f"Usable frame range: {start_offset} to {end_offset} ({usable_frames} frames, {usable_frames/fps:.1f}s)")
print()

# Scan with overlapping windows
window_size = 40  # ~1.3 seconds per window
step_size = 20    # ~0.67 second steps

print("=" * 70)
print("SCANNING FOR EACH TARGET SIGN")
print("=" * 70)

for target_sign in TARGET_SIGNS:
    print(f"\n--- Searching for {target_sign} ---")

    best_conf = 0
    best_range = None
    best_rank = 99

    results = []

    for start in range(start_offset, end_offset - window_size, step_size):
        end = start + window_size

        landmarks = extract_pose_segment(VIDEO_PATH, start, end)
        if len(landmarks) < 10:
            continue

        predictions = predict_segment(landmarks)

        # Find target sign in predictions
        for rank, p in enumerate(predictions[:5]):
            if p['gloss'].upper() == target_sign:
                results.append({
                    'start': start,
                    'end': end,
                    'conf': p['confidence'],
                    'rank': rank + 1,
                    'top1': predictions[0]['gloss']
                })

                if p['confidence'] > best_conf:
                    best_conf = p['confidence']
                    best_range = (start, end)
                    best_rank = rank + 1
                break

    # Show best result
    if best_range:
        print(f"  Best: frames {best_range[0]}-{best_range[1]} ({best_range[0]/fps:.1f}s - {best_range[1]/fps:.1f}s)")
        print(f"         {target_sign} at rank #{best_rank} with {best_conf*100:.1f}% confidence")

        # Show top 3 windows for this sign
        sorted_results = sorted(results, key=lambda x: x['conf'], reverse=True)[:5]
        print(f"  Top 5 windows:")
        for r in sorted_results:
            time_start = r['start'] / fps
            time_end = r['end'] / fps
            print(f"    frames {r['start']}-{r['end']} ({time_start:.1f}s-{time_end:.1f}s): "
                  f"#{r['rank']} {r['conf']*100:.1f}% (top1: {r['top1']})")
    else:
        print(f"  {target_sign} not found in top-5 predictions anywhere!")

# Now do a more focused scan around likely sign locations
print("\n" + "=" * 70)
print("TIMELINE ANALYSIS")
print("=" * 70)
print(f"\nAssuming signs are roughly evenly spaced in the middle {usable_frames/fps:.1f}s:")
print(f"  CHAIR: ~{start_offset/fps:.1f}s - {(start_offset + usable_frames//3)/fps:.1f}s")
print(f"  BLACK: ~{(start_offset + usable_frames//3)/fps:.1f}s - {(start_offset + 2*usable_frames//3)/fps:.1f}s")
print(f"  TABLE: ~{(start_offset + 2*usable_frames//3)/fps:.1f}s - {end_offset/fps:.1f}s")

# Test each third
thirds = [
    (start_offset, start_offset + usable_frames // 3, "CHAIR"),
    (start_offset + usable_frames // 3, start_offset + 2 * usable_frames // 3, "BLACK"),
    (start_offset + 2 * usable_frames // 3, end_offset, "TABLE")
]

print("\n--- Testing each third of video ---")
for region_start, region_end, expected_sign in thirds:
    print(f"\nRegion {region_start}-{region_end} ({region_start/fps:.1f}s-{region_end/fps:.1f}s) [Expected: {expected_sign}]")

    # Test with different window sizes within this region
    best_for_expected = None

    for win_size in [30, 40, 50, 60]:
        for start in range(region_start, region_end - win_size, 10):
            end = start + win_size
            landmarks = extract_pose_segment(VIDEO_PATH, start, end)
            if len(landmarks) < 10:
                continue
            predictions = predict_segment(landmarks)

            for rank, p in enumerate(predictions[:5]):
                if p['gloss'].upper() == expected_sign:
                    if best_for_expected is None or p['confidence'] > best_for_expected['conf']:
                        best_for_expected = {
                            'start': start, 'end': end,
                            'conf': p['confidence'], 'rank': rank + 1,
                            'top1': predictions[0]['gloss'],
                            'top1_conf': predictions[0]['confidence']
                        }
                    break

    if best_for_expected:
        b = best_for_expected
        print(f"  Best for {expected_sign}: frames {b['start']}-{b['end']} ({b['start']/fps:.1f}s-{b['end']/fps:.1f}s)")
        print(f"    {expected_sign} = #{b['rank']} at {b['conf']*100:.1f}%")
        print(f"    Top prediction: {b['top1']} at {b['top1_conf']*100:.1f}%")
    else:
        print(f"  {expected_sign} not found in this region!")

holistic.close()
print("\nDone!")
