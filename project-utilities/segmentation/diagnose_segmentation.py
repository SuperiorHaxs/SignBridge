#!/usr/bin/env python3
"""
Diagnose why HybridSegmenter is not detecting segments in the video.
"""

import sys
import os
import numpy as np
import cv2

# hybrid_segmenter is now in the same directory
from hybrid_segmenter import HybridSegmenter

VIDEO_PATH = r"C:\Users\ashwi\Downloads\IMG_1481.MOV"

print("=" * 70)
print("DIAGNOSING HYBRID SEGMENTER")
print("=" * 70)
print(f"Video: {VIDEO_PATH}")
print()

# Get video info
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print(f"Video info: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")
print(f"Duration: {total_frames/fps:.1f} seconds")
print()

# Calculate threshold scaling
base_threshold = 500000
base_pixels = 640 * 480
actual_pixels = width * height
scale_factor = actual_pixels / base_pixels
scaled_threshold = base_threshold * scale_factor

print(f"Threshold scaling:")
print(f"  Base: {base_threshold} (for {base_pixels} pixels)")
print(f"  Scaled: {scaled_threshold:.0f} (for {actual_pixels} pixels)")
print(f"  Scale factor: {scale_factor:.2f}x")
print()

# Analyze motion scores
print("Analyzing motion scores...")
cap = cv2.VideoCapture(VIDEO_PATH)
prev_gray = None
motion_scores = []
blur_size = 21
diff_threshold = 25

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    if prev_gray is not None:
        frame_delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_delta, diff_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        motion_score = float(np.sum(thresh))
    else:
        motion_score = 0

    motion_scores.append(motion_score)
    prev_gray = gray

cap.release()

motion_scores = np.array(motion_scores)
print(f"\nMotion score statistics:")
print(f"  Min: {motion_scores.min():.0f}")
print(f"  Max: {motion_scores.max():.0f}")
print(f"  Mean: {motion_scores.mean():.0f}")
print(f"  Median: {np.median(motion_scores):.0f}")
print(f"  Std: {motion_scores.std():.0f}")
print()

# Percentiles
p10 = np.percentile(motion_scores, 10)
p25 = np.percentile(motion_scores, 25)
p50 = np.percentile(motion_scores, 50)
p75 = np.percentile(motion_scores, 75)
p90 = np.percentile(motion_scores, 90)

print(f"Percentiles:")
print(f"  P10: {p10:.0f}")
print(f"  P25: {p25:.0f} <- ADAPTIVE THRESHOLD USED BY HYBRID SEGMENTER")
print(f"  P50: {p50:.0f}")
print(f"  P75: {p75:.0f}")
print(f"  P90: {p90:.0f}")
print()

# Show how many frames exceed different thresholds
print(f"Frames exceeding thresholds:")
print(f"  > scaled ({scaled_threshold:.0f}): {np.sum(motion_scores > scaled_threshold)} frames ({np.sum(motion_scores > scaled_threshold)/len(motion_scores)*100:.1f}%)")
print(f"  > P25 ({p25:.0f}): {np.sum(motion_scores > p25)} frames ({np.sum(motion_scores > p25)/len(motion_scores)*100:.1f}%)")
print(f"  > P50 ({p50:.0f}): {np.sum(motion_scores > p50)} frames ({np.sum(motion_scores > p50)/len(motion_scores)*100:.1f}%)")
print()

# The key issue: HybridSegmenter uses P25 as adaptive threshold
# If P25 is too high, no segments will be detected
adaptive_threshold = p25

# Show motion score timeline
print("Motion score timeline (simplified):")
window = 30  # 1 second windows
for i in range(0, len(motion_scores), window):
    end_idx = min(i + window, len(motion_scores))
    chunk = motion_scores[i:end_idx]
    avg_score = chunk.mean()
    max_score = chunk.max()
    time_sec = i / fps
    bar_len = int(max_score / (motion_scores.max() / 40)) if motion_scores.max() > 0 else 0
    bar = "#" * bar_len
    above_thresh = "*" if max_score > adaptive_threshold else " "
    print(f"  {time_sec:5.1f}s: avg={avg_score:8.0f} max={max_score:8.0f} {above_thresh} {bar}")

print()

# Test with different parameters
print("=" * 70)
print("TESTING HYBRID SEGMENTER WITH LIVE MODE PARAMETERS")
print("=" * 70)

segmenter = HybridSegmenter(
    motion_threshold=500000,  # Base threshold
    cooldown_frames=45,       # 1.5 seconds at 30 FPS
    min_sign_frames=12,
    max_sign_frames=150,
    padding_before=3,
    padding_after=3
)

# This is what's called in app.py for Live mode
segments = segmenter.detect_segments_from_video(VIDEO_PATH, verbose=True)
print(f"\nResult: {len(segments)} segments detected")

if not segments:
    print("\n" + "=" * 70)
    print("DIAGNOSIS: No segments detected!")
    print("=" * 70)
    print()
    print("Possible causes:")
    print("  1. Adaptive threshold (P25) is too high relative to actual signing motion")
    print("  2. Cooldown is too long (45 frames = 1.5 seconds)")
    print("  3. Video has high background motion noise raising the threshold")
    print()
    print("Let me test with adjusted parameters...")
    print()

    # Test with lower cooldown
    print("--- Testing with cooldown_frames=15 (0.5 sec) ---")
    segmenter2 = HybridSegmenter(
        motion_threshold=500000,
        cooldown_frames=15,  # Reduced
        min_sign_frames=12,
        max_sign_frames=150,
        padding_before=3,
        padding_after=3
    )
    segments2 = segmenter2.detect_segments_from_video(VIDEO_PATH, verbose=True)
    print(f"Result: {len(segments2)} segments detected")
    print()

    # Test with even lower cooldown
    print("--- Testing with cooldown_frames=8 (0.27 sec) ---")
    segmenter3 = HybridSegmenter(
        motion_threshold=500000,
        cooldown_frames=8,  # Very short
        min_sign_frames=10,
        max_sign_frames=150,
        padding_before=3,
        padding_after=3
    )
    segments3 = segmenter3.detect_segments_from_video(VIDEO_PATH, verbose=True)
    print(f"Result: {len(segments3)} segments detected")

print("\nDone!")
