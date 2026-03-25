"""
Concatenate three ASL sign videos (STOMACH + SICK + NOW) from the same signer
into one seamless sentence video with cross-fade transitions.
"""
import cv2
import numpy as np
import os

# Signer 12 videos in sentence order: STOMACH -> SICK -> NOW
videos = [
    r"D:\Projects\WLASL\datasets\wlasl-kaggle\videos\stomach\54886.mp4",
    r"D:\Projects\WLASL\datasets\wlasl-kaggle\videos\sick\51517.mp4",
    r"D:\Projects\WLASL\datasets\wlasl-kaggle\videos\now\39006.mp4",
]
labels = ["STOMACH", "SICK", "NOW"]

output_path = r"c:\Users\ashwi\Projects\WLASL-proj\asl-v1\temp_videos\sentence_stomach_sick_now.mp4"

# Cross-fade duration in frames
CROSSFADE_FRAMES = 3

def read_video_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, fps

# Read all videos
all_clips = []
fps_values = []
for i, vpath in enumerate(videos):
    frames, fps = read_video_frames(vpath)
    print(f"{labels[i]}: {len(frames)} frames, {fps} fps, "
          f"{frames[0].shape[1]}x{frames[0].shape[0]}" if frames else "EMPTY")
    all_clips.append(frames)
    fps_values.append(fps)

# Speed up slightly (1.15x) to reduce gaps between signs
base_fps = fps_values[0] if fps_values[0] > 0 else 25.0
target_fps = base_fps * 1.3

# Resize all frames to match the largest resolution
max_h = max(f[0].shape[0] for f in all_clips if f)
max_w = max(f[0].shape[1] for f in all_clips if f)

def resize_frames(frames, target_w, target_h):
    resized = []
    for f in frames:
        if f.shape[0] != target_h or f.shape[1] != target_w:
            resized.append(cv2.resize(f, (target_w, target_h)))
        else:
            resized.append(f)
    return resized

for i in range(len(all_clips)):
    all_clips[i] = resize_frames(all_clips[i], max_w, max_h)

# Aggressively trim rest-pose frames from start/end of each clip
# WLASL clips typically have ~15-25% idle frames at each end
TRIM_PERCENT_START = 0.22  # trim 22% from beginning
TRIM_PERCENT_END = 0.22    # trim 22% from end

for i in range(len(all_clips)):
    n = len(all_clips[i])
    start = int(n * TRIM_PERCENT_START)
    end = n - int(n * TRIM_PERCENT_END)
    all_clips[i] = all_clips[i][start:end]
    print(f"{labels[i]} after trim: {len(all_clips[i])} frames ({n - len(all_clips[i])} removed)")

# Build output with cross-fade transitions
output_frames = []

for clip_idx, clip in enumerate(all_clips):
    if clip_idx == 0:
        # First clip: add all frames except the last CROSSFADE_FRAMES
        output_frames.extend(clip[:-CROSSFADE_FRAMES])
    else:
        # Cross-fade: blend last CROSSFADE_FRAMES of previous result with
        # first CROSSFADE_FRAMES of this clip
        prev_tail = all_clips[clip_idx - 1][-CROSSFADE_FRAMES:]
        curr_head = clip[:CROSSFADE_FRAMES]

        n_blend = min(len(prev_tail), len(curr_head))
        for j in range(n_blend):
            alpha = j / n_blend  # 0.0 -> 1.0 (prev -> current)
            blended = cv2.addWeighted(prev_tail[j], 1.0 - alpha, curr_head[j], alpha, 0)
            output_frames.append(blended)

        # Rest of current clip after the crossfade portion
        if clip_idx < len(all_clips) - 1:
            output_frames.extend(clip[CROSSFADE_FRAMES:-CROSSFADE_FRAMES])
        else:
            output_frames.extend(clip[CROSSFADE_FRAMES:])

print(f"\nTotal output: {len(output_frames)} frames at {target_fps} fps "
      f"= {len(output_frames)/target_fps:.1f} seconds")

# Write output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_path, fourcc, target_fps, (max_w, max_h))

for frame in output_frames:
    writer.write(frame)

writer.release()
print(f"Written to: {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1024:.0f} KB")
