"""
Concatenate ASL sign videos into a full paragraph from Signer 11.

300-word waiter training lecture using only Signer 11's 53 available signs
from the wlasl_60_class_restaurant_model. Signs repeat as they appear
in the natural paragraph flow.
"""
import cv2
import numpy as np
import os

BASE = r"D:\Projects\WLASL\datasets\wlasl-kaggle\videos"

# Signer 11 unique videos (loaded once, referenced by label)
VIDEO_MAP = {
    "APPLE":      ("apple",      "03005"),
    "BAD":        ("bad",        "04717"),
    "BANANA":     ("banana",     "04903"),
    "BITTER":     ("bitter",     "06435"),
    "BREAKFAST":  ("breakfast",  "07628"),
    "BUT":        ("but",        "08431"),
    "BUTTER":     ("butter",     "08452"),
    "CHICKEN":    ("chicken",    "10405"),
    "CHOCOLATE":  ("chocolate",  "10583"),
    "COFFEE":     ("coffee",     "11563"),
    "COLD":       ("cold",       "11635"),
    "DAY":        ("day",        "14799"),
    "DOLLAR":     ("dollar",     "17132"),
    "DRINK":      ("drink",      "17720"),
    "EAT":        ("eat",        "18329"),
    "ENOUGH":     ("enough",     "19309"),
    "EVENING":    ("evening",    "19746"),
    "FISH":       ("fish",       "22120"),
    "FULL":       ("full",       "23776"),
    "HELLO":      ("hello",      "27177"),
    "HIS":        ("his",        "27556"),
    "HOUR":       ("hour",       "28143"),
    "HOW":        ("how",        "28210"),
    "HUNGRY":     ("hungry",     "28381"),
    "LUNCH":      ("lunch",      "34271"),
    "MILK":       ("milk",       "36050"),
    "MORE":       ("more",       "36834"),
    "MY":         ("my",         "37473"),
    "NOW":        ("now",        "38999"),
    "ORDER":      ("order",      "40177"),
    "PAY":        ("pay",        "41598"),
    "PEPPER":     ("pepper",     "41945"),
    "PLATE":      ("plate",      "43117"),
    "PLEASE":     ("please",     "43222"),
    "SALAD":      ("salad",      "49081"),
    "SANDWICH":   ("sandwich",   "49250"),
    "SAUCE":      ("sauce",      "49345"),
    "SIX":        ("six",        "51957"),
    "SUGAR":      ("sugar",      "55663"),
    "SUNDAY":     ("sunday",     "55776"),
    "TABLE":      ("table",      "56563"),
    "THEIR":      ("their",      "57728"),
    "THEY":       ("they",       "57880"),
    "TOMATO":     ("tomato",     "58791"),
    "TOMORROW":   ("tomorrow",   "58810"),
    "WAITER":     ("waiter",     "62092"),
    "WHAT":       ("what",       "62979"),
    "WHEN":       ("when",       "63072"),
    "WHERE":      ("where",      "63087"),
    "WHICH":      ("which",      "63108"),
    "WHO":        ("who",        "63236"),
    "WHY":        ("why",        "63287"),
    "YOUR":       ("your",       "64433"),
}

# Full paragraph sign sequence with repeats, in ASL gloss order.
# "P" marks sentence/phrase pauses.
#
# Waiter training lecture paragraph:
# "Hello, and welcome to your first day..."
SEQUENCE = [
    # "Hello, welcome your first day. My table how waiter train."
    "HELLO", "YOUR", "DAY", "MY", "TABLE", "HOW", "WAITER",
    "P",
    # "Hungry customer, what first they see? Their plate, their table, your."
    "HUNGRY", "WHAT", "THEY", "THEIR", "PLATE", "THEIR", "TABLE", "YOUR",
    # "Please hello."
    "PLEASE", "HELLO",
    "P",
    # "Breakfast when — coffee milk sugar order, banana apple."
    "BREAKFAST", "WHEN", "COFFEE", "MILK", "SUGAR", "ORDER", "BANANA", "APPLE",
    # "Which coffee — hot, cold?"
    "WHICH", "COFFEE", "COLD",
    # "They coffee more drink, but cold coffee bad."
    "THEY", "COFFEE", "MORE", "DRINK", "BUT", "COLD", "COFFEE", "BAD",
    # "Why? Cold coffee bitter."
    "WHY", "COLD", "COFFEE", "BITTER",
    # "How bitter coffee? More sugar, more milk, but enough enough."
    "HOW", "BITTER", "COFFEE", "MORE", "SUGAR", "MORE", "MILK", "BUT", "ENOUGH", "ENOUGH",
    "P",
    # "Lunch when full."
    "LUNCH", "WHEN", "FULL",
    # "They chicken sandwich, fish salad, tomato salad pepper sauce order."
    "THEY", "CHICKEN", "SANDWICH", "FISH", "SALAD", "TOMATO", "SALAD", "PEPPER", "SAUCE", "ORDER",
    # "His plate chicken butter tomato sauce."
    "HIS", "PLATE", "CHICKEN", "BUTTER", "TOMATO", "SAUCE",
    # "But which sauce — pepper sauce, tomato sauce?"
    "BUT", "WHICH", "SAUCE", "PEPPER", "SAUCE", "TOMATO", "SAUCE",
    # "Please your customer."
    "PLEASE", "YOUR",
    "P",
    # "Bad order — they cold fish, cold chicken eat."
    "BAD", "ORDER", "THEY", "COLD", "FISH", "COLD", "CHICKEN", "EAT",
    # "Why their plate cold? Who? Your waiter — your."
    "WHY", "THEIR", "PLATE", "COLD", "WHO", "YOUR", "WAITER", "YOUR",
    "P",
    # "Now, they eat, they more eat."
    "NOW", "THEY", "EAT", "THEY", "MORE", "EAT",
    # "More salad, more sandwich, more chicken. They hungry!"
    "MORE", "SALAD", "MORE", "SANDWICH", "MORE", "CHICKEN", "THEY", "HUNGRY",
    # "But they full, please more not."
    "BUT", "THEY", "FULL", "PLEASE", "MORE",
    # "Full full. Chocolate banana order, but enough."
    "FULL", "FULL", "CHOCOLATE", "BANANA", "ORDER", "BUT", "ENOUGH",
    "P",
    # "How they pay? Six dollar breakfast, but lunch more."
    "HOW", "THEY", "PAY", "SIX", "DOLLAR", "BREAKFAST", "BUT", "LUNCH", "MORE",
    # "Your waiter — what they order, when they order, where their table."
    "YOUR", "WAITER", "WHAT", "THEY", "ORDER", "WHEN", "THEY", "ORDER", "WHERE", "THEIR", "TABLE",
    # "Dollar plate day."
    "DOLLAR", "PLATE", "DAY",
    "P",
    # "Tomorrow sunday evening, full — table hour."
    "TOMORROW", "SUNDAY", "EVENING", "FULL", "TABLE", "HOUR",
    # "Who your waiter tomorrow?"
    "WHO", "YOUR", "WAITER", "TOMORROW",
    "P",
    # "They lunch eat early. Please breakfast eat, coffee now drink."
    "THEY", "LUNCH", "EAT", "PLEASE", "BREAKFAST", "EAT", "COFFEE", "NOW", "DRINK",
    # "Tomorrow, enough time not."
    "TOMORROW", "ENOUGH",
    "P",
    # "But how day — hungry customer, full plate, cold coffee, more chicken."
    "BUT", "HOW", "DAY", "HUNGRY", "FULL", "PLATE", "COLD", "COFFEE", "MORE", "CHICKEN",
    # "Hello."
    "HELLO",
]

output_path = r"c:\Users\ashwi\Projects\WLASL-proj\asl-v1\temp_videos\paragraph_signer11_restaurant.mp4"

CROSSFADE_FRAMES = 3
TRIM_PERCENT_START = 0.22
TRIM_PERCENT_END = 0.22
SPEED_MULTIPLIER = 1.3
PAUSE_SECONDS = 0.6  # brief pause between sentences

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

import gc

# Load each unique video once
video_cache = {}
fps_values = []
for label, (gloss, vid) in VIDEO_MAP.items():
    path = os.path.join(BASE, gloss, f"{vid}.mp4")
    frames, fps = read_video_frames(path)
    if not frames:
        print(f"ERROR: No frames for {label} at {path}")
        continue
    print(f"{label:15s}: {len(frames)} frames, {fps:.1f} fps, "
          f"{frames[0].shape[1]}x{frames[0].shape[0]}")
    video_cache[label] = (frames, fps)
    fps_values.append(fps)
    gc.collect()

# Count sign instances
sign_count = sum(1 for s in SEQUENCE if s != "P")
pause_count = sum(1 for s in SEQUENCE if s == "P")
print(f"\nSequence: {sign_count} sign instances, {pause_count} pauses, "
      f"{len(VIDEO_MAP)} unique signs")

# Target fps with speed multiplier
base_fps = fps_values[0] if fps_values[0] > 0 else 25.0
target_fps = base_fps * SPEED_MULTIPLIER
pause_frames = int(PAUSE_SECONDS * target_fps)

# Normalize resolution across all clips
max_h = max(f[0].shape[0] for f, _ in video_cache.values())
max_w = max(f[0].shape[1] for f, _ in video_cache.values())

# Resize cached clips
resized_cache = {}
for label, (frames, fps) in video_cache.items():
    resized = []
    for f in frames:
        if f.shape[0] != max_h or f.shape[1] != max_w:
            resized.append(cv2.resize(f, (max_w, max_h)))
        else:
            resized.append(f)
    resized_cache[label] = resized

# No histogram matching — Signer 11 clips are consistent

# Trim rest-pose frames
trimmed_cache = {}
for label, frames in resized_cache.items():
    n = len(frames)
    start = int(n * TRIM_PERCENT_START)
    end = n - int(n * TRIM_PERCENT_END)
    trimmed_cache[label] = frames[start:end]
    print(f"{label:15s} after trim: {len(trimmed_cache[label])} frames")

# Build the full clip sequence from SEQUENCE, expanding repeats
all_clips = []
labels = []
for item in SEQUENCE:
    if item == "P":
        continue
    all_clips.append(list(trimmed_cache[item]))
    labels.append(item)

# Track which sequence indices are followed by a pause
pause_after = set()
for i, item in enumerate(SEQUENCE):
    if item == "P":
        clip_idx = sum(1 for s in SEQUENCE[:i] if s != "P") - 1
        if clip_idx >= 0:
            pause_after.add(clip_idx)

print(f"\nBuilding video: {len(all_clips)} clips, {len(pause_after)} pauses")

# Build output with cross-fade transitions and track segment boundaries
output_frames = []
segment_boundaries = []

for clip_idx, clip in enumerate(all_clips):
    seg_start = len(output_frames)

    if clip_idx == 0:
        output_frames.extend(clip[:-CROSSFADE_FRAMES])
    else:
        prev_tail = all_clips[clip_idx - 1][-CROSSFADE_FRAMES:]
        curr_head = clip[:CROSSFADE_FRAMES]
        n_blend = min(len(prev_tail), len(curr_head))
        for j in range(n_blend):
            alpha = j / n_blend
            blended = cv2.addWeighted(prev_tail[j], 1.0 - alpha, curr_head[j], alpha, 0)
            output_frames.append(blended)
        if clip_idx < len(all_clips) - 1:
            output_frames.extend(clip[CROSSFADE_FRAMES:-CROSSFADE_FRAMES])
        else:
            output_frames.extend(clip[CROSSFADE_FRAMES:])

    seg_end = len(output_frames) - 1
    has_pause = clip_idx in pause_after
    segment_boundaries.append((seg_start, seg_end, labels[clip_idx], has_pause))

    if has_pause:
        last_frame = output_frames[-1]
        for _ in range(pause_frames):
            output_frames.append(last_frame)

duration = len(output_frames) / target_fps
print(f"\nTotal: {len(output_frames)} frames at {target_fps:.1f} fps = {duration:.1f} seconds")
print(f"  Sign instances: {len(all_clips)}, Pauses: {len(pause_after)} x {PAUSE_SECONDS}s")

# Write output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_path, fourcc, target_fps, (max_w, max_h))
for frame in output_frames:
    writer.write(frame)
writer.release()

print(f"Written to: {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1024:.0f} KB")

# Write segment metadata JSON for predict_sentence.py
metadata_path = output_path.replace('.mp4', '_segments.json')
metadata = {
    "source": "concatenate_restaurant_paragraph.py",
    "fps": target_fps,
    "total_frames": len(output_frames),
    "pause_duration_sec": PAUSE_SECONDS,
    "pause_frames": pause_frames,
    "segments": []
}
for seg_start, seg_end, label, has_pause in segment_boundaries:
    seg = {
        "gloss": label,
        "start_frame": seg_start,
        "end_frame": seg_end,
        "num_frames": seg_end - seg_start + 1
    }
    if has_pause:
        seg["sentence_break_after"] = True
    metadata["segments"].append(seg)

import json
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Segment metadata: {metadata_path}")
sentence_num = 1
for s, e, l, p in segment_boundaries:
    brk = " [SENTENCE BREAK]" if p else ""
    print(f"  S{sentence_num}: {l:10s}: frames {s}-{e} ({e-s+1} frames){brk}")
    if p:
        sentence_num += 1
