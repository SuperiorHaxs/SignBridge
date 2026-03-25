"""
Concatenate ASL sign videos into a paragraph using Signer 11.

Restaurant domain paragraph using Signer 11's available signs
from the wlasl_60_class_restaurant_model. Short ASL-style sentences
(3-5 signs each) with sentence boundary pauses.
"""
import cv2
import numpy as np
import os

BASE = r"D:\Projects\WLASL\datasets\wlasl-kaggle\videos"

# Signer 11 unique videos for restaurant model vocabulary (verified on disk)
VIDEO_MAP = {
    "APPLE":       ("apple",       "03005"),
    "BAD":         ("bad",         "04717"),
    "BANANA":      ("banana",      "04903"),
    "BREAKFAST":   ("breakfast",   "07628"),
    "BUT":         ("but",         "08431"),
    "BUTTER":      ("butter",      "08452"),
    "CHICKEN":     ("chicken",     "10405"),
    "CHOCOLATE":   ("chocolate",   "10583"),
    "COFFEE":      ("coffee",      "11563"),
    "COLD":        ("cold",        "11635"),
    "DAY":         ("day",         "14799"),
    "DOLLAR":      ("dollar",      "17132"),
    "DRINK":       ("drink",       "17720"),
    "EAT":         ("eat",         "18329"),
    "ENOUGH":      ("enough",      "19309"),
    "EVENING":     ("evening",     "19746"),
    "FISH":        ("fish",        "22120"),
    "FULL":        ("full",        "23776"),
    "HELLO":       ("hello",       "27177"),
    "HIS":         ("his",         "27556"),
    "HOUR":        ("hour",        "28143"),
    "HOW":         ("how",         "28210"),
    "HUNGRY":      ("hungry",      "28381"),
    "LUNCH":       ("lunch",       "34271"),
    "MILK":        ("milk",        "36050"),
    "MORE":        ("more",        "36834"),
    "MY":          ("my",          "37473"),
    "NOW":         ("now",         "38999"),
    "ORDER":       ("order",       "40177"),
    "PAY":         ("pay",         "41598"),
    "PEPPER":      ("pepper",      "41945"),
    "PLATE":       ("plate",       "43117"),
    "PLEASE":      ("please",      "43222"),
    "SALAD":       ("salad",       "49081"),
    "SANDWICH":    ("sandwich",    "49250"),
    "SAUCE":       ("sauce",       "49345"),
    "SIX":         ("six",         "51957"),
    "SUGAR":       ("sugar",       "55663"),
    "SUNDAY":      ("sunday",      "55776"),
    "TABLE":       ("table",       "56563"),
    "THEIR":       ("their",       "57728"),
    "THEY":        ("they",        "57880"),
    "TOMATO":      ("tomato",      "58791"),
    "TOMORROW":    ("tomorrow",    "58810"),
    "WAITER":      ("waiter",      "62092"),
    "WHAT":        ("what",        "62979"),
    "WHEN":        ("when",        "63072"),
    "WHERE":       ("where",       "63087"),
    "WHICH":       ("which",       "63108"),
    "WHO":         ("who",         "63236"),
    "WHY":         ("why",         "63287"),
    "YOUR":        ("your",        "64433"),
}

# Natural restaurant conversation using only 23 masked-reliable signs.
# Each sentence is a plausible restaurant utterance (2-5 signs).
# "P" marks sentence boundary pauses.
SEQUENCE = [
    # Greetings & Time
    "HELLO",                               # Hello
    "P",
    "HELLO", "HOW",                        # Hello, how are you?
    "P",
    "WHEN", "SUNDAY",                      # When is Sunday?
    "P",
    "NOW", "HOUR",                         # What hour is it now?
    "P",
    # Ordering
    "COFFEE", "NOW",                       # Coffee now
    "P",
    "COLD", "COFFEE",                      # Cold coffee
    "P",
    "COLD", "DRINK",                       # A cold drink
    "P",
    "FISH", "SANDWICH",                    # A fish sandwich
    "P",
    "TOMATO", "SALAD",                     # Tomato salad
    "P",
    "CHOCOLATE", "DRINK",                  # A chocolate drink
    "P",
    "FISH", "SALAD",                       # Fish salad
    "P",
    "MY", "SANDWICH",                      # My sandwich
    "P",
    "MY", "SALAD",                         # My salad
    "P",
    "MY", "COFFEE",                        # My coffee
    "P",
    # Questions
    "HOW", "COFFEE",                       # How is the coffee?
    "P",
    "HOW", "SALAD",                        # How is the salad?
    "P",
    "HOW", "FISH",                         # How is the fish?
    "P",
    "HOW", "SANDWICH",                     # How is the sandwich?
    "P",
    "HOW", "HUNGRY",                       # How hungry?
    "P",
    "WHY", "COLD",                         # Why is it cold?
    "P",
    "WHY", "BAD",                          # Why is it bad?
    "P",
    "WHEN", "DRINK",                       # When do we drink?
    "P",
    "HIS", "PLATE", "HOW",                # How is his plate?
    "P",
    "HIS", "COFFEE", "HOW",               # How is his coffee?
    "P",
    "HIS", "SANDWICH", "HOW",             # How is his sandwich?
    "P",
    # Complaints
    "COFFEE", "BAD",                       # The coffee is bad
    "P",
    "COFFEE", "COLD",                      # The coffee is cold
    "P",
    "FISH", "BAD",                         # The fish is bad
    "P",
    "FISH", "COLD",                        # The fish is cold
    "P",
    "SALAD", "BAD",                        # The salad is bad
    "P",
    "SANDWICH", "BAD",                     # The sandwich is bad
    "P",
    "SANDWICH", "COLD",                    # The sandwich is cold
    "P",
    "MY", "PLATE", "COLD",                # My plate is cold
    "P",
    "HIS", "PLATE", "BAD",                # His plate is bad
    "P",
    "HIS", "PLATE", "COLD",               # His plate is cold
    "P",
    "TOMATO", "BAD", "WHY",               # Why is the tomato bad?
    "P",
    "FISH", "COLD", "WHY",                # Why is the fish cold?
    "P",
    "COFFEE", "COLD", "WHY",              # Why is the coffee cold?
    "P",
    "SALAD", "BAD", "WHY",                # Why is the salad bad?
    "P",
    # States
    "HUNGRY", "NOW",                       # I am hungry now
    "P",
    "HUNGRY", "HOW",                       # How hungry?
    "P",
    "ENOUGH",                              # Enough
    "P",
    "ENOUGH", "DRINK",                     # Enough to drink
    "P",
    "ENOUGH", "COFFEE",                    # Enough coffee
    "P",
    "ENOUGH", "FISH",                      # Enough fish
    "P",
    "ENOUGH", "SANDWICH",                  # Had enough sandwiches
    "P",
    # Compound
    "BUT", "COLD",                         # But it is cold
    "P",
    "BUT", "BAD",                          # But it is bad
    "P",
    "BUT", "HUNGRY",                       # But I am hungry
    "P",
    "BUT", "ENOUGH",                       # But that is enough
    "P",
    "COFFEE", "COLD", "BUT", "DRINK",     # Coffee is cold but I'll drink it
    "P",
    "FISH", "BAD", "BUT", "HUNGRY",       # Fish is bad but I am hungry
    "P",
    "SANDWICH", "COLD", "BUT", "ENOUGH",  # Sandwich cold but enough
    "P",
    "COLD", "COFFEE", "NOW", "WHY",       # Why is the coffee cold now?
    "P",
    "SUNDAY", "HOW", "HUNGRY",            # How hungry on Sunday?
    "P",
    "MY", "CHOCOLATE", "DRINK", "NOW",    # My chocolate drink now
    "P",
    "HIS", "FISH", "SALAD", "BAD",        # His fish salad is bad
    "P",
    "HIS", "TOMATO", "SANDWICH", "COLD",  # His tomato sandwich is cold
    "P",
    "COLD", "SALAD", "BUT", "ENOUGH",     # Salad is cold but enough
    "P",
    "NOW", "COFFEE", "ENOUGH",            # Enough coffee for now
    "P",
    "SUNDAY", "FISH", "SANDWICH",          # Sunday fish sandwich
    "P",
    "HUNGRY", "BUT", "ENOUGH",            # Hungry but had enough
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
video_cache = {}  # label -> (trimmed_frames, fps)
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

# Histogram-match all clips to a reference to normalize brightness/contrast
def histogram_match(source, reference):
    """Match the histogram of source to reference, per channel."""
    result = np.zeros_like(source)
    for ch in range(3):
        s = source[:, :, ch].flatten()
        r = reference[:, :, ch].flatten()
        s_values, s_idx, s_counts = np.unique(s, return_inverse=True, return_counts=True)
        r_values, r_counts = np.unique(r, return_counts=True)
        s_cdf = np.cumsum(s_counts).astype(np.float64) / s.size
        r_cdf = np.cumsum(r_counts).astype(np.float64) / r.size
        mapping = np.interp(s_cdf, r_cdf, r_values)
        result[:, :, ch] = mapping[s_idx].reshape(source.shape[:2])
    return result.astype(np.uint8)

# No histogram matching — Signer 11 clips are already consistent enough

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
    all_clips.append(list(trimmed_cache[item]))  # copy so crossfade doesn't corrupt
    labels.append(item)

# Track which sequence indices are followed by a pause
pause_after = set()
for i, item in enumerate(SEQUENCE):
    if item == "P":
        # Find the previous non-P item's clip index
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
    "source": "concatenate_paragraph.py",
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
