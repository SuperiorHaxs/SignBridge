#!/usr/bin/env python3
"""
Overlay predicted glosses and LLM-generated sentences on a concatenated ASL video.

Takes the original concatenated video + prediction results and produces a new
video with two caption tracks:
  - Current predicted gloss (top of frame, large text)
  - Running LLM sentence (bottom of frame, appears after each sentence break)

Usage:
  python overlay_captions.py \
      --input temp_videos/paragraph_signer11_restaurant.mp4 \
      --segments temp_videos/paragraph_signer11_restaurant_segments.json \
      --predictions temp_videos/paragraph_signer11_restaurant_predictions.json \
      --output temp_videos/paragraph_signer11_restaurant_captioned.mp4

  Or with inline predict_sentence results:
  python overlay_captions.py \
      --input temp_videos/paragraph_signer11_restaurant.mp4 \
      --segments temp_videos/paragraph_signer11_restaurant_segments.json \
      --glosses "hello,hello,day,but,table,how,more" \
      --sentences "Hello, how was the day at the table?" \
      --output temp_videos/paragraph_signer11_restaurant_captioned.mp4

Predictions JSON format:
{
  "glosses": ["hello", "day", "table", ...],
  "sentences": [
    "Hello, how was the day at the table?",
    "They ordered chicken and salad.",
    ...
  ]
}
"""

import argparse
import json
import cv2
import numpy as np
import os
import textwrap


def load_video(path):
    """Load all frames and fps from a video file."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def draw_text_with_background(frame, text, font_scale, color, thickness,
                               bg_color=(0, 0, 0), bg_alpha=0.6, padding=4,
                               max_width=None, anchor="top-right"):
    """Draw text with a semi-transparent background box.

    anchor: "top-right", "bottom-left", "bottom-center"
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]

    # Word-wrap if max_width specified
    if max_width and text:
        char_width = cv2.getTextSize("A", font, font_scale, thickness)[0][0]
        chars_per_line = max(10, int(max_width / char_width))
        lines = textwrap.wrap(text, width=chars_per_line)
    else:
        lines = [text] if text else [""]

    if not lines:
        return frame

    # Calculate total text block size
    line_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    line_height = max(s[1] for s in line_sizes) + 6
    block_width = max(s[0] for s in line_sizes)
    block_height = line_height * len(lines)

    # Position based on anchor
    if anchor == "top-right":
        x = w - block_width - padding - 6
        y = 4
    elif anchor == "bottom-left":
        x = 6
        y = h - block_height - padding - 6
    elif anchor == "bottom-center":
        x = (w - block_width) // 2
        y = h - block_height - padding - 6
    else:
        x, y = 6, 4

    # Background rectangle
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + block_width + padding)
    y2 = min(h, y + block_height + padding)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)

    # Draw each line — left-aligned
    for i, line in enumerate(lines):
        lx = x
        ly = y + line_height * (i + 1) - 4
        # Shadow
        cv2.putText(frame, line, (lx + 1, ly + 1), font, font_scale, (0, 0, 0), thickness + 1)
        # Text
        cv2.putText(frame, line, (lx, ly), font, font_scale, color, thickness)

    return frame


def build_frame_lookup(segments):
    """Build frame-to-segment lookup from segments metadata."""
    # Find total frames
    max_frame = max(s['end_frame'] for s in segments) + 1

    # For each frame, which segment index does it belong to?
    frame_to_seg = [None] * (max_frame + 100)  # extra buffer for pauses
    for i, seg in enumerate(segments):
        for f in range(seg['start_frame'], seg['end_frame'] + 1):
            if f < len(frame_to_seg):
                frame_to_seg[f] = i

    return frame_to_seg


def build_sentence_map(segments, sentences):
    """Map each segment to its sentence based on sentence breaks."""
    seg_to_sentence = {}
    sentence_idx = 0

    for i, seg in enumerate(segments):
        if sentence_idx < len(sentences):
            seg_to_sentence[i] = sentence_idx

        if seg.get('sentence_break_after', False):
            sentence_idx += 1

    return seg_to_sentence


def main():
    parser = argparse.ArgumentParser(description="Overlay captions on ASL video")
    parser.add_argument("--input", required=True, help="Input concatenated video")
    parser.add_argument("--segments", required=True, help="Segments metadata JSON")
    parser.add_argument("--output", required=True, help="Output captioned video")

    # Prediction source (either JSON file or inline)
    parser.add_argument("--predictions", help="Predictions JSON file")
    parser.add_argument("--glosses", help="Comma-separated predicted glosses (inline)")
    parser.add_argument("--sentences", help="Pipe-separated LLM sentences (inline)")

    # Display options
    parser.add_argument("--show-ground-truth", action="store_true",
                        help="Show ground truth gloss from segments (green) alongside prediction (yellow)")
    parser.add_argument("--gloss-font-scale", type=float, default=0.45, help="Font scale for gloss text")
    parser.add_argument("--sentence-font-scale", type=float, default=0.4, help="Font scale for sentence text")

    args = parser.parse_args()

    # Load segments
    print(f"Loading segments: {args.segments}")
    with open(args.segments) as f:
        meta = json.load(f)
    segments = meta['segments']
    print(f"  {len(segments)} segments, {sum(1 for s in segments if s.get('sentence_break_after'))} sentence breaks")

    # Load predictions
    if args.predictions:
        print(f"Loading predictions: {args.predictions}")
        with open(args.predictions) as f:
            preds = json.load(f)
        pred_glosses = preds.get('glosses', [])
        pred_sentences = preds.get('sentences', [])
    elif args.glosses:
        pred_glosses = [g.strip() for g in args.glosses.split(',')]
        pred_sentences = [s.strip() for s in args.sentences.split('|')] if args.sentences else []
    else:
        print("ERROR: Provide either --predictions or --glosses")
        return 1

    print(f"  {len(pred_glosses)} predicted glosses, {len(pred_sentences)} sentences")

    # Validate
    if len(pred_glosses) != len(segments):
        print(f"WARNING: {len(pred_glosses)} glosses vs {len(segments)} segments — will use min")

    # Load video
    print(f"Loading video: {args.input}")
    frames, fps = load_video(args.input)
    h, w = frames[0].shape[:2]
    print(f"  {len(frames)} frames, {fps:.1f} fps, {w}x{h}")

    # Build lookups
    frame_to_seg = build_frame_lookup(segments)
    seg_to_sentence = build_sentence_map(segments, pred_sentences)

    # Build sentence word-reveal schedule:
    # For each sentence, map sign indices within that sentence to cumulative word count
    # so words appear left-to-right as signs are recognized
    sentence_sign_ranges = {}  # sent_idx -> (first_seg_idx, last_seg_idx)
    current_sent = 0
    first_seg = 0
    for i, seg in enumerate(segments):
        if seg.get('sentence_break_after', False):
            sentence_sign_ranges[current_sent] = (first_seg, i)
            current_sent += 1
            first_seg = i + 1
    # Last sentence (no trailing break)
    if first_seg < len(segments):
        sentence_sign_ranges[current_sent] = (first_seg, len(segments) - 1)

    current_sentence_text = ""
    current_sentence_idx = -1

    # Process each frame
    print("Rendering captions...")
    output_frames = []
    for frame_idx, frame in enumerate(frames):
        frame = frame.copy()

        seg_idx = frame_to_seg[frame_idx] if frame_idx < len(frame_to_seg) else None

        gloss_text = ""
        gt_text = ""
        if seg_idx is not None and seg_idx < len(segments):
            gt_text = segments[seg_idx]['gloss']
            if seg_idx < len(pred_glosses):
                gloss_text = pred_glosses[seg_idx]

            # Determine which sentence we're in and how much to reveal
            sent_idx = seg_to_sentence.get(seg_idx)
            if sent_idx is not None and sent_idx < len(pred_sentences):
                full_sentence = pred_sentences[sent_idx]
                words = full_sentence.split()

                if sent_idx in sentence_sign_ranges:
                    first_seg, last_seg = sentence_sign_ranges[sent_idx]
                    total_signs = last_seg - first_seg + 1
                    signs_so_far = seg_idx - first_seg + 1
                    # Reveal words proportionally to signs completed
                    words_to_show = max(1, int(len(words) * signs_so_far / total_signs))
                    current_sentence_text = ' '.join(words[:words_to_show])
                    current_sentence_idx = sent_idx
                else:
                    current_sentence_text = full_sentence

            elif current_sentence_idx >= 0:
                # In a pause gap — show the full previous sentence
                current_sentence_text = pred_sentences[current_sentence_idx]

        # Draw predicted gloss (top-right, small, caption-style)
        if gloss_text:
            if gloss_text.upper() == gt_text.upper():
                gloss_color = (0, 255, 0)  # green = correct
            else:
                gloss_color = (0, 255, 255)  # yellow = mismatch

            display_gloss = gloss_text.upper()
            if args.show_ground_truth and gloss_text.upper() != gt_text.upper():
                display_gloss = f"{gloss_text.upper()} (GT:{gt_text.upper()})"

            draw_text_with_background(
                frame, display_gloss,
                font_scale=args.gloss_font_scale,
                color=gloss_color,
                thickness=1,
                bg_alpha=0.5,
                anchor="top-right",
            )

        # Draw sentence (bottom-left, small, closed-caption style)
        if current_sentence_text:
            draw_text_with_background(
                frame, current_sentence_text,
                font_scale=args.sentence_font_scale,
                color=(255, 255, 255),
                thickness=1,
                bg_alpha=0.6,
                max_width=int(w * 0.92),
                anchor="bottom-left",
            )

        output_frames.append(frame)

    # Write output video
    print(f"Writing: {args.output}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
    for frame in output_frames:
        writer.write(frame)
    writer.release()

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    duration = len(output_frames) / fps
    print(f"Done: {len(output_frames)} frames, {duration:.1f}s, {file_size:.1f} MB")
    print(f"  Glosses shown: {len(pred_glosses)}")
    print(f"  Sentences shown: {len(pred_sentences)}")

    return 0


if __name__ == "__main__":
    exit(main())
