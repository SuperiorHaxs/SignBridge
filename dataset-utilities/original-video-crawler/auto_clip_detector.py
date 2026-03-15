#!/usr/bin/env python3
"""
Auto Clip Detector for ASL Videos

Automatically detects sign boundaries in ASL tutorial videos using:
1. Speech recognition (whisper) - finds when the target word is spoken,
   then clips the signing that follows
2. Motion detection (OpenCV) - detects hand movement start/end to trim
   dead frames from short single-sign videos

Requirements:
  pip install opencv-python-headless numpy

Optional (for speech-based detection):
  pip install openai-whisper
  OR
  pip install faster-whisper  (recommended, much faster)
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def detect_motion_boundaries(video_path, motion_threshold=5.0, min_motion_frames=5,
                              pad_sec=0.3):
    """
    Detect sign boundaries using frame-to-frame motion analysis.

    Works best for short videos (< 15s) with a single sign.
    Finds where significant motion starts and ends.

    Returns:
        (start_sec, end_sec) or None if detection fails
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Cannot open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prev_gray = None
    motion_scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_score = np.mean(diff)
            motion_scores.append(motion_score)
        else:
            motion_scores.append(0)

        prev_gray = gray

    cap.release()

    if not motion_scores:
        return None

    scores = np.array(motion_scores)
    # Adaptive threshold: use mean + 0.5 * std as minimum motion
    adaptive_thresh = max(motion_threshold, scores.mean() + 0.5 * scores.std())

    # Find frames with significant motion
    motion_mask = scores > adaptive_thresh

    # Find contiguous regions of motion
    motion_indices = np.where(motion_mask)[0]
    if len(motion_indices) < min_motion_frames:
        # Not enough motion detected — use full video
        return (0, total_frames / fps)

    # Find the largest contiguous motion region
    regions = []
    start_idx = motion_indices[0]
    prev_idx = motion_indices[0]

    for idx in motion_indices[1:]:
        if idx - prev_idx > fps * 0.5:  # gap > 0.5s = new region
            regions.append((start_idx, prev_idx))
            start_idx = idx
        prev_idx = idx
    regions.append((start_idx, prev_idx))

    # Pick the longest region
    longest = max(regions, key=lambda r: r[1] - r[0])
    start_frame, end_frame = longest

    # Add padding
    pad_frames = int(pad_sec * fps)
    start_frame = max(0, start_frame - pad_frames)
    end_frame = min(len(scores) - 1, end_frame + pad_frames)

    start_sec = round(start_frame / fps, 2)
    end_sec = round(end_frame / fps, 2)

    return (start_sec, end_sec)


def detect_with_whisper(video_path, target_word, sign_delay=0.3, sign_duration=2.5):
    """
    Use speech recognition to find when the target word is spoken.
    The sign typically happens right after or during the spoken word.

    Args:
        video_path: Path to video
        target_word: The ASL word to find
        sign_delay: Seconds after word is spoken that sign starts
        sign_duration: Expected duration of the sign

    Returns:
        List of (start_sec, end_sec) tuples for each occurrence
    """
    # Try faster-whisper first, fall back to openai-whisper
    try:
        return _detect_whisper_faster(video_path, target_word, sign_delay, sign_duration)
    except ImportError:
        pass

    try:
        return _detect_whisper_openai(video_path, target_word, sign_delay, sign_duration)
    except ImportError:
        print("  Neither faster-whisper nor openai-whisper installed.")
        print("  Install with: pip install faster-whisper")
        return []


def _detect_whisper_faster(video_path, target_word, sign_delay, sign_duration):
    """Use faster-whisper for word-level timestamp detection."""
    from faster_whisper import WhisperModel

    # Extract audio to temp wav
    audio_path = str(video_path) + '.temp.wav'
    cmd = ['ffmpeg', '-y', '-i', str(video_path), '-ac', '1', '-ar', '16000',
           '-f', 'wav', audio_path]
    subprocess.run(cmd, capture_output=True, timeout=30)

    try:
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path, word_timestamps=True)

        clips = []
        target_lower = target_word.lower()

        for segment in segments:
            if segment.words:
                for w in segment.words:
                    if target_lower in w.word.lower().strip('.,!?'):
                        start = round(w.start + sign_delay, 2)
                        end = round(start + sign_duration, 2)
                        clips.append((start, end))
                        print(f"    Found '{w.word}' at {w.start:.1f}s → clip {start}-{end}s")
    finally:
        Path(audio_path).unlink(missing_ok=True)

    return clips


def _detect_whisper_openai(video_path, target_word, sign_delay, sign_duration):
    """Use openai-whisper for word-level timestamp detection."""
    import whisper

    # Extract audio
    audio_path = str(video_path) + '.temp.wav'
    cmd = ['ffmpeg', '-y', '-i', str(video_path), '-ac', '1', '-ar', '16000',
           '-f', 'wav', audio_path]
    subprocess.run(cmd, capture_output=True, timeout=30)

    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, word_timestamps=True)

        clips = []
        target_lower = target_word.lower()

        for segment in result.get('segments', []):
            for w in segment.get('words', []):
                word_text = w.get('word', '').lower().strip('.,!? ')
                if target_lower in word_text:
                    start = round(w['start'] + sign_delay, 2)
                    end = round(start + sign_duration, 2)
                    clips.append((start, end))
                    print(f"    Found '{word_text}' at {w['start']:.1f}s → clip {start}-{end}s")
    finally:
        Path(audio_path).unlink(missing_ok=True)

    return clips


def auto_detect_clips(video_path, target_word, video_duration=None):
    """
    Automatically detect sign clip boundaries.

    Strategy:
    - Short videos (< 15s): Use motion detection (likely single-sign demo)
    - Longer videos: Use speech recognition to find word, then motion-refine

    Returns:
        List of {"start_sec": float, "end_sec": float, "method": str}
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return []

    # Get video duration if not provided
    if video_duration is None:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_duration = frames / fps if fps > 0 else 0
        cap.release()

    clips = []

    if video_duration < 15:
        # Short video — likely single sign, use motion detection
        print(f"  Short video ({video_duration:.1f}s) — using motion detection")
        boundaries = detect_motion_boundaries(video_path)
        if boundaries:
            clips.append({
                'start_sec': boundaries[0],
                'end_sec': boundaries[1],
                'method': 'motion',
            })
    else:
        # Longer video — try speech recognition first
        print(f"  Longer video ({video_duration:.1f}s) — trying speech recognition")
        whisper_clips = detect_with_whisper(video_path, target_word)

        if whisper_clips:
            for start, end in whisper_clips:
                clips.append({
                    'start_sec': start,
                    'end_sec': end,
                    'method': 'whisper',
                })
        else:
            # Fallback: motion detection on the full video
            print(f"  Speech detection failed — falling back to motion detection")
            boundaries = detect_motion_boundaries(video_path)
            if boundaries:
                clips.append({
                    'start_sec': boundaries[0],
                    'end_sec': boundaries[1],
                    'method': 'motion_fallback',
                })

    return clips


def auto_annotate_config(config_path, raw_videos_dir=None):
    """
    Scan downloaded videos and auto-generate clip_annotations in the config.

    Reads the config, finds all downloaded videos, runs auto-detection,
    and writes the clip timestamps back to the config.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        config = json.load(f)

    base_dir = config_path.parent
    if raw_videos_dir is None:
        raw_videos_dir = base_dir / config['output']['raw_videos_dir']
    raw_videos_dir = Path(raw_videos_dir)

    annotations = config.get('clip_annotations', {})
    total_clips = 0

    for word in config['target_words']:
        word_dir = raw_videos_dir / word.lower()
        if not word_dir.exists():
            continue

        existing = annotations.get(word, [])
        existing_urls = {c.get('url', '') for c in existing if c.get('start_sec') is not None}

        videos = list(word_dir.glob('*.mp4'))
        if not videos:
            continue

        print(f"\n{'='*50}")
        print(f"Auto-detecting clips for: {word.upper()} ({len(videos)} videos)")
        print(f"{'='*50}")

        word_annotations = list(existing)  # preserve existing

        for video_path in videos:
            vid_id = video_path.stem
            url = f'https://www.youtube.com/watch?v={vid_id}'

            # Skip if already annotated
            if url in existing_urls:
                print(f"  Already annotated: {vid_id}")
                continue

            print(f"\n  Processing: {vid_id}")
            detected = auto_detect_clips(video_path, word)

            for clip in detected:
                entry = {
                    'url': url,
                    'start_sec': clip['start_sec'],
                    'end_sec': clip['end_sec'],
                    'method': clip['method'],
                    'label': word,
                    'source_file': str(video_path.name),
                }
                word_annotations.append(entry)
                total_clips += 1
                print(f"    Clip: {clip['start_sec']}s - {clip['end_sec']}s ({clip['method']})")

        annotations[word] = word_annotations

    config['clip_annotations'] = annotations

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nAuto-detected {total_clips} clips. Updated: {config_path}")
    print("Review the annotations, then run: python crawl_asl_videos.py --clip")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Auto-detect ASL clip boundaries')
    parser.add_argument('--config', default=None, help='Path to crawler config')
    parser.add_argument('--video', help='Detect clips in a single video')
    parser.add_argument('--word', help='Target word (required with --video)')
    args = parser.parse_args()

    if args.video:
        if not args.word:
            print("ERROR: --word required with --video")
            sys.exit(1)
        clips = auto_detect_clips(args.video, args.word)
        print(f"\nDetected {len(clips)} clip(s):")
        for c in clips:
            print(f"  {c['start_sec']}s - {c['end_sec']}s ({c['method']})")
    else:
        base_dir = Path(__file__).parent
        config_path = Path(args.config) if args.config else base_dir / 'crawler_config.json'
        auto_annotate_config(config_path)
