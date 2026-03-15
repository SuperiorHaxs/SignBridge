#!/usr/bin/env python3
"""
ASL Video Captioning Script

Takes an ASL signing video, runs the full SignBridge pipeline, and outputs
the same video with burned-in English captions that:
  1. Show raw glosses progressively as each sign is recognized
  2. Every 3 signs, the LLM constructs a sentence that REPLACES the raw glosses
  3. Rolling, self-updating captions (like real closed captions)

Usage:
    python caption_video.py --video path/to/signing_video.mp4
    python caption_video.py --video path/to/signing_video.mp4 --output captioned.mp4
"""

import os
import sys
import json
import argparse
import tempfile
import subprocess
import pickle
import numpy as np
from pathlib import Path

# Load .env for API keys (manual parsing — python-dotenv may not be installed)
def _load_env_file(env_path):
    """Load key=value pairs from .env file into os.environ."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' in line:
            key, val = line.split('=', 1)
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val

_load_env_file(Path(__file__).parent.parent.parent / "project-utilities" / "llm_interface" / ".env")

# ── Project paths ──────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
APP_DIR = SCRIPT_DIR
PROJECT_ROOT = APP_DIR.parent.parent if APP_DIR.name == "closed-captions" else APP_DIR.parent
APPLICATIONS_DIR = PROJECT_ROOT / "applications"
PROJECT_UTILITIES_DIR = PROJECT_ROOT / "project-utilities"
MODELS_DIR = PROJECT_ROOT / "models"

sys.path.insert(0, str(PROJECT_UTILITIES_DIR / "segmentation"))
sys.path.insert(0, str(PROJECT_UTILITIES_DIR / "llm_interface"))
sys.path.insert(0, str(MODELS_DIR / "openhands-modernized" / "src"))
sys.path.insert(0, str(MODELS_DIR / "openhands-modernized" / "src" / "util"))
sys.path.insert(0, str(APPLICATIONS_DIR / "show-and-tell" / "scripts"))

# ── Lazy imports (heavy libs loaded only when needed) ──────────────────
cv2 = None
torch = None

def _ensure_cv2():
    global cv2
    if cv2 is None:
        import cv2 as _cv2
        cv2 = _cv2

def _ensure_torch():
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch

# ── Paths ──────────────────────────────────────────────────────────────
MODEL_CHECKPOINT = MODELS_DIR / "openhands-modernized" / "production-models" / "wlasl_43_class_50s_model"

VENV_SCRIPTS = Path(sys.executable).parent
SCRIPTS_SUBDIR = VENV_SCRIPTS / "Scripts"

def _find_exe(name):
    if (SCRIPTS_SUBDIR / name).exists():
        return SCRIPTS_SUBDIR / name
    if (VENV_SCRIPTS / name).exists():
        return VENV_SCRIPTS / name
    return SCRIPTS_SUBDIR / name

VIDEO_TO_POSE_EXE = _find_exe("video_to_pose.exe")

LLM_PROMPT_PATH = PROJECT_UTILITIES_DIR / "llm_interface" / "prompts" / "llm_prompt_closed_captions.txt"

# Caption settings
CAPTION_BUFFER_SIZE = 3  # signs before LLM constructs a sentence


# ═══════════════════════════════════════════════════════════════════════
# PIPELINE STEPS
# ═══════════════════════════════════════════════════════════════════════

def convert_video_to_pose(video_path: str, output_dir: str) -> str:
    """Convert video to .pose file using video_to_pose executable."""
    pose_path = os.path.join(output_dir, "video.pose")
    cmd = [str(VIDEO_TO_POSE_EXE), "--format", "mediapipe", "-i", video_path, "-o", pose_path]
    print(f"[1/5] Converting video to pose...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"video_to_pose failed: {result.stderr}")
    print(f"  Created: {pose_path}")
    return pose_path


def segment_video(video_path: str, pose_path: str, output_dir: str):
    """
    Segment video into individual signs.

    Returns:
        segments: list of (start_frame, end_frame) tuples
        segment_pose_files: list of .pose file paths for each segment
    """
    from hybrid_segmenter import HybridSegmenter

    segmenter = HybridSegmenter()

    # Get frame boundaries
    print(f"[2/5] Segmenting video into signs...")
    segments = segmenter.detect_segments_from_video(video_path, verbose=True)

    if not segments:
        raise RuntimeError("No sign segments detected in video")

    # Slice pose file at those boundaries
    seg_dir = os.path.join(output_dir, "segments")
    os.makedirs(seg_dir, exist_ok=True)
    segment_pose_files = segmenter.slice_pose_file(pose_path, segments, seg_dir, verbose=True)

    print(f"  Found {len(segments)} segments")
    return segments, segment_pose_files


def run_inference(segment_pose_files: list, checkpoint_path: str = None):
    """
    Run OpenHands model inference on each segment.

    Returns:
        list of prediction dicts: {'gloss', 'confidence', 'top_k_predictions'}
    """
    _ensure_torch()
    from openhands_modernized_inference import load_model_from_checkpoint, predict_pose_file
    from prepare_demo_sample import convert_pose_to_pickle

    if checkpoint_path is None:
        checkpoint_path = str(MODEL_CHECKPOINT)

    print(f"[3/5] Running model inference on {len(segment_pose_files)} segments...")
    model, id_to_gloss, masked_class_ids = load_model_from_checkpoint(checkpoint_path)

    predictions = []
    for i, pose_file in enumerate(segment_pose_files):
        # Convert .pose → .pkl
        pickle_path = convert_pose_to_pickle(pose_file)
        if pickle_path is None:
            predictions.append({
                'gloss': 'UNKNOWN',
                'confidence': 0.0,
                'top_k_predictions': []
            })
            continue

        # Predict
        result = predict_pose_file(
            pickle_path, model=model, tokenizer=id_to_gloss,
            masked_class_ids=masked_class_ids
        )
        gloss = result['gloss']
        conf = result['confidence']
        print(f"  Segment {i+1}: {gloss} ({conf*100:.1f}%)")
        predictions.append(result)

    return predictions


def build_captions(segments, predictions, fps: float):
    """
    Build caption timeline with progressive glosses then full sentence.

    Caption behavior:
    - As each sign is recognized, the gloss appears (building up: "TIME", "TIME SON", ...)
    - After the last sign in a batch ends, the LLM sentence replaces the glosses
    - The sentence holds until the next batch of signs starts
    """
    from llm_factory import create_llm_provider

    print(f"[4/5] Building captions (buffer={CAPTION_BUFFER_SIZE})...")

    # Load LLM
    llm_provider = create_llm_provider(
        provider="googleaistudio",
        model_name="gemini-2.0-flash"
    )

    prompt_template = LLM_PROMPT_PATH.read_text(encoding='utf-8')

    caption_events = []
    gloss_buffer = []       # accumulates glosses for current sentence batch
    prediction_buffer = []  # full prediction dicts for LLM context
    sentence_history = []   # previous sentences for context

    for seg_idx, (seg, pred) in enumerate(zip(segments, predictions)):
        start_frame, end_frame = seg
        gloss = pred['gloss']
        gloss_buffer.append(gloss)
        prediction_buffer.append(pred)

        # Progressive gloss display: show accumulated raw glosses
        gloss_text = " ".join(gloss_buffer).upper()

        # Check if this is the last sign in a batch (triggers sentence)
        is_last_overall = (seg_idx == len(segments) - 1)
        triggers_sentence = (len(gloss_buffer) >= CAPTION_BUFFER_SIZE) or is_last_overall

        if triggers_sentence:
            # Gloss only shows from this sign's start to this sign's END
            # (then the sentence takes over)
            caption_events.append({
                'start_frame': start_frame,
                'end_frame': end_frame,
                'text': gloss_text,
                'type': 'gloss'
            })

            # Build LLM sentence
            sentence = _call_llm_for_sentence(
                llm_provider, prompt_template, prediction_buffer, sentence_history
            )

            if sentence:
                sentence_history.append(sentence)

                # Sentence starts right when the last sign ENDS
                # and holds until the next sign starts (or 3s after video ends)
                if seg_idx + 1 < len(segments):
                    sentence_end = segments[seg_idx + 1][0]
                else:
                    sentence_end = end_frame + int(fps * 3)

                caption_events.append({
                    'start_frame': end_frame,
                    'end_frame': sentence_end,
                    'text': sentence,
                    'type': 'sentence'
                })

                print(f"  Sentence: \"{sentence}\"")

            # Reset buffers
            gloss_buffer = []
            prediction_buffer = []
        else:
            # Not the end of a batch — gloss holds until next sign starts
            if seg_idx + 1 < len(segments):
                next_start = segments[seg_idx + 1][0]
            else:
                next_start = end_frame + int(fps * 2)

            caption_events.append({
                'start_frame': start_frame,
                'end_frame': next_start,
                'text': gloss_text,
                'type': 'gloss'
            })

    return caption_events


def _call_llm_for_sentence(llm_provider, prompt_template, predictions, sentence_history):
    """Call LLM to construct a sentence from buffered predictions."""
    # Format gloss details
    details = []
    for i, pred in enumerate(predictions, 1):
        top_k = pred.get('top_k_predictions', [])
        if top_k:
            detail = f"Position {i}:\n"
            for j, p in enumerate(top_k[:3], 1):
                conf = p.get('confidence', 0) * 100
                detail += f"  Option {j}: '{p.get('gloss', 'UNKNOWN')}' (confidence: {conf:.1f}%)\n"
            details.append(detail)
        else:
            conf = pred.get('confidence', 0) * 100
            details.append(f"Position {i}: '{pred.get('gloss', 'UNKNOWN')}' (confidence: {conf:.1f}%)\n")

    gloss_details = "".join(details)

    # Build context section
    context_section = ""
    if sentence_history:
        context_section = "Previous Sentences (for context):\n"
        for i, sent in enumerate(sentence_history[-2:], 1):
            context_section += f"  {i}. \"{sent}\"\n"

    # Fill prompt
    prompt = prompt_template.replace('{context_section}', context_section)
    prompt = prompt.replace('{gloss_details}', gloss_details)

    try:
        response = llm_provider.generate(prompt)
        return _parse_llm_response(response)
    except Exception as e:
        print(f"  LLM error: {e}")
        # Fallback: just join the top-1 glosses
        return " ".join(p.get('gloss', '?') for p in predictions)


def _parse_llm_response(response: str) -> str:
    """Extract sentence from LLM JSON response."""
    import re

    text = response.strip()
    # Strip markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        result = json.loads(text)
        return result.get('sentence', text)
    except json.JSONDecodeError:
        # Try to find sentence in response
        match = re.search(r'"sentence"\s*:\s*"([^"]+)"', text)
        if match:
            return match.group(1)
        return text


def burn_captions_onto_video(video_path: str, caption_events: list, output_path: str):
    """
    Burn caption overlay onto video frames and write output video.

    Caption events are layered: 'sentence' type overrides 'gloss' type
    for the same frame range.
    """
    _ensure_cv2()

    print(f"[5/5] Burning captions onto video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use mp4v codec for broad compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Build frame→caption lookup (glosses and sentences occupy separate frame ranges)
    frame_captions = {}
    for evt in caption_events:
        for f in range(evt['start_frame'], min(evt['end_frame'], total_frames)):
            frame_captions[f] = evt['text']

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        caption_text = frame_captions.get(frame_idx, None)
        if caption_text:
            frame = _draw_caption(frame, caption_text, width, height)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"  Output: {output_path} ({frame_idx} frames)")


def _draw_caption(frame, text: str, width: int, height: int):
    """Draw closed-caption style text overlay on frame bottom."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, width / 1000)  # scale with video width
    thickness = max(1, int(width / 500))
    color = (255, 255, 255)
    bg_color = (0, 0, 0)

    # Word-wrap text to fit video width
    max_chars = int(width / (font_scale * 18))  # rough estimate
    lines = _wrap_text(text, max_chars)

    # Calculate text block dimensions
    line_height = int(35 * font_scale)
    padding = int(10 * font_scale)
    block_height = len(lines) * line_height + 2 * padding

    # Draw semi-transparent black background at bottom
    overlay = frame.copy()
    y_start = height - block_height - 20
    cv2.rectangle(overlay, (0, y_start), (width, height), bg_color, -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Draw text lines
    for i, line in enumerate(lines):
        y = y_start + padding + (i + 1) * line_height
        # Center the text
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        x = (width - text_size[0]) // 2
        # Black outline for readability
        cv2.putText(frame, line, (x, y), font, font_scale, (0, 0, 0), thickness + 2)
        # White text
        cv2.putText(frame, line, (x, y), font, font_scale, color, thickness)

    return frame


def _wrap_text(text: str, max_chars: int) -> list:
    """Simple word-wrap for caption text."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines if lines else [""]


# ═══════════════════════════════════════════════════════════════════════
# DEMO SAMPLE MODE (uses precomputed show-and-tell data)
# ═══════════════════════════════════════════════════════════════════════

DEMO_SAMPLES_DIR = APPLICATIONS_DIR / "show-and-tell" / "demo-data" / "samples"


def list_demo_samples() -> list:
    """List available demo samples from show-and-tell."""
    samples = []
    if not DEMO_SAMPLES_DIR.exists():
        return samples

    for sample_dir in sorted(DEMO_SAMPLES_DIR.iterdir()):
        meta_path = sample_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            precomputed = meta.get('precomputed', {})
            segments = precomputed.get('segments', [])
            if segments and (sample_dir / "pose_video.mp4").exists():
                glosses = [s.get('top_1', '?') for s in segments]
                samples.append({
                    'id': meta.get('id', sample_dir.name),
                    'name': meta.get('name', sample_dir.name),
                    'reference': meta.get('reference_sentence', ''),
                    'glosses': glosses,
                    'path': str(sample_dir),
                })
    return samples


def caption_demo_sample(sample_dir: str, output_path: str = None) -> str:
    """
    Caption a show-and-tell demo sample using its precomputed predictions.

    Uses the exact same model predictions from the show-and-tell demo,
    then burns captions onto the pose visualization video.
    """
    _ensure_cv2()

    sample_dir = Path(sample_dir)
    meta_path = sample_dir / "metadata.json"
    video_path = sample_dir / "pose_video.mp4"

    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json in {sample_dir}")
    if not video_path.exists():
        raise FileNotFoundError(f"No pose_video.mp4 in {sample_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    precomputed = meta.get('precomputed', {})
    seg_data = precomputed.get('segments', [])
    if not seg_data:
        raise RuntimeError("No precomputed segments found in metadata")

    if output_path is None:
        output_path = str(sample_dir / "captioned_video.mp4")

    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    sample_name = meta.get('name', sample_dir.name)
    print(f"Demo sample: {sample_name}")
    print(f"Video: {total_frames} frames at {fps:.1f} FPS")
    print(f"Segments: {len(seg_data)}")

    # Step 1: Use HybridSegmenter to detect frame boundaries from the pose video
    # (the pose video has visible motion from the stick figure animation)
    from hybrid_segmenter import HybridSegmenter
    segmenter = HybridSegmenter()
    print(f"\n[1/3] Detecting segment boundaries from pose video...")
    detected_segments = segmenter.detect_segments_from_video(str(video_path), verbose=True)

    # Match detected segments to precomputed predictions
    # If counts don't match, fall back to evenly distributing across video
    n_precomputed = len(seg_data)
    if len(detected_segments) == n_precomputed:
        segments = detected_segments
        print(f"  Segment counts match ({n_precomputed})")
    else:
        print(f"  Segment count mismatch: detected {len(detected_segments)}, precomputed {n_precomputed}")
        print(f"  Falling back to even distribution across {total_frames} frames")
        # Evenly distribute segments across the video
        usable = int(total_frames * 0.9)  # use 90% of video
        seg_len = usable // n_precomputed
        offset = int(total_frames * 0.05)  # 5% margin at start
        segments = []
        for i in range(n_precomputed):
            start = offset + i * seg_len
            end = start + seg_len - 1
            segments.append((start, min(end, total_frames - 1)))

    # Step 2: Build predictions from precomputed data (same format as run_inference)
    print(f"\n[2/3] Using precomputed predictions...")
    predictions = []
    for seg in seg_data:
        pred = {
            'gloss': seg.get('top_1', 'UNKNOWN'),
            'confidence': seg.get('confidence', 0.0),
            'top_k_predictions': seg.get('top_k', [])
        }
        print(f"  {pred['gloss']} ({pred['confidence']*100:.1f}%)")
        predictions.append(pred)

    # Step 3: Build captions + burn onto video
    print(f"\n[3/3] Building captions and burning onto video...")
    caption_events = build_captions(segments, predictions, fps)
    burn_captions_onto_video(str(video_path), caption_events, output_path)

    print(f"\nDone! Captioned video: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def caption_video(video_path: str, output_path: str = None) -> str:
    """
    Full pipeline: video → pose → segment → infer → caption → burn.

    Args:
        video_path: Path to input ASL signing video
        output_path: Path for output captioned video (default: <input>_captioned.mp4)

    Returns:
        Path to the output captioned video
    """
    _ensure_cv2()

    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_captioned.mp4"

    # Get video FPS for timestamp calculations
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Input: {video_path} ({fps:.1f} FPS)")

    # Create temp working directory
    with tempfile.TemporaryDirectory(prefix="asl_caption_") as tmp_dir:
        # Step 1: Video → Pose
        pose_path = convert_video_to_pose(video_path, tmp_dir)

        # Step 2: Segment
        segments, segment_pose_files = segment_video(video_path, pose_path, tmp_dir)

        # Step 3: Inference
        predictions = run_inference(segment_pose_files)

        # Step 4: Build caption timeline
        caption_events = build_captions(segments, predictions, fps)

        # Step 5: Burn captions onto video
        burn_captions_onto_video(video_path, caption_events, output_path)

    print(f"\nDone! Captioned video saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Caption an ASL signing video with English translations"
    )
    parser.add_argument(
        "--video", "-v", required=True,
        help="Path to input ASL signing video"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path for output captioned video (default: <input>_captioned.mp4)"
    )
    parser.add_argument(
        "--buffer-size", "-b", type=int, default=3,
        help="Number of signs before LLM sentence (default: 3)"
    )

    args = parser.parse_args()

    global CAPTION_BUFFER_SIZE
    CAPTION_BUFFER_SIZE = args.buffer_size

    caption_video(args.video, args.output)


if __name__ == "__main__":
    main()
