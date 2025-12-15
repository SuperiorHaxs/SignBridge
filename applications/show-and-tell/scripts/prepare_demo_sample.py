#!/usr/bin/env python3
"""
Prepare Demo Sample

This script creates demo samples either from:
1. A signing video file (--video mode)
2. Glosses from the test dataset (--glosses mode)

Usage (video mode):
    python prepare_demo_sample.py \
        --video path/to/signing_video.mp4 \
        --reference "The woman helps the deaf child" \
        --name "Woman Helps Child" \
        --id woman-helps-child

Usage (glosses mode):
    python prepare_demo_sample.py \
        --glosses "book,read,enjoy" \
        --reference "I enjoy reading books" \
        --name "Reading Books"

    python prepare_demo_sample.py --glosses "drink,water" --reference "Drink water"
    python prepare_demo_sample.py --list  # Show available glosses

Or for interactive mode:
    python prepare_demo_sample.py --interactive
"""

import os
import sys
import json
import copy
import shutil
import random
import argparse
import subprocess
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project paths
SCRIPT_DIR = Path(__file__).parent
APP_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = APP_DIR.parent.parent
APPLICATIONS_DIR = PROJECT_ROOT / "applications"
PROJECT_UTILITIES_DIR = PROJECT_ROOT / "project-utilities"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"

sys.path.insert(0, str(APPLICATIONS_DIR))
sys.path.insert(0, str(PROJECT_UTILITIES_DIR))
sys.path.insert(0, str(PROJECT_UTILITIES_DIR / "llm_interface"))
sys.path.insert(0, str(PROJECT_UTILITIES_DIR / "synthetic_evaluation"))
# OpenHands paths - src first (for openhands_modernized module), then util (for inference)
OPENHANDS_SRC = MODELS_DIR / "openhands-modernized" / "src"
sys.path.insert(0, str(OPENHANDS_SRC))
sys.path.insert(0, str(OPENHANDS_SRC / "util"))

# Paths to executables
VENV_SCRIPTS = Path(sys.executable).parent
VIDEO_TO_POSE_EXE = VENV_SCRIPTS / "video_to_pose.exe"
VISUALIZE_POSE_EXE = VENV_SCRIPTS / "visualize_pose.exe"

# LLM prompt path
LLM_PROMPT_PATH = APPLICATIONS_DIR / "llm_prompt_topk.txt"

# Import pose_format
try:
    from pose_format import Pose
    POSE_FORMAT_AVAILABLE = True
except ImportError:
    POSE_FORMAT_AVAILABLE = False
    print("WARNING: pose_format library not available")


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"  Running: {description}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False
    print(f"  Done: {description}")
    return True


def convert_pose_to_pickle(pose_file):
    """
    Convert .pose file to pickle format for model inference.
    Same logic as app.py convert_pose_to_pickle function.
    """
    try:
        from pose_format import Pose
        import numpy as np
        import pickle

        with open(pose_file, "rb") as f:
            buffer = f.read()
            pose = Pose.read(buffer)

        pose_data = pose.body.data

        if len(pose_data.shape) == 4:
            pose_sequence = pose_data[:, 0, :, :]
        else:
            pose_sequence = pose_data

        # Extract 75-point subset (pose + hands, exclude face)
        # MediaPipe formats:
        #   543: 33 pose + 468 face + 21 left hand + 21 right hand
        #   576: 33 pose + 468 face + 21 left hand + 21 right hand + 33 world coords
        if pose_sequence.shape[1] == 543:
            pose_75pt = np.concatenate([
                pose_sequence[:, 0:33, :],      # Pose landmarks
                pose_sequence[:, 501:522, :],   # Left hand landmarks
                pose_sequence[:, 522:543, :]    # Right hand landmarks
            ], axis=1)
        elif pose_sequence.shape[1] == 576:
            # Same as 543 format, just ignore the extra 33 world coords at the end
            pose_75pt = np.concatenate([
                pose_sequence[:, 0:33, :],      # Pose landmarks
                pose_sequence[:, 501:522, :],   # Left hand landmarks
                pose_sequence[:, 522:543, :]    # Right hand landmarks
            ], axis=1)
        elif pose_sequence.shape[1] == 75:
            pose_75pt = pose_sequence
        else:
            print(f"  WARNING: Unknown keypoint count {pose_sequence.shape[1]}, using as-is")
            pose_75pt = pose_sequence

        # Create pickle file
        pickle_path = str(pose_file).replace('.pose', '.pkl')

        pickle_data = {
            'keypoints': pose_75pt[:, :, :2],
            'confidences': pose_75pt[:, :, 2] if pose_75pt.shape[2] > 2 else np.ones(pose_75pt.shape[:2]),
            'gloss': 'UNKNOWN'
        }

        with open(pickle_path, 'wb') as f:
            pickle.dump(pickle_data, f)

        return pickle_path

    except Exception as e:
        print(f"  Error converting pose to pickle: {e}")
        return None


def resample_video_to_30fps(input_path: str, output_path: str, target_fps: int = 30):
    """Resample video to target FPS using imageio with H.264 codec."""
    try:
        import imageio.v3 as iio
        import numpy as np
        import cv2

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"  ERROR: Failed to open video: {input_path}")
            return False

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"  Video: {width}x{height}, {original_fps} FPS, {frame_count} frames")

        if original_fps > target_fps * 1.5:
            frame_skip = max(1, int(original_fps / target_fps))
        else:
            frame_skip = 1

        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            frame_idx += 1

        cap.release()

        if not frames:
            print("  ERROR: No frames extracted")
            return False

        print(f"  Writing {len(frames)} frames at {target_fps} FPS")
        frames_array = np.stack(frames)
        iio.imwrite(output_path, frames_array, fps=target_fps, codec='libx264', plugin='pyav')
        print(f"  Created: {output_path}")
        return True

    except Exception as e:
        print(f"  ERROR: Video resampling failed: {e}")
        return False


# ============================================================================
# GLOSS-BASED SAMPLE CREATION FUNCTIONS
# ============================================================================

def load_settings():
    """Load settings from config/settings.json"""
    settings_path = CONFIG_DIR / "settings.json"
    if settings_path.exists():
        with open(settings_path) as f:
            return json.load(f)
    return {}


def get_dataset_path(num_classes=50):
    """Get the path to the pose test dataset."""
    settings = load_settings()

    # Try settings first - use data_root which points to the dataset base
    if 'data_root' in settings:
        base_path = Path(settings['data_root']) / "dataset_splits"
    else:
        base_path = PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "dataset_splits"

    # Build path to test poses
    test_path = base_path / f"{num_classes}_classes" / "original" / f"pose_split_{num_classes}_class" / "test"

    if not test_path.exists():
        print(f"ERROR: Test dataset not found at: {test_path}")
        return None

    return test_path


def find_pose_file_for_gloss(gloss: str, test_path: Path) -> Path:
    """Find a pose file for the given gloss in the test dataset."""
    gloss_lower = gloss.lower().strip()
    gloss_dir = test_path / gloss_lower

    if not gloss_dir.exists():
        # Try to find case-insensitive match
        for d in test_path.iterdir():
            if d.is_dir() and d.name.lower() == gloss_lower:
                gloss_dir = d
                break
        else:
            return None

    # Get all pose files in the gloss directory
    pose_files = list(gloss_dir.glob("*.pose"))

    if not pose_files:
        return None

    # Return a random pose file
    return random.choice(pose_files)


def load_pose_file(pose_path: Path) -> Pose:
    """Load a pose file."""
    with open(pose_path, "rb") as f:
        return Pose.read(f.read())


def normalize_pose_to_center(pose: Pose, target_center=(256, 256), target_scale=200) -> Pose:
    """
    Normalize a pose to be centered at a target position with consistent scale.
    This ensures poses from different videos align properly when concatenated.

    Args:
        pose: Input Pose object
        target_center: (x, y) center position in output space
        target_scale: Target scale (distance from center to extremities)

    Returns:
        New Pose object with normalized coordinates
    """
    data = pose.body.data.copy()

    # Handle 4D data (frames, people, keypoints, dims)
    if len(data.shape) == 4:
        # Work with first person
        coords = data[:, 0, :, :2]  # (frames, keypoints, 2)
    else:
        coords = data[:, :, :2]  # (frames, keypoints, 2)

    # Find bounding box across all frames (ignoring zero/invalid points)
    valid_mask = np.any(coords != 0, axis=-1)  # (frames, keypoints)

    if not np.any(valid_mask):
        # No valid points, return as-is
        return pose

    # Get valid coordinates
    valid_coords = coords[valid_mask]  # (N, 2)

    # Calculate current bounding box
    min_xy = np.min(valid_coords, axis=0)
    max_xy = np.max(valid_coords, axis=0)
    current_center = (min_xy + max_xy) / 2
    current_size = np.max(max_xy - min_xy)

    if current_size < 1e-6:
        # Degenerate case, return as-is
        return pose

    # Calculate scale factor
    scale_factor = target_scale / (current_size / 2)

    # Apply transformation: center and scale
    # new_coord = (old_coord - current_center) * scale_factor + target_center
    if len(data.shape) == 4:
        # Transform x,y coordinates for all frames and people
        for p in range(data.shape[1]):
            xy = data[:, p, :, :2]
            # Only transform non-zero points
            mask = np.any(xy != 0, axis=-1, keepdims=True)
            transformed = (xy - current_center) * scale_factor + np.array(target_center)
            data[:, p, :, :2] = np.where(mask, transformed, xy)
    else:
        xy = data[:, :, :2]
        mask = np.any(xy != 0, axis=-1, keepdims=True)
        transformed = (xy - current_center) * scale_factor + np.array(target_center)
        data[:, :, :2] = np.where(mask, transformed, xy)

    # Create new pose with normalized data
    normalized_pose = Pose(
        header=copy.deepcopy(pose.header),
        body=type(pose.body)(
            data=data,
            confidence=pose.body.confidence.copy() if pose.body.confidence is not None else None,
            fps=pose.body.fps
        )
    )

    return normalized_pose


def create_rest_pose(reference_pose: Pose, duration_seconds: float = 1.5) -> Pose:
    """
    Create a 'rest' pose segment with hands at rest position.
    Uses the last frame of the reference pose as template.
    """
    fps = reference_pose.body.fps
    num_frames = int(duration_seconds * fps)

    # Get last frame as template (keep all dimensions)
    if len(reference_pose.body.data.shape) == 4:
        last_frame = reference_pose.body.data[-1:, :, :, :]
    else:
        last_frame = reference_pose.body.data[-1:, :, :]

    # Repeat the last frame for the rest duration
    rest_data = np.repeat(last_frame, num_frames, axis=0)

    # Handle confidence
    if reference_pose.body.confidence is not None:
        rest_confidence = np.repeat(reference_pose.body.confidence[-1:], num_frames, axis=0)
    else:
        rest_confidence = None

    # Create rest pose with same structure
    rest_pose = Pose(
        header=copy.deepcopy(reference_pose.header),
        body=type(reference_pose.body)(
            data=rest_data,
            confidence=rest_confidence,
            fps=fps
        )
    )

    return rest_pose


def concatenate_poses(poses: list, gap_seconds: float = 1.5, normalize: bool = True) -> Pose:
    """
    Concatenate multiple pose files with rest gaps between them.

    Args:
        poses: List of Pose objects
        gap_seconds: Seconds of rest between each sign
        normalize: If True, normalize each pose to consistent center/scale before concatenating

    Returns:
        Single concatenated Pose object
    """
    if not poses:
        raise ValueError("No poses to concatenate")

    if len(poses) == 1:
        if normalize:
            return normalize_pose_to_center(poses[0])
        return poses[0]

    # Normalize each pose to consistent coordinate space before concatenating
    if normalize:
        poses = [normalize_pose_to_center(p) for p in poses]

    # Use first pose as reference for structure
    reference = poses[0]
    fps = reference.body.fps

    # Build concatenated data
    all_data = []
    all_confidence = []

    for i, pose in enumerate(poses):
        # Add the sign pose data
        all_data.append(pose.body.data)
        if pose.body.confidence is not None:
            all_confidence.append(pose.body.confidence)

        # Add rest gap after each sign (except last)
        if i < len(poses) - 1:
            rest_pose = create_rest_pose(pose, gap_seconds)
            all_data.append(rest_pose.body.data)
            if rest_pose.body.confidence is not None:
                all_confidence.append(rest_pose.body.confidence)

    # Concatenate along frame axis
    concatenated_data = np.concatenate(all_data, axis=0)
    concatenated_confidence = np.concatenate(all_confidence, axis=0) if all_confidence else None

    # Create final pose
    result = Pose(
        header=copy.deepcopy(reference.header),
        body=type(reference.body)(
            data=concatenated_data,
            confidence=concatenated_confidence,
            fps=fps
        )
    )

    return result


def list_available_glosses(num_classes: int = 50):
    """List all available glosses in the test dataset."""
    test_path = get_dataset_path(num_classes)
    if test_path is None:
        return

    glosses = sorted([d.name for d in test_path.iterdir() if d.is_dir()])

    print(f"\nAvailable glosses in {num_classes}-class dataset ({len(glosses)} total):")
    print("-" * 60)

    # Print in columns
    cols = 5
    for i in range(0, len(glosses), cols):
        row = glosses[i:i+cols]
        print("  " + "  ".join(f"{g:12s}" for g in row))

    print()


def prepare_sample_from_glosses(glosses, reference, name, sample_id, output_dir, description="", gap_seconds=1.5, num_classes=50):
    """Create a demo sample from a list of glosses."""

    if not POSE_FORMAT_AVAILABLE:
        print("ERROR: pose_format library is required for glosses mode")
        return False

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Creating Demo Sample from Glosses")
    print(f"{'='*60}")
    print(f"Glosses: {', '.join(glosses)}")
    print(f"Reference: {reference}")
    print(f"Gap between signs: {gap_seconds} seconds")
    print(f"Output: {output_dir}")
    print()

    # Step 1: Find pose files for each gloss
    print("[1/7] Finding pose files for glosses...")
    test_path = get_dataset_path(num_classes)
    if test_path is None:
        return False

    pose_files = []
    found_glosses = []
    missing_glosses = []

    for gloss in glosses:
        pose_file = find_pose_file_for_gloss(gloss, test_path)
        if pose_file:
            pose_files.append(pose_file)
            found_glosses.append(gloss.upper())
            print(f"  {gloss.upper()}: {pose_file.name}")
        else:
            missing_glosses.append(gloss)
            print(f"  {gloss.upper()}: NOT FOUND")

    if missing_glosses:
        print(f"\nERROR: Could not find pose files for: {', '.join(missing_glosses)}")
        print(f"Use --list to see available glosses")
        return False

    # Step 2: Load and concatenate poses
    print(f"\n[2/7] Loading and concatenating {len(pose_files)} pose files...")
    poses = [load_pose_file(pf) for pf in pose_files]

    for i, (pose, gloss) in enumerate(zip(poses, found_glosses)):
        frames = pose.body.data.shape[0]
        fps = pose.body.fps
        duration = frames / fps
        print(f"  {gloss}: {frames} frames, {fps} FPS, {duration:.2f}s")

    # Create TWO versions:
    # 1. Original coordinates for segmentation/prediction (model expects original scale)
    # 2. Normalized coordinates for visualization (centered display)
    concatenated_pose_original = concatenate_poses(poses, gap_seconds, normalize=False)
    concatenated_pose_normalized = concatenate_poses(poses, gap_seconds, normalize=True)

    total_frames = concatenated_pose_original.body.data.shape[0]
    total_duration = total_frames / concatenated_pose_original.body.fps
    print(f"  Concatenated: {total_frames} frames, {total_duration:.2f}s total")

    # Step 3: Save concatenated pose (original for segmentation/prediction)
    print(f"\n[3/7] Saving concatenated pose file...")
    pose_path = output_dir / "capture.pose"
    with open(pose_path, 'wb') as f:
        concatenated_pose_original.write(f)
    print(f"  Saved: {pose_path}")

    # Also save normalized version for visualization
    pose_path_viz = output_dir / "capture_viz.pose"
    with open(pose_path_viz, 'wb') as f:
        concatenated_pose_normalized.write(f)

    # Step 4: Visualize full pose (use normalized version for proper centering)
    print(f"\n[4/7] Generating pose visualization...")
    pose_video_raw = output_dir / "pose_video_raw.mp4"
    pose_video = output_dir / "pose_video.mp4"

    cmd = [
        str(VISUALIZE_POSE_EXE),
        "-i", str(pose_path_viz),  # Use normalized version
        "-o", str(pose_video_raw),
        "--normalize"
    ]
    if run_command(cmd, "visualize_pose"):
        resample_video_to_30fps(str(pose_video_raw), str(pose_video))
        pose_video_raw.unlink(missing_ok=True)
        pose_path_viz.unlink(missing_ok=True)  # Clean up temp file

    # Step 5: Segment the pose (using motion-based since no video source)
    print(f"\n[5/7] Segmenting pose file...")
    try:
        sys.path.insert(0, str(APP_DIR))
        from motion_based_segmenter import MotionBasedSegmenter

        # min_rest_duration must be less than actual gap frames
        # gap_seconds=1.5 at ~24 FPS = ~36 frames, so use 30 to be safe
        segmenter = MotionBasedSegmenter(
            velocity_threshold=0.02,
            min_sign_duration=10,
            max_sign_duration=120,
            min_rest_duration=30,  # ~1 second - less than 1.5s gap at any FPS
            padding_before=3,
            padding_after=3
        )

        segment_files = segmenter.segment_pose_file(
            str(pose_path),
            str(segments_dir)
        )

        segments_data = []
        for i, seg_file in enumerate(segment_files):
            seg_id = i + 1
            segments_data.append({
                'segment_id': seg_id,
                'pose_file': f"segments/segment_{seg_id:03d}.pose",
                'video_file': f"segments/segment_{seg_id:03d}.mp4",
                'expected_gloss': found_glosses[i] if i < len(found_glosses) else "UNKNOWN"
            })

        print(f"  Detected {len(segments_data)} segments (expected {len(glosses)})")

    except Exception as e:
        print(f"  ERROR: Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 6: Visualize segments and run predictions
    print(f"\n[6/7] Visualizing segments and running predictions...")
    try:
        from openhands_modernized_inference import load_model_from_checkpoint, predict_pose_file

        checkpoint_path = MODELS_DIR / "training-scripts" / "models" / f"wlasl_{num_classes}_class_model"
        print(f"  Loading model from: {checkpoint_path}")
        model, tokenizer = load_model_from_checkpoint(str(checkpoint_path))

        for seg in segments_data:
            seg_id = seg['segment_id']
            seg_pose = segments_dir / f"segment_{seg_id:03d}.pose"
            seg_pose_viz = segments_dir / f"segment_{seg_id:03d}_viz.pose"
            seg_video_raw = segments_dir / f"segment_{seg_id:03d}_raw.mp4"
            seg_video = segments_dir / f"segment_{seg_id:03d}.mp4"

            if seg_pose.exists():
                # In glosses mode, the segments come from an already-normalized
                # concatenated pose, so we don't need to re-normalize.
                # Just visualize the segment directly.
                cmd = [
                    str(VISUALIZE_POSE_EXE),
                    "-i", str(seg_pose),
                    "-o", str(seg_video_raw),
                    "--normalize"
                ]
                if run_command(cmd, f"visualize segment {seg_id}"):
                    resample_video_to_30fps(str(seg_video_raw), str(seg_video))
                    seg_video_raw.unlink(missing_ok=True)

                # Predict using original pose (model expects original coordinates)
                pickle_path = convert_pose_to_pickle(str(seg_pose))
                if pickle_path:
                    prediction = predict_pose_file(
                        pickle_path,
                        model=model,
                        tokenizer=tokenizer
                    )
                    seg['top_1'] = prediction['gloss']
                    seg['confidence'] = prediction['confidence']
                    seg['top_k'] = prediction['top_k_predictions'][:5]

                    expected = seg.get('expected_gloss', '?')
                    match = "OK" if seg['top_1'].upper() == expected.upper() else "MISMATCH"
                    print(f"  Segment {seg_id}: {seg['top_1']} ({seg['confidence']:.1%}) [{match}, expected: {expected}]")
                else:
                    seg['top_1'] = "UNKNOWN"
                    seg['confidence'] = 0.0
                    seg['top_k'] = []

    except Exception as e:
        print(f"  ERROR: Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        for seg in segments_data:
            seg['top_1'] = seg.get('expected_gloss', 'UNKNOWN')
            seg['confidence'] = 0.0
            seg['top_k'] = []

    # Build raw sentence
    raw_sentence = " ".join([seg['top_1'] for seg in segments_data])
    print(f"  Raw sentence: {raw_sentence}")

    # Step 7: Run LLM construction and evaluation
    print(f"\n[7/7] Running LLM construction and evaluation...")
    llm_sentence = raw_sentence
    evaluation = {
        "raw": {"bleu": 0.0, "bert": 0.0, "quality": 0.0, "composite": 0.0},
        "llm": {"bleu": 0.0, "bert": 0.0, "quality": 0.0, "composite": 0.0}
    }

    try:
        from llm_factory import create_llm_provider

        if LLM_PROMPT_PATH.exists():
            prompt_template = LLM_PROMPT_PATH.read_text(encoding='utf-8')
        else:
            prompt_template = "Convert these ASL glosses to English: {gloss_details}"

        # Format gloss details
        gloss_details = []
        for i, seg in enumerate(segments_data, 1):
            top_k = seg.get('top_k', [])
            if top_k:
                detail = f"Position {i}:\n"
                for j, p in enumerate(top_k[:3], 1):
                    conf = p.get('confidence', 0) * 100
                    detail += f"  Option {j}: '{p['gloss']}' (confidence: {conf:.1f}%)\n"
                gloss_details.append(detail)

        prompt = prompt_template.format(gloss_details="".join(gloss_details))

        llm = create_llm_provider("googleaistudio")
        response = llm.generate(prompt)

        # Parse response
        response_text = response.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            result = json.loads(response_text)
            llm_sentence = result.get('sentence', raw_sentence)
        except json.JSONDecodeError:
            llm_sentence = response_text

        print(f"  LLM sentence: {llm_sentence}")

    except Exception as e:
        print(f"  WARNING: LLM construction failed: {e}")

    # Calculate evaluation metrics
    try:
        from calculate_sent_bleu import calculate_bleu_score
        evaluation["raw"]["bleu"] = calculate_bleu_score(raw_sentence, reference)
        evaluation["llm"]["bleu"] = calculate_bleu_score(llm_sentence, reference)
        print(f"  BLEU - Raw: {evaluation['raw']['bleu']:.1f}, LLM: {evaluation['llm']['bleu']:.1f}")
    except Exception as e:
        print(f"  WARNING: BLEU calculation failed: {e}")

    try:
        from bert_score import score as bert_score_fn
        _, _, raw_bert = bert_score_fn([raw_sentence], [reference], lang="en", verbose=False)
        _, _, llm_bert = bert_score_fn([llm_sentence], [reference], lang="en", verbose=False)
        evaluation["raw"]["bert"] = float(raw_bert[0]) * 100
        evaluation["llm"]["bert"] = float(llm_bert[0]) * 100
        print(f"  BERT - Raw: {evaluation['raw']['bert']:.1f}, LLM: {evaluation['llm']['bert']:.1f}")
    except Exception as e:
        print(f"  WARNING: BERTScore calculation failed: {e}")

    # Quality score (GPT-2 perplexity)
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        import torch

        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        gpt2_model.eval()

        def get_quality_score(text):
            if not text.strip():
                return 0.0
            inputs = gpt2_tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = gpt2_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
            perplexity = min(np.exp(loss), 1000)
            # Convert perplexity to 0-100 score (lower perplexity = higher score)
            return max(0, 100 * (1 - np.log(perplexity) / np.log(1000)))

        evaluation["raw"]["quality"] = get_quality_score(raw_sentence)
        evaluation["llm"]["quality"] = get_quality_score(llm_sentence)
        print(f"  Quality - Raw: {evaluation['raw']['quality']:.1f}, LLM: {evaluation['llm']['quality']:.1f}")
    except Exception as e:
        print(f"  WARNING: Quality calculation failed: {e}")

    # Gloss Accuracy (% of expected glosses present in generated sentence)
    def calculate_gloss_accuracy(sentence, expected_glosses):
        """Calculate what % of expected glosses appear in the sentence."""
        sentence_lower = sentence.lower()
        found = sum(1 for g in expected_glosses if g.lower() in sentence_lower)
        return (found / len(expected_glosses) * 100) if expected_glosses else 0.0

    evaluation["raw"]["gloss_accuracy"] = calculate_gloss_accuracy(raw_sentence, found_glosses)
    evaluation["llm"]["gloss_accuracy"] = calculate_gloss_accuracy(llm_sentence, found_glosses)
    print(f"  Gloss Accuracy - Raw: {evaluation['raw']['gloss_accuracy']:.1f}%, LLM: {evaluation['llm']['gloss_accuracy']:.1f}%")

    # Composite score
    weights = {'bleu': 0.15, 'bert': 0.20, 'quality': 0.30, 'gloss_accuracy': 0.35}
    for key in ['raw', 'llm']:
        evaluation[key]['composite'] = (
            evaluation[key]['bleu'] * weights['bleu'] +
            evaluation[key]['bert'] * weights['bert'] +
            evaluation[key]['quality'] * weights['quality'] +
            evaluation[key]['gloss_accuracy'] * weights['gloss_accuracy']
        )
    print(f"  Composite - Raw: {evaluation['raw']['composite']:.1f}, LLM: {evaluation['llm']['composite']:.1f}")

    # Save metadata
    print(f"\nSaving metadata...")
    metadata = {
        "id": sample_id,
        "name": name,
        "description": description or f"Demo sample from glosses: {', '.join(found_glosses)}",
        "reference_sentence": reference,
        "source_glosses": found_glosses,
        "created": datetime.now().strftime("%Y-%m-%d"),
        "precomputed": {
            "pose_file": "capture.pose",
            "pose_video": "pose_video.mp4",
            "segments": segments_data,
            "raw_sentence": raw_sentence,
            "llm_sentence": llm_sentence,
            "evaluation": evaluation
        }
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {metadata_path}")

    # Update samples.json index
    samples_json_path = APP_DIR / "demo-data" / "samples.json"
    samples_json_path.parent.mkdir(parents=True, exist_ok=True)

    if samples_json_path.exists():
        with open(samples_json_path) as f:
            samples_index = json.load(f)
    else:
        samples_index = {"samples": []}

    existing_ids = [s['id'] for s in samples_index['samples']]
    if sample_id not in existing_ids:
        samples_index['samples'].append({
            "id": sample_id,
            "name": name,
            "thumbnail": "pose_video.mp4"
        })
        with open(samples_json_path, 'w') as f:
            json.dump(samples_index, f, indent=2)
        print(f"  Updated: {samples_json_path}")

    print(f"\n{'='*60}")
    print(f"SUCCESS: Demo sample created!")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Glosses: {' '.join(found_glosses)}")
    print(f"Raw sentence: {raw_sentence}")
    print(f"LLM sentence: {llm_sentence}")
    print(f"Reference: {reference}")

    return True


# ============================================================================
# VIDEO-BASED SAMPLE CREATION (original function)
# ============================================================================

def prepare_sample(video_path, reference, name, sample_id, output_dir, description="", num_classes=50):
    """Run full pipeline and save as demo sample."""

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}")
        return False

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Preparing demo sample: {name}")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Reference: {reference}")
    print(f"Output: {output_dir}")
    print()

    # Step 1: Copy original video
    print("[1/8] Copying original video...")
    original_video_dest = output_dir / f"original_video{video_path.suffix}"
    shutil.copy2(video_path, original_video_dest)
    print(f"  Copied to: {original_video_dest}")

    # Step 2: Convert video to pose
    print("\n[2/8] Converting video to pose...")
    pose_path = output_dir / "capture.pose"
    cmd = [
        str(VIDEO_TO_POSE_EXE),
        "-i", str(video_path),
        "-o", str(pose_path),
        "--format", "mediapipe"
    ]
    if not run_command(cmd, "video_to_pose"):
        return False

    # Step 3: Visualize full pose
    print("\n[3/8] Generating pose visualization...")
    pose_video_raw = output_dir / "pose_video_raw.mp4"
    cmd = [
        str(VISUALIZE_POSE_EXE),
        "-i", str(pose_path),
        "-o", str(pose_video_raw),
        "--normalize"
    ]
    if not run_command(cmd, "visualize_pose"):
        return False

    # Resample to 30fps
    pose_video = output_dir / "pose_video.mp4"
    if not resample_video_to_30fps(str(pose_video_raw), str(pose_video)):
        return False

    # Clean up raw video
    pose_video_raw.unlink(missing_ok=True)

    # Step 4: Segment using hybrid approach (pixel motion + pose slicing)
    print("\n[4/8] Segmenting using hybrid approach (video motion + pose slicing)...")
    try:
        # Add show-and-tell app directory for hybrid_segmenter import
        sys.path.insert(0, str(APP_DIR))
        from hybrid_segmenter import HybridSegmenter

        segmenter = HybridSegmenter(
            motion_threshold=500000,   # Pixel sum threshold for motion
            cooldown_frames=45,        # 1.5 seconds at 30 FPS to end sign
            min_sign_frames=12,        # Minimum frames for valid sign
            max_sign_frames=150,       # Maximum frames before splitting
            padding_before=3,
            padding_after=3
        )

        # Hybrid segmentation: detect motion in video, slice pose file
        segment_files = segmenter.segment_video_and_pose(
            str(video_path),           # Original video for motion detection
            str(pose_path),            # Pose file to slice
            str(segments_dir)
        )

        # Convert to the data structure we need
        segments_data = []
        for i, seg_file in enumerate(segment_files):
            seg_id = i + 1
            segments_data.append({
                'segment_id': seg_id,
                'pose_file': f"segments/segment_{seg_id:03d}.pose",
                'video_file': f"segments/segment_{seg_id:03d}.mp4"
            })

        print(f"  Found {len(segments_data)} segments")

    except Exception as e:
        print(f"  ERROR: Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Visualize each segment
    print("\n[5/8] Generating segment visualizations...")
    for seg in segments_data:
        seg_id = seg['segment_id']
        seg_pose = segments_dir / f"segment_{seg_id:03d}.pose"
        seg_video_raw = segments_dir / f"segment_{seg_id:03d}_raw.mp4"
        seg_video = segments_dir / f"segment_{seg_id:03d}.mp4"

        if seg_pose.exists():
            cmd = [
                str(VISUALIZE_POSE_EXE),
                "-i", str(seg_pose),
                "-o", str(seg_video_raw),
                "--normalize"
            ]
            if run_command(cmd, f"visualize segment {seg_id}"):
                resample_video_to_30fps(str(seg_video_raw), str(seg_video))
                seg_video_raw.unlink(missing_ok=True)

    # Step 6: Run model predictions
    print("\n[6/8] Running model predictions...")
    try:
        from openhands_modernized_inference import load_model_from_checkpoint, predict_pose_file

        checkpoint_path = MODELS_DIR / "training-scripts" / "models" / f"wlasl_{num_classes}_class_model"
        print(f"  Loading model from: {checkpoint_path}")
        model, tokenizer = load_model_from_checkpoint(str(checkpoint_path))

        for seg in segments_data:
            seg_id = seg['segment_id']
            seg_pose = segments_dir / f"segment_{seg_id:03d}.pose"
            if seg_pose.exists():
                # Convert .pose to pickle format (same as app.py convert_pose_to_pickle)
                pickle_path = convert_pose_to_pickle(str(seg_pose))
                if pickle_path:
                    prediction = predict_pose_file(
                        pickle_path,
                        model=model,
                        tokenizer=tokenizer
                    )
                    seg['top_1'] = prediction['gloss']
                    seg['confidence'] = prediction['confidence']
                    seg['top_k'] = prediction['top_k_predictions'][:5]
                    print(f"  Segment {seg_id}: {seg['top_1']} ({seg['confidence']:.2%})")
                else:
                    seg['top_1'] = "UNKNOWN"
                    seg['confidence'] = 0.0
                    seg['top_k'] = []

    except Exception as e:
        print(f"  ERROR: Model prediction failed: {e}")
        import traceback
        traceback.print_exc()
        # Continue without predictions
        for seg in segments_data:
            seg['top_1'] = "UNKNOWN"
            seg['confidence'] = 0.0
            seg['top_k'] = []

    # Build raw sentence from predictions
    raw_sentence = " ".join([seg['top_1'] for seg in segments_data])
    print(f"  Raw sentence: {raw_sentence}")

    # Step 7: Run LLM construction
    print("\n[7/8] Constructing sentence with LLM...")
    llm_sentence = ""
    try:
        from llm_factory import create_llm_provider

        # Load prompt template
        if LLM_PROMPT_PATH.exists():
            prompt_template = LLM_PROMPT_PATH.read_text(encoding='utf-8')
        else:
            prompt_template = "Convert these ASL glosses to English: {glosses}"

        # Format top-k data for prompt
        topk_data = []
        for seg in segments_data:
            topk_data.append({
                "segment": seg['segment_id'],
                "predictions": seg.get('top_k', [])
            })

        # Create prompt
        prompt = prompt_template.replace("{topk_predictions}", json.dumps(topk_data, indent=2))

        # Call LLM
        llm = create_llm_provider("googleaistudio")
        llm_sentence = llm.generate(prompt)
        print(f"  LLM sentence: {llm_sentence}")

    except Exception as e:
        print(f"  WARNING: LLM construction failed: {e}")
        llm_sentence = raw_sentence  # Fallback to raw

    # Step 8: Calculate evaluation metrics
    print("\n[8/8] Calculating evaluation metrics...")
    evaluation = {
        "raw": {"bleu": 0.0, "bert": 0.0, "quality": 0.0, "gloss_accuracy": 0.0, "composite": 0.0},
        "llm": {"bleu": 0.0, "bert": 0.0, "quality": 0.0, "gloss_accuracy": 0.0, "composite": 0.0}
    }

    try:
        # BLEU scores - use direct sentence comparison
        from calculate_sent_bleu import calculate_bleu_score
        raw_bleu = calculate_bleu_score(raw_sentence, reference)
        llm_bleu = calculate_bleu_score(llm_sentence, reference)
        evaluation["raw"]["bleu"] = raw_bleu
        evaluation["llm"]["bleu"] = llm_bleu
        print(f"  BLEU - Raw: {raw_bleu:.1f}, LLM: {llm_bleu:.1f}")
    except Exception as e:
        print(f"  WARNING: BLEU calculation failed: {e}")

    try:
        # BERTScore
        from bert_score import score as bert_score_fn
        _, _, raw_bert = bert_score_fn([raw_sentence], [reference], lang="en", verbose=False)
        _, _, llm_bert = bert_score_fn([llm_sentence], [reference], lang="en", verbose=False)
        evaluation["raw"]["bert"] = float(raw_bert[0]) * 100
        evaluation["llm"]["bert"] = float(llm_bert[0]) * 100
        print(f"  BERT - Raw: {evaluation['raw']['bert']:.1f}, LLM: {evaluation['llm']['bert']:.1f}")
    except Exception as e:
        print(f"  WARNING: BERTScore calculation failed: {e}")

    # Quality score (GPT-2 perplexity)
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        import torch

        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        gpt2_model.eval()

        def get_quality_score(text):
            if not text.strip():
                return 0.0
            inputs = gpt2_tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = gpt2_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
            perplexity = min(np.exp(loss), 1000)
            # Convert perplexity to 0-100 score (lower perplexity = higher score)
            return max(0, 100 * (1 - np.log(perplexity) / np.log(1000)))

        evaluation["raw"]["quality"] = get_quality_score(raw_sentence)
        evaluation["llm"]["quality"] = get_quality_score(llm_sentence)
        print(f"  Quality - Raw: {evaluation['raw']['quality']:.1f}, LLM: {evaluation['llm']['quality']:.1f}")
    except Exception as e:
        print(f"  WARNING: Quality calculation failed: {e}")

    # Gloss Accuracy (% of predicted glosses present in reference sentence)
    # For video mode, we check if the predicted glosses appear in the reference
    def calculate_gloss_accuracy(sentence, predicted_glosses):
        """Calculate what % of predicted glosses appear in the sentence."""
        sentence_lower = sentence.lower()
        found = sum(1 for g in predicted_glosses if g.lower() in sentence_lower)
        return (found / len(predicted_glosses) * 100) if predicted_glosses else 0.0

    predicted_glosses = [seg['top_1'] for seg in segments_data if seg.get('top_1') and seg['top_1'] != 'UNKNOWN']
    evaluation["raw"]["gloss_accuracy"] = calculate_gloss_accuracy(raw_sentence, predicted_glosses)
    evaluation["llm"]["gloss_accuracy"] = calculate_gloss_accuracy(llm_sentence, predicted_glosses)
    print(f"  Gloss Accuracy - Raw: {evaluation['raw']['gloss_accuracy']:.1f}%, LLM: {evaluation['llm']['gloss_accuracy']:.1f}%")

    # Composite score (same weights as glosses mode)
    weights = {'bleu': 0.15, 'bert': 0.20, 'quality': 0.30, 'gloss_accuracy': 0.35}
    for key in ['raw', 'llm']:
        evaluation[key]['composite'] = (
            evaluation[key]['bleu'] * weights['bleu'] +
            evaluation[key]['bert'] * weights['bert'] +
            evaluation[key]['quality'] * weights['quality'] +
            evaluation[key]['gloss_accuracy'] * weights['gloss_accuracy']
        )
    print(f"  Composite - Raw: {evaluation['raw']['composite']:.1f}, LLM: {evaluation['llm']['composite']:.1f}")

    # Create metadata.json
    print("\nSaving metadata...")
    metadata = {
        "id": sample_id,
        "name": name,
        "description": description or f"Demo sample: {name}",
        "reference_sentence": reference,
        "created": datetime.now().strftime("%Y-%m-%d"),
        "original_video": original_video_dest.name,
        "precomputed": {
            "pose_file": "capture.pose",
            "pose_video": "pose_video.mp4",
            "segments": segments_data,
            "raw_sentence": raw_sentence,
            "llm_sentence": llm_sentence,
            "evaluation": evaluation
        }
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {metadata_path}")

    # Update samples.json index
    samples_json_path = APP_DIR / "demo-data" / "samples.json"
    if samples_json_path.exists():
        with open(samples_json_path) as f:
            samples_index = json.load(f)
    else:
        samples_index = {"samples": []}

    # Check if sample already exists in index
    existing_ids = [s['id'] for s in samples_index['samples']]
    if sample_id not in existing_ids:
        samples_index['samples'].append({
            "id": sample_id,
            "name": name,
            "thumbnail": "pose_video.mp4"
        })
        with open(samples_json_path, 'w') as f:
            json.dump(samples_index, f, indent=2)
        print(f"  Updated: {samples_json_path}")

    print(f"\n{'='*60}")
    print(f"SUCCESS: Demo sample prepared!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Reference: {reference}")
    print(f"Raw sentence: {raw_sentence}")
    print(f"LLM sentence: {llm_sentence}")

    return True


def interactive_mode():
    """Interactive mode for preparing samples."""
    print("\n=== Demo Sample Preparation (Interactive Mode) ===\n")

    video_path = input("Video file path: ").strip()
    if not video_path:
        print("ERROR: Video path is required")
        return

    reference = input("Reference sentence: ").strip()
    if not reference:
        print("ERROR: Reference sentence is required")
        return

    name = input("Sample name (e.g., 'Woman Helps Child'): ").strip()
    if not name:
        name = "Untitled Sample"

    # Generate ID from name
    default_id = name.lower().replace(" ", "-").replace("'", "")
    sample_id = input(f"Sample ID [{default_id}]: ").strip() or default_id

    description = input("Description (optional): ").strip()

    default_output = APP_DIR / "demo-data" / "samples" / sample_id
    output_dir = input(f"Output directory [{default_output}]: ").strip() or str(default_output)

    prepare_sample(video_path, reference, name, sample_id, output_dir, description)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare demo sample for ASL Show & Tell",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Video mode:
    python prepare_demo_sample.py --video video.mp4 --reference "Hello world" --name "Hello" --id hello

  Glosses mode:
    python prepare_demo_sample.py --glosses "book,read" --reference "Read a book"
    python prepare_demo_sample.py --glosses "drink,water" --reference "Drink water" --name "Drinking"
    python prepare_demo_sample.py --list  # Show available glosses
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--video", help="Path to signing video file (video mode)")
    mode_group.add_argument("--glosses", "-g", help="Comma-separated glosses from test dataset (glosses mode)")
    mode_group.add_argument("--list", "-l", action="store_true", help="List available glosses and exit")
    mode_group.add_argument("--interactive", action="store_true", help="Run in interactive mode")

    # Common arguments
    parser.add_argument("--reference", "-r", help="Reference English sentence")
    parser.add_argument("--name", "-n", help="Display name for the sample")
    parser.add_argument("--id", help="Sample ID (default: generated from name or glosses)")
    parser.add_argument("--output", "-o", help="Output directory for the sample")
    parser.add_argument("--description", "-d", help="Optional description", default="")

    # Glosses mode specific
    parser.add_argument("--gap", type=float, default=1.5, help="Gap between signs in seconds (default: 1.5)")
    parser.add_argument("--classes", type=int, default=50, choices=[20, 50, 100], help="Number of classes (default: 50)")

    args = parser.parse_args()

    # List available glosses
    if args.list:
        list_available_glosses(args.classes)
        return

    # Interactive mode
    if args.interactive:
        interactive_mode()
        return

    # Glosses mode
    if args.glosses:
        if not args.reference:
            print("ERROR: --reference is required")
            parser.print_help()
            return

        # Parse glosses
        glosses = [g.strip() for g in args.glosses.split(",")]

        # Generate defaults
        if not args.name:
            args.name = " ".join([g.capitalize() for g in glosses])

        if not args.id:
            args.id = "-".join([g.lower() for g in glosses])

        output_dir = args.output or (APP_DIR / "demo-data" / "samples" / args.id)

        prepare_sample_from_glosses(
            glosses=glosses,
            reference=args.reference,
            name=args.name,
            sample_id=args.id,
            output_dir=output_dir,
            description=args.description,
            gap_seconds=args.gap,
            num_classes=args.classes
        )
        return

    # Video mode
    if args.video:
        if not args.reference:
            print("ERROR: --reference is required")
            parser.print_help()
            return

        if not args.name:
            print("ERROR: --name is required for video mode")
            parser.print_help()
            return

        if not args.id:
            args.id = args.name.lower().replace(" ", "-").replace("'", "")

        output_dir = args.output or (APP_DIR / "demo-data" / "samples" / args.id)
        prepare_sample(args.video, args.reference, args.name, args.id, output_dir, args.description, args.classes)
        return

    # No mode selected
    print("Usage:")
    print("  Glosses mode: python prepare_demo_sample.py --glosses 'book,read' --reference 'Read a book'")
    print("  Video mode:   python prepare_demo_sample.py --video video.mp4 --reference 'Hello' --name 'Hello'")
    print("  List glosses: python prepare_demo_sample.py --list")
    print()
    parser.print_help()


if __name__ == "__main__":
    main()
