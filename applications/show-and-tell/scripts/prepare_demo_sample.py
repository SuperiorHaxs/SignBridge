#!/usr/bin/env python3
"""
Prepare Demo Sample

This script takes a signing video and reference sentence, runs the full pipeline,
and saves all outputs as a pre-computed demo sample.

Usage:
    python prepare_demo_sample.py \
        --video path/to/signing_video.mp4 \
        --reference "The woman helps the deaf child" \
        --name "Woman Helps Child" \
        --id woman-helps-child \
        --output ../demo-data/samples/woman-helps-child/

Or for interactive mode:
    python prepare_demo_sample.py --interactive
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
import pickle
from pathlib import Path
from datetime import datetime

# Add project paths
SCRIPT_DIR = Path(__file__).parent
APP_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = APP_DIR.parent.parent
APPLICATIONS_DIR = PROJECT_ROOT / "applications"
PROJECT_UTILITIES_DIR = PROJECT_ROOT / "project-utilities"
MODELS_DIR = PROJECT_ROOT / "models"

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
        if pose_sequence.shape[1] == 543:
            pose_75pt = np.concatenate([
                pose_sequence[:, 0:33, :],      # Pose landmarks
                pose_sequence[:, 501:522, :],   # Left hand landmarks
                pose_sequence[:, 522:543, :]    # Right hand landmarks
            ], axis=1)
        elif pose_sequence.shape[1] == 75:
            pose_75pt = pose_sequence
        else:
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


def prepare_sample(video_path, reference, name, sample_id, output_dir, description=""):
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

    # Step 4: Segment the pose
    print("\n[4/8] Segmenting pose...")
    try:
        from motion_based_segmenter import MotionBasedSegmenter

        segmenter = MotionBasedSegmenter(
            velocity_threshold=0.02,
            min_sign_duration=10,
            min_rest_duration=5,
            padding_before=3,
            padding_after=3
        )

        # segment_pose_file returns list of file paths
        segment_files = segmenter.segment_pose_file(
            str(pose_path),
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

        checkpoint_path = MODELS_DIR / "training-scripts" / "models" / "wlasl_50_class_model"
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
        llm = create_llm_provider("google")
        llm_sentence = llm.generate(prompt)
        print(f"  LLM sentence: {llm_sentence}")

    except Exception as e:
        print(f"  WARNING: LLM construction failed: {e}")
        llm_sentence = raw_sentence  # Fallback to raw

    # Step 8: Calculate evaluation metrics
    print("\n[8/8] Calculating evaluation metrics...")
    evaluation = {
        "raw": {"bleu": 0.0, "bert": 0.0, "quality": 0.0, "composite": 0.0},
        "llm": {"bleu": 0.0, "bert": 0.0, "quality": 0.0, "composite": 0.0}
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

    try:
        # Quality score (GPT-2 perplexity)
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        import torch

        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        gpt2_model.eval()

        def get_quality_score(text):
            inputs = gpt2_tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = gpt2_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
            perplexity = min(torch.exp(torch.tensor(loss)).item(), 1000)
            return max(0, 100 - perplexity)

        evaluation["raw"]["quality"] = get_quality_score(raw_sentence)
        evaluation["llm"]["quality"] = get_quality_score(llm_sentence)
        print(f"  Quality - Raw: {evaluation['raw']['quality']:.1f}, LLM: {evaluation['llm']['quality']:.1f}")
    except Exception as e:
        print(f"  WARNING: Quality calculation failed: {e}")

    # Composite score
    weights = {'bleu': 0.15, 'bert': 0.20, 'quality': 0.40, 'composite': 0.25}
    for key in ['raw', 'llm']:
        evaluation[key]['composite'] = (
            evaluation[key]['bleu'] * weights['bleu'] +
            evaluation[key]['bert'] * weights['bert'] +
            evaluation[key]['quality'] * weights['quality']
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
    parser = argparse.ArgumentParser(description="Prepare demo sample for ASL Show & Tell")
    parser.add_argument("--video", help="Path to signing video file")
    parser.add_argument("--reference", help="Reference English sentence")
    parser.add_argument("--name", help="Display name for the sample")
    parser.add_argument("--id", help="Sample ID (used in URLs and folder names)")
    parser.add_argument("--output", help="Output directory for the sample")
    parser.add_argument("--description", help="Optional description", default="")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.video and args.reference and args.name and args.id:
        output_dir = args.output or (APP_DIR / "demo-data" / "samples" / args.id)
        prepare_sample(args.video, args.reference, args.name, args.id, output_dir, args.description)
    else:
        print("Usage: python prepare_demo_sample.py --video VIDEO --reference SENTENCE --name NAME --id ID")
        print("   or: python prepare_demo_sample.py --interactive")
        parser.print_help()


if __name__ == "__main__":
    main()
