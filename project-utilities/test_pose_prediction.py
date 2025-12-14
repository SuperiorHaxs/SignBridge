#!/usr/bin/env python3
"""
test_pose_prediction.py

Simple CLI utility to test model prediction on a single .pose file.
Handles conversion to pickle format internally.

Usage:
    python test_pose_prediction.py path/to/file.pose
    python test_pose_prediction.py path/to/file.pose --checkpoint path/to/model
    python test_pose_prediction.py path/to/file.pose --top-k 10
"""

import os
import sys
import argparse
import pickle
import tempfile
from pathlib import Path

# Add paths for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

sys.path.insert(0, str(MODELS_DIR / "openhands-modernized" / "src"))
sys.path.insert(0, str(MODELS_DIR / "openhands-modernized" / "src" / "util"))


def convert_pose_to_pickle(pose_file: str) -> str:
    """
    Convert .pose file to pickle format for model inference.
    Returns path to temporary pickle file.
    """
    from pose_format import Pose
    import numpy as np

    with open(pose_file, "rb") as f:
        buffer = f.read()
        pose = Pose.read(buffer)

    pose_data = pose.body.data

    # Handle different shapes
    if len(pose_data.shape) == 4:
        # (frames, people, keypoints, dimensions) -> take first person
        pose_sequence = pose_data[:, 0, :, :]
    else:
        pose_sequence = pose_data

    # Extract 75-point subset (pose + hands, exclude face)
    # MediaPipe format: 33 pose + 468 face + 21 left hand + 21 right hand = 543 total
    if pose_sequence.shape[1] == 543:
        pose_75pt = np.concatenate([
            pose_sequence[:, 0:33, :],      # Pose landmarks
            pose_sequence[:, 501:522, :],   # Left hand landmarks
            pose_sequence[:, 522:543, :]    # Right hand landmarks
        ], axis=1)
    elif pose_sequence.shape[1] == 576:
        # 576 = 33 pose + 468 face + 21 left + 21 right + 33 world pose
        pose_75pt = np.concatenate([
            pose_sequence[:, 0:33, :],      # Pose landmarks
            pose_sequence[:, 501:522, :],   # Left hand landmarks
            pose_sequence[:, 522:543, :]    # Right hand landmarks
        ], axis=1)
    elif pose_sequence.shape[1] == 75:
        pose_75pt = pose_sequence
    else:
        print(f"  Warning: Unexpected keypoint count: {pose_sequence.shape[1]}")
        pose_75pt = pose_sequence

    # Create temp pickle file
    fd, pickle_path = tempfile.mkstemp(suffix='.pkl')
    os.close(fd)

    pickle_data = {
        'keypoints': pose_75pt[:, :, :2],  # x, y only
        'confidences': pose_75pt[:, :, 2] if pose_75pt.shape[2] > 2 else None,
        'gloss': 'UNKNOWN'
    }

    with open(pickle_path, 'wb') as f:
        pickle.dump(pickle_data, f)

    return pickle_path


def main():
    parser = argparse.ArgumentParser(
        description="Test model prediction on a single .pose file"
    )
    parser.add_argument(
        "pose_file",
        help="Path to .pose file to test"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default=str(MODELS_DIR / "training-scripts" / "models" / "wlasl_100_class_model"),
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of top predictions to show (default: 5)"
    )

    args = parser.parse_args()

    pose_file = Path(args.pose_file)
    if not pose_file.exists():
        print(f"ERROR: File not found: {pose_file}")
        sys.exit(1)

    if not pose_file.suffix == '.pose':
        print(f"ERROR: Expected .pose file, got: {pose_file.suffix}")
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print("=" * 60)
    print("Pose File Prediction Test")
    print("=" * 60)
    print(f"Pose file: {pose_file}")
    print(f"Checkpoint: {checkpoint_path}")
    print()

    # Step 1: Get pose file info
    print("[1/3] Reading pose file...")
    from pose_format import Pose
    with open(pose_file, "rb") as f:
        pose = Pose.read(f.read())

    pose_data = pose.body.data
    if len(pose_data.shape) == 4:
        frames, people, keypoints, dims = pose_data.shape
        print(f"  Shape: {frames} frames, {people} people, {keypoints} keypoints, {dims} dims")
    else:
        frames, keypoints, dims = pose_data.shape
        print(f"  Shape: {frames} frames, {keypoints} keypoints, {dims} dims")
    print(f"  FPS: {pose.body.fps}")

    # Step 2: Convert to pickle
    print("\n[2/3] Converting to pickle format...")
    pickle_path = convert_pose_to_pickle(str(pose_file))
    print(f"  Temp pickle: {pickle_path}")

    # Verify pickle
    with open(pickle_path, 'rb') as f:
        pkl_data = pickle.load(f)
    print(f"  Keypoints shape: {pkl_data['keypoints'].shape}")

    # Step 3: Run prediction
    print("\n[3/3] Running model prediction...")
    from openhands_modernized_inference import load_model_from_checkpoint, predict_pose_file

    print(f"  Loading model...")
    model, tokenizer = load_model_from_checkpoint(str(checkpoint_path))

    print(f"  Predicting...")
    result = predict_pose_file(pickle_path, model=model, tokenizer=tokenizer)

    # Clean up temp file
    os.unlink(pickle_path)

    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\nTop-1 Prediction: {result['gloss']}")
    print(f"Confidence: {result['confidence']:.2%}")

    print(f"\nTop-{args.top_k} Predictions:")
    print("-" * 40)
    for i, pred in enumerate(result['top_k_predictions'][:args.top_k], 1):
        bar_len = int(pred['confidence'] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {i}. {pred['gloss']:15s} {pred['confidence']:6.2%} {bar}")

    print("\n" + "=" * 60)

    return result


if __name__ == "__main__":
    main()
