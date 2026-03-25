"""
Diagnostic: Check if velocity features and hand-detected flags
can separate confused sign pairs:
  - NIGHT vs DAY
  - PLEASE vs HEART
  - STOMACH vs YESTERDAY

Loads pose data from the 12-sign paragraph video and compares with
ground-truth videos of the confused-with signs.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models" / "openhands-modernized" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dataset-utilities" / "landmarks-extraction"))

from pose_format import Pose


def extract_hand_velocity(pose_83):
    """Compute per-frame velocity for left and right hands."""
    left_hand = pose_83[:, 33:54, :2]
    right_hand = pose_83[:, 54:75, :2]

    left_centroid = np.mean(left_hand, axis=1)
    right_centroid = np.mean(right_hand, axis=1)

    left_vel = np.linalg.norm(np.diff(left_centroid, axis=0), axis=1)
    right_vel = np.linalg.norm(np.diff(right_centroid, axis=0), axis=1)

    fingertip_offsets = [4, 8, 12, 16, 20]
    left_tips = left_hand[:, fingertip_offsets, :]
    right_tips = right_hand[:, fingertip_offsets, :]
    left_tip_vel = np.mean([np.linalg.norm(np.diff(left_tips[:, i, :], axis=0), axis=1) for i in range(5)], axis=0)
    right_tip_vel = np.mean([np.linalg.norm(np.diff(right_tips[:, i, :], axis=0), axis=1) for i in range(5)], axis=0)

    return {
        'left_centroid_vel_mean': float(np.mean(left_vel)),
        'left_centroid_vel_max': float(np.max(left_vel)),
        'right_centroid_vel_mean': float(np.mean(right_vel)),
        'right_centroid_vel_max': float(np.max(right_vel)),
        'left_tip_vel_mean': float(np.mean(left_tip_vel)),
        'right_tip_vel_mean': float(np.mean(right_tip_vel)),
        'vel_ratio_LR': float(np.mean(left_vel) / max(np.mean(right_vel), 1e-6)),
    }


def extract_hand_detected_flags(pose_83):
    """Detect which hand is active per frame."""
    left_hand = pose_83[:, 33:54, :2]
    right_hand = pose_83[:, 54:75, :2]

    left_present = np.any(np.abs(left_hand) > 1e-4, axis=(1, 2))
    right_present = np.any(np.abs(right_hand) > 1e-4, axis=(1, 2))

    left_centroid = np.mean(left_hand, axis=1)
    right_centroid = np.mean(right_hand, axis=1)

    left_disp = np.zeros(len(pose_83))
    right_disp = np.zeros(len(pose_83))
    left_disp[1:] = np.linalg.norm(np.diff(left_centroid, axis=0), axis=1)
    right_disp[1:] = np.linalg.norm(np.diff(right_centroid, axis=0), axis=1)

    active_thresh = 0.005
    left_active = left_disp > active_thresh
    right_active = right_disp > active_thresh

    left_var = np.mean(np.var(left_hand, axis=0))
    right_var = np.mean(np.var(right_hand, axis=0))

    return {
        'left_present_pct': float(np.mean(left_present) * 100),
        'right_present_pct': float(np.mean(right_present) * 100),
        'left_active_pct': float(np.mean(left_active) * 100),
        'right_active_pct': float(np.mean(right_active) * 100),
        'left_position_var': float(left_var),
        'right_position_var': float(right_var),
        'both_hands_active_pct': float(np.mean(left_active & right_active) * 100),
        'one_hand_only_pct': float(np.mean(left_active ^ right_active) * 100),
        'handedness': 'both' if np.mean(left_active & right_active) > 0.3 else ('left' if np.mean(left_active) > np.mean(right_active) else 'right'),
    }


def load_pose_segment(pose_path, start_frame, end_frame):
    """Load a segment from a .pose file and convert to 83-point format."""
    with open(pose_path, 'rb') as f:
        pose = Pose.read(f.read())
    data = np.array(pose.body.data)[:, 0, :, :]
    segment = data[start_frame:end_frame + 1]

    pose_indices = list(range(0, 33))
    left_hand_indices = list(range(501, 522))
    right_hand_indices = list(range(522, 543))
    face_indices = [33 + i for i in [1, 61, 291, 152, 107, 336, 33, 263]]
    all_indices = pose_indices + left_hand_indices + right_hand_indices + face_indices
    return segment[:, all_indices, :3]


def load_video_as_pose(video_path):
    """Convert a video to pose, return 83-point format."""
    import subprocess, tempfile, os

    with tempfile.NamedTemporaryFile(suffix='.pose', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run([
            'C:/Users/ashwi/Projects/WLASL-proj/asl-v1/venv/Scripts/video_to_pose.exe',
            '-i', video_path, '-o', tmp_path, '--format', 'mediapipe'
        ], capture_output=True, check=True)

        with open(tmp_path, 'rb') as f:
            pose = Pose.read(f.read())
        data = np.array(pose.body.data)[:, 0, :, :]

        pose_indices = list(range(0, 33))
        left_hand_indices = list(range(501, 522))
        right_hand_indices = list(range(522, 543))
        face_indices = [33 + i for i in [1, 61, 291, 152, 107, 336, 33, 263]]
        all_indices = pose_indices + left_hand_indices + right_hand_indices + face_indices
        return data[:, all_indices, :3]
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def print_comparison(label_a, features_a, label_b, features_b, feature_type):
    """Pretty-print comparison between two signs."""
    print(f"\n  {feature_type}:")
    keys = [k for k in features_a if not k.endswith('_timeseries')]
    for key in keys:
        va, vb = features_a[key], features_b[key]
        if isinstance(va, str):
            sep = "** YES **" if va != vb else "no"
            print(f"    {key:<28} {va:>12} {vb:>12}  {sep}")
        else:
            delta = abs(va - vb)
            rel = delta / max(abs(va), abs(vb), 1e-6) * 100
            sep = "** YES **" if rel > 30 else ("maybe" if rel > 15 else "no")
            print(f"    {key:<28} {va:>12.4f} {vb:>12.4f}  delta={delta:.4f}  {sep} ({rel:.0f}%)")


def main():
    pose_path = "c:/Users/ashwi/Projects/WLASL-proj/asl-v1/temp_videos/paragraph_signer13_12signs.pose"
    segments_path = "c:/Users/ashwi/Projects/WLASL-proj/asl-v1/temp_videos/paragraph_signer13_12signs_segments.json"
    video_base = "D:/Projects/WLASL/datasets/wlasl-kaggle/videos"

    with open(segments_path) as f:
        meta = json.load(f)
    segments = {s['gloss']: s for s in meta['segments']}

    confused_pairs = [
        ("NIGHT", "DAY"),
        ("PLEASE", "HEART"),
        ("STOMACH", "YESTERDAY"),
    ]

    print("Loading paragraph pose segments...")
    our_signs = {}
    for gloss in ["NIGHT", "PLEASE", "STOMACH"]:
        seg = segments[gloss]
        our_signs[gloss] = load_pose_segment(pose_path, seg['start_frame'], seg['end_frame'])
        print(f"  {gloss}: {our_signs[gloss].shape}")

    print("\nLoading confused-with sign videos...")
    confused_with = {}
    for gloss in ["day", "heart", "yesterday"]:
        video_dir = Path(video_base) / gloss
        videos = sorted(video_dir.glob("*.mp4"))
        if videos:
            print(f"  Converting {gloss} from {videos[0].name}...")
            confused_with[gloss.upper()] = load_video_as_pose(str(videos[0]))
            print(f"  {gloss.upper()}: {confused_with[gloss.upper()].shape}")

    print("\n" + "=" * 70)
    print("  CAN NEW FEATURES SEPARATE CONFUSED SIGN PAIRS?")
    print("  (>30% delta = YES, 15-30% = maybe, <15% = no)")
    print("=" * 70)

    for our_gloss, confused_gloss in confused_pairs:
        if our_gloss not in our_signs or confused_gloss not in confused_with:
            print(f"\n  SKIP: {our_gloss} vs {confused_gloss}")
            continue

        pose_ours = our_signs[our_gloss]
        pose_confused = confused_with[confused_gloss]

        print(f"\n{'#' * 70}")
        print(f"  {our_gloss} (true) vs {confused_gloss} (predicted)")
        print(f"  Frames: {our_gloss}={len(pose_ours)}, {confused_gloss}={len(pose_confused)}")
        print(f"{'#' * 70}")

        vel_ours = extract_hand_velocity(pose_ours)
        vel_confused = extract_hand_velocity(pose_confused)
        print_comparison(our_gloss, vel_ours, confused_gloss, vel_confused, "VELOCITY")

        hand_ours = extract_hand_detected_flags(pose_ours)
        hand_confused = extract_hand_detected_flags(pose_confused)
        print_comparison(our_gloss, hand_ours, confused_gloss, hand_confused, "HAND DETECTION")

    print(f"\n{'=' * 70}")
    print("  SIGN DESCRIPTIONS (what the features should capture):")
    print("  PLEASE: one hand circles chest  |  HEART: both hands on chest")
    print("  NIGHT: hand arcs downward        |  DAY: hand arcs upward")
    print("  STOMACH: hand pats stomach       |  YESTERDAY: thumb brushes cheek")
    print("=" * 70)


if __name__ == "__main__":
    main()
