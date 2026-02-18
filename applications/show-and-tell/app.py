#!/usr/bin/env python3
"""
Show-and-Tell Demo Application
A Flask wrapper around existing ASL pipeline libraries.

This application demonstrates the complete ASL-to-English translation pipeline
in a step-by-step workflow for science fair presentation.

NO business logic duplication - all calls go to existing libraries.
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, session

# ============================================================================
# PATH CONFIGURATION - Add project paths for imports
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
APPLICATIONS_DIR = PROJECT_ROOT / "applications"
PROJECT_UTILITIES_DIR = PROJECT_ROOT / "project-utilities"
DATASET_UTILITIES_DIR = PROJECT_ROOT / "dataset-utilities"
MODELS_DIR = PROJECT_ROOT / "models"

# Add paths for imports
sys.path.insert(0, str(APPLICATIONS_DIR))
sys.path.insert(0, str(PROJECT_UTILITIES_DIR))
sys.path.insert(0, str(PROJECT_UTILITIES_DIR / "llm_interface"))
sys.path.insert(0, str(PROJECT_UTILITIES_DIR / "segmentation"))
sys.path.insert(0, str(DATASET_UTILITIES_DIR / "categorization"))
sys.path.insert(0, str(PROJECT_UTILITIES_DIR / "synthetic_evaluation"))
sys.path.insert(0, str(MODELS_DIR / "openhands-modernized" / "src"))
sys.path.insert(0, str(MODELS_DIR / "openhands-modernized" / "src" / "util"))

# ============================================================================
# IMPORTS FROM EXISTING LIBRARIES (no duplication)
# ============================================================================

# Import segmenters
# - Hybrid: uses pixel motion detection on video + pose slicing (preferred)
# - Motion-based: uses pose keypoint velocity (fallback for pose-only uploads)
from hybrid_segmenter import HybridSegmenter
from motion_based_segmenter import MotionBasedSegmenter

# Import model inference functions
from openhands_modernized_inference import load_model_from_checkpoint, predict_pose_file

# Import common camera processor (uses VIDEO_TO_POSE_EXE for better hand detection)
from camera_processor import get_camera_processor, CameraProcessor

# Import LLM interface
from llm_factory import create_llm_provider

# Import gloss categorization
from gloss_categorizer import GlossCategorizer

# Import evaluation metrics from the modular library
from evaluation_metrics import (
    calculate_gloss_accuracy as metrics_gloss_accuracy,
    calculate_bleu_score,
    calculate_bert_score,
    calculate_quality_score as metrics_quality_score,
    calculate_composite_score_v3 as metrics_composite_score,
    calculate_coverage_v2 as calculate_coverage,
    QualityScorer,
    METRIC_WEIGHTS,
    BLEU_AVAILABLE,
    BERTSCORE_AVAILABLE,
    QUALITY_SCORING_AVAILABLE,
)

print(f"Metrics availability: BLEU={BLEU_AVAILABLE}, BERTScore={BERTSCORE_AVAILABLE}, Quality={QUALITY_SCORING_AVAILABLE}")

# Pose format for file handling
try:
    from pose_format import Pose
    POSE_FORMAT_AVAILABLE = True
except ImportError:
    POSE_FORMAT_AVAILABLE = False
    print("WARNING: pose_format not available")

import numpy as np
import pickle
import cv2
import io
import threading
import time

# Import Caption Service for closed-captions mode
from caption_service import CaptionService, config as cc_config

# MediaPipe for in-memory pose extraction (fast path)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("WARNING: MediaPipe not available - using slow path for pose extraction")

# ============================================================================
# PERSISTENT MEDIAPIPE INSTANCE (for fast in-memory pose extraction)
# ============================================================================

_holistic_instance = None
_holistic_lock = threading.Lock()

def get_holistic():
    """Get or create persistent MediaPipe holistic instance (thread-safe)."""
    global _holistic_instance
    if not MEDIAPIPE_AVAILABLE:
        return None

    with _holistic_lock:
        if _holistic_instance is None:
            print("[MediaPipe] Initializing persistent Holistic model...")
            mp_holistic = mp.solutions.holistic
            _holistic_instance = mp_holistic.Holistic(
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
                model_complexity=1
            )
            print("[MediaPipe] Holistic model ready")
        return _holistic_instance


def extract_poses_from_frames(frames, target_frames=15):
    """
    Extract 83-point pose landmarks from video frames using MediaPipe.

    Returns numpy array of shape (num_frames, 83, 4) where:
    - 83 = 33 body + 21 left hand + 21 right hand + 8 face
    - 4 = x, y, z, visibility/confidence

    Downsamples to target_frames to keep processing fast (~130ms per frame).
    With target_frames=15, processing takes ~2 seconds max.
    """
    holistic = get_holistic()
    if holistic is None:
        return None

    # 8 minimal face landmark indices from MediaPipe Face Mesh (468 total)
    # These are: nose tip, mouth corners, chin, eyebrows, eye corners
    FACE_INDICES = [1, 61, 291, 152, 107, 336, 33, 263]

    # Downsample if too many frames (each frame takes ~100ms)
    num_frames = len(frames)
    if num_frames > target_frames:
        # Sample evenly across the video to preserve temporal coverage
        step = num_frames / target_frames
        indices = [int(i * step) for i in range(target_frames)]
        frames_to_process = [frames[i] for i in indices]
        print(f"[MediaPipe] Downsampled {num_frames} -> {len(frames_to_process)} frames")
    else:
        frames_to_process = frames

    pose_sequence = []
    # Debug counters for detection stats
    body_count = 0
    left_hand_count = 0
    right_hand_count = 0
    face_count = 0

    for frame in frames_to_process:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        landmarks = []

        # Body pose (33 landmarks)
        if results.pose_landmarks:
            body_count += 1
            for lm in results.pose_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
        else:
            landmarks.extend([[0, 0, 0, 0]] * 33)

        # Left hand (21 landmarks) - hand landmarks DON'T have visibility, use 1.0
        if results.left_hand_landmarks:
            left_hand_count += 1
            for lm in results.left_hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z, 1.0])  # hands don't have visibility
        else:
            landmarks.extend([[0, 0, 0, 0]] * 21)

        # Right hand (21 landmarks) - hand landmarks DON'T have visibility, use 1.0
        if results.right_hand_landmarks:
            right_hand_count += 1
            for lm in results.right_hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z, 1.0])  # hands don't have visibility
        else:
            landmarks.extend([[0, 0, 0, 0]] * 21)

        # 8 minimal face landmarks (indices 75-82)
        if results.face_landmarks:
            face_count += 1
            for idx in FACE_INDICES:
                lm = results.face_landmarks.landmark[idx]
                landmarks.append([lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0])
        else:
            landmarks.extend([[0, 0, 0, 0]] * 8)

        pose_sequence.append(landmarks)

    # Log detection stats - critical for debugging hand detection issues
    total = len(frames_to_process)
    print(f"[MediaPipe] Detection: body={body_count}/{total}, L_hand={left_hand_count}/{total}, R_hand={right_hand_count}/{total}, face={face_count}/{total}")

    return np.array(pose_sequence) if pose_sequence else None


def decode_video_to_frames(video_bytes):
    """
    Decode video bytes to list of frames using OpenCV.
    Works with WebM and MP4 formats.

    This is the fast in-memory alternative to saving files to disk.
    """
    # Write to temporary file (OpenCV needs file path)
    # But we use a very fast in-memory approach
    temp_path = tempfile.mktemp(suffix='.webm')
    try:
        with open(temp_path, 'wb') as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            # Try as MP4
            temp_mp4 = tempfile.mktemp(suffix='.mp4')
            with open(temp_mp4, 'wb') as f:
                f.write(video_bytes)
            cap = cv2.VideoCapture(temp_mp4)
            os.unlink(temp_mp4)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return frames
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def poses_to_pickle_data(pose_array):
    """
    Convert pose numpy array to pickle format expected by model.

    Input: (num_frames, 83, 4) array where 4 = x, y, z, visibility
    Output: dict with 'keypoints' (x,y,z) and 'confidences'
    """
    if pose_array is None or len(pose_array) == 0:
        return None

    return {
        'keypoints': pose_array[:, :, :3],      # x, y, z (keep z for 43-class model)
        'confidences': pose_array[:, :, 3],     # visibility/confidence
        'gloss': 'UNKNOWN'
    }


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths to existing executables (in venv/Scripts)
VENV_SCRIPTS = Path(sys.executable).parent
VIDEO_TO_POSE_EXE = VENV_SCRIPTS / "video_to_pose.exe"
VISUALIZE_POSE_EXE = VENV_SCRIPTS / "visualize_pose.exe"

# FFmpeg path - try common locations
FFMPEG_EXE = None
ffmpeg_candidates = [
    Path(os.environ.get('LOCALAPPDATA', '')) / "Temp" / "ffmpeg-static-win64-gpl" / "bin" / "ffmpeg.exe",
    Path("C:/ffmpeg/bin/ffmpeg.exe"),
    Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
    Path(os.environ.get('USERPROFILE', '')) / "ffmpeg" / "bin" / "ffmpeg.exe",
]
for candidate in ffmpeg_candidates:
    if candidate.exists():
        FFMPEG_EXE = candidate
        break
if FFMPEG_EXE is None:
    # Fall back to PATH
    FFMPEG_EXE = "ffmpeg"

# Path to existing LLM prompt
LLM_PROMPT_PATH = PROJECT_UTILITIES_DIR / "llm_interface" / "prompts" / "llm_prompt_topk.txt"

# Model checkpoint path - use production models folder
CHECKPOINT_PATH = MODELS_DIR / "openhands-modernized" / "production-models" / "wlasl_43_class_50s_model"

# METRIC_WEIGHTS imported from evaluation_metrics library

# ============================================================================
# LIVE DEMO MODE CONFIGURATION
# Easy to modify - change these settings to tune real-time sign detection
# ============================================================================

# Segmentation type for Live Demo mode: 'motion' or 'time'
# - 'motion': Detect signs based on hand motion/velocity (more natural signing)
# - 'time': Fixed time intervals (useful for controlled demos)
LIVE_SEGMENTATION_TYPE = 'motion'

# Motion-based segmentation settings (used when LIVE_SEGMENTATION_TYPE = 'motion')
LIVE_MOTION_CONFIG = {
    'cooldown_ms': 1000,            # Milliseconds of no motion to end sign
    'min_sign_ms': 500,             # Minimum sign duration in milliseconds
    'max_sign_ms': 5000,            # Maximum sign duration before forced end
    'motion_threshold': 30,         # Pixel difference threshold for motion
    'motion_area_threshold': 0.02,  # Fraction of frame that must change
}

# Time-based segmentation settings (used when LIVE_SEGMENTATION_TYPE = 'time')
LIVE_TIME_CONFIG = {
    'sign_duration_ms': 2500,       # Expected sign duration in milliseconds
    'pause_duration_ms': 1250,      # Expected pause between signs in milliseconds
    'startup_trim_ms': 1000,        # Milliseconds to ignore at start
}

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)
app.secret_key = os.urandom(24)


@app.after_request
def add_permissions_headers(response):
    """Allow camera/microphone access in HF Spaces iframe"""
    response.headers['Permissions-Policy'] = 'camera=*, microphone=*'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

# Temp directory for this app
TEMP_DIR = Path(__file__).parent / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# Global model cache (loaded once)
_model_cache = {"model": None, "tokenizer": None, "num_classes": None, "masked_class_ids": None}

# Global quality scorer (from evaluation_metrics library)
_quality_scorer = None


def get_model():
    """Get cached model or load it"""
    if _model_cache["model"] is None:
        print("Loading model from checkpoint...")
        # vocab_size will be read from config.json in checkpoint directory
        _model_cache["model"], _model_cache["tokenizer"], _model_cache["masked_class_ids"] = load_model_from_checkpoint(
            str(CHECKPOINT_PATH)
        )
        # Load num_classes from model config
        config_path = CHECKPOINT_PATH / "config.json"
        if config_path.exists():
            import json
            with open(config_path) as f:
                config = json.load(f)
            _model_cache["num_classes"] = config.get("vocab_size", 100)
        else:
            _model_cache["num_classes"] = 100  # fallback
        print(f"Model loaded successfully ({_model_cache['num_classes']} classes)")
    return _model_cache["model"], _model_cache["tokenizer"]


def get_model_num_classes():
    """Get number of classes in loaded model"""
    if _model_cache["num_classes"] is None:
        get_model()  # Ensure model is loaded
    return _model_cache["num_classes"]


def get_quality_scorer():
    """Get cached quality scorer (using evaluation_metrics library)"""
    global _quality_scorer
    if not QUALITY_SCORING_AVAILABLE:
        return None
    if _quality_scorer is None:
        _quality_scorer = QualityScorer(verbose=True)
    return _quality_scorer


def get_session_dir():
    """Get or create session-specific temp directory"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    session_dir = TEMP_DIR / session['session_id']
    session_dir.mkdir(exist_ok=True)
    (session_dir / "segments").mkdir(exist_ok=True)
    return session_dir


def get_video_info(video_path: str):
    """Get video FPS and frame count for playback rate calculation."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        return {
            'fps': fps,
            'frame_count': frame_count,
            'duration': frame_count / fps if fps > 0 else 0,
            'playback_rate': fps / 30.0  # Rate to slow down to 30 FPS
        }
    except Exception as e:
        print(f"Failed to get video info: {e}")
        return None


def resample_video_to_30fps(input_path: str, output_path: str, target_fps: int = 30):
    """
    Resample video to target FPS by skipping frames and re-encoding with H.264.

    The visualize_pose.exe outputs at ~1000 FPS. We need to:
    1. Skip frames to get to target FPS
    2. Re-encode with H.264 codec (browser compatible)

    Uses imageio with pyav plugin which supports H.264.
    """
    try:
        import imageio.v3 as iio
        import numpy as np

        # Read video properties
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"ERROR: Failed to open video: {input_path}")
            return False

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Sanity check FPS - visualize_pose outputs at ~1000 FPS which is expected
        # But if this is being called on webcam video, FPS might be wrong
        if original_fps > 500:
            print(f"Resampling video: {width}x{height}, {original_fps} FPS (high - likely pose viz) -> {target_fps} FPS, {frame_count} frames")
        elif original_fps > 60 or original_fps < 10:
            print(f"WARNING: Unusual FPS ({original_fps}), assuming 30 FPS")
            original_fps = 30

        print(f"Resampling video: {width}x{height}, {original_fps} FPS -> {target_fps} FPS, {frame_count} frames")

        # Calculate frame skip - only if FPS is significantly higher
        if original_fps > target_fps * 1.5:
            frame_skip = max(1, int(original_fps / target_fps))
            print(f"Frame skip: every {frame_skip} frames")
        else:
            frame_skip = 1
            print(f"No frame skip needed (original FPS is close to target)")

        # Collect frames with skipping
        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                # Convert BGR to RGB for imageio
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            frame_idx += 1

        cap.release()

        if not frames:
            print("ERROR: No frames extracted")
            return False

        print(f"Writing {len(frames)} frames at {target_fps} FPS using imageio+pyav")

        # Stack frames into numpy array
        frames_array = np.stack(frames)

        # Write with imageio using pyav backend (H.264)
        iio.imwrite(
            output_path,
            frames_array,
            fps=target_fps,
            codec='libx264',
            plugin='pyav'
        )

        print(f"SUCCESS: Created H.264 video: {output_path}")
        return True

    except ImportError as e:
        print(f"WARNING: imageio/pyav not available ({e}), falling back to OpenCV")
        return resample_video_opencv(input_path, output_path, target_fps)
    except Exception as e:
        print(f"ERROR: Video resampling with imageio failed: {e}")
        import traceback
        traceback.print_exc()
        # Try OpenCV fallback
        print("Attempting OpenCV fallback...")
        return resample_video_opencv(input_path, output_path, target_fps)


def convert_webm_to_mp4(input_path: str, output_path: str) -> bool:
    """
    Convert WebM video to MP4 format for OpenCV compatibility.

    WebM (VP8/VP9) videos recorded by browsers often can't be read properly
    by OpenCV on Windows. This converts them to H.264 MP4.
    """
    try:
        import imageio.v3 as iio

        print(f"  Reading WebM: {input_path}")

        # Read all frames from WebM
        frames = iio.imread(input_path, plugin='pyav')

        if len(frames) == 0:
            print("  ERROR: No frames read from WebM")
            return False

        # Get properties
        # Try to get FPS from metadata, default to 30
        try:
            meta = iio.immeta(input_path, plugin='pyav')
            fps = meta.get('fps', 30)
        except:
            fps = 30

        # IMPORTANT: Browser webcam recordings often have incorrect FPS metadata
        # MediaRecorder sometimes reports 1000 FPS or other wrong values
        # If FPS is unreasonable, assume 30 FPS (typical webcam)
        if fps > 60 or fps < 10:
            print(f"  WARNING: Unreasonable FPS ({fps}), assuming 30 FPS for webcam recording")
            fps = 30

        print(f"  Read {len(frames)} frames at {fps} FPS")

        # Write as H.264 MP4
        iio.imwrite(
            output_path,
            frames,
            fps=fps,
            codec='libx264',
            plugin='pyav'
        )

        print(f"  SUCCESS: Converted to MP4: {output_path}")
        return True

    except ImportError:
        print("  WARNING: imageio/pyav not available for WebM conversion")
        # Try OpenCV fallback
        return convert_webm_opencv(input_path, output_path)
    except Exception as e:
        print(f"  ERROR: WebM conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_webm_opencv(input_path: str, output_path: str) -> bool:
    """Fallback WebM conversion using OpenCV (may not work for all WebM files)."""
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("  OpenCV fallback: Cannot open WebM")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30  # Default

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            print("  OpenCV fallback: Invalid dimensions")
            cap.release()
            return False

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print("  OpenCV fallback: Cannot create output")
            cap.release()
            return False

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

        print(f"  OpenCV fallback: Wrote {frame_count} frames")
        return frame_count > 0

    except Exception as e:
        print(f"  OpenCV fallback failed: {e}")
        return False


def resample_video_opencv(input_path: str, output_path: str, target_fps: int = 30):
    """
    Fallback: Resample video using OpenCV.
    Note: Uses mp4v codec which may not work in all browsers.
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Failed to open video: {input_path}")
            return False

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_skip = max(1, int(original_fps / target_fps))

        # Try H264 codec first (may work on some systems)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, float(target_fps), (width, height))

        if not out.isOpened():
            # Fallback to mp4v
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, float(target_fps), (width, height))

        if not out.isOpened():
            print("Failed to create video writer")
            cap.release()
            return False

        frame_idx = 0
        frames_written = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                out.write(frame)
                frames_written += 1
            frame_idx += 1

        cap.release()
        out.release()

        print(f"Created video with {frames_written} frames (OpenCV fallback): {output_path}")
        return True

    except Exception as e:
        print(f"OpenCV fallback failed: {e}")
        return False


# ============================================================================
# PAGE ROUTES - 5-Phase Workflow
# ============================================================================

@app.route('/')
def convert_page():
    """Phase 1: Convert Video to Pose"""
    return render_template('convert.html')


@app.route('/segment')
def segment_page():
    """Phase 2: Segment the Pose"""
    return render_template('segment.html')


@app.route('/predict')
def predict_page():
    """Phase 3: Predict Top-K Glosses"""
    return render_template('predict.html')


@app.route('/construct')
def construct_page():
    """Phase 4: Construct Sentence with LLM"""
    return render_template('construct.html')


@app.route('/evaluate')
def evaluate_page():
    """Phase 5: Evaluate Results"""
    return render_template('evaluate.html')


# ============================================================================
# LIVE MODE PAGE ROUTES
# ============================================================================

@app.route('/live-setup')
def live_setup_page():
    """Live Mode Phase 1: Reference Sentence Setup"""
    return render_template('live_setup.html')


@app.route('/live-learn')
def live_learn_page():
    """Live Mode Phase 2: Learn to Sign"""
    return render_template('live_learn.html')


@app.route('/live-record')
def live_record_page():
    """Live Mode Phase 3: Record Signs (webcam)"""
    return render_template('live_record.html')


# ============================================================================
# DEMO DATA DIRECTORY
# ============================================================================

DEMO_DATA_DIR = Path(__file__).parent / "demo-data"
DEMO_SAMPLES_DIR = DEMO_DATA_DIR / "samples"


# ============================================================================
# DEMO MODE API ROUTES
# ============================================================================

@app.route('/api/samples')
def get_samples():
    """Get list of available demo samples with original video info"""
    samples_json = DEMO_DATA_DIR / "samples.json"
    if samples_json.exists():
        with open(samples_json) as f:
            data = json.load(f)

        # Check each sample for original video
        for sample in data.get('samples', []):
            sample_dir = DEMO_SAMPLES_DIR / sample['id']
            original_video = None

            # Check for common video extensions (prefer .mp4 for better browser compatibility)
            for ext in ['.mp4', '.MP4', '.webm', '.MOV', '.mov', '.avi']:
                video_path = sample_dir / f"original_video{ext}"
                if video_path.exists():
                    original_video = f"original_video{ext}"
                    break

            sample['has_original_video'] = original_video is not None
            sample['original_video'] = original_video

        return jsonify(data)
    return jsonify({"samples": []})


@app.route('/api/samples/<sample_id>')
def get_sample(sample_id):
    """Get full metadata for a specific sample"""
    metadata_path = DEMO_SAMPLES_DIR / sample_id / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Sample not found"}), 404


@app.route('/demo-data/samples/<path:filepath>')
def serve_demo_file(filepath):
    """Serve files from demo-data/samples directory"""
    return send_from_directory(DEMO_SAMPLES_DIR, filepath)


@app.route('/api/upload-pose', methods=['POST'])
def upload_pose():
    """Handle direct pose file upload (skip video conversion)"""
    try:
        session_dir = get_session_dir()

        if 'pose_file' not in request.files:
            return jsonify({"success": False, "error": "No pose file provided"}), 400

        pose_file = request.files['pose_file']
        pose_path = session_dir / "capture.pose"
        pose_file.save(str(pose_path))

        # Generate pose visualization
        viz_path_raw = session_dir / "pose_video_raw.mp4"
        viz_path = session_dir / "pose_video.mp4"

        cmd = [
            str(VISUALIZE_POSE_EXE),
            "-i", str(pose_path),
            "-o", str(viz_path_raw),
            "--normalize"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Visualization failed, but pose file is uploaded - continue anyway
            print(f"WARNING: Pose visualization failed: {result.stderr}")
        else:
            # Resample to 30fps
            resample_video_to_30fps(str(viz_path_raw), str(viz_path))

        return jsonify({
            "success": True,
            "pose_file": str(pose_path)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/convert', methods=['POST'])
def convert_video():
    """
    Convert uploaded video to pose using EXISTING video_to_pose.exe
    Then visualize using EXISTING visualize_pose.exe
    """
    try:
        session_dir = get_session_dir()

        # Check if using demo mode
        if request.form.get('demo_mode') == 'true':
            # Use pre-recorded demo files
            demo_dir = Path(__file__).parent / "static" / "demo"
            demo_pose = demo_dir / "sample.pose"
            demo_video = demo_dir / "sample_pose.mp4"

            if demo_pose.exists() and demo_video.exists():
                # Copy to session directory
                shutil.copy(demo_pose, session_dir / "capture.pose")
                shutil.copy(demo_video, session_dir / "pose_video.mp4")

                return jsonify({
                    "success": True,
                    "pose_video_url": f"/temp/{session['session_id']}/pose_video.mp4",
                    "pose_file": str(session_dir / "capture.pose"),
                    "demo_mode": True
                })
            else:
                return jsonify({"success": False, "error": "Demo files not found"}), 404

        # Save uploaded video
        if 'video' not in request.files:
            return jsonify({"success": False, "error": "No video file provided"}), 400

        video_file = request.files['video']
        # Get the original extension
        original_filename = video_file.filename or 'video.webm'
        ext = Path(original_filename).suffix.lower() or '.webm'
        video_path = session_dir / f"capture{ext}"
        video_file.save(str(video_path))

        print(f"Saved uploaded video: {video_path}")

        # For video_to_pose compatibility, use the file directly
        # Most video formats are supported
        input_video = video_path

        # Convert WebM to MP4 for OpenCV compatibility in segmentation step
        # (video_to_pose can handle WebM, but OpenCV in hybrid_segmenter often can't)
        if ext == '.webm':
            print("WebM detected, converting to MP4 for segmentation compatibility...")
            mp4_path = session_dir / "capture.mp4"
            if convert_webm_to_mp4(str(video_path), str(mp4_path)):
                print(f"Converted to MP4: {mp4_path}")
                # Keep original for video_to_pose, but MP4 will be used by segmenter
            else:
                print("WARNING: WebM conversion failed")

        # Check for fast mode (skip visualization for faster processing)
        fast_mode = request.form.get('fast_mode', 'false').lower() == 'true'
        print(f"Fast mode: {fast_mode}")

        # Step 1: Convert video to pose using EXISTING video_to_pose.exe
        pose_path = session_dir / "capture.pose"

        cmd = [
            str(VIDEO_TO_POSE_EXE),
            "-i", str(input_video),
            "-o", str(pose_path),
            "--format", "mediapipe"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({
                "success": False,
                "error": f"video_to_pose failed: {result.stderr}"
            }), 500

        # Step 2: Visualize pose (skip if fast_mode is enabled)
        viz_path = session_dir / "pose_video.mp4"
        pose_video_url = None

        if not fast_mode:
            viz_path_raw = session_dir / "pose_video_raw.mp4"

            cmd = [
                str(VISUALIZE_POSE_EXE),
                "-i", str(pose_path),
                "-o", str(viz_path_raw),
                "--normalize"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return jsonify({
                    "success": False,
                    "error": f"visualize_pose failed: {result.stderr}"
                }), 500

            # Step 3: Resample to 30 FPS with H.264 codec for browser compatibility
            # visualize_pose.exe outputs at ~1000 FPS which browsers can't handle
            if not resample_video_to_30fps(str(viz_path_raw), str(viz_path)):
                return jsonify({
                    "success": False,
                    "error": "Failed to resample video to 30 FPS"
                }), 500

            # Verify the resampled video properties
            verify_cap = cv2.VideoCapture(str(viz_path))
            if verify_cap.isOpened():
                final_fps = verify_cap.get(cv2.CAP_PROP_FPS)
                final_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                verify_cap.release()
                print(f"VERIFICATION: Final video has {final_frames} frames at {final_fps} FPS")
            else:
                print("WARNING: Could not verify resampled video")

            pose_video_url = f"/temp/{session['session_id']}/pose_video.mp4"
        else:
            print("FAST MODE: Skipping pose visualization")

        return jsonify({
            "success": True,
            "pose_video_url": pose_video_url,
            "pose_file": str(pose_path),
            "fast_mode": fast_mode
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/segment', methods=['POST'])
def segment_pose():
    """
    Segment pose file using hybrid segmentation:
    1. Detect motion boundaries from original video (pixel-based)
    2. Slice pose file at those boundaries

    This is more robust than pose-keypoint-velocity-based segmentation.
    """
    try:
        session_dir = get_session_dir()
        data = request.get_json()
        pose_file = data.get('pose_file', str(session_dir / "capture.pose"))
        fast_mode = data.get('fast_mode', False)
        print(f"Segment API - Fast mode: {fast_mode}")

        if not Path(pose_file).exists():
            return jsonify({"success": False, "error": "Pose file not found"}), 404

        segments_dir = session_dir / "segments"

        # Clear previous segments
        for f in segments_dir.glob("*"):
            f.unlink()

        # Find the original video file for motion detection
        # Prefer MP4 over WebM since WebM often has OpenCV compatibility issues
        video_file = None
        for ext in ['.mp4', '.avi', '.mov', '.MOV', '.webm']:
            candidate = session_dir / f"capture{ext}"
            if candidate.exists():
                video_file = str(candidate)
                break

        # Also check for converted MP4
        converted_mp4 = session_dir / "capture_converted.mp4"
        if converted_mp4.exists():
            video_file = str(converted_mp4)

        if video_file:
            # Check if WebM needs conversion (OpenCV often can't read WebM properly on Windows)
            if video_file.endswith('.webm'):
                print("WebM detected in segment API, converting to MP4...")
                mp4_path = session_dir / "capture_converted.mp4"
                if convert_webm_to_mp4(video_file, str(mp4_path)):
                    video_file = str(mp4_path)
                    print(f"Using converted MP4: {video_file}")
                else:
                    print("WARNING: WebM conversion failed, trying with original")

            # PREFERRED: Use hybrid segmenter (pixel motion detection + pose slicing)
            # More robust since it uses actual pixel motion, not pose keypoint velocity
            print("Using HYBRID segmentation (video motion + pose slicing)")
            segmenter = HybridSegmenter(
                motion_threshold=500000,   # Pixel sum threshold for motion
                cooldown_frames=45,        # 1.5 seconds at 30 FPS to end sign
                min_sign_frames=12,        # Minimum frames for valid sign
                max_sign_frames=150,       # Maximum frames before splitting
                padding_before=3,
                padding_after=3
            )

            segment_files = segmenter.segment_video_and_pose(
                video_file,
                pose_file,
                str(segments_dir),
                verbose=True
            )
        else:
            # FALLBACK: Use motion-based segmenter (pose keypoint velocity)
            # Used when only a pose file is uploaded without source video
            print("Using MOTION-BASED segmentation (pose keypoint velocity) - no source video available")
            segmenter = MotionBasedSegmenter(
                velocity_threshold=0.02,
                min_sign_duration=10,
                max_sign_duration=120,
                min_rest_duration=45,  # 1.5 seconds at 30 FPS
                padding_before=3,
                padding_after=3
            )

            segment_files = segmenter.segment_pose_file(
                pose_file,
                str(segments_dir),
                verbose=True
            )

        if not segment_files:
            return jsonify({
                "success": False,
                "error": "No segments detected. Try signing more clearly with pauses between words."
            }), 400

        # Load model once for all predictions
        model, tokenizer = get_model()

        results = []
        for i, seg_file in enumerate(segment_files):
            seg_path = Path(seg_file)
            video_url = None

            # Visualize segment (skip if fast_mode is enabled)
            if not fast_mode:
                seg_video_raw = str(seg_path.with_suffix('.raw.mp4'))

                subprocess.run([
                    str(VISUALIZE_POSE_EXE),
                    "-i", str(seg_file),
                    "-o", seg_video_raw,
                    "--normalize"
                ], capture_output=True)

                # Resample to 30 FPS with H.264 for browser compatibility
                seg_video = str(seg_path.with_suffix('.mp4'))
                resample_video_to_30fps(seg_video_raw, seg_video)
                video_url = f"/temp/{session['session_id']}/segments/{seg_path.stem}.mp4"

            # Convert pose to pickle format for prediction
            pickle_path = convert_pose_to_pickle(seg_file)

            if pickle_path:
                # Use EXISTING predict_pose_file function
                prediction = predict_pose_file(
                    pickle_path,
                    model=model,
                    tokenizer=tokenizer
                )

                results.append({
                    "segment_id": i + 1,
                    "video_url": video_url,
                    "top_1": prediction['gloss'],
                    "confidence": prediction['confidence'],
                    "top_k": prediction['top_k_predictions'][:3]  # Top-3
                })
            else:
                results.append({
                    "segment_id": i + 1,
                    "video_url": video_url,
                    "top_1": "UNKNOWN",
                    "confidence": 0.0,
                    "top_k": []
                })

        # Store results in session for later use
        session['segment_results'] = results

        return jsonify({
            "success": True,
            "segments": results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


def convert_pose_to_pickle(pose_file, debug=False):
    """
    Convert .pose file to pickle format for model inference.
    Extracts 83 keypoints (33 body + 21 left hand + 21 right hand + 8 face).
    """
    # 8 minimal face landmark indices from MediaPipe Face Mesh
    # In 543-format, face landmarks start at index 33
    FACE_MESH_INDICES = [1, 61, 291, 152, 107, 336, 33, 263]
    FACE_543_INDICES = [33 + i for i in FACE_MESH_INDICES]

    try:
        with open(pose_file, "rb") as f:
            buffer = f.read()
            pose = Pose.read(buffer)

        pose_data = pose.body.data

        if len(pose_data.shape) == 4:
            pose_sequence = pose_data[:, 0, :, :]
        else:
            pose_sequence = pose_data

        if debug:
            print(f"[DEBUG] Pose file: {pose_file}")
            print(f"[DEBUG] Original shape: {pose_data.shape}")
            print(f"[DEBUG] After squeeze: {pose_sequence.shape}")

        # Extract 83-point subset (33 body + 21 left hand + 21 right hand + 8 face)
        if pose_sequence.shape[1] == 543:
            pose_83pt = np.concatenate([
                pose_sequence[:, 0:33, :],            # Pose landmarks (33)
                pose_sequence[:, 501:522, :],         # Left hand landmarks (21)
                pose_sequence[:, 522:543, :],         # Right hand landmarks (21)
                pose_sequence[:, FACE_543_INDICES, :] # Face landmarks (8)
            ], axis=1)
        elif pose_sequence.shape[1] == 83:
            pose_83pt = pose_sequence
        elif pose_sequence.shape[1] == 75:
            # Pad with zeros for face landmarks
            frames = pose_sequence.shape[0]
            coords = pose_sequence.shape[2]
            pose_83pt = np.zeros((frames, 83, coords), dtype=np.float32)
            pose_83pt[:, :75, :] = pose_sequence
        else:
            pose_83pt = pose_sequence

        if debug:
            print(f"[DEBUG] After 83pt extraction: {pose_83pt.shape}")

        # Create pickle file
        pickle_path = str(pose_file).replace('.pose', '.pkl')

        pickle_data = {
            'keypoints': pose_83pt[:, :, :3],  # x, y, z
            'confidences': pose_83pt[:, :, 3] if pose_83pt.shape[2] > 3 else np.ones(pose_83pt.shape[:2]),
            'gloss': 'UNKNOWN'
        }

        if debug:
            print(f"[DEBUG] Pickle keypoints shape: {pickle_data['keypoints'].shape}")
            print(f"[DEBUG] Pickle confidences shape: {pickle_data['confidences'].shape}")

        with open(pickle_path, 'wb') as f:
            pickle.dump(pickle_data, f)

        return pickle_path

    except Exception as e:
        print(f"Error converting pose to pickle: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.route('/api/process-full', methods=['POST'])
def process_full_pipeline():
    """
    Full pipeline for Live (fast) mode:
    Video → Pose → Segment → Predict → Construct
    All in one API call, skipping visualizations.
    """
    try:
        session_dir = get_session_dir()

        # Save uploaded video
        if 'video' not in request.files:
            return jsonify({"success": False, "error": "No video file provided"}), 400

        video_file = request.files['video']
        original_filename = video_file.filename or 'video.webm'
        ext = Path(original_filename).suffix.lower() or '.webm'
        video_path = session_dir / f"capture{ext}"
        video_file.save(str(video_path))

        print(f"[FAST MODE] Saved video: {video_path}")

        # Check if WebM needs conversion (OpenCV often can't read WebM properly on Windows)
        if ext == '.webm':
            print(f"[FAST MODE] WebM detected, converting to MP4 for OpenCV compatibility...")
            mp4_path = session_dir / "capture.mp4"
            if convert_webm_to_mp4(str(video_path), str(mp4_path)):
                video_path = mp4_path
                print(f"[FAST MODE] Converted to: {video_path}")
            else:
                print(f"[FAST MODE] WARNING: WebM conversion failed, trying with original")

        # Step 1: Convert video to pose
        pose_path = session_dir / "capture.pose"
        cmd = [
            str(VIDEO_TO_POSE_EXE),
            "-i", str(video_path),
            "-o", str(pose_path),
            "--format", "mediapipe"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({
                "success": False,
                "error": f"Video to pose conversion failed: {result.stderr}"
            }), 500

        print(f"[FAST MODE] Converted to pose: {pose_path}")

        # Step 2: Segment pose file
        segments_dir = session_dir / "segments"
        segments_dir.mkdir(exist_ok=True)

        # Clear previous segments
        for f in segments_dir.glob("*"):
            f.unlink()

        # Use hybrid segmenter if video available
        # Lower thresholds for better detection of casual signing
        segmenter = HybridSegmenter(
            motion_threshold=500000,
            cooldown_frames=20,       # Reduced from 45 - detect shorter pauses (~0.67s)
            min_sign_frames=10,       # Reduced from 12 - allow shorter signs
            max_sign_frames=150,
            padding_before=3,
            padding_after=3
        )

        print(f"[FAST MODE] Running hybrid segmentation on {video_path}")
        print(f"[FAST MODE] Pose file: {pose_path}")

        segment_files = segmenter.segment_video_and_pose(
            str(video_path),
            str(pose_path),
            str(segments_dir),
            verbose=True
        )

        print(f"[FAST MODE] Segmentation result: {len(segment_files) if segment_files else 0} segments")

        if not segment_files:
            # Try with even more relaxed parameters
            print("[FAST MODE] No segments with default params, trying relaxed params...")
            segmenter_relaxed = HybridSegmenter(
                motion_threshold=300000,  # Lower threshold
                cooldown_frames=12,       # Very short cooldown
                min_sign_frames=8,        # Allow very short signs
                max_sign_frames=200,
                padding_before=5,
                padding_after=5
            )
            segment_files = segmenter_relaxed.segment_video_and_pose(
                str(video_path),
                str(pose_path),
                str(segments_dir),
                verbose=True
            )
            print(f"[FAST MODE] Relaxed segmentation result: {len(segment_files) if segment_files else 0} segments")

        if not segment_files:
            return jsonify({
                "success": False,
                "error": "No segments detected. Try signing more clearly with pauses between words. Check server console for details."
            }), 400

        print(f"[FAST MODE] Found {len(segment_files)} segments")

        # Step 3: Load model and predict for each segment using camera processor
        model, tokenizer = get_model()
        processor = get_camera_processor(model, tokenizer)

        predictions = []
        for i, seg_file in enumerate(segment_files):
            # Use camera processor for consistent pose-to-pickle and prediction
            pickle_path = processor.pose_to_pickle(seg_file)

            if pickle_path:
                prediction = processor.predict(str(pickle_path))

                if prediction:
                    predictions.append({
                        "segment_id": i + 1,
                        "top_1": prediction['gloss'],
                        "confidence": prediction['confidence'],
                        "top_k": prediction['top_k_predictions'][:3]
                    })
                else:
                    predictions.append({
                        "segment_id": i + 1,
                        "top_1": "UNKNOWN",
                        "confidence": 0.0,
                        "top_k": []
                    })
            else:
                predictions.append({
                    "segment_id": i + 1,
                    "top_1": "UNKNOWN",
                    "confidence": 0.0,
                    "top_k": []
                })

        print(f"[FAST MODE] Predictions: {[p['top_1'] for p in predictions]}")

        # Step 4: Construct sentence with LLM
        raw_sentence = " ".join([p['top_1'] for p in predictions])

        # Load prompt template
        with open(LLM_PROMPT_PATH, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        gloss_details = format_gloss_details_for_prompt(predictions)
        prompt = prompt_template.format(gloss_details=gloss_details, context_section="")

        # Call LLM
        llm = create_llm_provider(
            provider="googleaistudio",
            model_name="gemini-2.0-flash",
            max_tokens=500,
            timeout=30
        )

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

        print(f"[FAST MODE] Raw: {raw_sentence}")
        print(f"[FAST MODE] LLM: {llm_sentence}")

        # Store in session for evaluate page
        session['raw_sentence'] = raw_sentence
        session['llm_sentence'] = llm_sentence
        session['segment_results'] = predictions

        return jsonify({
            "success": True,
            "predictions": predictions,
            "raw_sentence": raw_sentence,
            "llm_sentence": llm_sentence,
            "segment_count": len(predictions)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/process-pose-full', methods=['POST'])
def process_pose_full_pipeline():
    """
    Full pipeline for Live (fast) mode with pose file input:
    Pose → Segment → Predict → Construct
    All in one API call, skipping visualizations.
    """
    try:
        session_dir = get_session_dir()

        # Save uploaded pose file
        if 'pose_file' not in request.files:
            return jsonify({"success": False, "error": "No pose file provided"}), 400

        pose_file = request.files['pose_file']
        pose_path = session_dir / "capture.pose"
        pose_file.save(str(pose_path))

        print(f"[FAST MODE - POSE] Saved pose file: {pose_path}")

        # Step 1: Segment pose file (no video available, use motion-based segmenter)
        segments_dir = session_dir / "segments"
        segments_dir.mkdir(exist_ok=True)

        # Clear previous segments
        for f in segments_dir.glob("*"):
            f.unlink()

        # Use motion-based segmenter for pose-only input
        # Settings match the normal /api/segment endpoint for pose-only uploads
        segmenter = MotionBasedSegmenter(
            velocity_threshold=0.02,
            min_sign_duration=10,
            max_sign_duration=120,
            min_rest_duration=45,  # 1.5 seconds at 30 FPS (match normal mode)
            padding_before=3,
            padding_after=3
        )

        segment_files = segmenter.segment_pose_file(
            str(pose_path),
            str(segments_dir),
            verbose=True
        )

        if not segment_files:
            return jsonify({
                "success": False,
                "error": "No segments detected. Try signing more clearly with pauses between words."
            }), 400

        print(f"[FAST MODE - POSE] Found {len(segment_files)} segments")

        # Debug: Log segment file info
        for seg_file in segment_files:
            try:
                with open(seg_file, "rb") as f:
                    buffer = f.read()
                    pose = Pose.read(buffer)
                    frames = pose.body.data.shape[0]
                    print(f"[FAST MODE - POSE] Segment {seg_file}: {frames} frames")
            except Exception as e:
                print(f"[FAST MODE - POSE] Could not read segment info: {e}")

        # Step 2: Load model and predict for each segment
        model, tokenizer = get_model()

        predictions = []
        for i, seg_file in enumerate(segment_files):
            # Convert pose to pickle for prediction (no visualization)
            # Enable debug for first segment to see the data shape
            pickle_path = convert_pose_to_pickle(seg_file, debug=(i == 0))

            if pickle_path:
                prediction = predict_pose_file(
                    pickle_path,
                    model=model,
                    tokenizer=tokenizer
                )

                print(f"[FAST MODE - POSE] Segment {i+1}: {prediction['gloss']} ({prediction['confidence']:.2%})")

                predictions.append({
                    "segment_id": i + 1,
                    "top_1": prediction['gloss'],
                    "confidence": prediction['confidence'],
                    "top_k": prediction['top_k_predictions'][:3]
                })
            else:
                print(f"[FAST MODE - POSE] Segment {i+1}: FAILED to convert to pickle")
                predictions.append({
                    "segment_id": i + 1,
                    "top_1": "UNKNOWN",
                    "confidence": 0.0,
                    "top_k": []
                })

        print(f"[FAST MODE - POSE] All Predictions: {[p['top_1'] for p in predictions]}")

        # Step 3: Construct sentence with LLM
        raw_sentence = " ".join([p['top_1'] for p in predictions])

        # Load prompt template
        with open(LLM_PROMPT_PATH, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        gloss_details = format_gloss_details_for_prompt(predictions)
        prompt = prompt_template.format(gloss_details=gloss_details, context_section="")

        # Call LLM
        llm = create_llm_provider(
            provider="googleaistudio",
            model_name="gemini-2.0-flash",
            max_tokens=500,
            timeout=30
        )

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

        print(f"[FAST MODE - POSE] Raw: {raw_sentence}")
        print(f"[FAST MODE - POSE] LLM: {llm_sentence}")

        # Store in session for evaluate page
        session['raw_sentence'] = raw_sentence
        session['llm_sentence'] = llm_sentence
        session['segment_results'] = predictions

        return jsonify({
            "success": True,
            "predictions": predictions,
            "raw_sentence": raw_sentence,
            "llm_sentence": llm_sentence,
            "segment_count": len(predictions)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/construct', methods=['POST'])
def construct_sentence():
    """
    Construct sentence using EXISTING LLM interface and EXISTING prompt
    """
    try:
        data = request.get_json()
        predictions = data.get('predictions', [])

        if not predictions:
            # Try to get from session
            predictions = session.get('segment_results', [])

        if not predictions:
            return jsonify({"success": False, "error": "No predictions available"}), 400

        # Raw concatenation (Top-1 words joined)
        raw_sentence = " ".join([p['top_1'] for p in predictions])

        # Load EXISTING prompt template
        with open(LLM_PROMPT_PATH, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        # Format gloss details (same format as predict_sentence.py line 933-949)
        gloss_details = format_gloss_details_for_prompt(predictions)
        prompt = prompt_template.format(gloss_details=gloss_details, context_section="")

        # Use EXISTING LLM provider (Google AI Studio)
        llm = create_llm_provider(
            provider="googleaistudio",
            model_name="gemini-2.0-flash",
            max_tokens=500,
            timeout=30
        )

        response = llm.generate(prompt)

        # Parse JSON response
        # Clean up response - remove markdown code blocks if present
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
            selections = result.get('selections', [p['top_1'] for p in predictions])
        except json.JSONDecodeError:
            # If JSON parsing fails, use raw response as sentence
            llm_sentence = response_text
            selections = [p['top_1'] for p in predictions]

        # Store for evaluation
        session['raw_sentence'] = raw_sentence
        session['llm_sentence'] = llm_sentence
        session['selected_glosses'] = selections  # LLM-selected glosses for Effective GA

        return jsonify({
            "success": True,
            "raw_sentence": raw_sentence,
            "llm_sentence": llm_sentence,
            "selections": selections
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


def format_gloss_details_for_prompt(predictions):
    """
    Format predictions for LLM prompt.
    Same format as predict_sentence.py line 933-949
    """
    details = []
    for i, pred in enumerate(predictions, 1):
        top_k = pred.get('top_k', [])
        if top_k:
            detail = f"Position {i}:\n"
            for j, p in enumerate(top_k, 1):
                conf = p.get('confidence', 0) * 100
                detail += f"  Option {j}: '{p['gloss']}' (confidence: {conf:.1f}%)\n"
            details.append(detail)
        else:
            details.append(f"Position {i}: '{pred['top_1']}' (confidence: {pred['confidence']*100:.1f}%)\n")

    return "".join(details)


# ============================================================================
# EVALUATION METRICS API (using evaluation_metrics library)
# ============================================================================

@app.route('/api/evaluate', methods=['POST'])
def evaluate_sentences():
    """
    Calculate all evaluation metrics using the modular evaluation_metrics library.
    Compares raw concatenated glosses vs LLM-constructed sentence against reference.

    Returns metrics for both raw and LLM sentences including:
    - Gloss Accuracy: Model GA (top-1) for raw, Effective GA (LLM-selected) for LLM
    - Coverage F1 (semantic overlap with reference)
    - Plausibility Score (GPT-2 fluency)
    - CTQI v3 (human-validated: GA × CF1 × (0.5 + 0.5 × P × GA))
    """
    try:
        data = request.get_json()
        raw_sentence = data.get('raw_sentence', session.get('raw_sentence', ''))
        llm_sentence = data.get('llm_sentence', session.get('llm_sentence', ''))
        reference = data.get('reference', '')
        predicted_glosses = data.get('predicted_glosses', [])  # Model top-1 predictions
        selected_glosses = data.get('selected_glosses', session.get('selected_glosses', []))  # LLM-selected from top-k
        expected_glosses = data.get('expected_glosses', [])    # Ground truth glosses

        if not reference:
            return jsonify({"success": False, "error": "Reference sentence required"}), 400

        # Get quality scorer (cached)
        scorer = get_quality_scorer()

        # If predicted_glosses not provided, extract from raw_sentence (model top-1)
        if not predicted_glosses and raw_sentence:
            predicted_glosses = raw_sentence.lower().split()

        # If selected_glosses not provided, default to predicted_glosses
        if not selected_glosses:
            selected_glosses = predicted_glosses

        # If expected_glosses not provided, derive from reference (content words, lemmatized)
        if not expected_glosses and reference:
            # Use same lemmatization as coverage calculation for consistency
            import re
            try:
                from nltk.stem import WordNetLemmatizer
                lemmatizer = WordNetLemmatizer()
                # Extract content words from reference (filter stopwords)
                stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                            'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
                            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
                            'during', 'before', 'after', 'above', 'below', 'between', 'under',
                            'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
                            'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                            'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because', 'until',
                            'while', 'although', 'though', 'since', 'unless', 'i', 'me', 'my',
                            'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
                            'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
                            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                            'these', 'those', 'am'}
                words = re.findall(r'\b[a-zA-Z]+\b', reference.lower())
                expected_glosses = [lemmatizer.lemmatize(w, 'v') for w in words if w not in stopwords]
            except ImportError:
                # Fallback: just use content words without lemmatization
                words = re.findall(r'\b[a-zA-Z]+\b', reference.lower())
                stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'for',
                            'on', 'with', 'at', 'by', 'from', 'and', 'or', 'but', 'his', 'her', 'its'}
                expected_glosses = [w for w in words if w not in stopwords]

        # Calculate Model Gloss Accuracy (top-1 predictions vs expected)
        model_ga_result = metrics_gloss_accuracy(predicted_glosses, expected_glosses)
        model_gloss_accuracy = model_ga_result.get('accuracy', 0.0)

        # Calculate Effective Gloss Accuracy (LLM-selected vs expected)
        effective_ga_result = metrics_gloss_accuracy(selected_glosses, expected_glosses)
        effective_gloss_accuracy = effective_ga_result.get('accuracy', 0.0)

        # Calculate all metrics for raw sentence (concatenated glosses)
        raw_bleu = calculate_bleu_score(raw_sentence, reference) or 0.0
        raw_bert = calculate_bert_score(raw_sentence, reference) or 0.0
        raw_quality = metrics_quality_score(raw_sentence, scorer=scorer) or 0.0
        raw_coverage = calculate_coverage(reference, raw_sentence)

        raw_metrics = {
            'bleu': raw_bleu,
            'bert': raw_bert,
            'model_gloss_accuracy': model_gloss_accuracy,
            'effective_gloss_accuracy': model_gloss_accuracy,  # Raw uses top-1, same as Model GA
            'quality': raw_quality,
            'coverage_recall': raw_coverage['recall'] or 0.0,
            'coverage_precision': raw_coverage['precision'] or 0.0,
            'coverage_f1': raw_coverage['f1'] or 0.0,
            'missing_words': raw_coverage['missing_words'],
            'hallucinated_words': raw_coverage['hallucinated_words'],
        }

        # Calculate CTQI v3 for raw (uses Model GA)
        # Formula: GA × CF1 × (0.5 + 0.5 × P/100 × GA/100) × 100
        raw_composite = metrics_composite_score(
            gloss_accuracy=model_gloss_accuracy,
            coverage_f1=raw_coverage['f1'] or 0.0,
            plausibility=raw_quality
        ) or 0.0
        raw_metrics['composite'] = raw_composite

        # Calculate all metrics for LLM sentence
        llm_bleu = calculate_bleu_score(llm_sentence, reference) or 0.0
        llm_bert = calculate_bert_score(llm_sentence, reference) or 0.0
        llm_quality = metrics_quality_score(llm_sentence, scorer=scorer) or 0.0
        llm_coverage = calculate_coverage(reference, llm_sentence)

        llm_metrics = {
            'bleu': llm_bleu,
            'bert': llm_bert,
            'model_gloss_accuracy': model_gloss_accuracy,  # Same as raw (model accuracy)
            'effective_gloss_accuracy': effective_gloss_accuracy,  # LLM-selected glosses
            'quality': llm_quality,
            'coverage_recall': llm_coverage['recall'] or 0.0,
            'coverage_precision': llm_coverage['precision'] or 0.0,
            'coverage_f1': llm_coverage['f1'] or 0.0,
            'missing_words': llm_coverage['missing_words'],
            'hallucinated_words': llm_coverage['hallucinated_words'],
        }

        # Calculate CTQI v3 for LLM (uses Effective GA)
        # Formula: GA × CF1 × (0.5 + 0.5 × P/100 × GA/100) × 100
        llm_composite = metrics_composite_score(
            gloss_accuracy=effective_gloss_accuracy,
            coverage_f1=llm_coverage['f1'] or 0.0,
            plausibility=llm_quality
        ) or 0.0
        llm_metrics['composite'] = llm_composite

        return jsonify({
            "success": True,
            "raw": raw_metrics,
            "llm": llm_metrics
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset session and clean up temp files"""
    try:
        session_dir = get_session_dir()
        if session_dir.exists():
            shutil.rmtree(session_dir)
        session.clear()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Directory for saved sessions (persistent, not deleted on reset)
SAVED_SESSIONS_DIR = Path(__file__).parent / "saved-sessions"
SAVED_SESSIONS_DIR.mkdir(exist_ok=True)


@app.route('/api/save-session', methods=['POST'])
def save_session():
    """
    Save current session to a persistent location for later use with prepare_demo_sample.py --from-session
    """
    try:
        session_dir = get_session_dir()

        if not session_dir.exists():
            return jsonify({"success": False, "error": "No active session to save"}), 400

        # Check if there are segment files (.pose or .pkl) or at least a capture file
        segments_dir = session_dir / "segments"
        segment_poses = list(segments_dir.glob("*.pose")) if segments_dir.exists() else []
        segment_pkls = list(segments_dir.glob("*.pkl")) if segments_dir.exists() else []
        has_segments = bool(segment_poses or segment_pkls)

        has_capture = (
            (session_dir / "capture.pose").exists() or
            (session_dir / "capture.mp4").exists() or
            (session_dir / "capture.webm").exists()
        )

        if not has_segments and not has_capture:
            return jsonify({"success": False, "error": "No session data found. Process a video first."}), 400

        # Warn if no segments (will need re-segmentation with --from-session)
        needs_resegment = not has_segments and has_capture

        # Get metadata from request
        data = request.get_json() or {}
        reference = data.get('reference', '')
        raw_sentence = data.get('raw_sentence', '')
        llm_sentence = data.get('llm_sentence', '')

        # Create timestamped folder name
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"session_{timestamp}"
        saved_path = SAVED_SESSIONS_DIR / session_name

        # Copy entire session directory
        shutil.copytree(session_dir, saved_path)

        # For Live Demo mode: Create capture.mp4 from segment videos if needed
        saved_segments_dir = saved_path / "segments"
        capture_mp4 = saved_path / "capture.mp4"
        capture_pose = saved_path / "capture.pose"

        if not capture_mp4.exists() and saved_segments_dir.exists():
            segment_videos = sorted(saved_segments_dir.glob("segment_*.mp4"))
            if segment_videos:
                print(f"Creating capture.mp4 from {len(segment_videos)} segment videos...")
                try:
                    # Create concat list for ffmpeg (use forward slashes for Windows compatibility)
                    concat_list = saved_path / "concat_list.txt"
                    with open(concat_list, 'w') as f:
                        for vid in segment_videos:
                            # Use as_posix() to convert backslashes to forward slashes
                            f.write(f"file '{vid.absolute().as_posix()}'\n")

                    # Concatenate videos
                    concat_cmd = [
                        str(FFMPEG_EXE), "-y", "-f", "concat", "-safe", "0",
                        "-i", str(concat_list),
                        "-c", "copy", str(capture_mp4)
                    ]
                    result = subprocess.run(concat_cmd, capture_output=True, text=True)

                    if result.returncode != 0:
                        # Try with re-encoding if copy fails
                        concat_cmd = [
                            str(FFMPEG_EXE), "-y", "-f", "concat", "-safe", "0",
                            "-i", str(concat_list),
                            "-c:v", "libx264", "-preset", "fast",
                            str(capture_mp4)
                        ]
                        result = subprocess.run(concat_cmd, capture_output=True, text=True)

                    if result.returncode == 0:
                        print(f"  Created: {capture_mp4.name}")
                    else:
                        print(f"  Failed to create capture.mp4: {result.stderr}")

                    # Clean up concat list
                    concat_list.unlink(missing_ok=True)

                except Exception as e:
                    print(f"  Error creating capture.mp4: {e}")

        # Create capture.pose from capture.mp4 if needed
        if capture_mp4.exists() and not capture_pose.exists():
            print("Creating capture.pose from capture.mp4...")
            try:
                cmd = [
                    str(VIDEO_TO_POSE_EXE),
                    "-i", str(capture_mp4),
                    "-o", str(capture_pose),
                    "--format", "mediapipe"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  Created: {capture_pose.name}")
                else:
                    print(f"  Failed to create capture.pose: {result.stderr}")
            except Exception as e:
                print(f"  Error creating capture.pose: {e}")

        # Create segment .pose files from segment .mp4 files if needed
        if saved_segments_dir.exists():
            segment_videos = sorted(saved_segments_dir.glob("segment_*.mp4"))
            if segment_videos:
                print(f"Creating segment .pose files from {len(segment_videos)} videos...")
                for vid in segment_videos:
                    pose_file = vid.with_suffix('.pose')
                    if not pose_file.exists():
                        try:
                            cmd = [
                                str(VIDEO_TO_POSE_EXE),
                                "-i", str(vid),
                                "-o", str(pose_file),
                                "--format", "mediapipe"
                            ]
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            if result.returncode == 0:
                                print(f"  Created: {pose_file.name}")
                            else:
                                print(f"  Failed: {pose_file.name} - {result.stderr[:100]}")
                        except Exception as e:
                            print(f"  Error creating {pose_file.name}: {e}")

        # Save metadata including predictions
        predictions = session.get('live_predictions', [])
        metadata = {
            "saved_at": datetime.now().isoformat(),
            "reference": reference,
            "raw_sentence": raw_sentence,
            "llm_sentence": llm_sentence,
            "predictions": predictions,
            "model_classes": get_model_num_classes(),
            "original_session_id": session.get('session_id', 'unknown')
        }
        metadata_path = saved_path / "session_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Predictions saved: {len(predictions)}")

        print(f"Session saved to: {saved_path}")
        print(f"  Segments: {len(segment_poses)} .pose, {len(segment_pkls)} .pkl")
        print(f"  Has capture: {has_capture}, Needs re-segment: {needs_resegment}")

        return jsonify({
            "success": True,
            "session_name": session_name,
            "saved_path": str(saved_path),
            "segment_count": len(segment_poses) + len(segment_pkls),
            "has_capture": has_capture,
            "needs_resegment": needs_resegment
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# LIVE DEMO MODE API ROUTES
# Real-time sign detection and gloss-only prediction (no LLM)
# ============================================================================

@app.route('/api/live-reset', methods=['POST'])
def reset_live_demo():
    """Reset live demo state (clears sign count, predictions, and segments for new session)"""
    try:
        session['live_sign_count'] = 0
        session['live_predictions'] = []
        session.modified = True
        session_dir = get_session_dir()
        segments_dir = session_dir / "segments"
        if segments_dir.exists():
            for f in segments_dir.glob("*"):
                f.unlink()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/live-config')
def get_live_config():
    """
    Get live demo configuration for client-side sign detection.
    Returns segmentation type and parameters.
    """
    return jsonify({
        "segmentation_type": LIVE_SEGMENTATION_TYPE,
        "motion_config": LIVE_MOTION_CONFIG,
        "time_config": LIVE_TIME_CONFIG
    })


# ============================================================================
# LIVE MODE SETUP API ROUTES
# ============================================================================

# Load gloss categorizer (master categories filtered to model's glosses at runtime)
try:
    gloss_categorizer = GlossCategorizer.load_master()
except FileNotFoundError:
    print("WARNING: Master categorization file not found. Falling back to seed categories.")
    gloss_categorizer = GlossCategorizer.from_seed()
except Exception as e:
    print(f"WARNING: Failed to load gloss categorizer: {e}. Falling back to seed categories.")
    gloss_categorizer = GlossCategorizer.from_seed()


@app.route('/api/reference-sentences')
def get_reference_sentences():
    """
    Get list of preset reference sentences from demo samples.
    Returns sentences with their glosses for dropdown selection.
    """
    sentences = []
    samples_json = DEMO_DATA_DIR / "samples.json"

    if samples_json.exists():
        with open(samples_json) as f:
            data = json.load(f)

        for sample in data.get('samples', []):
            sample_id = sample['id']
            metadata_path = DEMO_SAMPLES_DIR / sample_id / "metadata.json"

            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)

                # Extract glosses from segments
                glosses = []
                precomputed = metadata.get('precomputed', {})
                for seg in precomputed.get('segments', []):
                    gloss = seg.get('top_1') or seg.get('expected_gloss')
                    if gloss:
                        glosses.append(gloss)

                sentences.append({
                    "id": sample_id,
                    "name": sample.get('name', sample_id),
                    "reference": metadata.get('reference_sentence', ''),
                    "glosses": glosses
                })

    return jsonify({"sentences": sentences})


@app.route('/api/available-glosses')
def get_available_glosses():
    """
    Get all available glosses organized by category.
    Uses the production model's class mapping and filters master categories dynamically.
    """
    # Load class mapping from production model
    class_mapping_path = CHECKPOINT_PATH / "class_index_mapping.json"
    all_glosses = []

    if class_mapping_path.exists():
        with open(class_mapping_path) as f:
            mapping = json.load(f)
        all_glosses = list(mapping.values())

    # Filter master categories to only this model's glosses
    if gloss_categorizer and all_glosses:
        categories = gloss_categorizer.filter_by_glosses(all_glosses)
    else:
        categories = {}

    return jsonify({
        "categories": categories,
        "all_glosses": sorted(all_glosses)
    })


@app.route('/api/gloss-tutorials', methods=['POST'])
def get_gloss_tutorials():
    """
    Get tutorial video/pose URLs for a list of glosses.
    Priority: 1) Sign Bank videos > 2) preset demo segment videos > 3) dataset pose files (visualized on-demand).
    """
    data = request.get_json()
    glosses = data.get('glosses', [])

    if not glosses:
        return jsonify({"error": "No glosses provided"}), 400

    # First, check Sign Bank for user-recorded videos (highest priority)
    gloss_to_sign_bank = {}
    if SIGN_BANK_DIR.exists():
        for gloss in glosses:
            gloss_lower = gloss.lower()
            mp4_path = SIGN_BANK_DIR / f"{gloss_lower}.mp4"
            webm_path = SIGN_BANK_DIR / f"{gloss_lower}.webm"

            if mp4_path.exists():
                gloss_to_sign_bank[gloss] = {
                    "video_url": f"/api/sign-bank/video/{gloss_lower}",
                    "source": "sign_bank"
                }
            elif webm_path.exists():
                gloss_to_sign_bank[gloss] = {
                    "video_url": f"/api/sign-bank/video/{gloss_lower}",
                    "source": "sign_bank"
                }

    # Build mapping of gloss -> demo sample segment video
    # Two-pass approach: first collect originals, then fallback to pose visualizations
    gloss_to_demo = {}
    gloss_to_pose_fallback = {}
    samples_json = DEMO_DATA_DIR / "samples.json"

    if samples_json.exists():
        with open(samples_json) as f:
            samples_data = json.load(f)

        # Also check all sample directories, not just those in samples.json
        all_sample_dirs = set()
        for sample in samples_data.get('samples', []):
            all_sample_dirs.add(sample['id'])
        # Add any directories not in samples.json
        for d in DEMO_SAMPLES_DIR.iterdir():
            if d.is_dir() and (d / "metadata.json").exists():
                all_sample_dirs.add(d.name)

        for sample_id in all_sample_dirs:
            metadata_path = DEMO_SAMPLES_DIR / sample_id / "metadata.json"

            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)

                segments = metadata.get('precomputed', {}).get('segments', [])
                # Use LLM selections if available, otherwise fall back to top_1
                selections = metadata.get('precomputed', {}).get('selections', [])

                for idx, seg in enumerate(segments):
                    # Priority: LLM selection > top_1 > expected_gloss
                    if selections and idx < len(selections):
                        gloss = selections[idx]
                    else:
                        gloss = seg.get('top_1') or seg.get('expected_gloss')

                    if not gloss:
                        continue

                    video_file = seg.get('video_file')
                    if not video_file:
                        continue

                    # Check for original video first
                    original_video_file = video_file.replace('.mp4', '_original.mp4')
                    original_video_path = DEMO_SAMPLES_DIR / sample_id / original_video_file
                    video_path = DEMO_SAMPLES_DIR / sample_id / video_file

                    if original_video_path.exists():
                        # Found original webcam recording - always prefer this
                        if gloss not in gloss_to_demo:
                            gloss_to_demo[gloss] = {
                                "sample_id": sample_id,
                                "video_url": f"/demo-data/samples/{sample_id}/{original_video_file}",
                                "source": "demo"
                            }
                    elif video_path.exists():
                        # Store as fallback (pose visualization)
                        if gloss not in gloss_to_pose_fallback:
                            gloss_to_pose_fallback[gloss] = {
                                "sample_id": sample_id,
                                "video_url": f"/demo-data/samples/{sample_id}/{video_file}",
                                "source": "demo"
                            }

        # Merge: use originals first, then fill in with pose fallbacks
        for gloss, data in gloss_to_pose_fallback.items():
            if gloss not in gloss_to_demo:
                gloss_to_demo[gloss] = data

    # Build tutorials response - Priority: sign_bank > demo > dataset
    tutorials = []
    for gloss in glosses:
        # 1. Check Sign Bank first (user-recorded videos)
        if gloss in gloss_to_sign_bank:
            tutorials.append({
                "gloss": gloss,
                "video_url": gloss_to_sign_bank[gloss]["video_url"],
                "source": "sign_bank",
                "available": True
            })
        # 2. Check demo samples
        elif gloss in gloss_to_demo:
            tutorials.append({
                "gloss": gloss,
                "video_url": gloss_to_demo[gloss]["video_url"],
                "source": "demo",
                "available": True
            })
        else:
            # Check if we can generate from dataset pose files
            pose_dir = PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "pose_files_by_gloss" / gloss
            if pose_dir.exists():
                pose_files = list(pose_dir.glob("*.pose"))
                if pose_files:
                    tutorials.append({
                        "gloss": gloss,
                        "pose_file": str(pose_files[0]),  # Use first available
                        "source": "dataset",
                        "available": True,
                        "needs_visualization": True
                    })
                else:
                    tutorials.append({
                        "gloss": gloss,
                        "available": False,
                        "reason": "No pose files found"
                    })
            else:
                tutorials.append({
                    "gloss": gloss,
                    "available": False,
                    "reason": "Gloss not in dataset"
                })

    return jsonify({"tutorials": tutorials})


@app.route('/api/visualize-gloss', methods=['POST'])
def visualize_gloss():
    """
    Generate visualization video for a gloss from dataset pose file (on-demand).
    Returns URL to the generated video.
    """
    data = request.get_json()
    gloss = data.get('gloss')
    pose_file = data.get('pose_file')

    if not gloss or not pose_file:
        return jsonify({"error": "Missing gloss or pose_file"}), 400

    pose_path = Path(pose_file)
    if not pose_path.exists():
        return jsonify({"error": "Pose file not found"}), 404

    # Generate visualization in session temp directory
    session_dir = get_session_dir()
    viz_dir = session_dir / "gloss_tutorials"
    viz_dir.mkdir(exist_ok=True)

    output_raw = viz_dir / f"{gloss}_raw.mp4"
    output_final = viz_dir / f"{gloss}.mp4"

    # Check if already generated
    if output_final.exists():
        return jsonify({
            "success": True,
            "video_url": f"/temp/{session['session_id']}/gloss_tutorials/{gloss}.mp4"
        })

    # Generate visualization
    cmd = [
        str(VISUALIZE_POSE_EXE),
        "-i", str(pose_path),
        "-o", str(output_raw),
        "--normalize"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return jsonify({"error": f"Visualization failed: {result.stderr[:200]}"}), 500

    # Resample to 30fps
    resample_cmd = [
        str(FFMPEG_EXE), "-y",
        "-i", str(output_raw),
        "-vf", "fps=30",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an", str(output_final)
    ]

    result = subprocess.run(resample_cmd, capture_output=True, text=True)
    output_raw.unlink(missing_ok=True)

    if result.returncode != 0:
        return jsonify({"error": f"Video encoding failed: {result.stderr[:200]}"}), 500

    return jsonify({
        "success": True,
        "video_url": f"/temp/{session['session_id']}/gloss_tutorials/{gloss}.mp4"
    })


@app.route('/api/generate-reference', methods=['POST'])
def generate_reference_sentence():
    """
    Generate a natural English reference sentence from a list of glosses using LLM.
    """
    data = request.get_json()
    glosses = data.get('glosses', [])

    if not glosses:
        return jsonify({"error": "No glosses provided"}), 400

    # Use the existing LLM construct endpoint logic
    raw_sentence = " ".join(glosses)

    try:
        # Create simple prompt for reference generation
        prompt = f"""Convert these ASL glosses into a natural, grammatically correct English sentence.

Glosses: {raw_sentence}

Rules:
- Use all the glosses in order
- Add appropriate filler words (the, a, is, are, etc.) for natural English
- Use proper grammar and punctuation
- Keep it simple and natural

English sentence:"""

        print(f"[LLM] Generating reference for glosses: {glosses}")

        # Call LLM
        llm = create_llm_provider(
            provider="googleaistudio",
            model_name="gemini-2.0-flash",
            max_tokens=100,
            timeout=30
        )
        response = llm.generate(prompt)

        # Clean up response
        generated = response.strip()
        # Remove quotes if present
        if generated.startswith('"') and generated.endswith('"'):
            generated = generated[1:-1]

        print(f"[LLM] Generated: {generated}")

        return jsonify({
            "success": True,
            "glosses": glosses,
            "raw_sentence": raw_sentence,
            "reference": generated
        })

    except Exception as e:
        # Fallback: just capitalize and add period
        print(f"[LLM] ERROR: {e}")
        fallback = raw_sentence.capitalize()
        if not fallback.endswith(('.', '?', '!')):
            fallback += '.'

        return jsonify({
            "success": True,
            "glosses": glosses,
            "raw_sentence": raw_sentence,
            "reference": fallback,
            "fallback": True,
            "error": str(e)
        })


@app.route('/api/process-sign', methods=['POST'])
def process_sign():
    """
    Process a single sign video clip and return gloss prediction.
    NO LLM sentence construction - just raw gloss prediction.

    FAST PATH: Uses in-memory MediaPipe instead of subprocess calls.
    This is the lightweight endpoint for real-time live demo mode.

    Input: video blob (WebM/MP4)
    Output: { gloss, confidence, top_k }
    """
    import time
    start_time = time.time()

    try:
        if 'video' not in request.files:
            return jsonify({"success": False, "error": "No video provided"}), 400

        video_file = request.files['video']
        video_bytes = video_file.read()

        # FAST PATH: Use in-memory MediaPipe if available
        if MEDIAPIPE_AVAILABLE:
            # Step 1: Decode video to frames in memory
            t1 = time.time()
            frames = decode_video_to_frames(video_bytes)
            decode_time = time.time() - t1

            if not frames or len(frames) == 0:
                return jsonify({
                    "success": False,
                    "error": "Failed to decode video - no frames extracted"
                }), 400

            # Step 2: Extract poses using persistent MediaPipe
            t2 = time.time()
            pose_array = extract_poses_from_frames(frames)
            pose_time = time.time() - t2

            if pose_array is None or len(pose_array) == 0:
                return jsonify({
                    "success": False,
                    "error": "Failed to extract poses from frames"
                }), 500

            # Step 3: Convert to pickle format in memory
            pickle_data = poses_to_pickle_data(pose_array)

            # Step 4: Save pickle to temp file for model (model expects file path)
            t3 = time.time()
            session_dir = get_session_dir()
            segments_dir = session_dir / "segments"
            segments_dir.mkdir(exist_ok=True)

            # Track sign count for sequential naming
            if 'live_sign_count' not in session:
                session['live_sign_count'] = 0
            session['live_sign_count'] += 1
            sign_num = session['live_sign_count']

            # Save to segments folder with sequential naming (preserved for session save)
            pickle_path = segments_dir / f"segment_{sign_num:03d}.pkl"

            with open(pickle_path, 'wb') as f:
                pickle.dump(pickle_data, f)

            # Also save video and pose for session saving
            video_path = segments_dir / f"segment_{sign_num:03d}.webm"
            with open(video_path, 'wb') as f:
                f.write(video_bytes)

            # Convert to MP4 for better compatibility
            mp4_path = segments_dir / f"segment_{sign_num:03d}.mp4"
            if convert_webm_to_mp4(str(video_path), str(mp4_path)):
                video_path.unlink()  # Remove webm, keep mp4

            # Save pose array as .pose file using pose-format
            try:
                from pose_format import Pose
                from pose_format.pose_header import PoseHeader, PoseHeaderDimensions, PoseHeaderComponent
                import numpy as np

                # Create pose header for MediaPipe format
                dimensions = PoseHeaderDimensions(width=1, height=1, depth=1)

                # Define body components (simplified - just the keypoints we use)
                components = [
                    PoseHeaderComponent(
                        name="pose_landmarks",
                        points=[f"point_{i}" for i in range(33)],
                        limbs=[],
                        colors=[],
                        format="XYZC"
                    ),
                    PoseHeaderComponent(
                        name="left_hand_landmarks",
                        points=[f"point_{i}" for i in range(21)],
                        limbs=[],
                        colors=[],
                        format="XYZC"
                    ),
                    PoseHeaderComponent(
                        name="right_hand_landmarks",
                        points=[f"point_{i}" for i in range(21)],
                        limbs=[],
                        colors=[],
                        format="XYZC"
                    )
                ]

                header = PoseHeader(version=0.1, dimensions=dimensions, components=components)

                # Reshape pose_array to pose-format structure
                # pose_array shape is (frames, 75, 3) - we need (frames, 1, points, dims)
                frames_count = pose_array.shape[0]

                # Split into body (33), left hand (21), right hand (21)
                body_data = pose_array[:, :33, :].reshape(frames_count, 1, 33, 3)
                left_hand_data = pose_array[:, 33:54, :].reshape(frames_count, 1, 21, 3)
                right_hand_data = pose_array[:, 54:75, :].reshape(frames_count, 1, 21, 3)

                # Add confidence channel (all 1s)
                body_conf = np.ones((frames_count, 1, 33, 1), dtype=np.float32)
                left_conf = np.ones((frames_count, 1, 21, 1), dtype=np.float32)
                right_conf = np.ones((frames_count, 1, 21, 1), dtype=np.float32)

                body_full = np.concatenate([body_data, body_conf], axis=-1)
                left_full = np.concatenate([left_hand_data, left_conf], axis=-1)
                right_full = np.concatenate([right_hand_data, right_conf], axis=-1)

                # Create pose object - this is simplified, may need adjustment
                pose_path = segments_dir / f"segment_{sign_num:03d}.pose"
                # For now, skip pose file creation - it's complex and .pkl works for predictions
            except Exception as e:
                print(f"  Note: Could not save .pose file: {e}")

            # Step 5: Run model inference
            model, tokenizer = get_model()
            prediction = predict_pose_file(
                str(pickle_path),
                model=model,
                tokenizer=tokenizer
            )
            model_time = time.time() - t3

            # Store prediction in session for later saving
            if 'live_predictions' not in session:
                session['live_predictions'] = []
            session['live_predictions'].append({
                'segment_id': sign_num,
                'gloss': prediction['gloss'],
                'confidence': prediction['confidence'],
                'top_k': prediction['top_k_predictions'][:5]
            })
            session.modified = True

            # Don't delete - keep for session saving

            total_time = time.time() - start_time
            print(f"[FAST] Sign processed in {total_time*1000:.0f}ms "
                  f"(decode:{decode_time*1000:.0f}ms, pose:{pose_time*1000:.0f}ms, model:{model_time*1000:.0f}ms) "
                  f"- {len(frames)} frames -> {prediction['gloss']}")

            return jsonify({
                "success": True,
                "gloss": prediction['gloss'],
                "confidence": prediction['confidence'],
                "top_k": prediction['top_k_predictions'][:3],
                "timing_ms": int(total_time * 1000)
            })

        # SLOW PATH: Fallback to file-based processing if MediaPipe not available
        else:
            print("[SLOW] Using file-based processing (MediaPipe not available)")
            session_dir = get_session_dir()
            segments_dir = session_dir / "segments"
            segments_dir.mkdir(exist_ok=True)

            # Track sign count for sequential naming
            if 'live_sign_count' not in session:
                session['live_sign_count'] = 0
            session['live_sign_count'] += 1
            sign_num = session['live_sign_count']

            video_path = segments_dir / f"segment_{sign_num:03d}.webm"

            with open(video_path, 'wb') as f:
                f.write(video_bytes)

            # Convert WebM to MP4 if needed
            mp4_path = segments_dir / f"segment_{sign_num:03d}.mp4"
            if convert_webm_to_mp4(str(video_path), str(mp4_path)):
                video_path.unlink()  # Remove webm, keep mp4
                video_path = mp4_path

            # Use video_to_pose.exe
            pose_path = segments_dir / f"segment_{sign_num:03d}.pose"
            cmd = [
                str(VIDEO_TO_POSE_EXE),
                "-i", str(video_path),
                "-o", str(pose_path),
                "--format", "mediapipe"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return jsonify({
                    "success": False,
                    "error": f"Pose extraction failed: {result.stderr}"
                }), 500

            pickle_path = convert_pose_to_pickle(str(pose_path))
            if not pickle_path:
                return jsonify({
                    "success": False,
                    "error": "Failed to convert pose to pickle format"
                }), 500

            model, tokenizer = get_model()
            prediction = predict_pose_file(
                pickle_path,
                model=model,
                tokenizer=tokenizer
            )

            # Keep segment files for session saving (only clean up temp video)

            total_time = time.time() - start_time
            print(f"[SLOW] Sign processed in {total_time*1000:.0f}ms -> {prediction['gloss']}")

            return jsonify({
                "success": True,
                "gloss": prediction['gloss'],
                "confidence": prediction['confidence'],
                "top_k": prediction['top_k_predictions'][:3],
                "timing_ms": int(total_time * 1000)
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/process-signs-batch', methods=['POST'])
def process_signs_batch():
    """
    Process a full video and return all detected sign predictions.
    NO LLM sentence construction - just raw gloss predictions for each segment.

    This is for Live Demo mode where we want to show predictions without LLM.

    Input: video blob (WebM/MP4)
    Output: { predictions: [{gloss, confidence, top_k}, ...], raw_sentence }
    """
    try:
        session_dir = get_session_dir()

        if 'video' not in request.files:
            return jsonify({"success": False, "error": "No video provided"}), 400

        video_file = request.files['video']
        original_filename = video_file.filename or 'recording.webm'
        ext = Path(original_filename).suffix.lower() or '.webm'
        video_path = session_dir / f"capture{ext}"
        video_file.save(str(video_path))

        # Convert WebM to MP4 if needed
        if ext == '.webm':
            mp4_path = session_dir / "capture.mp4"
            if convert_webm_to_mp4(str(video_path), str(mp4_path)):
                video_path = mp4_path

        # Step 1: Convert video to pose
        pose_path = session_dir / "capture.pose"
        cmd = [
            str(VIDEO_TO_POSE_EXE),
            "-i", str(video_path),
            "-o", str(pose_path),
            "--format", "mediapipe"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({
                "success": False,
                "error": f"Pose extraction failed: {result.stderr}"
            }), 500

        # Step 2: Segment based on configuration
        segments_dir = session_dir / "segments"
        segments_dir.mkdir(exist_ok=True)

        # Clear previous segments
        for f in segments_dir.glob("*"):
            f.unlink()

        # Use hybrid segmenter (video motion + pose slicing)
        segmenter = HybridSegmenter(
            motion_threshold=500000,
            cooldown_frames=int(LIVE_MOTION_CONFIG['cooldown_ms'] / 33.3),  # Convert ms to frames at ~30fps
            min_sign_frames=int(LIVE_MOTION_CONFIG['min_sign_ms'] / 33.3),
            max_sign_frames=int(LIVE_MOTION_CONFIG['max_sign_ms'] / 33.3),
            padding_before=3,
            padding_after=3
        )

        segment_files = segmenter.segment_video_and_pose(
            str(video_path),
            str(pose_path),
            str(segments_dir),
            verbose=True
        )

        if not segment_files:
            return jsonify({
                "success": False,
                "error": "No signs detected. Try signing more clearly with pauses."
            }), 400

        # Step 3: Predict each segment
        model, tokenizer = get_model()
        predictions = []

        for i, seg_file in enumerate(segment_files):
            pickle_path = convert_pose_to_pickle(seg_file)

            if pickle_path:
                prediction = predict_pose_file(
                    pickle_path,
                    model=model,
                    tokenizer=tokenizer
                )

                predictions.append({
                    "segment_id": i + 1,
                    "gloss": prediction['gloss'],
                    "confidence": prediction['confidence'],
                    "top_k": prediction['top_k_predictions'][:3]
                })
            else:
                predictions.append({
                    "segment_id": i + 1,
                    "gloss": "UNKNOWN",
                    "confidence": 0.0,
                    "top_k": []
                })

        # Build raw sentence (just concatenated glosses)
        raw_sentence = " ".join([p['gloss'] for p in predictions])

        return jsonify({
            "success": True,
            "predictions": predictions,
            "raw_sentence": raw_sentence,
            "segment_count": len(predictions)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# CLOSED-CAPTIONS MODE
# ============================================================================

# Global caption service instances (per session)
_caption_services = {}
_caption_services_lock = threading.Lock()


def _get_cc_session_id():
    """Get session ID from request body, query params, or Flask session.
    Cookies may be blocked in iframes (tracking prevention), so accept
    session_id from request data as a fallback."""
    # Try request JSON body
    if request.is_json and request.json and request.json.get('session_id'):
        return request.json['session_id']
    # Try form data
    if request.form.get('session_id'):
        return request.form['session_id']
    # Try query params
    if request.args.get('session_id'):
        return request.args['session_id']
    # Fall back to Flask session cookie
    return session.get('session_id')


def get_caption_service(session_id: str) -> CaptionService:
    """Get or create caption service for session."""
    with _caption_services_lock:
        if session_id not in _caption_services:
            session_dir = TEMP_DIR / session_id / "captions"
            session_dir.mkdir(parents=True, exist_ok=True)

            # Get model (uses cached global model)
            model, tokenizer = get_model()

            # Create service
            service = CaptionService(
                session_dir=session_dir,
                model=model,
                tokenizer=tokenizer
            )
            _caption_services[session_id] = service

        return _caption_services[session_id]


@app.route('/closed-captions')
def closed_captions_page():
    """Closed-captions mode page."""
    return render_template('closed_captions.html')


@app.route('/api/cc/start', methods=['POST'])
def cc_start_session():
    """Start closed-captions session."""
    try:
        session_id = session.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id

        service = get_caption_service(session_id)
        service.start()

        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Caption service started'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/cc/stop', methods=['POST'])
def cc_stop_session():
    """Stop closed-captions session."""
    session_id = _get_cc_session_id()
    if not session_id:
        return jsonify({'success': False, 'error': 'No active session'}), 400

    final_caption = ''
    with _caption_services_lock:
        if session_id in _caption_services:
            service = _caption_services[session_id]
            service.stop()  # This flushes remaining glosses
            final_caption = service.get_current_caption()

    return jsonify({
        'success': True,
        'message': 'Caption service stopped',
        'final_caption': final_caption
    })


@app.route('/api/cc/reset', methods=['POST'])
def cc_reset_session():
    """Reset closed-captions session."""
    session_id = _get_cc_session_id()
    if not session_id:
        session_id = str(uuid.uuid4())

    with _caption_services_lock:
        if session_id in _caption_services:
            _caption_services[session_id].reset()
            del _caption_services[session_id]

    return jsonify({'success': True, 'message': 'Session reset'})


@app.route('/api/cc/process-sign', methods=['POST'])
def cc_process_sign():
    """
    Process a single sign video blob for closed-captions.

    Uses common camera processor (VIDEO_TO_POSE_EXE) for consistent,
    high-quality pose extraction matching Live Mode.
    """
    session_id = _get_cc_session_id()
    if not session_id:
        session_id = str(uuid.uuid4())

    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video provided'}), 400

    video_file = request.files['video']
    video_bytes = video_file.read()

    if len(video_bytes) == 0:
        return jsonify({'success': False, 'error': 'Empty video'}), 400

    try:
        # Get model and camera processor
        model, tokenizer = get_model()
        processor = get_camera_processor(model, tokenizer)

        # Use common camera processor (same as Live Mode)
        # This uses VIDEO_TO_POSE_EXE for better hand detection
        prediction = processor.process_video_bytes(video_bytes, video_format='webm')

        if prediction is None:
            return jsonify({'success': False, 'error': 'Could not process video'}), 400

        # Get caption service and add gloss
        service = get_caption_service(session_id)

        gloss_data = {
            'gloss': prediction['gloss'],
            'confidence': prediction['confidence'],
            'top_k': prediction['top_k_predictions'][:3]
        }

        # Add to context and write to file
        service.context.add_gloss(gloss_data)
        service._write_gloss_to_file(gloss_data)

        return jsonify({
            'success': True,
            'gloss': prediction['gloss'],
            'confidence': prediction['confidence'],
            'top_k': prediction['top_k_predictions'][:3]
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/cc/get-caption', methods=['GET'])
def cc_get_caption():
    """Get current caption from sentence file."""
    session_id = _get_cc_session_id()
    if not session_id:
        return jsonify({'caption': '', 'gloss_count': 0, 'gloss_status': []})

    service = get_caption_service(session_id)

    # Read from file (what the file watcher writes)
    caption = service.get_sentence_file_content()
    state = service.get_state()

    return jsonify({
        'caption': caption,
        'running_caption': state['running_caption'],
        'gloss_count': state['gloss_count'],
        'processed_count': state['processed_count'],
        'is_running': state['is_running'],
        'gloss_status': state.get('gloss_status', [])
    })


@app.route('/api/cc/debug', methods=['GET'])
def cc_debug():
    """Debug endpoint showing caption service internal state."""
    session_id = _get_cc_session_id()
    if not session_id:
        return jsonify({'error': 'No session_id provided', 'active_sessions': list(_caption_services.keys())})

    with _caption_services_lock:
        if session_id not in _caption_services:
            return jsonify({
                'error': f'No caption service for session {session_id}',
                'active_sessions': list(_caption_services.keys())[:5]
            })

        service = _caption_services[session_id]
        state = service.get_state()

        # Check file contents
        gloss_file_exists = service.gloss_file.exists()
        gloss_file_content = ''
        gloss_file_lines = 0
        if gloss_file_exists:
            try:
                gloss_file_content = service.gloss_file.read_text(encoding='utf-8')
                gloss_file_lines = len([l for l in gloss_file_content.strip().split('\n') if l.strip()])
            except Exception as e:
                gloss_file_content = f'Error reading: {e}'

        sentence_file_exists = service.sentence_file.exists()
        sentence_content = ''
        if sentence_file_exists:
            try:
                sentence_content = service.sentence_file.read_text(encoding='utf-8')
            except Exception as e:
                sentence_content = f'Error reading: {e}'

        # LLM status
        llm_info = {
            'initialized': service._llm_initialized,
            'provider_set': service.llm_provider is not None,
            'prompt_loaded': service.prompt_template is not None,
            'prompt_length': len(service.prompt_template) if service.prompt_template else 0,
            'last_error': service._last_llm_error,
        }

        return jsonify({
            'session_id': session_id,
            'state': state,
            'gloss_file': str(service.gloss_file),
            'gloss_file_exists': gloss_file_exists,
            'gloss_file_lines': gloss_file_lines,
            'sentence_file': str(service.sentence_file),
            'sentence_file_exists': sentence_file_exists,
            'sentence_content': sentence_content,
            'llm': llm_info,
            'file_watcher_running': service._file_watcher_running,
            'file_watcher_thread_alive': service._file_watcher_thread.is_alive() if service._file_watcher_thread else False,
        })


@app.route('/api/cc/config', methods=['GET'])
def cc_get_config():
    """Get closed-captions configuration for frontend."""
    return jsonify({
        'motion_config': cc_config.MOTION_CONFIG,
        'caption_buffer_size': cc_config.CAPTION_BUFFER_SIZE,
        'file_check_interval': cc_config.FILE_CHECK_INTERVAL,
        'min_translation_delay': cc_config.MIN_TRANSLATION_DELAY,
    })


# ============================================================================
# DIAGNOSTICS
# ============================================================================

@app.route('/api/diagnostics', methods=['GET'])
def diagnostics():
    """Check pipeline component status for debugging deployment issues."""
    checks = {}

    # Check model
    try:
        model, tokenizer = get_model()
        checks['model_loaded'] = model is not None
        checks['tokenizer_loaded'] = tokenizer is not None
        checks['num_classes'] = len(tokenizer) if tokenizer else 0
    except Exception as e:
        checks['model_loaded'] = False
        checks['model_error'] = str(e)

    # Check video_to_pose executable
    import shutil
    from camera_processor import VIDEO_TO_POSE_EXE
    checks['video_to_pose_path'] = str(VIDEO_TO_POSE_EXE)
    checks['video_to_pose_found'] = shutil.which(str(VIDEO_TO_POSE_EXE)) is not None or os.path.isfile(str(VIDEO_TO_POSE_EXE))

    # Check video_to_pose runs
    try:
        result = subprocess.run([str(VIDEO_TO_POSE_EXE), '--help'], capture_output=True, text=True, timeout=10)
        checks['video_to_pose_runs'] = True
        checks['video_to_pose_returncode'] = result.returncode
    except FileNotFoundError:
        checks['video_to_pose_runs'] = False
        checks['video_to_pose_error'] = 'FileNotFoundError - executable not found'
    except Exception as e:
        checks['video_to_pose_runs'] = False
        checks['video_to_pose_error'] = str(e)

    # Check ffmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        checks['ffmpeg_available'] = result.returncode == 0
    except Exception:
        checks['ffmpeg_available'] = False

    # Check mediapipe
    try:
        import mediapipe
        checks['mediapipe_available'] = True
        checks['mediapipe_version'] = mediapipe.__version__
    except ImportError:
        checks['mediapipe_available'] = False

    # Check pose_format
    try:
        import pose_format
        checks['pose_format_available'] = True
    except ImportError:
        checks['pose_format_available'] = False

    # Platform info
    import platform
    checks['platform'] = platform.system()
    checks['python_version'] = platform.python_version()

    # LLM / Caption Service checks
    checks['google_api_key_set'] = bool(os.environ.get('GOOGLE_API_KEY'))
    checks['llm_provider_config'] = cc_config.LLM_PROVIDER

    # Check if LLM prompt file exists
    prompt_path = PROJECT_ROOT / "project-utilities" / "llm_interface" / "prompts" / "llm_prompt_closed_captions.txt"
    checks['llm_prompt_exists'] = prompt_path.exists()
    checks['llm_prompt_path'] = str(prompt_path)

    # Check if llm_factory can be imported
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "project-utilities" / "llm_interface"))
        from llm_factory import create_llm_provider as _test_import
        checks['llm_factory_importable'] = True
    except Exception as e:
        checks['llm_factory_importable'] = False
        checks['llm_factory_error'] = str(e)

    # Check if google-generativeai is installed
    try:
        import google.generativeai
        checks['google_generativeai_installed'] = True
    except ImportError:
        checks['google_generativeai_installed'] = False

    # Check python-dotenv
    try:
        import dotenv
        checks['dotenv_installed'] = True
    except ImportError:
        checks['dotenv_installed'] = False

    # Active caption services
    with _caption_services_lock:
        active_sessions = {}
        for sid, svc in _caption_services.items():
            state = svc.get_state()
            active_sessions[sid[:8] + '...'] = {
                'is_running': state['is_running'],
                'gloss_count': state['gloss_count'],
                'processed_count': state['processed_count'],
                'running_caption': state['running_caption'],
            }
        checks['active_caption_sessions'] = active_sessions

    return jsonify(checks)


# ============================================================================
# STATIC FILE SERVING
# ============================================================================

@app.route('/temp/<session_id>/<path:filename>')
def serve_temp_file(session_id, filename):
    """Serve temporary files (pose videos, etc.)"""
    return send_from_directory(TEMP_DIR / session_id, filename)


@app.route('/temp/<session_id>/segments/<filename>')
def serve_segment_file(session_id, filename):
    """Serve segment video files"""
    return send_from_directory(TEMP_DIR / session_id / "segments", filename)


# ============================================================================
# SIGN BANK - Record and manage individual sign videos
# ============================================================================

# Sign bank storage directory (persistent across sessions)
SIGN_BANK_DIR = Path(__file__).parent / "sign-bank"
SIGN_BANK_DIR.mkdir(exist_ok=True)


@app.route('/sign-bank')
def sign_bank_page():
    """Sign Bank page for recording individual signs"""
    return render_template('sign_bank.html')


@app.route('/api/sign-bank/list')
def sign_bank_list():
    """List all recorded signs in the sign bank"""
    recorded_words = []

    if SIGN_BANK_DIR.exists():
        for video_file in SIGN_BANK_DIR.glob("*.mp4"):
            recorded_words.append(video_file.stem.lower())

    return jsonify({
        'success': True,
        'recorded_words': sorted(recorded_words),
        'count': len(recorded_words)
    })


@app.route('/api/sign-bank/save', methods=['POST'])
def sign_bank_save():
    """Save a recorded sign video to the sign bank"""
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400

        video_file = request.files['video']
        gloss = request.form.get('gloss', '').strip().lower()

        if not gloss:
            return jsonify({'success': False, 'error': 'No gloss provided'}), 400

        # Save the video file
        video_path = SIGN_BANK_DIR / f"{gloss}.webm"
        video_file.save(str(video_path))

        # Convert WebM to MP4 for better compatibility
        mp4_path = SIGN_BANK_DIR / f"{gloss}.mp4"

        try:
            # Try to convert using ffmpeg
            result = subprocess.run([
                FFMPEG_EXE, '-y', '-i', str(video_path),
                '-c:v', 'libx264', '-preset', 'fast',
                '-c:a', 'aac', '-strict', 'experimental',
                str(mp4_path)
            ], capture_output=True, timeout=30)

            if result.returncode == 0:
                # Remove the WebM file after successful conversion
                video_path.unlink()
                print(f"[SignBank] Saved and converted: {gloss}.mp4")
            else:
                # Keep WebM if conversion fails
                print(f"[SignBank] Conversion failed, keeping WebM: {gloss}")
                mp4_path = video_path
        except Exception as e:
            print(f"[SignBank] Conversion error: {e}")
            mp4_path = video_path

        return jsonify({
            'success': True,
            'gloss': gloss,
            'path': str(mp4_path)
        })

    except Exception as e:
        print(f"[SignBank] Save error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sign-bank/video/<gloss>')
def sign_bank_video(gloss):
    """Serve a sign bank video file"""
    gloss = gloss.lower()

    # Try MP4 first, then WebM
    mp4_path = SIGN_BANK_DIR / f"{gloss}.mp4"
    webm_path = SIGN_BANK_DIR / f"{gloss}.webm"

    if mp4_path.exists():
        return send_from_directory(SIGN_BANK_DIR, f"{gloss}.mp4")
    elif webm_path.exists():
        return send_from_directory(SIGN_BANK_DIR, f"{gloss}.webm")
    else:
        return jsonify({'success': False, 'error': 'Video not found'}), 404


@app.route('/api/sign-bank/delete/<gloss>', methods=['DELETE'])
def sign_bank_delete(gloss):
    """Delete a sign from the sign bank"""
    gloss = gloss.lower()

    deleted = False
    for ext in ['mp4', 'webm']:
        path = SIGN_BANK_DIR / f"{gloss}.{ext}"
        if path.exists():
            path.unlink()
            deleted = True

    if deleted:
        return jsonify({'success': True, 'gloss': gloss})
    else:
        return jsonify({'success': False, 'error': 'Video not found'}), 404


@app.route('/api/sign-bank/has/<gloss>')
def sign_bank_has(gloss):
    """Check if a sign exists in the sign bank"""
    gloss = gloss.lower()

    mp4_path = SIGN_BANK_DIR / f"{gloss}.mp4"
    webm_path = SIGN_BANK_DIR / f"{gloss}.webm"

    exists = mp4_path.exists() or webm_path.exists()

    return jsonify({
        'success': True,
        'gloss': gloss,
        'exists': exists
    })


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("ASL Show-and-Tell Demo Application")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Video to Pose: {VIDEO_TO_POSE_EXE} (exists: {VIDEO_TO_POSE_EXE.exists()})")
    print(f"Visualize Pose: {VISUALIZE_POSE_EXE} (exists: {VISUALIZE_POSE_EXE.exists()})")
    ffmpeg_exists = Path(FFMPEG_EXE).exists() if FFMPEG_EXE != "ffmpeg" else "in PATH"
    print(f"FFmpeg: {FFMPEG_EXE} (exists: {ffmpeg_exists})")
    print(f"Model Checkpoint: {CHECKPOINT_PATH} (exists: {CHECKPOINT_PATH.exists()})")
    print(f"LLM Prompt: {LLM_PROMPT_PATH} (exists: {LLM_PROMPT_PATH.exists()})")
    print(f"BLEU Available: {BLEU_AVAILABLE}")
    print(f"BERTScore Available: {BERTSCORE_AVAILABLE}")
    print(f"Quality Scoring Available: {QUALITY_SCORING_AVAILABLE}")
    print("=" * 60)
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on http://localhost:{port}")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=port)
