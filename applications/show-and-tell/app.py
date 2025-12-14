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
MODELS_DIR = PROJECT_ROOT / "models"

# Add paths for imports
sys.path.insert(0, str(APPLICATIONS_DIR))
sys.path.insert(0, str(PROJECT_UTILITIES_DIR))
sys.path.insert(0, str(PROJECT_UTILITIES_DIR / "llm_interface"))
sys.path.insert(0, str(PROJECT_UTILITIES_DIR / "synthetic_evaluation"))
sys.path.insert(0, str(MODELS_DIR / "openhands-modernized" / "src"))
sys.path.insert(0, str(MODELS_DIR / "openhands-modernized" / "src" / "util"))

# ============================================================================
# IMPORTS FROM EXISTING LIBRARIES (no duplication)
# ============================================================================

# Import motion-based segmenter
from motion_based_segmenter import MotionBasedSegmenter

# Import model inference functions
from openhands_modernized_inference import load_model_from_checkpoint, predict_pose_file

# Import LLM interface
from llm_factory import create_llm_provider

# Import BLEU calculator
try:
    from calculate_sent_bleu import calculate_bleu_from_glosses
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("WARNING: BLEU calculator not available")

# Import BERTScore
try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("WARNING: BERTScore not available")

# Import quality scoring (GPT-2)
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
    QUALITY_SCORING_AVAILABLE = True
except ImportError:
    QUALITY_SCORING_AVAILABLE = False
    print("WARNING: Quality scoring not available")

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

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths to existing executables (in venv/Scripts)
VENV_SCRIPTS = Path(sys.executable).parent
VIDEO_TO_POSE_EXE = VENV_SCRIPTS / "video_to_pose.exe"
VISUALIZE_POSE_EXE = VENV_SCRIPTS / "visualize_pose.exe"

# Path to existing LLM prompt
LLM_PROMPT_PATH = APPLICATIONS_DIR / "llm_prompt_topk.txt"

# Model checkpoint path (50-class model)
CHECKPOINT_PATH = MODELS_DIR / "training-scripts" / "models" / "wlasl_50_class_model"

# Metric weights (from evaluate_synthetic_dataset.py)
METRIC_WEIGHTS = {
    'quality': 0.4,
    'coverage_f1': 0.25,
    'bertscore': 0.2,
    'bleu': 0.15
}

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Temp directory for this app
TEMP_DIR = Path(__file__).parent / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# Global model cache (loaded once)
_model_cache = {"model": None, "tokenizer": None}

# Global quality scorer cache
_quality_scorer = {"model": None, "tokenizer": None}


def get_model():
    """Get cached model or load it"""
    if _model_cache["model"] is None:
        print("Loading model from checkpoint...")
        # vocab_size will be read from config.json in checkpoint directory
        _model_cache["model"], _model_cache["tokenizer"] = load_model_from_checkpoint(
            str(CHECKPOINT_PATH)
        )
        print("Model loaded successfully")
    return _model_cache["model"], _model_cache["tokenizer"]


def get_quality_scorer():
    """Get cached quality scorer or load it"""
    if not QUALITY_SCORING_AVAILABLE:
        return None, None
    if _quality_scorer["model"] is None:
        print("Loading GPT-2 for quality scoring...")
        _quality_scorer["tokenizer"] = GPT2Tokenizer.from_pretrained('gpt2')
        _quality_scorer["model"] = GPT2LMHeadModel.from_pretrained('gpt2')
        _quality_scorer["model"].eval()
        print("Quality scorer loaded")
    return _quality_scorer["model"], _quality_scorer["tokenizer"]


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
# DEMO DATA DIRECTORY
# ============================================================================

DEMO_DATA_DIR = Path(__file__).parent / "demo-data"
DEMO_SAMPLES_DIR = DEMO_DATA_DIR / "samples"


# ============================================================================
# DEMO MODE API ROUTES
# ============================================================================

@app.route('/api/samples')
def get_samples():
    """Get list of available demo samples"""
    samples_json = DEMO_DATA_DIR / "samples.json"
    if samples_json.exists():
        with open(samples_json) as f:
            return jsonify(json.load(f))
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

        # Step 2: Visualize pose using EXISTING visualize_pose.exe
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
        viz_path = session_dir / "pose_video.mp4"
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

        return jsonify({
            "success": True,
            "pose_video_url": f"/temp/{session['session_id']}/pose_video.mp4",
            "pose_file": str(pose_path)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/segment', methods=['POST'])
def segment_pose():
    """
    Segment pose file using EXISTING MotionBasedSegmenter class
    Predict using EXISTING predict_pose_file function
    """
    try:
        session_dir = get_session_dir()
        data = request.get_json()
        pose_file = data.get('pose_file', str(session_dir / "capture.pose"))

        if not Path(pose_file).exists():
            return jsonify({"success": False, "error": "Pose file not found"}), 404

        segments_dir = session_dir / "segments"

        # Clear previous segments
        for f in segments_dir.glob("*"):
            f.unlink()

        # Use EXISTING MotionBasedSegmenter (from motion_based_segmenter.py)
        segmenter = MotionBasedSegmenter(
            velocity_threshold=0.02,
            min_sign_duration=10,
            min_rest_duration=5,
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

            # Visualize segment using EXISTING visualize_pose.exe
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
                    "video_url": f"/temp/{session['session_id']}/segments/{seg_path.stem}.mp4",
                    "top_1": prediction['gloss'],
                    "confidence": prediction['confidence'],
                    "top_k": prediction['top_k_predictions'][:3]  # Top-3
                })
            else:
                results.append({
                    "segment_id": i + 1,
                    "video_url": f"/temp/{session['session_id']}/segments/{seg_path.stem}.mp4",
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


def convert_pose_to_pickle(pose_file):
    """
    Convert .pose file to pickle format for model inference.
    Uses same logic as EndToEndPipeline.step3_convert_segments_to_pickle
    """
    try:
        with open(pose_file, "rb") as f:
            buffer = f.read()
            pose = Pose.read(buffer)

        pose_data = pose.body.data

        if len(pose_data.shape) == 4:
            pose_sequence = pose_data[:, 0, :, :]
        else:
            pose_sequence = pose_data

        # Extract 75-point subset (same as predict_sentence.py line 706-712)
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
            'confidences': pose_75pt[:, :, 3] if pose_75pt.shape[2] > 3 else np.ones(pose_75pt.shape[:2]),
            'gloss': 'UNKNOWN'
        }

        with open(pickle_path, 'wb') as f:
            pickle.dump(pickle_data, f)

        return pickle_path

    except Exception as e:
        print(f"Error converting pose to pickle: {e}")
        return None


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
        prompt = prompt_template.format(gloss_details=gloss_details)

        # Use EXISTING LLM provider (Google AI Studio)
        llm = create_llm_provider(
            provider="googleaistudio",
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


@app.route('/api/evaluate', methods=['POST'])
def evaluate_sentences():
    """
    Calculate metrics using EXISTING evaluation functions
    """
    try:
        data = request.get_json()
        raw_sentence = data.get('raw_sentence', session.get('raw_sentence', ''))
        llm_sentence = data.get('llm_sentence', session.get('llm_sentence', ''))
        reference = data.get('reference', '')

        if not reference:
            return jsonify({"success": False, "error": "Reference sentence required"}), 400

        # Calculate metrics for both sentences
        raw_metrics = calculate_all_metrics(raw_sentence, reference)
        llm_metrics = calculate_all_metrics(llm_sentence, reference)

        # Calculate composite scores using EXISTING weights
        raw_composite = calculate_composite_score(raw_metrics)
        llm_composite = calculate_composite_score(llm_metrics)

        return jsonify({
            "success": True,
            "raw": {**raw_metrics, "composite": raw_composite},
            "llm": {**llm_metrics, "composite": llm_composite}
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


def calculate_all_metrics(sentence, reference):
    """Calculate BLEU, BERT, and Quality scores"""
    metrics = {}

    # BLEU Score (using sacrebleu)
    try:
        import sacrebleu
        bleu = sacrebleu.sentence_bleu(sentence, [reference])
        metrics['bleu'] = bleu.score
    except Exception as e:
        print(f"BLEU calculation failed: {e}")
        metrics['bleu'] = 0.0

    # BERT Score
    if BERTSCORE_AVAILABLE:
        try:
            P, R, F1 = bert_score_fn(
                [sentence], [reference],
                lang='en',
                verbose=False
            )
            metrics['bert'] = F1.item() * 100
        except Exception as e:
            print(f"BERTScore calculation failed: {e}")
            metrics['bert'] = 0.0
    else:
        metrics['bert'] = 0.0

    # Quality Score (reference-free, using GPT-2 perplexity)
    if QUALITY_SCORING_AVAILABLE:
        try:
            metrics['quality'] = calculate_quality_score(sentence)
        except Exception as e:
            print(f"Quality score calculation failed: {e}")
            metrics['quality'] = 0.0
    else:
        metrics['quality'] = 0.0

    return metrics


def calculate_quality_score(sentence):
    """
    Calculate reference-free quality score using GPT-2 perplexity.
    Lower perplexity = more fluent = higher quality.
    Same approach as evaluate_synthetic_dataset.py
    """
    model, tokenizer = get_quality_scorer()
    if model is None:
        return 0.0

    try:
        inputs = tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss.item()

        # Convert loss to perplexity
        perplexity = np.exp(loss)

        # Convert perplexity to 0-100 score (lower perplexity = higher score)
        # Typical perplexity range: 10-1000
        # Score = 100 * (1 - log(perplexity)/log(1000))
        score = max(0, min(100, 100 * (1 - np.log(perplexity) / np.log(1000))))

        return score

    except Exception as e:
        print(f"Quality score error: {e}")
        return 0.0


def calculate_composite_score(metrics):
    """
    Calculate weighted composite score using EXISTING weights
    """
    # Only use available metrics
    score = 0.0
    total_weight = 0.0

    if 'quality' in metrics and metrics['quality'] > 0:
        score += METRIC_WEIGHTS['quality'] * metrics['quality']
        total_weight += METRIC_WEIGHTS['quality']

    if 'bert' in metrics and metrics['bert'] > 0:
        score += METRIC_WEIGHTS['bertscore'] * metrics['bert']
        total_weight += METRIC_WEIGHTS['bertscore']

    if 'bleu' in metrics and metrics['bleu'] > 0:
        score += METRIC_WEIGHTS['bleu'] * metrics['bleu']
        total_weight += METRIC_WEIGHTS['bleu']

    if total_weight > 0:
        return score / total_weight * (total_weight / sum(METRIC_WEIGHTS.values()))
    return 0.0


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
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("ASL Show-and-Tell Demo Application")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Video to Pose: {VIDEO_TO_POSE_EXE} (exists: {VIDEO_TO_POSE_EXE.exists()})")
    print(f"Visualize Pose: {VISUALIZE_POSE_EXE} (exists: {VISUALIZE_POSE_EXE.exists()})")
    print(f"Model Checkpoint: {CHECKPOINT_PATH} (exists: {CHECKPOINT_PATH.exists()})")
    print(f"LLM Prompt: {LLM_PROMPT_PATH} (exists: {LLM_PROMPT_PATH.exists()})")
    print(f"BLEU Available: {BLEU_AVAILABLE}")
    print(f"BERTScore Available: {BERTSCORE_AVAILABLE}")
    print(f"Quality Scoring Available: {QUALITY_SCORING_AVAILABLE}")
    print("=" * 60)
    print("Starting server on http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
