# -*- coding: utf-8 -*-
"""
predict_sentence.py
Real-time sign language translation from webcam to English sentence

Pipeline flow (Real-time Mode):
1. Capture video from webcam continuously
2. Detect signing activity (motion detection)
3. Buffer frames during active signing
4. Convert buffered frames to pose data using MediaPipe
5. Segment poses (auto-detect sign boundaries)
6. Predict individual glosses using OpenHands model
7. Construct grammatically correct English sentence with Gemini API

Pipeline flow (File Mode):
1. Accept input video file
2. Convert video to pose file using video_to_pose
3. Segment pose file using pose_segment2.py with auto-detect
4. Convert pose segments to pickle segments
5. Predict individual glosses using OpenHands model
6. Use Gemini API to construct grammatically correct English sentence

Usage:
    Real-time webcam: python predict_sentence.py --webcam [--checkpoint path] [--gemini-api-key key]
    File mode:        python predict_sentence.py input_video.mp4 [--checkpoint path] [--gemini-api-key key]
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
import glob
import pickle
from pathlib import Path
import json
import time
import threading
from queue import Queue

# Import motion-based segmenter
from motion_based_segmenter import MotionBasedSegmenter
from collections import deque

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import OpenCV for webcam capture
try:
    import cv2
    CV2_AVAILABLE = True
    print("SUCCESS: OpenCV imported successfully")
except ImportError as e:
    CV2_AVAILABLE = False
    print(f"ERROR: Failed to import OpenCV: {e}")
    print("INSTALL: Run 'pip install opencv-python' to enable webcam mode")

# Import numpy for frame processing
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    print("SUCCESS: NumPy imported successfully")
except ImportError as e:
    NUMPY_AVAILABLE = False
    print(f"ERROR: Failed to import NumPy: {e}")

# Import MediaPipe for real-time pose estimation
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("SUCCESS: MediaPipe imported successfully")
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"ERROR: Failed to import MediaPipe: {e}")
    print("INSTALL: Run 'pip install mediapipe' to enable real-time pose estimation")

# Import required modules
try:
    from openhands_modernized import predict_pose_file, load_model_from_checkpoint
    print("SUCCESS: OpenHands model imported successfully")
except ImportError as e:
    print(f"ERROR: Failed to import OpenHands model: {e}")
    sys.exit(1)

try:
    from pose_format import Pose
    print("SUCCESS: Pose format library imported")
except ImportError as e:
    print(f"ERROR: Failed to import pose_format: {e}")
    sys.exit(1)

# Optional: Check if Gemini API is available (lazy import)
GEMINI_AVAILABLE = None  # Will be checked when needed


class MotionDetector:
    """Detect signing activity using motion detection"""

    def __init__(self, motion_threshold=500, cooldown_frames=10):
        self.motion_threshold = motion_threshold
        self.cooldown_frames = cooldown_frames
        self.prev_frame = None
        self.frames_since_motion = 0

    def detect_motion(self, frame):
        """
        Detect if there's significant motion in the frame.
        Returns: (is_active, motion_score)
        """
        if not NUMPY_AVAILABLE or not CV2_AVAILABLE:
            return True, 0  # Fallback: always active

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Initialize on first frame
        if self.prev_frame is None:
            self.prev_frame = gray
            return False, 0

        # Calculate frame difference
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Calculate motion score
        motion_score = np.sum(thresh)

        # Update previous frame
        self.prev_frame = gray

        # Determine if signing is active
        is_active = motion_score > self.motion_threshold

        if is_active:
            self.frames_since_motion = 0
        else:
            self.frames_since_motion += 1

        return is_active, motion_score

    def is_signing_complete(self):
        """Check if signing has stopped (cooldown period elapsed)"""
        return self.frames_since_motion >= self.cooldown_frames


class WebcamCapture:
    """Capture video frames from webcam with buffering"""

    def __init__(self, camera_index=0, buffer_seconds=10, fps=30):
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required for webcam capture")

        self.camera_index = camera_index
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.max_buffer_size = int(buffer_seconds * fps)

        self.cap = None
        self.frame_buffer = deque(maxlen=self.max_buffer_size)
        self.is_capturing = False
        self.capture_thread = None

    def start(self):
        """Start webcam capture"""
        print(f"WEBCAM: Starting capture from camera {self.camera_index}")

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        print("WEBCAM: Capture started successfully")

    def _capture_loop(self):
        """Continuous capture loop (runs in separate thread)"""
        while self.is_capturing:
            ret, frame = self.cap.read()
            if ret:
                self.frame_buffer.append(frame.copy())
            else:
                print("WARNING: Failed to read frame from camera")
                time.sleep(0.1)

    def get_latest_frame(self):
        """Get the most recent frame"""
        if len(self.frame_buffer) > 0:
            return self.frame_buffer[-1]
        return None

    def get_buffered_frames(self):
        """Get all buffered frames"""
        return list(self.frame_buffer)

    def clear_buffer(self):
        """Clear the frame buffer"""
        self.frame_buffer.clear()

    def stop(self):
        """Stop webcam capture"""
        print("WEBCAM: Stopping capture")
        self.is_capturing = False

        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)

        if self.cap:
            self.cap.release()

        print("WEBCAM: Capture stopped")


class RealTimePoseEstimator:
    """Real-time pose estimation using MediaPipe"""

    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required for real-time pose estimation")

        # Initialize MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )

    def process_frame(self, frame):
        """
        Process a single frame and extract pose landmarks.
        Returns: pose_data (numpy array) or None if no pose detected
        """
        if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
            return None

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.holistic.process(image_rgb)

        # Extract landmarks
        pose_landmarks = []

        # Pose (33 landmarks)
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                pose_landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            pose_landmarks = [[0, 0, 0, 0]] * 33

        # Left hand (21 landmarks)
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                pose_landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            pose_landmarks = pose_landmarks + [[0, 0, 0, 0]] * 21

        # Right hand (21 landmarks)
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                pose_landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            pose_landmarks = pose_landmarks + [[0, 0, 0, 0]] * 21

        return np.array(pose_landmarks), results

    def process_frames(self, frames):
        """
        Process multiple frames and return pose sequence.
        Returns: pose_sequence (frames, 75, 4) numpy array
        """
        pose_sequence = []

        for frame in frames:
            pose_data, _ = self.process_frame(frame)
            if pose_data is not None:
                pose_sequence.append(pose_data)

        if len(pose_sequence) > 0:
            return np.array(pose_sequence)
        return None

    def draw_landmarks(self, frame, results):
        """Draw pose landmarks on frame for visualization"""
        if not CV2_AVAILABLE:
            return frame

        annotated_frame = frame.copy()

        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS
            )

        # Draw hand landmarks
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS
            )

        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS
            )

        return annotated_frame

    def close(self):
        """Release MediaPipe resources"""
        if hasattr(self, 'holistic'):
            self.holistic.close()


class EndToEndPipeline:
    """End-to-end sign language video to sentence pipeline"""

    def __init__(
        self,
        checkpoint_path=None,
        gemini_api_key=None,
        segmentation_method="auto",
        velocity_threshold=0.02,
        min_sign_duration=10,
        use_top_k=1
    ):
        self.checkpoint_path = checkpoint_path
        self.gemini_api_key = gemini_api_key
        self.segmentation_method = segmentation_method
        self.velocity_threshold = velocity_threshold
        self.min_sign_duration = min_sign_duration
        self.use_top_k = use_top_k
        self.model = None
        self.tokenizer = None
        self.temp_dir = None

        # Initialize motion-based segmenter if needed
        if segmentation_method == "motion":
            self.motion_segmenter = MotionBasedSegmenter(
                velocity_threshold=velocity_threshold,
                min_sign_duration=min_sign_duration
            )
        else:
            self.motion_segmenter = None

        # Load checkpoint model if provided
        if checkpoint_path:
            self._load_checkpoint_model()

    def _load_checkpoint_model(self):
        """Load model from checkpoint"""
        print(f"STEP: Loading model from checkpoint: {self.checkpoint_path}")
        try:
            self.model, self.tokenizer = load_model_from_checkpoint(self.checkpoint_path)
            print("SUCCESS: Checkpoint model loaded successfully")
        except Exception as e:
            print(f"ERROR: Failed to load checkpoint: {e}")
            print("WARNING: Falling back to default model")

    def _create_temp_directory(self):
        """Create temporary working directory"""
        self.temp_dir = tempfile.mkdtemp(prefix="pipeline_")
        print(f"DEBUG: Created temporary directory: {self.temp_dir}")
        return self.temp_dir

    def _cleanup_temp_directory(self):
        """Clean up temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"DEBUG: Cleaned up temporary directory: {self.temp_dir}")

    def step1_video_to_pose(self, input_video):
        """Step 1: Convert video to pose file using video_to_pose"""
        print("\n" + "="*60)
        print("STEP 1: Converting video to pose file")
        print("="*60)

        if not os.path.exists(input_video):
            raise FileNotFoundError(f"Input video not found: {input_video}")

        # Create pose output path
        video_name = Path(input_video).stem
        pose_output = os.path.join(self.temp_dir, f"{video_name}.pose")

        print(f"INPUT: Video file: {input_video}")
        print(f"OUTPUT: Pose file: {pose_output}")

        # Run video_to_pose command - use full path to ensure correct environment
        # Get the Scripts directory from the current Python executable
        scripts_dir = os.path.join(os.path.dirname(sys.executable))
        video_to_pose_path = os.path.join(scripts_dir, "video_to_pose.exe")

        cmd = [
            video_to_pose_path,
            "-i", input_video,
            "-o", pose_output,
            "--format", "mediapipe"
        ]

        print(f"COMMAND: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("SUCCESS: Video to pose conversion completed")
            if result.stdout:
                print(f"STDOUT: {result.stdout}")

            # Verify pose file was created
            if os.path.exists(pose_output):
                file_size = os.path.getsize(pose_output)
                print(f"RESULT: Pose file created ({file_size:,} bytes)")
                return pose_output
            else:
                raise RuntimeError("Pose file was not created")

        except subprocess.CalledProcessError as e:
            print(f"ERROR: video_to_pose failed with exit code {e.returncode}")
            print(f"STDERR OUTPUT:")
            print("="*60)
            print(e.stderr)
            print("="*60)
            if e.stdout:
                print(f"STDOUT OUTPUT:")
                print("="*60)
                print(e.stdout)
                print("="*60)
            raise

    def step2_segment_pose(self, pose_file):
        """Step 2: Segment pose file using selected segmentation method"""
        print("\n" + "="*60)
        print(f"STEP 2: Segmenting pose file (method: {self.segmentation_method})")
        print("="*60)

        segments_dir = os.path.join(self.temp_dir, "segments")
        os.makedirs(segments_dir, exist_ok=True)

        print(f"INPUT: Pose file: {pose_file}")
        print(f"OUTPUT: Segments directory: {segments_dir}")

        # Use motion-based segmentation if selected
        if self.segmentation_method == "motion":
            return self._segment_pose_motion_based(pose_file, segments_dir)

        # Step 2a: Generate EAF file from pose using pose_to_segments
        eaf_file = os.path.join(self.temp_dir, "auto_segments.eaf")
        print(f"\nSTEP 2a: Auto-generating EAF file from pose data...")
        print(f"OUTPUT: {eaf_file}")

        cmd_eaf = [
            "pose_to_segments",
            "--pose", pose_file,
            "--elan", eaf_file
        ]

        print(f"COMMAND: {' '.join(cmd_eaf)}")

        try:
            result = subprocess.run(cmd_eaf, capture_output=True, text=True, check=True)
            print("SUCCESS: EAF file generated from pose data")
            if result.stdout:
                print(f"STDOUT: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: pose_to_segments failed with exit code {e.returncode}")
            print(f"STDERR: {e.stderr}")
            raise

        # Step 2b: Use the generated EAF file with pose_segment2.py
        print(f"\nSTEP 2b: Extracting pose segments using EAF annotations...")
        cmd = [
            sys.executable, "pose_segment2.py",
            eaf_file,
            pose_file,
            "--output-dir", segments_dir,
            "--intelligent",           # Use intelligent processing
            "--auto-detect",           # Auto-detect concatenated vs continuous
            "--no-videos"              # Skip visualization for speed
        ]

        print(f"COMMAND: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("SUCCESS: Pose segmentation completed")
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)

            # Find generated segment files
            segment_files = glob.glob(os.path.join(segments_dir, "*.pose"))
            segment_files.sort()

            if segment_files:
                print(f"RESULT: {len(segment_files)} segments created:")
                for i, seg_file in enumerate(segment_files):
                    file_name = os.path.basename(seg_file)
                    file_size = os.path.getsize(seg_file)
                    print(f"  {i+1:2d}. {file_name} ({file_size:,} bytes)")
                return segment_files
            else:
                raise RuntimeError("No segment files were created")

        except subprocess.CalledProcessError as e:
            print(f"ERROR: pose_segment2.py failed with exit code {e.returncode}")
            print(f"STDERR: {e.stderr}")
            raise

    def _segment_pose_motion_based(self, pose_file, segments_dir):
        """Segment pose file using motion-based velocity detection"""
        print(f"\nUsing motion-based segmentation:")
        print(f"  Velocity threshold: {self.velocity_threshold}")
        print(f"  Min sign duration: {self.min_sign_duration} frames")

        try:
            # Use motion-based segmenter
            segment_files = self.motion_segmenter.segment_pose_file(
                pose_file,
                segments_dir,
                verbose=True
            )

            if segment_files:
                print(f"\nRESULT: {len(segment_files)} segments created:")
                for i, seg_file in enumerate(segment_files):
                    file_name = os.path.basename(seg_file)
                    file_size = os.path.getsize(seg_file)
                    print(f"  {i+1:2d}. {file_name} ({file_size:,} bytes)")
                return segment_files
            else:
                raise RuntimeError("No segment files were created by motion-based segmenter")

        except Exception as e:
            print(f"ERROR: Motion-based segmentation failed: {e}")
            raise

    def step3_convert_segments_to_pickle(self, segment_files):
        """Step 3: Convert pose segments to pickle format"""
        print("\n" + "="*60)
        print("STEP 3: Converting pose segments to pickle format")
        print("="*60)

        pickle_files = []

        for i, pose_file in enumerate(segment_files):
            print(f"\nProcessing segment {i+1}/{len(segment_files)}: {os.path.basename(pose_file)}")

            try:
                # Load pose file
                with open(pose_file, "rb") as f:
                    buffer = f.read()
                    pose = Pose.read(buffer)

                # Extract pose data
                pose_data = pose.body.data
                print(f"DEBUG: Full pose data shape: {pose_data.shape}")

                if len(pose_data.shape) == 4:
                    # (frames, people, keypoints, dimensions) -> take first person
                    pose_sequence = pose_data[:, 0, :, :]
                else:
                    pose_sequence = pose_data

                print(f"DEBUG: Extracted sequence shape: {pose_sequence.shape}")

                # Extract 75-point subset (pose + hands, exclude face)
                # MediaPipe format: 33 pose + 468 face + 21 left hand + 21 right hand = 543 total
                # We need: 33 pose + 21 left hand + 21 right hand = 75 keypoints
                if pose_sequence.shape[1] == 543:
                    # Extract pose (0:33), left hand (501:522), right hand (522:543)
                    pose_75pt = np.concatenate([
                        pose_sequence[:, 0:33, :],      # Pose landmarks
                        pose_sequence[:, 501:522, :],   # Left hand landmarks
                        pose_sequence[:, 522:543, :]    # Right hand landmarks
                    ], axis=1)
                    print(f"DEBUG: Extracted 75-point subset: {pose_75pt.shape}")
                elif pose_sequence.shape[1] == 75:
                    # Already 75 keypoints
                    pose_75pt = pose_sequence
                    print(f"DEBUG: Already 75 keypoints: {pose_75pt.shape}")
                else:
                    print(f"WARNING: Unexpected keypoint count: {pose_sequence.shape[1]}, using as-is")
                    pose_75pt = pose_sequence

                # Create pickle file
                pickle_filename = os.path.basename(pose_file).replace('.pose', '.pkl')
                pickle_path = os.path.join(self.temp_dir, pickle_filename)

                pickle_data = {
                    'keypoints': pose_75pt[:, :, :2],  # x, y coordinates only
                    'confidences': pose_75pt[:, :, 3] if pose_75pt.shape[2] > 3 else np.ones(pose_75pt.shape[:2]),  # visibility scores
                    'gloss': 'UNKNOWN'  # Placeholder
                }

                with open(pickle_path, 'wb') as f:
                    pickle.dump(pickle_data, f)

                pickle_files.append(pickle_path)
                print(f"SUCCESS: Created {pickle_filename} with 75 keypoints")

            except Exception as e:
                print(f"ERROR: Failed to convert {os.path.basename(pose_file)}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\nRESULT: {len(pickle_files)} pickle files created")
        return pickle_files

    def step4_predict_glosses(self, pickle_files):
        """Step 4: Predict individual glosses using OpenHands model"""
        print("\n" + "="*60)
        print("STEP 4: Predicting individual glosses")
        print("="*60)

        predictions = []
        successful_predictions = 0

        for i, pickle_file in enumerate(pickle_files):
            print(f"\nPredicting segment {i+1}/{len(pickle_files)}: {os.path.basename(pickle_file)}")

            try:
                # Use checkpoint model if available, otherwise default model
                if self.model and self.tokenizer:
                    result = predict_pose_file(pickle_file, model=self.model, tokenizer=self.tokenizer)
                else:
                    result = predict_pose_file(pickle_file)

                if result:
                    if isinstance(result, tuple) and len(result) == 2:
                        prediction, confidence = result
                        print(f"SUCCESS: '{prediction}' (confidence: {confidence:.2%})")
                        predictions.append(prediction)
                    elif isinstance(result, dict):
                        # Handle dict response from predict_pose_file
                        prediction = result.get('gloss', '<UNKNOWN>')
                        confidence = result.get('confidence', 0.0)
                        top_k_preds = result.get('top_k_predictions', [])

                        if self.use_top_k > 1 and top_k_preds:
                            # Use top-k predictions
                            top_k = top_k_preds[:self.use_top_k]
                            print(f"SUCCESS: Top-{self.use_top_k} predictions:")
                            for j, pred in enumerate(top_k, 1):
                                print(f"  {j}. '{pred['gloss']}' (confidence: {pred['confidence']:.2%})")
                            # Store as dict with top-k info
                            predictions.append({
                                'top_prediction': prediction,
                                'confidence': confidence,
                                'top_k': top_k
                            })
                        else:
                            # Use only top-1
                            print(f"SUCCESS: '{prediction}' (confidence: {confidence:.2%})")
                            predictions.append(prediction)
                    else:
                        prediction = str(result)
                        print(f"SUCCESS: '{prediction}'")
                        predictions.append(prediction)

                    successful_predictions += 1
                else:
                    print("ERROR: No prediction returned")
                    predictions.append("<UNKNOWN>")

            except Exception as e:
                print(f"ERROR: Prediction failed: {e}")
                predictions.append("<UNKNOWN>")

        print(f"\nRESULT: {successful_predictions}/{len(pickle_files)} successful predictions")
        print("PREDICTED GLOSSES:")
        for i, pred in enumerate(predictions):
            if isinstance(pred, dict):
                # Top-k prediction
                print(f"  {i+1:2d}. SUCCESS: '{pred['top_prediction']}' (+ {len(pred['top_k'])-1} alternatives)")
            else:
                # Single prediction
                status = "SUCCESS" if pred != "<UNKNOWN>" else "ERROR"
                print(f"  {i+1:2d}. {status}: '{pred}'")

        return predictions

    def step5_construct_sentence_with_gemini(self, glosses):
        """Step 5: Use Gemini API to construct grammatically correct sentence"""
        print("\n" + "="*60)
        print("STEP 5: Constructing sentence with Gemini API")
        print("="*60)

        # Process glosses - extract strings and top-k info
        processed_glosses = []
        has_top_k = False

        for g in glosses:
            if g == "<UNKNOWN>":
                continue
            elif isinstance(g, dict):
                # Top-k prediction
                processed_glosses.append(g)
                has_top_k = True
            else:
                # Simple string
                processed_glosses.append(g)

        if not processed_glosses:
            print("ERROR: No valid glosses to construct sentence")
            return "No valid predictions available"

        # Format for display
        if has_top_k:
            display_glosses = []
            for g in processed_glosses:
                if isinstance(g, dict):
                    # Extract gloss from dict (could have 'gloss' or 'top_prediction')
                    display_glosses.append(g.get('gloss', g.get('top_prediction', 'UNKNOWN')))
                else:
                    display_glosses.append(g)
            print(f"INPUT GLOSSES: {', '.join(display_glosses)} (with top-k alternatives)")
        else:
            print(f"INPUT GLOSSES: {', '.join(processed_glosses)}")

        if not self.gemini_api_key:
            print("WARNING: No Gemini API key provided")
            print("FALLBACK: Joining glosses with spaces")
            # Extract simple strings for fallback
            fallback_glosses = []
            for g in processed_glosses:
                if isinstance(g, dict):
                    fallback_glosses.append(g.get('gloss', g.get('top_prediction', 'UNKNOWN')))
                else:
                    fallback_glosses.append(g)
            return " ".join(fallback_glosses)

        # Check if Gemini is available (dynamic import)
        global GEMINI_AVAILABLE
        if GEMINI_AVAILABLE is None:
            try:
                import importlib
                module_name = 'google.' + 'generativeai'
                genai_module = importlib.import_module(module_name)
                GEMINI_AVAILABLE = True
            except ImportError:
                GEMINI_AVAILABLE = False

        if not GEMINI_AVAILABLE:
            print("ERROR: Gemini AI library not installed")
            print("INSTALL: Run 'pip install google-generativeai' to enable Gemini API")
            print("FALLBACK: Joining glosses with spaces")
            # Extract simple strings for fallback
            fallback_glosses = []
            for g in processed_glosses:
                if isinstance(g, dict):
                    fallback_glosses.append(g.get('gloss', g.get('top_prediction', 'UNKNOWN')))
                else:
                    fallback_glosses.append(g)
            return " ".join(fallback_glosses)

        try:
            # Import and configure Gemini dynamically
            import importlib
            module_name = 'google.' + 'generativeai'
            genai = importlib.import_module(module_name)
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')

            # Create prompt based on whether we have top-k predictions
            if has_top_k:
                # Build detailed input with top-k alternatives
                gloss_details = []
                for i, g in enumerate(processed_glosses, 1):
                    if isinstance(g, dict):
                        detail = f"Position {i}:\n"
                        # Handle both 'top_k' and 'top_k_predictions' keys
                        top_k_list = g.get('top_k_predictions', g.get('top_k', []))
                        if top_k_list:
                            for j, pred in enumerate(top_k_list, 1):
                                detail += f"  Option {j}: '{pred['gloss']}' (confidence: {pred['confidence']:.1%})\n"
                        else:
                            # Fallback to main gloss
                            detail += f"  '{g.get('gloss', 'UNKNOWN')}'\n"
                        gloss_details.append(detail)
                    else:
                        gloss_details.append(f"Position {i}:\n  '{g}'")

                prompt = f"""
You are an expert in American Sign Language (ASL) and English grammar.
Given the following ASL glosses in sequential order, construct a natural, grammatically correct English sentence that conveys the intended meaning.

For each position, you are given multiple prediction options with confidence scores. Use the confidence scores and contextual understanding to select the most appropriate word for each position.

ASL Gloss Predictions:
{''.join(gloss_details)}

Instructions:
1. Consider that ASL has different grammar rules than English
2. For each position, choose the most contextually appropriate option from the given predictions
3. Use the confidence scores as a guide, but prioritize semantic and grammatical coherence
4. Add appropriate articles (a, an, the) where needed
5. Add appropriate prepositions and conjunctions
6. Ensure proper verb tenses and subject-verb agreement
7. Make the sentence sound natural and professional

Return only the constructed English sentence, nothing else.
"""
            else:
                # Simple prompt for single predictions
                prompt = f"""
You are an expert in American Sign Language (ASL) and English grammar.
Given the following ASL glosses in order, construct a natural, grammatically correct English sentence that conveys the intended meaning.

ASL Glosses: {', '.join(processed_glosses)}

Instructions:
1. Consider that ASL has different grammar rules than English
2. Add appropriate articles (a, an, the) where needed
3. Add appropriate prepositions and conjunctions
4. Ensure proper verb tenses and subject-verb agreement
5. Make the sentence sound natural and professional
6. If some glosses seem unclear, use context to infer the most likely meaning

Return only the constructed English sentence, nothing else.
"""

            print("PROMPT: Sending request to Gemini API...")
            response = model.generate_content(prompt)
            sentence = response.text.strip()

            print(f"SUCCESS: Gemini response received")
            print(f"CONSTRUCTED SENTENCE: '{sentence}'")

            return sentence

        except Exception as e:
            print(f"ERROR: Gemini API call failed: {e}")
            print("FALLBACK: Joining glosses with spaces")
            # Extract simple strings for fallback
            fallback_glosses = []
            for g in processed_glosses:
                if isinstance(g, dict):
                    fallback_glosses.append(g.get('gloss', g.get('top_prediction', 'UNKNOWN')))
                else:
                    fallback_glosses.append(g)
            return " ".join(fallback_glosses)

    def run_pipeline(self, input_video):
        """Run the complete end-to-end pipeline"""
        print("Sign Language Video to English Sentence Pipeline")
        print("="*60)
        print(f"INPUT VIDEO: {input_video}")
        if self.checkpoint_path:
            print(f"CHECKPOINT: {self.checkpoint_path}")
        if self.gemini_api_key:
            print("GEMINI API: Enabled")
        else:
            print("GEMINI API: Disabled (fallback to word joining)")
        print()

        try:
            # Create temporary working directory
            self._create_temp_directory()

            # Step 1: Video to pose
            pose_file = self.step1_video_to_pose(input_video)

            # Step 2: Pose segmentation
            segment_files = self.step2_segment_pose(pose_file)

            # Step 3: Convert to pickle
            pickle_files = self.step3_convert_segments_to_pickle(segment_files)

            # Step 4: Predict glosses
            glosses = self.step4_predict_glosses(pickle_files)

            # Step 5: Construct sentence
            final_sentence = self.step5_construct_sentence_with_gemini(glosses)

            # Final results
            print("\n" + "="*60)
            print("FINAL RESULTS")
            print("="*60)
            print(f"INPUT VIDEO: {input_video}")
            print(f"SEGMENTS PROCESSED: {len(segment_files)}")
            print(f"PREDICTED GLOSSES: {', '.join([g for g in glosses if g != '<UNKNOWN>'])}")
            print(f"FINAL SENTENCE: '{final_sentence}'")

            return final_sentence

        except Exception as e:
            print(f"\nERROR: Pipeline failed: {e}")
            raise
        finally:
            # Clean up temporary files
            self._cleanup_temp_directory()

    def run_webcam_pipeline(self, camera_index=0):
        """Run real-time webcam sign language translation"""
        print("="*70)
        print("REAL-TIME SIGN LANGUAGE TRANSLATOR")
        print("="*70)
        print("MODE: Webcam real-time translation")
        if self.checkpoint_path:
            print(f"CHECKPOINT: {self.checkpoint_path}")
        if self.gemini_api_key:
            print("GEMINI API: Enabled")
        else:
            print("GEMINI API: Disabled (fallback to word joining)")
        print()

        print("INSTRUCTIONS:")
        print("  - Position yourself in front of the camera")
        print("  - Perform each sign clearly")
        print("  - Pause 0.5-1 second between signs (hands at rest)")
        print("  - Click webcam window, then press 'q' or ESC to quit")
        print("="*70)
        print()

        # Check dependencies
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is not installed. Run: pip install opencv-python")
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is not installed. Run: pip install mediapipe")
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy is not installed. Run: pip install numpy")

        # Initialize components
        webcam = WebcamCapture(camera_index=camera_index)
        motion_detector = MotionDetector(motion_threshold=1000000, cooldown_frames=7)  # 7 frames = 0.23s at 30fps
        pose_estimator = RealTimePoseEstimator()

        # Processing queue for non-blocking pose extraction
        processing_queue = Queue()
        detected_glosses = []
        worker_stop_flag = threading.Event()

        # Start background worker thread
        worker_thread = threading.Thread(
            target=self._processing_worker,
            args=(processing_queue, detected_glosses, pose_estimator, worker_stop_flag),
            daemon=True
        )
        worker_thread.start()

        # State variables
        is_signing = False
        signing_frames = []

        try:
            # Start webcam
            webcam.start()
            time.sleep(1.0)  # Allow camera to warm up
            print("WEBCAM: Started. Press 'q' or ESC to quit")
            print()

            # Main loop - KEEP SIMPLE, NO BLOCKING
            while True:
                # Get frame
                frame = webcam.get_latest_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Detect motion
                is_active, motion_score = motion_detector.detect_motion(frame)

                # Draw pose for visualization (lightweight)
                pose_data, results = pose_estimator.process_frame(frame)
                display_frame = pose_estimator.draw_landmarks(frame, results)

                # Add status overlay
                status_color = (0, 255, 0) if is_active else (0, 0, 255)
                status_text = "SIGNING" if is_signing else "WAITING"
                cv2.putText(display_frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                cv2.putText(display_frame, f"Motion: {int(motion_score)}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Display glosses
                if detected_glosses:
                    gloss_strings = []
                    for g in detected_glosses[-5:]:
                        if isinstance(g, dict):
                            gloss_strings.append(g.get('gloss', 'UNKNOWN'))
                        else:
                            gloss_strings.append(str(g))
                    cv2.putText(display_frame, f"Glosses: {' '.join(gloss_strings)}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # State machine
                if is_active and not is_signing:
                    print("SIGN: Started")
                    is_signing = True
                    signing_frames = []

                if is_signing:
                    signing_frames.append(frame.copy())

                    # Check completion: motion stopped AND cooldown elapsed
                    if not is_active and motion_detector.is_signing_complete():
                        print(f"SIGN: Completed ({len(signing_frames)} frames)")

                        # Queue frames for background processing (NO BLOCKING!)
                        processing_queue.put(signing_frames.copy())

                        # Reset immediately - UI stays responsive
                        is_signing = False
                        signing_frames = []

                # Display frame
                cv2.imshow('Sign Language Translator', display_frame)

                # Check quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:
                    print("\nQUIT: User pressed quit key")
                    break

        except KeyboardInterrupt:
            print("\nQUIT: Interrupted by user")

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Stop worker thread
            worker_stop_flag.set()
            processing_queue.put(None)  # Signal worker to exit
            worker_thread.join(timeout=2.0)

            # Cleanup
            webcam.stop()
            pose_estimator.close()
            cv2.destroyAllWindows()
            time.sleep(0.2)  # Brief wait for cleanup

            # Final results
            if detected_glosses:
                print("\n" + "="*70)
                print("FINAL RESULTS")
                print("="*70)

                gloss_strings = []
                for g in detected_glosses:
                    if isinstance(g, dict):
                        gloss_strings.append(g.get('gloss', 'UNKNOWN'))
                    else:
                        gloss_strings.append(str(g))
                print(f"DETECTED GLOSSES: {', '.join(gloss_strings)}")

                try:
                    final_sentence = self.step5_construct_sentence_with_gemini(detected_glosses)
                    print(f"FINAL SENTENCE: '{final_sentence}'")
                except Exception as e:
                    print(f"ERROR: Gemini failed: {e}")

    def _processing_worker(self, queue, glosses_list, pose_estimator, stop_flag):
        """Background worker thread - processes frames from queue (MediaPipe safe)"""
        while not stop_flag.is_set():
            try:
                # Wait for frames from queue (with timeout to check stop flag)
                frames = queue.get(timeout=0.5)

                if frames is None:  # Exit signal
                    break

                # Extract pose (safe: only one worker thread uses MediaPipe)
                pose_sequence = pose_estimator.process_frames(frames)

                if pose_sequence is not None and len(pose_sequence) > 0:
                    # Run model prediction
                    self._predict_gloss_sync(pose_sequence, glosses_list)
                else:
                    print("WARNING: No pose extracted")

                queue.task_done()

            except:
                # Timeout - check stop flag and continue
                continue

    def _predict_gloss_sync(self, pose_sequence, glosses_list):
        """Predict gloss from pose sequence (synchronous)"""
        try:
            # Create temporary pickle file
            self._create_temp_directory()
            pickle_path = os.path.join(self.temp_dir, f"temp_sign_{len(glosses_list)}.pkl")

            pickle_data = {
                'keypoints': pose_sequence[:, :, :2],  # x, y coordinates
                'confidences': pose_sequence[:, :, 2],  # confidence scores
                'gloss': 'UNKNOWN'
            }

            with open(pickle_path, 'wb') as f:
                pickle.dump(pickle_data, f)

            # Run prediction
            if self.model and self.tokenizer:
                result = predict_pose_file(pickle_path, model=self.model, tokenizer=self.tokenizer)
            else:
                result = predict_pose_file(pickle_path)

            # Process and log result
            if result and isinstance(result, dict):
                prediction = result.get('gloss', 'UNKNOWN')
                confidence = result.get('confidence', 0.0)
                top_k_preds = result.get('top_k_predictions', [])

                print(f"PREDICTED: '{prediction}' ({confidence:.1%})")

                # Log top-k
                if top_k_preds:
                    k = min(self.use_top_k or 3, len(top_k_preds))
                    print(f"TOP-{k}:")
                    for i, pred in enumerate(top_k_preds[:k], 1):
                        print(f"  {i}. {pred.get('gloss', '?')} ({pred.get('confidence', 0):.1%})")

                # Append to glosses
                glosses_list.append(result)
            else:
                print(f"PREDICTED: {result}")
                glosses_list.append({'gloss': str(result), 'confidence': 0.0})

        except Exception as e:
            print(f"ERROR: Prediction failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="End-to-end sign language video to English sentence pipeline"
    )
    parser.add_argument(
        "input_video",
        nargs='?',
        help="Input video file path (not required if --webcam is used)"
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Use webcam for real-time sign language translation"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index for webcam mode (default: 0)"
    )
    parser.add_argument(
        "--checkpoint",
        help="Path to model checkpoint directory (e.g., './checkpoints/checkpoint_epoch_5')"
    )
    parser.add_argument(
        "--gemini-api-key",
        help="Gemini API key for sentence construction (optional, or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--output",
        help="Output file to save the final sentence (file mode only)"
    )
    parser.add_argument(
        "--segmentation-method",
        choices=["auto", "motion"],
        default="auto",
        help="Segmentation method: 'auto' (pose_to_segments), 'motion' (velocity-based) (default: auto)"
    )
    parser.add_argument(
        "--velocity-threshold",
        type=float,
        default=0.02,
        help="Velocity threshold for motion-based segmentation (default: 0.02)"
    )
    parser.add_argument(
        "--min-sign-duration",
        type=int,
        default=10,
        help="Minimum frames for valid sign in motion-based segmentation (default: 10)"
    )
    parser.add_argument(
        "--use-top-k",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Use top-k predictions from model (1=top-1 only, 3=top-3, default: 1)"
    )

    args = parser.parse_args()

    # Get Gemini API key from args or environment variable
    gemini_api_key = args.gemini_api_key or os.environ.get('GEMINI_API_KEY')

    # Validate arguments
    if args.webcam:
        # Webcam mode - no input video needed
        pass
    elif args.input_video:
        # File mode - validate input video exists
        if not os.path.exists(args.input_video):
            print(f"ERROR: Input video not found: {args.input_video}")
            return 1
    else:
        print("ERROR: Either provide an input video file or use --webcam flag")
        parser.print_help()
        return 1

    try:
        # Create pipeline
        pipeline = EndToEndPipeline(
            checkpoint_path=args.checkpoint,
            gemini_api_key=gemini_api_key,
            segmentation_method=args.segmentation_method,
            velocity_threshold=args.velocity_threshold,
            min_sign_duration=args.min_sign_duration,
            use_top_k=args.use_top_k
        )

        if args.webcam:
            # Run webcam mode
            pipeline.run_webcam_pipeline(camera_index=args.camera_index)
        else:
            # Run file mode
            result_sentence = pipeline.run_pipeline(args.input_video)

            # Save output if requested
            if args.output:
                try:
                    with open(args.output, 'w') as f:
                        f.write(f"Input Video: {args.input_video}\n")
                        f.write(f"Final Sentence: {result_sentence}\n")
                    print(f"\nSAVED: Results saved to: {args.output}")
                except Exception as e:
                    print(f"ERROR: Failed to save output: {e}")

            print(f"\nSUCCESS: Pipeline completed successfully")
            print(f"FINAL RESULT: '{result_sentence}'")

    except KeyboardInterrupt:
        print("\n\nSTOPPED: Pipeline interrupted by user")
    except Exception as e:
        print(f"\nERROR: Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())