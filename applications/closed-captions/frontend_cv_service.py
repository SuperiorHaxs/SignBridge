"""
Frontend Computer Vision Service (Port 3000)
Handles webcam capture, pose estimation, sign detection, and display
"""

import os
import sys
import time
import threading
from queue import Queue
from pathlib import Path
from collections import deque
import pickle
import tempfile
import requests
import json

from flask import Flask, request, jsonify

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "models" / "openhands-modernized" / "src"))

# Import OpenCV and dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("ERROR: OpenCV not available")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("ERROR: NumPy not available")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("ERROR: MediaPipe not available")

try:
    from openhands_modernized import predict_pose_file, load_model_from_checkpoint
except ImportError as e:
    print(f"ERROR: Failed to import OpenHands model: {e}")
    sys.exit(1)

# Configuration
BACKEND_URL = "http://localhost:4000"
MOTION_THRESHOLD = 2000000
COOLDOWN_FRAMES = 12
MIN_SIGN_FRAMES = 15
SIGN_DEBOUNCE_TIME = 0.5
ENABLE_AUTO_TRANSLATE = True
CAPTION_BUFFER_SIZE = 3
MIN_PREDICTION_CONFIDENCE = 0.0

# File-based communication (non-blocking)
GLOSS_FILE = "detected_glosses.txt"
SENTENCE_FILE = "translated_sentence.txt"

# Flask app for receiving sentence updates
app = Flask(__name__)
app_ready = threading.Event()

# Shared state
current_sentence = ""
current_sentence_lock = threading.Lock()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'frontend_cv_service'}), 200

@app.route('/sentence_update', methods=['POST'])
def sentence_update():
    """Receive sentence updates from backend"""
    global current_sentence
    try:
        data = request.get_json()
        sentence = data.get('sentence', '')

        with current_sentence_lock:
            current_sentence = sentence

        print(f"\n[FRONTEND] Received sentence from backend: '{sentence}'")
        return jsonify({'success': True}), 200
    except Exception as e:
        print(f"[FRONTEND] Error receiving sentence: {e}")
        return jsonify({'error': str(e)}), 500


class MotionDetector:
    """Detect signing activity using motion detection"""

    def __init__(self, motion_threshold=500, cooldown_frames=10):
        self.motion_threshold = motion_threshold
        self.cooldown_frames = cooldown_frames
        self.prev_frame = None
        self.frames_since_motion = 0

    def detect_motion(self, frame):
        """Detect if there's significant motion in the frame"""
        if not NUMPY_AVAILABLE or not CV2_AVAILABLE:
            return True, 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False, 0

        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        motion_score = np.sum(thresh)
        self.prev_frame = gray

        is_active = motion_score > self.motion_threshold

        if is_active:
            self.frames_since_motion = 0
        else:
            self.frames_since_motion += 1

        return is_active, motion_score

    def is_signing_complete(self):
        """Check if signing has stopped"""
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
        print(f"[CV] Starting webcam capture from camera {self.camera_index}")

        # Try different camera backends
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

        for backend in backends:
            print(f"[CV] Trying backend: {backend}")
            self.cap = cv2.VideoCapture(self.camera_index, backend)

            if self.cap.isOpened():
                # Test read a frame
                ret, test_frame = self.cap.read()
                if ret:
                    print(f"[CV] Successfully opened camera with backend {backend}")
                    break
                else:
                    print(f"[CV] Camera opened but can't read frames with backend {backend}")
                    self.cap.release()
            else:
                print(f"[CV] Failed to open camera with backend {backend}")

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index} with any backend. Try a different --camera-index")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        print("[CV] Webcam capture started successfully")

    def _capture_loop(self):
        """Continuous capture loop"""
        error_count = 0
        while self.is_capturing:
            ret, frame = self.cap.read()
            if ret:
                self.frame_buffer.append(frame.copy())
                error_count = 0  # Reset error count on success
            else:
                error_count += 1
                if error_count % 30 == 1:  # Print every ~1 second (30 frames)
                    print(f"[CV] WARNING: Failed to read frame (error count: {error_count})")
                time.sleep(0.1)

    def get_latest_frame(self):
        """Get the most recent frame"""
        if len(self.frame_buffer) > 0:
            return self.frame_buffer[-1]
        return None

    def stop(self):
        """Stop webcam capture"""
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()


class RealTimePoseEstimator:
    """Real-time pose estimation using MediaPipe"""

    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required")

        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )

    def process_frame(self, frame):
        """Process a single frame and extract pose landmarks"""
        if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
            return None

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)

        pose_landmarks = []

        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                pose_landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            pose_landmarks = [[0, 0, 0, 0]] * 33

        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                pose_landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            pose_landmarks = pose_landmarks + [[0, 0, 0, 0]] * 21

        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                pose_landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            pose_landmarks = pose_landmarks + [[0, 0, 0, 0]] * 21

        return np.array(pose_landmarks), results

    def process_frames(self, frames):
        """Process multiple frames and return pose sequence"""
        pose_sequence = []

        for frame in frames:
            pose_data, _ = self.process_frame(frame)
            if pose_data is not None:
                pose_sequence.append(pose_data)

        if len(pose_sequence) > 0:
            return np.array(pose_sequence)
        return None

    def draw_landmarks(self, frame, results):
        """Draw pose landmarks on frame"""
        if not CV2_AVAILABLE:
            return frame

        annotated_frame = frame.copy()

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS
            )

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


class SignLanguageApp:
    """Main application for sign language detection"""

    def __init__(self, checkpoint_path=None, camera_index=0):
        self.checkpoint_path = checkpoint_path
        self.camera_index = camera_index
        self.model = None
        self.tokenizer = None
        self.temp_dir = tempfile.mkdtemp(prefix="cv_service_")
        self.gloss_file_lock = threading.Lock()

        if checkpoint_path:
            self._load_checkpoint_model()

        # Initialize gloss file
        self._init_gloss_file()

    def _init_gloss_file(self):
        """Initialize the gloss file"""
        # Clear both gloss and sentence files for a fresh start
        with open(GLOSS_FILE, 'w') as f:
            f.write("")  # Empty file
        with open(SENTENCE_FILE, 'w') as f:
            f.write("")  # Empty file
        print(f"[CV] Initialized gloss file: {GLOSS_FILE}")
        print(f"[CV] Cleared sentence file: {SENTENCE_FILE}")

    def _load_checkpoint_model(self):
        """Load model from checkpoint"""
        print(f"[CV] Loading model from: {self.checkpoint_path}")
        try:
            self.model, self.tokenizer = load_model_from_checkpoint(self.checkpoint_path)
            print("[CV] Model loaded successfully")
        except Exception as e:
            print(f"[CV] Failed to load model: {e}")

    def _write_gloss_to_file(self, gloss_data):
        """Write a single gloss to file asynchronously (non-blocking)"""
        def write_async():
            try:
                with self.gloss_file_lock:
                    with open(GLOSS_FILE, 'a') as f:
                        f.write(json.dumps(gloss_data) + '\n')
                print(f"[CV] ✓ Wrote gloss to file: {gloss_data.get('gloss', 'UNKNOWN')}")
            except Exception as e:
                print(f"[CV] Error writing gloss to file: {e}")

        # Run in background thread - never blocks camera
        threading.Thread(target=write_async, daemon=True).start()

    def _predict_gloss_sync(self, pose_sequence, glosses_list):
        """Predict gloss from pose sequence"""
        try:
            pickle_path = os.path.join(self.temp_dir, f"temp_sign_{len(glosses_list)}.pkl")

            pickle_data = {
                'keypoints': pose_sequence[:, :, :2],
                'confidences': pose_sequence[:, :, 2],
                'gloss': 'UNKNOWN'
            }

            with open(pickle_path, 'wb') as f:
                pickle.dump(pickle_data, f)

            if self.model and self.tokenizer:
                result = predict_pose_file(pickle_path, model=self.model, tokenizer=self.tokenizer)
            elif self.checkpoint_path:
                result = predict_pose_file(pickle_path, checkpoint_path=self.checkpoint_path)
            else:
                raise ValueError("Either model or checkpoint_path must be provided")

            if result and isinstance(result, dict):
                prediction = result.get('gloss', 'UNKNOWN')
                confidence = result.get('confidence', 0.0)
                top_k_preds = result.get('top_k_predictions', [])

                print(f"[CV] PREDICTED: '{prediction}' ({confidence:.1%})")

                if top_k_preds:
                    print(f"[CV] TOP-3:")
                    for i, pred in enumerate(top_k_preds[:3], 1):
                        print(f"  {i}. {pred.get('gloss', '?')} ({pred.get('confidence', 0):.1%})")

                if confidence >= MIN_PREDICTION_CONFIDENCE:
                    # Write to file asynchronously - NEVER blocks camera
                    self._write_gloss_to_file(result)
                    glosses_list.append(result)  # Also keep in memory for display
                else:
                    print(f"[CV] Prediction rejected - low confidence")
            else:
                print(f"[CV] PREDICTED: {result}")
                gloss_dict = {'gloss': str(result), 'confidence': 0.0}
                self._write_gloss_to_file(gloss_dict)
                glosses_list.append(gloss_dict)

        except Exception as e:
            print(f"[CV] Prediction error: {e}")

    def _processing_worker(self, queue, glosses_list, pose_estimator, stop_flag, processing_event):
        """Background worker for processing signs"""
        while not stop_flag.is_set():
            try:
                frames = queue.get(timeout=0.5)

                if frames is None:
                    break

                pose_sequence = pose_estimator.process_frames(frames)

                if pose_sequence is not None and len(pose_sequence) > 0:
                    self._predict_gloss_sync(pose_sequence, glosses_list)
                else:
                    print("[CV] No pose extracted")

                if processing_event is not None:
                    processing_event.clear()

                queue.task_done()

            except:
                if processing_event is not None:
                    processing_event.clear()
                continue

    def _read_sentence_from_file(self):
        """Read translated sentence from file (non-blocking)"""
        try:
            if os.path.exists(SENTENCE_FILE):
                with open(SENTENCE_FILE, 'r') as f:
                    content = f.read().strip()
                    if content:
                        return content
        except Exception as e:
            pass
        return None

    def run(self):
        """Run the computer vision application"""
        global current_sentence

        print("="*70)
        print("FRONTEND COMPUTER VISION SERVICE (Port 3000)")
        print("="*70)
        print(f"Model: {'Loaded' if self.model else 'Default'}")
        print(f"Backend URL: {BACKEND_URL}")
        print()
        print("INSTRUCTIONS:")
        print("  - Perform signs clearly in front of camera")
        print("  - Pause 0.5-1s between signs")
        if ENABLE_AUTO_TRANSLATE and CAPTION_BUFFER_SIZE > 0:
            print(f"  - AUTO-TRANSLATE: Every {CAPTION_BUFFER_SIZE} words")
        print("  - Press SPACEBAR to force translate")
        print("  - Press 'q' or ESC to quit")
        print("="*70)

        # Initialize components
        webcam = WebcamCapture(camera_index=self.camera_index)
        motion_detector = MotionDetector(
            motion_threshold=MOTION_THRESHOLD,
            cooldown_frames=COOLDOWN_FRAMES
        )
        pose_estimator = RealTimePoseEstimator()

        processing_queue = Queue()
        detected_glosses = []
        worker_stop_flag = threading.Event()
        processing_event = threading.Event()
        last_sign_time = 0
        is_signing = False
        signing_frames = []
        is_translating = False

        # Start background worker
        worker_thread = threading.Thread(
            target=self._processing_worker,
            args=(processing_queue, detected_glosses, pose_estimator, worker_stop_flag, processing_event),
            daemon=True
        )
        worker_thread.start()

        try:
            webcam.start()
            time.sleep(1.0)
            print("[CV] Ready. Press 'q' to quit\n")

            while True:
                frame = webcam.get_latest_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                is_active, motion_score = motion_detector.detect_motion(frame)
                pose_data, results = pose_estimator.process_frame(frame)
                display_frame = pose_estimator.draw_landmarks(frame, results)

                h, w = display_frame.shape[:2]

                # Status indicator
                if processing_event.is_set():
                    status_text = "PROCESSING..."
                    cv2.circle(display_frame, (w - 50, 30), 15, (0, 165, 255), -1)
                elif is_signing:
                    status_text = "SIGNING"
                    cv2.circle(display_frame, (w - 50, 30), 15, (0, 255, 0), -1)
                else:
                    status_text = "READY"
                    cv2.circle(display_frame, (w - 50, 30), 15, (0, 255, 0), -1)

                cv2.putText(display_frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Motion: {int(motion_score)}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Display sentence from backend
                with current_sentence_lock:
                    sentence_to_display = current_sentence

                if is_translating:
                    cv2.putText(display_frame, "Translating...", (10, h - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                elif sentence_to_display:
                    # Word wrap
                    max_chars = 80
                    words = sentence_to_display.split()
                    lines = []
                    current_line = ""

                    for word in words:
                        test_line = current_line + (" " if current_line else "") + word
                        if len(test_line) <= max_chars:
                            current_line = test_line
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = word

                    if current_line:
                        lines.append(current_line)

                    # Display lines
                    y_start = h - 50 - (len(lines) - 1) * 30
                    for i, line in enumerate(lines[-3:]):  # Show last 3 lines
                        cv2.putText(display_frame, line, (10, y_start + (i * 30)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Check for updated sentence from file (non-blocking)
                # Backend watches the gloss file and writes to sentence file
                new_sentence = self._read_sentence_from_file()
                if new_sentence and new_sentence != current_sentence:
                    with current_sentence_lock:
                        current_sentence = new_sentence
                    print(f"\n[CV] ✓ New sentence received: '{new_sentence}'")

                # Sign detection state machine
                can_start_signing = not is_signing and not processing_event.is_set()

                if is_active and can_start_signing:
                    print("[CV] Sign started")
                    is_signing = True
                    signing_frames = []

                if is_signing:
                    signing_frames.append(frame.copy())

                    if not is_active and motion_detector.is_signing_complete():
                        print(f"[CV] Sign completed ({len(signing_frames)} frames)")

                        if len(signing_frames) < MIN_SIGN_FRAMES:
                            print(f"[CV] Sign rejected - too short")
                            is_signing = False
                            signing_frames = []
                        elif SIGN_DEBOUNCE_TIME > 0 and (time.time() - last_sign_time) < SIGN_DEBOUNCE_TIME:
                            print(f"[CV] Sign rejected - too soon")
                            is_signing = False
                            signing_frames = []
                        else:
                            processing_queue.put(signing_frames.copy())
                            last_sign_time = time.time()
                            processing_event.set()
                            is_signing = False
                            signing_frames = []

                cv2.imshow('Sign Language Translator - Frontend', display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:
                    print("\n[CV] Quit signal received")
                    break
                # No manual translation needed - backend watches file automatically

        except KeyboardInterrupt:
            print("\n[CV] Interrupted")
        except Exception as e:
            print(f"\n[CV] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            worker_stop_flag.set()
            processing_queue.put(None)
            worker_thread.join(timeout=2.0)
            webcam.stop()
            pose_estimator.close()
            cv2.destroyAllWindows()

            print("\n[CV] Cleanup complete")


def run_flask_server():
    """Run Flask server in background thread"""
    app.run(host='0.0.0.0', port=3000, debug=False, use_reloader=False, threaded=True)


if __name__ == '__main__':
    import argparse

    # Default model path
    DEFAULT_CHECKPOINT = str(Path(__file__).parent.parent / "models" / "training-scripts" / "models" / "wlasl_100_class_model")

    parser = argparse.ArgumentParser(description="Frontend Computer Vision Service")
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT, help='Path to model checkpoint')
    parser.add_argument('--camera-index', type=int, default=0, help='Camera index')
    args = parser.parse_args()

    # Start Flask server in background
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()
    time.sleep(2)  # Let Flask start

    # Run CV application
    app_instance = SignLanguageApp(
        checkpoint_path=args.checkpoint,
        camera_index=args.camera_index
    )
    app_instance.run()
