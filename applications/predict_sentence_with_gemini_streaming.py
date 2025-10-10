# -*- coding: utf-8 -*-
"""
predict_sentence_with_gemini_streaming.py
Enhanced version of predict_sentence.py with real-time Gemini streaming for video conferences

Key Enhancements:
- Streaming Gemini responses (low latency)
- Smart buffering (pause detection, question words)
- Real-time conversation display
- Latency tracking and optimization

Usage:
    python predict_sentence_with_gemini_streaming.py --webcam --gemini-api-key YOUR_KEY [--checkpoint path]
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np

# Import base predict_sentence components
from predict_sentence import (
    WebcamCapture,
    MotionDetector,
    RealTimePoseEstimator,
    EndToEndPipeline,
    CV2_AVAILABLE,
    MEDIAPIPE_AVAILABLE,
    NUMPY_AVAILABLE
)

# Import Gemini conversation manager
from gemini_conversation_manager import GeminiConversationManager, wrap_text


class EnhancedWebcamPipeline(EndToEndPipeline):
    """
    Enhanced webcam pipeline with real-time Gemini streaming.

    Extends EndToEndPipeline with:
    - Gemini conversation manager
    - Streaming response display
    - Improved UI with word buffer and latency stats
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gemini_manager = None

    def run_webcam_pipeline_with_streaming(self, camera_index=0):
        """Run real-time webcam translation with Gemini streaming"""
        print("="*70)
        print("REAL-TIME ASL VIDEO CONFERENCING")
        print("With Gemini Streaming Integration")
        print("="*70)
        print("MODE: Webcam real-time translation")
        if self.checkpoint_path:
            print(f"CHECKPOINT: {self.checkpoint_path}")
        if self.gemini_api_key:
            print("GEMINI API: Enabled (Streaming)")
        else:
            print("GEMINI API: Disabled")
        print()

        print("INSTRUCTIONS:")
        print("  - Position yourself in front of the camera")
        print("  - Sign naturally - Gemini will respond in real-time")
        print("  - Pause briefly between signs for best results")
        print("  - Press 'q' or ESC to quit")
        print("="*70)
        print()

        # Check dependencies
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV required. Install: pip install opencv-python")
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe required. Install: pip install mediapipe")
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy required. Install: pip install numpy")

        # Initialize Gemini manager
        if self.gemini_api_key:
            self.gemini_manager = GeminiConversationManager(
                api_key=self.gemini_api_key,
                model_name='gemini-2.5-flash'  # Use same model as predict_sentence.py
            )
            self.gemini_manager.start()
            print("[Gemini] Manager started")
        else:
            print("[Warning] No Gemini API key - streaming disabled")
            self.gemini_manager = None

        # Initialize components
        webcam = WebcamCapture(camera_index=camera_index)
        motion_detector = MotionDetector(motion_threshold=1000000, cooldown_frames=10)
        pose_estimator = RealTimePoseEstimator()

        # State variables
        is_signing = False
        signing_frames = []
        detected_glosses = []

        # Gemini display state
        current_gemini_text = ""
        gemini_complete = True

        # Prediction stability (prevent jitter)
        last_prediction = None
        prediction_stable_count = 0
        STABILITY_THRESHOLD = 3  # Require 3 consecutive frames

        try:
            # Start webcam
            webcam.start()
            time.sleep(1.0)
            print("WEBCAM: Started. Press 'q' or ESC to quit\n")

            while True:
                # Get latest frame
                frame = webcam.get_latest_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Detect motion
                is_active, motion_score = motion_detector.detect_motion(frame)

                # Process frame for pose
                pose_data, results = pose_estimator.process_frame(frame)
                display_frame = pose_estimator.draw_landmarks(frame, results)

                # === Sign Detection State Machine ===
                if is_active and not is_signing:
                    print("SIGN: Started")
                    is_signing = True
                    signing_frames = []
                    last_prediction = None
                    prediction_stable_count = 0

                if is_signing:
                    signing_frames.append(frame.copy())

                    # Check completion
                    if not is_active and motion_detector.is_signing_complete():
                        print(f"SIGN: Completed ({len(signing_frames)} frames)")

                        # Process sign (non-blocking)
                        self._process_sign_frames_async(
                            signing_frames,
                            detected_glosses,
                            pose_estimator
                        )

                        is_signing = False
                        signing_frames = []

                # === Gemini Integration ===
                if self.gemini_manager and detected_glosses:
                    # Get latest prediction
                    latest_gloss = detected_glosses[-1]
                    if isinstance(latest_gloss, dict):
                        predicted_word = latest_gloss.get('gloss', 'UNKNOWN')
                        confidence = latest_gloss.get('confidence', 0.0)
                    else:
                        predicted_word = str(latest_gloss)
                        confidence = 0.0

                    # Stability check (prevent sending jittery predictions)
                    if predicted_word == last_prediction:
                        prediction_stable_count += 1
                    else:
                        prediction_stable_count = 0
                        last_prediction = predicted_word

                    # Send to Gemini when stable
                    if prediction_stable_count == STABILITY_THRESHOLD:
                        self.gemini_manager.add_signed_word(predicted_word, confidence)
                        print(f"[Sign → Gemini] '{predicted_word}' ({confidence:.2f})")
                        prediction_stable_count = 0  # Reset

                # Get Gemini response (non-blocking)
                if self.gemini_manager:
                    gemini_update, is_complete = self.gemini_manager.get_display_text()
                    if gemini_update is not None:
                        current_gemini_text = gemini_update
                        gemini_complete = is_complete

                # === UI Rendering ===
                self._render_ui(
                    display_frame,
                    is_signing,
                    is_active,
                    motion_score,
                    detected_glosses,
                    current_gemini_text,
                    gemini_complete
                )

                # Display
                cv2.imshow('ASL Video Conference', display_frame)

                # Check quit
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
            # Cleanup
            if self.gemini_manager:
                self.gemini_manager.stop()

            webcam.stop()
            pose_estimator.close()
            cv2.destroyAllWindows()
            time.sleep(0.2)

            # Final results
            if detected_glosses:
                print("\n" + "="*70)
                print("FINAL RESULTS")
                print("="*70)

                gloss_strings = [
                    g.get('gloss', 'UNKNOWN') if isinstance(g, dict) else str(g)
                    for g in detected_glosses
                ]
                print(f"DETECTED GLOSSES: {', '.join(gloss_strings)}")

                if current_gemini_text:
                    print(f"FINAL GEMINI RESPONSE: '{current_gemini_text}'")

    def _process_sign_frames_async(self, frames, glosses_list, pose_estimator):
        """Process sign frames asynchronously (lightweight)"""
        try:
            # Extract pose sequence
            pose_sequence = pose_estimator.process_frames(frames)

            if pose_sequence is not None and len(pose_sequence) > 0:
                # Predict (sync for now - could be made async if needed)
                self._predict_gloss_sync(pose_sequence, glosses_list)
            else:
                print("WARNING: No pose extracted from sign")

        except Exception as e:
            print(f"ERROR: Sign processing failed: {e}")

    def _render_ui(self, frame, is_signing, is_active, motion_score, glosses, gemini_text, gemini_complete):
        """Render enhanced UI with Gemini streaming"""
        h, w = frame.shape[:2]

        # === Top Section: Status & Stats ===
        y_offset = 30

        # Signing status
        status_color = (0, 255, 0) if is_signing else (0, 0, 255)
        status_text = "SIGNING" if is_signing else "WAITING"
        cv2.putText(frame, status_text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        y_offset += 30

        # Motion score
        cv2.putText(frame, f"Motion: {int(motion_score)}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25

        # === Middle Section: Detected Glosses ===
        if glosses:
            recent_glosses = glosses[-5:]  # Last 5
            gloss_strings = []
            for g in recent_glosses:
                if isinstance(g, dict):
                    word = g.get('gloss', 'UNKNOWN')
                    conf = g.get('confidence', 0.0)
                    gloss_strings.append(f"{word}({conf:.0%})")
                else:
                    gloss_strings.append(str(g))

            cv2.putText(frame, f"Signed: {' '.join(gloss_strings)}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 30

        # === Word Buffer Preview (if Gemini enabled) ===
        if self.gemini_manager:
            buffer_preview = self.gemini_manager.get_buffer_preview(max_words=5)
            if buffer_preview:
                cv2.putText(frame, f"Buffer: {buffer_preview}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 25

        # === Bottom Section: Gemini Response ===
        if self.gemini_manager:
            gemini_y = h - 150

            # Streaming indicator
            is_streaming = self.gemini_manager.is_streaming
            status_symbol = "●" if is_streaming else "○"
            color = (0, 255, 255) if is_streaming else (0, 255, 0)

            cv2.putText(frame, f"{status_symbol} Gemini:", (10, gemini_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Response text (word-wrapped)
            if gemini_text:
                wrapped_lines = wrap_text(gemini_text, max_chars=55)
                for i, line in enumerate(wrapped_lines[:4]):  # Max 4 lines
                    cv2.putText(frame, line, (10, gemini_y + 30 + (i * 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Latency stats (top-right corner)
            avg_latency = self.gemini_manager.get_avg_latency()
            if avg_latency > 0:
                cv2.putText(frame, f"Latency: {avg_latency:.2f}s",
                            (w - 180, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Real-time ASL video conferencing with Gemini streaming"
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        default=True,
        help="Use webcam for real-time translation (default)"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--checkpoint",
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--gemini-api-key",
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )

    args = parser.parse_args()

    # Get Gemini API key
    gemini_api_key = args.gemini_api_key or os.environ.get('GEMINI_API_KEY')

    if not gemini_api_key:
        print("WARNING: No Gemini API key provided")
        print("Set GEMINI_API_KEY environment variable or use --gemini-api-key flag")
        print("Proceeding without Gemini integration...\n")

    try:
        # Create enhanced pipeline
        pipeline = EnhancedWebcamPipeline(
            checkpoint_path=args.checkpoint,
            gemini_api_key=gemini_api_key
        )

        # Run enhanced webcam pipeline
        pipeline.run_webcam_pipeline_with_streaming(camera_index=args.camera_index)

    except KeyboardInterrupt:
        print("\n\nSTOPPED: Interrupted by user")
    except Exception as e:
        print(f"\nERROR: Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
