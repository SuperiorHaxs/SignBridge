#!/usr/bin/env python3
"""
camera_processor.py

Common camera and pose processing module for the Show-and-Tell application.
Used by both Live Mode and Closed Captions Mode.

This module provides:
- Video to pose conversion using pose-format library (VIDEO_TO_POSE_EXE)
- Pose to pickle conversion for model inference
- Model prediction wrapper
- Temporary file management
"""

import os
import sys
import shutil
import subprocess
import tempfile
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# Pose format library
from pose_format import Pose

# Get paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Find video_to_pose executable (cross-platform)
def _find_video_to_pose():
    """Locate video_to_pose executable on Windows or Linux."""
    # Check PATH first (works on Linux/Docker where pip installs to /usr/local/bin)
    found = shutil.which("video_to_pose")
    if found:
        return found
    # Windows venv fallback
    win_exe = PROJECT_ROOT / "venv" / "Scripts" / "video_to_pose.exe"
    if win_exe.exists():
        return str(win_exe)
    # Linux venv fallback
    linux_exe = PROJECT_ROOT / "venv" / "bin" / "video_to_pose"
    if linux_exe.exists():
        return str(linux_exe)
    # Last resort: hope it's on PATH at runtime
    return "video_to_pose"

VIDEO_TO_POSE_EXE = _find_video_to_pose()


class CameraProcessor:
    """
    Unified camera/pose processing for ASL sign recognition.

    Uses VIDEO_TO_POSE_EXE for pose extraction (same as Live Mode)
    which provides better hand detection than in-app MediaPipe.
    """

    def __init__(self, model=None, tokenizer=None, temp_dir: Optional[Path] = None):
        """
        Initialize the camera processor.

        Args:
            model: Pre-loaded OpenHands model (optional, can set later)
            tokenizer: Model's id_to_gloss mapping (optional, can set later)
            temp_dir: Directory for temporary files (optional, uses system temp)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "asl_camera_processor"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Import prediction function
        from openhands_modernized_inference import predict_pose_file
        self._predict_pose_file = predict_pose_file

    def set_model(self, model, tokenizer):
        """Set or update the model and tokenizer."""
        self.model = model
        self.tokenizer = tokenizer

    def video_bytes_to_pose(self, video_bytes: bytes, video_format: str = 'webm') -> Optional[Path]:
        """
        Convert video bytes to .pose file using VIDEO_TO_POSE_EXE.

        This is the same method used by Live Mode, which provides
        better hand detection than in-app MediaPipe.

        Args:
            video_bytes: Raw video bytes
            video_format: Video format extension (webm, mp4, etc.)

        Returns:
            Path to the generated .pose file, or None on failure
        """
        import time
        timestamp = int(time.time() * 1000)

        # Save video to temp file
        video_path = self.temp_dir / f"video_{timestamp}.{video_format}"
        with open(video_path, 'wb') as f:
            f.write(video_bytes)

        # Convert WebM to MP4 if needed (OpenCV compatibility)
        if video_format.lower() == 'webm':
            mp4_path = self.temp_dir / f"video_{timestamp}.mp4"
            if self._convert_webm_to_mp4(str(video_path), str(mp4_path)):
                video_path.unlink()  # Remove webm
                video_path = mp4_path

        # Run video_to_pose
        pose_path = self.temp_dir / f"pose_{timestamp}.pose"

        try:
            cmd = [
                str(VIDEO_TO_POSE_EXE),
                "-i", str(video_path),
                "-o", str(pose_path),
                "--format", "mediapipe"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                print(f"[CameraProcessor] video_to_pose failed: {result.stderr}")
                return None

            if not pose_path.exists():
                print(f"[CameraProcessor] Pose file not created")
                return None

            print(f"[CameraProcessor] Created pose file: {pose_path}")
            return pose_path

        except subprocess.TimeoutExpired:
            print(f"[CameraProcessor] video_to_pose timeout")
            return None
        except Exception as e:
            print(f"[CameraProcessor] Error: {e}")
            return None
        finally:
            # Cleanup video file
            try:
                if video_path.exists():
                    video_path.unlink()
            except:
                pass

    def video_file_to_pose(self, video_path: str) -> Optional[Path]:
        """
        Convert video file to .pose file using VIDEO_TO_POSE_EXE.

        Args:
            video_path: Path to video file

        Returns:
            Path to the generated .pose file, or None on failure
        """
        import time
        timestamp = int(time.time() * 1000)

        video_path = Path(video_path)

        # Convert WebM to MP4 if needed
        if video_path.suffix.lower() == '.webm':
            mp4_path = self.temp_dir / f"video_{timestamp}.mp4"
            if self._convert_webm_to_mp4(str(video_path), str(mp4_path)):
                video_path = mp4_path

        # Run video_to_pose
        pose_path = self.temp_dir / f"pose_{timestamp}.pose"

        try:
            cmd = [
                str(VIDEO_TO_POSE_EXE),
                "-i", str(video_path),
                "-o", str(pose_path),
                "--format", "mediapipe"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                print(f"[CameraProcessor] video_to_pose failed: {result.stderr}")
                return None

            print(f"[CameraProcessor] Created pose file: {pose_path}")
            return pose_path

        except Exception as e:
            print(f"[CameraProcessor] Error: {e}")
            return None

    def pose_to_pickle(self, pose_path: str, output_path: Optional[str] = None) -> Optional[Path]:
        """
        Convert .pose file to pickle format for model inference.

        Extracts 83 keypoints (33 body + 21 left hand + 21 right hand + 8 face)
        from the 543-point MediaPipe format for the 43-class model.

        Args:
            pose_path: Path to .pose file
            output_path: Optional output pickle path (auto-generated if not provided)

        Returns:
            Path to the generated pickle file, or None on failure
        """
        # 8 minimal face landmark indices from MediaPipe Face Mesh (468 total)
        # These are: nose tip, mouth corners, chin, eyebrows, eye corners
        # In 543-format, face landmarks start at index 33
        FACE_MESH_INDICES = [1, 61, 291, 152, 107, 336, 33, 263]
        FACE_543_INDICES = [33 + i for i in FACE_MESH_INDICES]  # [34, 94, 324, 185, 140, 369, 66, 296]

        try:
            with open(pose_path, "rb") as f:
                buffer = f.read()
                pose = Pose.read(buffer)

            pose_data = pose.body.data

            # Remove person dimension if present
            if len(pose_data.shape) == 4:
                pose_sequence = pose_data[:, 0, :, :]
            else:
                pose_sequence = pose_data

            # Extract 83-point subset from 543-point format
            # 83 = 33 body + 21 left hand + 21 right hand + 8 face
            if pose_sequence.shape[1] == 543:
                pose_83pt = np.concatenate([
                    pose_sequence[:, 0:33, :],           # Pose landmarks (33)
                    pose_sequence[:, 501:522, :],        # Left hand landmarks (21)
                    pose_sequence[:, 522:543, :],        # Right hand landmarks (21)
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

            # Create pickle data
            pickle_data = {
                'keypoints': pose_83pt[:, :, :3],  # x, y, z
                'confidences': pose_83pt[:, :, 3] if pose_83pt.shape[2] > 3 else np.ones(pose_83pt.shape[:2]),
                'gloss': 'UNKNOWN'
            }

            # Determine output path
            if output_path:
                pickle_path = Path(output_path)
            else:
                pickle_path = Path(str(pose_path).replace('.pose', '.pkl'))

            with open(pickle_path, 'wb') as f:
                pickle.dump(pickle_data, f)

            print(f"[CameraProcessor] Created pickle: {pickle_path} (shape: {pickle_data['keypoints'].shape})")
            return pickle_path

        except Exception as e:
            print(f"[CameraProcessor] Error converting pose to pickle: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict(self, pickle_path: str) -> Optional[Dict[str, Any]]:
        """
        Run model prediction on a pickle file.

        Args:
            pickle_path: Path to pickle file

        Returns:
            Dict with 'gloss', 'confidence', 'top_k_predictions', or None on failure
        """
        if self.model is None or self.tokenizer is None:
            print("[CameraProcessor] Model not loaded")
            return None

        try:
            result = self._predict_pose_file(
                str(pickle_path),
                model=self.model,
                tokenizer=self.tokenizer
            )
            return result
        except Exception as e:
            print(f"[CameraProcessor] Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_video_bytes(self, video_bytes: bytes, video_format: str = 'webm') -> Optional[Dict[str, Any]]:
        """
        Full pipeline: video bytes -> pose -> pickle -> prediction.

        This is the main method for processing a single sign video.

        Args:
            video_bytes: Raw video bytes
            video_format: Video format extension

        Returns:
            Prediction dict or None on failure
        """
        # Step 1: Video to pose
        pose_path = self.video_bytes_to_pose(video_bytes, video_format)
        if not pose_path:
            return None

        # Step 2: Pose to pickle
        pickle_path = self.pose_to_pickle(str(pose_path))
        if not pickle_path:
            self._cleanup_file(pose_path)
            return None

        # Step 3: Predict
        result = self.predict(str(pickle_path))

        # Cleanup temp files
        self._cleanup_file(pose_path)
        self._cleanup_file(pickle_path)

        return result

    def process_video_file(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Full pipeline: video file -> pose -> pickle -> prediction.

        Args:
            video_path: Path to video file

        Returns:
            Prediction dict or None on failure
        """
        # Step 1: Video to pose
        pose_path = self.video_file_to_pose(video_path)
        if not pose_path:
            return None

        # Step 2: Pose to pickle
        pickle_path = self.pose_to_pickle(str(pose_path))
        if not pickle_path:
            self._cleanup_file(pose_path)
            return None

        # Step 3: Predict
        result = self.predict(str(pickle_path))

        # Cleanup temp files
        self._cleanup_file(pose_path)
        self._cleanup_file(pickle_path)

        return result

    def _convert_webm_to_mp4(self, webm_path: str, mp4_path: str) -> bool:
        """Convert WebM to MP4 using ffmpeg."""
        try:
            cmd = [
                'ffmpeg', '-y', '-i', webm_path,
                '-c:v', 'libx264', '-preset', 'ultrafast',
                '-c:a', 'aac', mp4_path
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            return result.returncode == 0 and Path(mp4_path).exists()
        except:
            return False

    def _cleanup_file(self, file_path: Path):
        """Safely remove a temporary file."""
        try:
            if file_path and file_path.exists():
                file_path.unlink()
        except:
            pass

    def cleanup_temp_dir(self):
        """Remove all files in the temp directory."""
        try:
            for f in self.temp_dir.glob("*"):
                f.unlink()
        except:
            pass


# Singleton instance for shared use
_camera_processor_instance = None

def get_camera_processor(model=None, tokenizer=None) -> CameraProcessor:
    """Get or create the shared CameraProcessor instance."""
    global _camera_processor_instance
    if _camera_processor_instance is None:
        _camera_processor_instance = CameraProcessor()
    if model is not None and tokenizer is not None:
        _camera_processor_instance.set_model(model, tokenizer)
    return _camera_processor_instance
