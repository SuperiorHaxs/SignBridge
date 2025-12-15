#!/usr/bin/env python3
"""
hybrid_segmenter.py
Hybrid Video-Pose Segmentation

Combines pixel-based motion detection (robust) with pose slicing (efficient):
1. Analyze video frames to detect motion/rest transitions
2. Get segment timestamps (start/end frames)
3. Slice the pre-converted pose file at those timestamps

This avoids the issues with pose-keypoint-velocity-based segmentation
while keeping pose extraction as a single pass.
"""

import os
import copy
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("WARNING: OpenCV not available for hybrid segmentation")

try:
    from pose_format import Pose
    POSE_FORMAT_AVAILABLE = True
except ImportError:
    POSE_FORMAT_AVAILABLE = False
    print("WARNING: pose_format not available")


class HybridSegmenter:
    """
    Segment videos using pixel-based motion detection, then slice pose files.

    Adapted from frontend_cv_service.py MotionDetector but designed for
    post-processing rather than real-time use.
    """

    def __init__(
        self,
        motion_threshold: int = 500000,
        cooldown_frames: int = 45,
        min_sign_frames: int = 12,
        max_sign_frames: int = 150,
        padding_before: int = 3,
        padding_after: int = 3,
        blur_size: int = 21,
        diff_threshold: int = 25
    ):
        """
        Initialize hybrid segmenter.

        Args:
            motion_threshold: Pixel sum threshold for "active" motion detection.
                             Higher = less sensitive. Default 500000 works for
                             most webcam videos (640x480).
            cooldown_frames: Frames of no motion before sign is considered complete.
                             Default 45 = 1.5 seconds at 30 FPS.
            min_sign_frames: Minimum frames for a valid sign segment.
            max_sign_frames: Maximum frames for a single sign (splits if exceeded).
            padding_before: Frames to add before detected sign start.
            padding_after: Frames to add after detected sign end.
            blur_size: Gaussian blur kernel size for noise reduction.
            diff_threshold: Threshold for frame difference binarization.
        """
        self.motion_threshold = motion_threshold
        self.cooldown_frames = cooldown_frames
        self.min_sign_frames = min_sign_frames
        self.max_sign_frames = max_sign_frames
        self.padding_before = padding_before
        self.padding_after = padding_after
        self.blur_size = blur_size
        self.diff_threshold = diff_threshold

    def detect_motion_in_frame(
        self,
        frame: np.ndarray,
        prev_gray: Optional[np.ndarray]
    ) -> Tuple[bool, float, np.ndarray]:
        """
        Detect if there's significant motion in the frame.

        Args:
            frame: BGR frame from video
            prev_gray: Previous grayscale frame (None for first frame)

        Returns:
            (is_active, motion_score, current_gray)
        """
        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

        if prev_gray is None:
            return False, 0.0, gray

        # Calculate frame difference
        frame_delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_delta, self.diff_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Sum of thresholded pixels as motion score
        motion_score = float(np.sum(thresh))
        is_active = motion_score > self.motion_threshold

        return is_active, motion_score, gray

    def detect_segments_from_video(
        self,
        video_path: str,
        verbose: bool = True
    ) -> List[Tuple[int, int]]:
        """
        Analyze video and detect segment boundaries using pixel motion.

        Args:
            video_path: Path to video file
            verbose: Print progress information

        Returns:
            List of (start_frame, end_frame) tuples
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required for video motion detection")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if verbose:
            print(f"Analyzing video: {total_frames} frames at {fps:.1f} FPS")
            print(f"Motion threshold: {self.motion_threshold}")

        # State machine for detecting signs
        segments = []
        in_sign = False
        sign_start = 0
        frames_since_motion = 0
        prev_gray = None

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            is_active, motion_score, prev_gray = self.detect_motion_in_frame(
                frame, prev_gray
            )

            if is_active:
                if not in_sign:
                    # Motion started - potential sign beginning
                    sign_start = frame_idx
                    in_sign = True
                    if verbose:
                        print(f"  Frame {frame_idx}: Motion started (score: {motion_score:.0f})")
                frames_since_motion = 0
            else:
                if in_sign:
                    frames_since_motion += 1

                    # Check if sign is complete (enough frames of no motion)
                    if frames_since_motion >= self.cooldown_frames:
                        sign_end = frame_idx - frames_since_motion
                        sign_duration = sign_end - sign_start

                        if sign_duration >= self.min_sign_frames:
                            # Valid sign - add padding
                            start_padded = max(0, sign_start - self.padding_before)
                            end_padded = min(total_frames - 1, sign_end + self.padding_after)
                            segments.append((start_padded, end_padded))

                            if verbose:
                                print(f"  Frame {sign_end}: Sign complete ({sign_duration} frames) -> segment {len(segments)}")
                        else:
                            if verbose:
                                print(f"  Frame {sign_end}: Motion too short ({sign_duration} frames), skipping")

                        in_sign = False
                        frames_since_motion = 0

            # Handle very long signs by splitting
            if in_sign and (frame_idx - sign_start) >= self.max_sign_frames:
                # Force end the current sign
                start_padded = max(0, sign_start - self.padding_before)
                end_padded = min(total_frames - 1, frame_idx + self.padding_after)
                segments.append((start_padded, end_padded))

                if verbose:
                    print(f"  Frame {frame_idx}: Sign too long, splitting -> segment {len(segments)}")

                in_sign = False
                frames_since_motion = 0

            frame_idx += 1

        # Handle case where video ends during a sign
        if in_sign:
            sign_end = frame_idx - 1
            sign_duration = sign_end - sign_start
            if sign_duration >= self.min_sign_frames:
                start_padded = max(0, sign_start - self.padding_before)
                segments.append((start_padded, sign_end))
                if verbose:
                    print(f"  End of video: Final sign ({sign_duration} frames) -> segment {len(segments)}")

        cap.release()

        if verbose:
            print(f"Detected {len(segments)} segments")

        return segments

    def slice_pose_file(
        self,
        pose_path: str,
        segments: List[Tuple[int, int]],
        output_dir: str,
        verbose: bool = True
    ) -> List[str]:
        """
        Slice a pose file at the given frame boundaries.

        Args:
            pose_path: Path to .pose file
            segments: List of (start_frame, end_frame) tuples
            output_dir: Directory to save segment files
            verbose: Print progress information

        Returns:
            List of paths to created segment files
        """
        if not POSE_FORMAT_AVAILABLE:
            raise RuntimeError("pose_format library is required")

        os.makedirs(output_dir, exist_ok=True)

        # Load pose file
        with open(pose_path, "rb") as f:
            buffer = f.read()
            pose = Pose.read(buffer)

        total_frames = pose.body.data.shape[0]
        fps = pose.body.fps

        if verbose:
            print(f"Pose file: {total_frames} frames at {fps} FPS")

        segment_files = []

        for i, (start, end) in enumerate(segments):
            # Clamp to valid frame range
            start = max(0, min(start, total_frames - 1))
            end = max(start + 1, min(end, total_frames - 1))

            # Extract segment data
            segment_data = pose.body.data[start:end+1]

            # Handle confidence data
            if hasattr(pose.body, 'confidence') and pose.body.confidence is not None:
                segment_confidence = pose.body.confidence[start:end+1]
            else:
                segment_confidence = None

            # Create segment header
            segment_header = copy.deepcopy(pose.header)

            # Create new pose object
            segmented_pose = Pose(
                header=segment_header,
                body=type(pose.body)(
                    data=segment_data,
                    confidence=segment_confidence,
                    fps=pose.body.fps
                )
            )

            # Save segment
            segment_file = os.path.join(output_dir, f"segment_{i+1:03d}.pose")
            with open(segment_file, 'wb') as f:
                segmented_pose.write(f)

            segment_files.append(segment_file)

            if verbose:
                duration = end - start + 1
                print(f"  Segment {i+1}: frames {start}-{end} ({duration} frames) -> {os.path.basename(segment_file)}")

        return segment_files

    def segment_video_and_pose(
        self,
        video_path: str,
        pose_path: str,
        output_dir: str,
        verbose: bool = True
    ) -> List[str]:
        """
        Complete hybrid segmentation pipeline.

        1. Detect motion boundaries in video
        2. Slice pose file at those boundaries

        Args:
            video_path: Path to video file
            pose_path: Path to corresponding .pose file
            output_dir: Directory to save segment files
            verbose: Print progress information

        Returns:
            List of paths to created segment pose files
        """
        if verbose:
            print("=" * 60)
            print("Hybrid Segmentation (Video Motion + Pose Slicing)")
            print("=" * 60)

        # Step 1: Detect segments from video
        if verbose:
            print("\nStep 1: Detecting motion boundaries in video...")
        segments = self.detect_segments_from_video(video_path, verbose)

        if not segments:
            if verbose:
                print("WARNING: No segments detected in video")
            return []

        # Step 2: Slice pose file
        if verbose:
            print(f"\nStep 2: Slicing pose file into {len(segments)} segments...")
        segment_files = self.slice_pose_file(pose_path, segments, output_dir, verbose)

        if verbose:
            print(f"\nSUCCESS: Created {len(segment_files)} segment files")

        return segment_files


def main():
    """Command-line interface for hybrid segmentation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid video-pose segmentation using pixel motion detection"
    )

    parser.add_argument(
        "video_file",
        help="Input video file"
    )

    parser.add_argument(
        "pose_file",
        help="Input pose file (.pose)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Output directory for segment files"
    )

    parser.add_argument(
        "--motion-threshold",
        type=int,
        default=500000,
        help="Motion detection threshold (default: 500000)"
    )

    parser.add_argument(
        "--cooldown-frames",
        type=int,
        default=8,
        help="Frames of no motion to end sign (default: 8)"
    )

    parser.add_argument(
        "--min-sign-frames",
        type=int,
        default=12,
        help="Minimum frames for valid sign (default: 12)"
    )

    parser.add_argument(
        "--padding",
        type=int,
        default=3,
        help="Frames to add before/after segments (default: 3)"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Create segmenter
    segmenter = HybridSegmenter(
        motion_threshold=args.motion_threshold,
        cooldown_frames=args.cooldown_frames,
        min_sign_frames=args.min_sign_frames,
        padding_before=args.padding,
        padding_after=args.padding
    )

    # Run segmentation
    segment_files = segmenter.segment_video_and_pose(
        args.video_file,
        args.pose_file,
        args.output_dir,
        verbose=not args.quiet
    )

    if not args.quiet:
        print(f"\nCreated {len(segment_files)} segment files in {args.output_dir}")


if __name__ == "__main__":
    main()
