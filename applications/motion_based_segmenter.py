#!/usr/bin/env python3
"""
Motion-based pose segmentation for sign language videos.

This module provides an alternative to pose_to_segments for detecting sign boundaries
by analyzing hand velocity and motion patterns in pose data.

Algorithm:
1. Calculate hand velocity (distance moved between consecutive frames)
2. Smooth velocities using moving average to reduce noise
3. Detect low-motion periods (velocity below threshold)
4. Find sign boundaries at transitions from high to low motion
5. Extract segments with configurable padding
"""

import numpy as np
import json
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

# Import pose_format for loading .pose files
try:
    from pose_format import Pose
    POSE_FORMAT_AVAILABLE = True
except ImportError:
    POSE_FORMAT_AVAILABLE = False
    print("WARNING: pose_format not available, motion-based segmentation requires it")
    print("Install with: pip install pose-format")


class MotionBasedSegmenter:
    """Segment pose sequences based on hand motion velocity."""

    def __init__(
        self,
        velocity_threshold: float = 0.02,
        min_sign_duration: int = 10,
        max_sign_duration: int = 120,
        min_rest_duration: int = 5,
        padding_before: int = 5,
        padding_after: int = 5,
        smoothing_window: int = 5
    ):
        """
        Initialize motion-based segmenter.

        Args:
            velocity_threshold: Velocity below this = "at rest" (normalized 0-1)
            min_sign_duration: Minimum frames for a valid sign
            max_sign_duration: Maximum frames for a single sign
            min_rest_duration: Minimum frames of rest between signs
            padding_before: Frames to add before detected sign start
            padding_after: Frames to add after detected sign end
            smoothing_window: Window size for velocity smoothing
        """
        self.velocity_threshold = velocity_threshold
        self.min_sign_duration = min_sign_duration
        self.max_sign_duration = max_sign_duration
        self.min_rest_duration = min_rest_duration
        self.padding_before = padding_before
        self.padding_after = padding_after
        self.smoothing_window = smoothing_window

    def load_pose_file(self, pose_file: str) -> np.ndarray:
        """Load pose data from .pose file using pose_format library."""
        if not POSE_FORMAT_AVAILABLE:
            raise RuntimeError("pose_format library is required but not installed")

        # Load binary pose file
        with open(pose_file, "rb") as f:
            buffer = f.read()
            pose = Pose.read(buffer)

        # Extract pose data
        pose_data = pose.body.data

        # Handle different shapes
        if len(pose_data.shape) == 4:
            # (frames, people, keypoints, dimensions) -> take first person
            pose_sequence = pose_data[:, 0, :, :]
        else:
            pose_sequence = pose_data

        # Extract 75-point subset if needed (pose + hands, exclude face)
        # MediaPipe format: 33 pose + 468 face + 21 left hand + 21 right hand = 543 total
        # We need: 33 pose + 21 left hand + 21 right hand = 75 keypoints
        if pose_sequence.shape[1] == 543:
            # Extract pose (0:33), left hand (501:522), right hand (522:543)
            pose_75pt = np.concatenate([
                pose_sequence[:, 0:33, :],      # Pose landmarks
                pose_sequence[:, 501:522, :],   # Left hand landmarks
                pose_sequence[:, 522:543, :]    # Right hand landmarks
            ], axis=1)
        elif pose_sequence.shape[1] == 75:
            # Already 75 keypoints
            pose_75pt = pose_sequence
        else:
            # Use as-is for other formats
            pose_75pt = pose_sequence

        return pose_75pt

    def calculate_hand_velocity(self, pose_sequence: np.ndarray) -> np.ndarray:
        """
        Calculate hand velocity for each frame.

        Uses hand keypoints (typically indices 4, 8, 12, 16, 20 for right hand,
        and 25, 29, 33, 37, 41 for left hand in MediaPipe Holistic).

        Args:
            pose_sequence: Array of shape (num_frames, num_keypoints, 3)

        Returns:
            velocities: Array of shape (num_frames,) with average hand velocity per frame
        """
        num_frames = len(pose_sequence)
        velocities = np.zeros(num_frames)

        # Hand keypoint indices (simplified - using wrist and fingertips)
        # Adjust these based on your actual keypoint layout
        right_hand_indices = [4, 8, 12, 16, 20]  # Right hand keypoints
        left_hand_indices = [25, 29, 33, 37, 41]  # Left hand keypoints
        hand_indices = right_hand_indices + left_hand_indices

        for i in range(1, num_frames):
            # Calculate displacement for each hand keypoint
            displacements = []

            for kp_idx in hand_indices:
                if kp_idx < pose_sequence.shape[1]:
                    # Get x, y coordinates
                    prev_pos = pose_sequence[i-1, kp_idx, :2]
                    curr_pos = pose_sequence[i, kp_idx, :2]
                    confidence = pose_sequence[i, kp_idx, 2]

                    # Only use keypoints with sufficient confidence
                    if confidence > 0.5:
                        displacement = np.linalg.norm(curr_pos - prev_pos)
                        displacements.append(displacement)

            # Average displacement across all visible hand keypoints
            if displacements:
                velocities[i] = np.mean(displacements)

        return velocities

    def smooth_velocities(self, velocities: np.ndarray) -> np.ndarray:
        """Apply moving average smoothing to velocity signal."""
        window = self.smoothing_window
        smoothed = np.convolve(velocities, np.ones(window)/window, mode='same')
        return smoothed

    def detect_sign_boundaries(self, velocities: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect sign start and end frames based on velocity thresholds.

        Args:
            velocities: Smoothed velocity array

        Returns:
            segments: List of (start_frame, end_frame) tuples
        """
        num_frames = len(velocities)
        is_moving = velocities > self.velocity_threshold

        segments = []
        in_sign = False
        sign_start = 0
        rest_counter = 0

        for i in range(num_frames):
            if is_moving[i]:
                if not in_sign:
                    # Motion started - potential sign beginning
                    sign_start = i
                    in_sign = True
                rest_counter = 0
            else:
                # No motion - at rest
                if in_sign:
                    rest_counter += 1

                    # If rest period is long enough, end the sign
                    if rest_counter >= self.min_rest_duration:
                        sign_end = i - rest_counter
                        sign_duration = sign_end - sign_start

                        # Validate sign duration
                        if (self.min_sign_duration <= sign_duration <= self.max_sign_duration):
                            # Add padding
                            start_padded = max(0, sign_start - self.padding_before)
                            end_padded = min(num_frames - 1, sign_end + self.padding_after)
                            segments.append((start_padded, end_padded))

                        in_sign = False
                        rest_counter = 0

        # Handle case where signing continues until end of video
        if in_sign:
            sign_end = num_frames - 1
            sign_duration = sign_end - sign_start
            if sign_duration >= self.min_sign_duration:
                start_padded = max(0, sign_start - self.padding_before)
                segments.append((start_padded, sign_end))

        return segments

    def segment_pose_file(
        self,
        pose_file: str,
        output_dir: str,
        verbose: bool = True
    ) -> List[str]:
        """
        Segment a pose file and save individual segments.

        Args:
            pose_file: Path to input .pose file
            output_dir: Directory to save segment files
            verbose: Print progress information

        Returns:
            segment_files: List of paths to created segment files
        """
        os.makedirs(output_dir, exist_ok=True)

        if verbose:
            print(f"Loading pose file: {pose_file}")

        # Load the original pose object for header info
        with open(pose_file, "rb") as f:
            buffer = f.read()
            original_pose = Pose.read(buffer)

        # Load pose data as numpy array
        pose_sequence = self.load_pose_file(pose_file)

        if verbose:
            print(f"Pose sequence shape: {pose_sequence.shape}")

        # Calculate velocities
        velocities = self.calculate_hand_velocity(pose_sequence)
        smoothed_velocities = self.smooth_velocities(velocities)

        if verbose:
            print(f"Velocity range: {smoothed_velocities.min():.4f} - {smoothed_velocities.max():.4f}")
            print(f"Velocity threshold: {self.velocity_threshold:.4f}")

        # Detect boundaries
        segments = self.detect_sign_boundaries(smoothed_velocities)

        if verbose:
            print(f"Detected {len(segments)} sign segments")

        # Get original pose data for proper slicing
        original_data = original_pose.body.data

        # Save segments
        segment_files = []
        for i, (start, end) in enumerate(segments):
            # Slice the original pose data maintaining original structure
            if len(original_data.shape) == 4:
                # (frames, people, keypoints, dimensions)
                segment_data = original_data[start:end+1, :, :, :]
            else:
                # (frames, keypoints, dimensions) - add people dimension
                segment_data = original_data[start:end+1]
                # Expand to 4D: (frames, people=1, keypoints, dimensions)
                segment_data = np.expand_dims(segment_data, axis=1)

            # Get confidence data if available
            if hasattr(original_pose.body, 'confidence'):
                if len(original_pose.body.confidence.shape) == 3:
                    # (frames, people, keypoints)
                    segment_confidence = original_pose.body.confidence[start:end+1, :, :]
                else:
                    # (frames, keypoints) - add people dimension
                    segment_confidence = original_pose.body.confidence[start:end+1]
                    segment_confidence = np.expand_dims(segment_confidence, axis=1)
            else:
                segment_confidence = None

            # Create segment header (copy and update frame count)
            import copy
            segment_header = copy.deepcopy(original_pose.header)
            if hasattr(segment_header, "frames"):
                segment_header.frames = segment_data.shape[0]
            elif hasattr(segment_header, "length"):
                segment_header.length = segment_data.shape[0]

            # Create new pose object for segment
            segmented_pose = Pose(
                header=segment_header,
                body=type(original_pose.body)(
                    data=segment_data,
                    confidence=segment_confidence,
                    fps=original_pose.body.fps
                )
            )

            # Create segment filename
            segment_file = os.path.join(output_dir, f"segment_{i+1:03d}.pose")

            # Save segment in binary Pose format
            with open(segment_file, 'wb') as f:
                segmented_pose.write(f)

            segment_files.append(segment_file)

            if verbose:
                duration = end - start + 1
                print(f"  Segment {i+1}: frames {start}-{end} ({duration} frames) -> {os.path.basename(segment_file)}")

        return segment_files


def main():
    """Command-line interface for motion-based segmentation."""
    parser = argparse.ArgumentParser(
        description="Segment pose files based on hand motion velocity"
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
        "--velocity-threshold",
        type=float,
        default=0.02,
        help="Velocity threshold for detecting rest (default: 0.02)"
    )

    parser.add_argument(
        "--min-sign-duration",
        type=int,
        default=10,
        help="Minimum frames for valid sign (default: 10)"
    )

    parser.add_argument(
        "--max-sign-duration",
        type=int,
        default=120,
        help="Maximum frames for single sign (default: 120)"
    )

    parser.add_argument(
        "--min-rest-duration",
        type=int,
        default=5,
        help="Minimum rest frames between signs (default: 5)"
    )

    parser.add_argument(
        "--padding-before",
        type=int,
        default=5,
        help="Frames to add before sign start (default: 5)"
    )

    parser.add_argument(
        "--padding-after",
        type=int,
        default=5,
        help="Frames to add after sign end (default: 5)"
    )

    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Window size for velocity smoothing (default: 5)"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Create segmenter
    segmenter = MotionBasedSegmenter(
        velocity_threshold=args.velocity_threshold,
        min_sign_duration=args.min_sign_duration,
        max_sign_duration=args.max_sign_duration,
        min_rest_duration=args.min_rest_duration,
        padding_before=args.padding_before,
        padding_after=args.padding_after,
        smoothing_window=args.smoothing_window
    )

    # Process file
    segment_files = segmenter.segment_pose_file(
        args.pose_file,
        args.output_dir,
        verbose=not args.quiet
    )

    if not args.quiet:
        print(f"\nSUCCESS: Created {len(segment_files)} segment files in {args.output_dir}")


if __name__ == "__main__":
    main()
