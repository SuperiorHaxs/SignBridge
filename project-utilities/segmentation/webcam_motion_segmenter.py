#!/usr/bin/env python3
"""
webcam_motion_segmenter.py
Motion-Based Video Segmentation for Webcam Recordings

Detects sign boundaries using pixel-based motion detection.
Works by detecting when motion starts (sign begins) and when
motion stops for a period (sign ends).

This is useful for natural webcam recordings where sign timing
may vary.

Usage:
    python webcam_motion_segmenter.py video.mp4 --output-dir segments/
    python webcam_motion_segmenter.py video.mp4 --pose-file capture.pose --output-dir segments/
"""

import os
import copy
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from pose_format import Pose
    POSE_FORMAT_AVAILABLE = True
except ImportError:
    POSE_FORMAT_AVAILABLE = False


class MotionBasedSegmenter:
    """
    Segment videos using pixel-based motion detection.

    Detects signing activity by measuring frame-to-frame pixel changes.
    Signs are segmented when:
    1. Motion exceeds threshold (sign starts)
    2. Motion drops below threshold for cooldown period (sign ends)
    """

    def __init__(
        self,
        motion_threshold: int = 500000,
        cooldown_frames: int = 30,
        min_sign_frames: int = 15,
        max_sign_frames: int = 150,
        padding_before: int = 5,
        padding_after: int = 5,
        blur_size: int = 21,
        diff_threshold: int = 25,
        startup_trim_seconds: float = 1.0,
        rampdown_trim_seconds: float = 1.0
    ):
        """
        Initialize motion-based segmenter.

        Args:
            motion_threshold: Pixel sum threshold for "active" motion.
                             Higher = less sensitive. Default 500000 for 640x480.
            cooldown_frames: Frames of no motion before sign ends (default: 30 ~1s at 30fps)
            min_sign_frames: Minimum frames for valid sign (default: 15 ~0.5s)
            max_sign_frames: Maximum frames before forced split (default: 150 ~5s)
            padding_before: Frames to add before sign start (default: 5)
            padding_after: Frames to add after sign end (default: 5)
            blur_size: Gaussian blur kernel size (default: 21)
            diff_threshold: Frame difference threshold (default: 25)
            startup_trim_seconds: Seconds to ignore at video start (default: 1.0)
            rampdown_trim_seconds: Seconds to ignore at video end (default: 1.0)
        """
        self.motion_threshold = motion_threshold
        self.cooldown_frames = cooldown_frames
        self.min_sign_frames = min_sign_frames
        self.max_sign_frames = max_sign_frames
        self.padding_before = padding_before
        self.padding_after = padding_after
        self.blur_size = blur_size
        self.diff_threshold = diff_threshold
        self.startup_trim_seconds = startup_trim_seconds
        self.rampdown_trim_seconds = rampdown_trim_seconds

    def detect_motion_in_frame(
        self,
        frame: np.ndarray,
        prev_gray: Optional[np.ndarray]
    ) -> Tuple[bool, float, np.ndarray]:
        """
        Detect motion between current and previous frame.

        Args:
            frame: Current BGR frame
            prev_gray: Previous grayscale frame (None for first frame)

        Returns:
            (is_active, motion_score, current_gray)
        """
        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

        if prev_gray is None:
            return False, 0.0, gray

        # Frame difference
        frame_delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_delta, self.diff_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Motion score = sum of thresholded pixels
        motion_score = float(np.sum(thresh))

        return motion_score > self.motion_threshold, motion_score, gray

    def detect_segments_from_video(
        self,
        video_path: str,
        verbose: bool = True
    ) -> List[Tuple[int, int]]:
        """
        Analyze video to detect sign segment boundaries.

        Args:
            video_path: Path to video file
            verbose: Print progress

        Returns:
            List of (start_frame, end_frame) tuples
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        # Calculate trim frames
        startup_trim_frames = int(self.startup_trim_seconds * fps)
        rampdown_trim_frames = int(self.rampdown_trim_seconds * fps)
        usable_start = startup_trim_frames
        usable_end = total_frames - rampdown_trim_frames

        # Scale threshold for resolution
        base_pixels = 640 * 480
        actual_pixels = width * height
        scale_factor = actual_pixels / base_pixels
        effective_threshold = self.motion_threshold * scale_factor

        if verbose:
            print(f"Video: {total_frames} frames at {fps:.1f} FPS ({duration:.2f}s)")
            print(f"Resolution: {width}x{height}")
            print(f"Usable range: frames {usable_start}-{usable_end} "
                  f"(trimming {self.startup_trim_seconds}s start, {self.rampdown_trim_seconds}s end)")
            print(f"Motion threshold: {effective_threshold:.0f} (scaled for resolution)")

        # First pass: collect motion scores for adaptive thresholding
        motion_scores = []
        prev_gray = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, score, prev_gray = self.detect_motion_in_frame(frame, prev_gray)
            motion_scores.append(score)

        # Adaptive threshold based on distribution
        if motion_scores:
            # Use percentile-based threshold
            usable_scores = motion_scores[usable_start:usable_end]
            if usable_scores:
                p25 = np.percentile(usable_scores, 25)
                p75 = np.percentile(usable_scores, 75)
                median = np.median(usable_scores)

                # Threshold between 25th percentile and median
                adaptive_threshold = (p25 + median) / 2

                if verbose:
                    print(f"Motion score percentiles: p25={p25:.0f}, median={median:.0f}, p75={p75:.0f}")
                    print(f"Using adaptive threshold: {adaptive_threshold:.0f}")

                effective_threshold = adaptive_threshold

        # Detect valleys (rest periods) using smoothed signal
        valleys = []
        if motion_scores:
            window_size = max(5, int(fps * 0.5))  # 0.5 second window
            smoothed = np.convolve(motion_scores, np.ones(window_size)/window_size, mode='same')

            # Find local minima
            min_gap = int(fps * 0.5)  # At least 0.5s between valleys
            last_valley = -min_gap

            for i in range(window_size, len(smoothed) - window_size):
                if i < usable_start or i > usable_end:
                    continue
                if i - last_valley < min_gap:
                    continue

                local_region = smoothed[max(0, i-window_size):min(len(smoothed), i+window_size+1)]
                if len(local_region) > 0 and smoothed[i] == np.min(local_region):
                    local_max = np.max(local_region)
                    if local_max > 0 and smoothed[i] < local_max * 0.3:
                        valleys.append(i)
                        last_valley = i

            if valleys and verbose:
                print(f"Detected {len(valleys)} rest valleys")

        valley_set = set(valleys)

        # Reset for second pass
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # State machine
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

            # Skip frames outside usable range
            if frame_idx < usable_start or frame_idx > usable_end:
                _, _, prev_gray = self.detect_motion_in_frame(frame, prev_gray)
                frame_idx += 1
                continue

            _, motion_score, prev_gray = self.detect_motion_in_frame(frame, prev_gray)
            is_active = motion_score > effective_threshold

            # Force rest at valleys
            is_valley = frame_idx in valley_set
            if is_valley:
                is_active = False

            if is_active:
                if not in_sign:
                    sign_start = frame_idx
                    in_sign = True
                    if verbose:
                        print(f"  Frame {frame_idx}: Sign started (motion: {motion_score:.0f})")
                frames_since_motion = 0
            else:
                if in_sign:
                    frames_since_motion += 1

                    # Shorter cooldown at valleys
                    required_cooldown = self.cooldown_frames // 2 if is_valley else self.cooldown_frames

                    if frames_since_motion >= required_cooldown:
                        sign_end = frame_idx - frames_since_motion
                        sign_duration = sign_end - sign_start

                        if sign_duration >= self.min_sign_frames:
                            start_padded = max(usable_start, sign_start - self.padding_before)
                            end_padded = min(usable_end, sign_end + self.padding_after)
                            segments.append((start_padded, end_padded))

                            if verbose:
                                print(f"  Frame {sign_end}: Sign ended ({sign_duration} frames) "
                                      f"-> segment {len(segments)}")
                        elif verbose:
                            print(f"  Frame {sign_end}: Too short ({sign_duration} frames), skipping")

                        in_sign = False
                        frames_since_motion = 0

            # Force split long signs
            if in_sign and (frame_idx - sign_start) >= self.max_sign_frames:
                start_padded = max(usable_start, sign_start - self.padding_before)
                end_padded = min(usable_end, frame_idx + self.padding_after)
                segments.append((start_padded, end_padded))

                if verbose:
                    print(f"  Frame {frame_idx}: Sign too long, splitting -> segment {len(segments)}")

                in_sign = False
                frames_since_motion = 0

            frame_idx += 1

        # Handle sign at end of video
        if in_sign:
            sign_end = min(frame_idx - 1, usable_end)
            sign_duration = sign_end - sign_start

            if sign_duration >= self.min_sign_frames:
                start_padded = max(usable_start, sign_start - self.padding_before)
                segments.append((start_padded, sign_end))

                if verbose:
                    print(f"  End: Final sign ({sign_duration} frames) -> segment {len(segments)}")

        cap.release()

        if verbose:
            print(f"Detected {len(segments)} segments")

        return segments

    def segment_video(
        self,
        video_path: str,
        output_dir: str,
        verbose: bool = True
    ) -> List[str]:
        """
        Segment video file based on motion detection.

        Args:
            video_path: Path to input video
            output_dir: Output directory for segments
            verbose: Print progress

        Returns:
            List of segment video file paths
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required")

        os.makedirs(output_dir, exist_ok=True)

        if verbose:
            print("=" * 60)
            print("Motion-Based Video Segmentation")
            print("=" * 60)

        # Detect segments
        segments = self.detect_segments_from_video(video_path, verbose)

        if not segments:
            if verbose:
                print("No segments detected")
            return []

        # Extract segments
        if verbose:
            print(f"\nExtracting {len(segments)} video segments...")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        segment_files = []

        for i, (start_frame, end_frame) in enumerate(segments):
            output_path = os.path.join(output_dir, f"segment_{i+1:03d}.mp4")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for _ in range(end_frame - start_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            out.release()
            segment_files.append(output_path)

            if verbose:
                duration = (end_frame - start_frame + 1) / fps
                print(f"  Created: {os.path.basename(output_path)} ({duration:.2f}s)")

        cap.release()

        if verbose:
            print(f"\nSUCCESS: Created {len(segment_files)} segments")

        return segment_files

    def segment_pose_file(
        self,
        pose_path: str,
        output_dir: str,
        video_path: Optional[str] = None,
        verbose: bool = True
    ) -> List[str]:
        """
        Segment pose file. If video_path provided, uses motion detection
        from video. Otherwise uses pose-based velocity detection.

        Args:
            pose_path: Path to .pose file
            output_dir: Output directory for segments
            video_path: Optional video file for motion detection
            verbose: Print progress

        Returns:
            List of segment pose file paths
        """
        if not POSE_FORMAT_AVAILABLE:
            raise RuntimeError("pose_format library is required")

        os.makedirs(output_dir, exist_ok=True)

        # Load pose
        with open(pose_path, "rb") as f:
            pose = Pose.read(f.read())

        fps = pose.body.fps
        total_frames = pose.body.data.shape[0]

        if verbose:
            print("=" * 60)
            print("Motion-Based Pose Segmentation")
            print("=" * 60)
            print(f"Pose: {total_frames} frames at {fps} FPS")

        # Get segments from video or pose
        if video_path and os.path.exists(video_path):
            if verbose:
                print("Using video for motion detection...")
            segments = self.detect_segments_from_video(video_path, verbose)
        else:
            if verbose:
                print("Using pose-based velocity detection...")
            segments = self._detect_segments_from_pose(pose, verbose)

        if not segments:
            if verbose:
                print("No segments detected")
            return []

        # Extract segments
        if verbose:
            print(f"\nExtracting {len(segments)} pose segments...")

        segment_files = []

        for i, (start_frame, end_frame) in enumerate(segments):
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, total_frames - 1))

            segment_data = pose.body.data[start_frame:end_frame + 1]

            if hasattr(pose.body, 'confidence') and pose.body.confidence is not None:
                segment_confidence = pose.body.confidence[start_frame:end_frame + 1]
            else:
                segment_confidence = None

            segment_pose = Pose(
                header=copy.deepcopy(pose.header),
                body=type(pose.body)(
                    data=segment_data,
                    confidence=segment_confidence,
                    fps=fps
                )
            )

            output_path = os.path.join(output_dir, f"segment_{i+1:03d}.pose")
            with open(output_path, 'wb') as f:
                segment_pose.write(f)

            segment_files.append(output_path)

            if verbose:
                duration = (end_frame - start_frame + 1) / fps
                print(f"  Created: {os.path.basename(output_path)} ({duration:.2f}s)")

        if verbose:
            print(f"\nSUCCESS: Created {len(segment_files)} segments")

        return segment_files

    def _detect_segments_from_pose(
        self,
        pose: 'Pose',
        verbose: bool = True
    ) -> List[Tuple[int, int]]:
        """
        Detect segments using pose keypoint velocity.

        Fallback when no video is available.
        """
        fps = pose.body.fps
        data = pose.body.data
        total_frames = data.shape[0]

        # Calculate trim frames
        startup_trim_frames = int(self.startup_trim_seconds * fps)
        rampdown_trim_frames = int(self.rampdown_trim_seconds * fps)
        usable_start = startup_trim_frames
        usable_end = total_frames - rampdown_trim_frames

        if verbose:
            print(f"Usable range: frames {usable_start}-{usable_end}")

        # Extract hand keypoints for velocity calculation
        # Assuming 75-point format: 33 body + 21 left hand + 21 right hand
        if data.shape[1] >= 75:
            # Use hands for velocity
            left_hand = data[:, 33:54, :2]
            right_hand = data[:, 54:75, :2]
        else:
            # Use all points
            left_hand = data[:, :, :2]
            right_hand = data[:, :, :2]

        # Calculate velocity
        velocities = []
        for i in range(1, total_frames):
            left_center = np.mean(left_hand[i], axis=0)
            right_center = np.mean(right_hand[i], axis=0)
            prev_left = np.mean(left_hand[i-1], axis=0)
            prev_right = np.mean(right_hand[i-1], axis=0)

            left_vel = np.linalg.norm(left_center - prev_left)
            right_vel = np.linalg.norm(right_center - prev_right)

            velocities.append(max(left_vel, right_vel))

        velocities = [0] + velocities  # Pad first frame

        # Smooth velocities
        window = max(3, int(fps * 0.1))
        smoothed = np.convolve(velocities, np.ones(window)/window, mode='same')

        # Adaptive threshold
        usable_velocities = smoothed[usable_start:usable_end]
        if len(usable_velocities) > 0:
            threshold = np.percentile(usable_velocities, 30)
        else:
            threshold = 0.02

        if verbose:
            print(f"Velocity threshold: {threshold:.4f}")

        # Detect segments
        segments = []
        in_sign = False
        sign_start = 0
        frames_below = 0

        for i in range(usable_start, usable_end):
            is_active = smoothed[i] > threshold

            if is_active:
                if not in_sign:
                    sign_start = i
                    in_sign = True
                frames_below = 0
            else:
                if in_sign:
                    frames_below += 1
                    if frames_below >= self.cooldown_frames:
                        sign_end = i - frames_below
                        duration = sign_end - sign_start

                        if duration >= self.min_sign_frames:
                            start_padded = max(usable_start, sign_start - self.padding_before)
                            end_padded = min(usable_end, sign_end + self.padding_after)
                            segments.append((start_padded, end_padded))

                        in_sign = False
                        frames_below = 0

        # Handle end
        if in_sign:
            sign_end = usable_end
            duration = sign_end - sign_start
            if duration >= self.min_sign_frames:
                start_padded = max(usable_start, sign_start - self.padding_before)
                segments.append((start_padded, sign_end))

        if verbose:
            print(f"Detected {len(segments)} segments from pose velocity")

        return segments


def main():
    parser = argparse.ArgumentParser(
        description="Motion-based video/pose segmentation for webcam recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Segment video
  python webcam_motion_segmenter.py video.mp4 --output-dir segments/

  # Segment pose file using video for motion detection
  python webcam_motion_segmenter.py capture.pose --video video.mp4 --output-dir segments/

  # Segment pose file using pose-based velocity
  python webcam_motion_segmenter.py capture.pose --output-dir segments/

  # Adjust sensitivity
  python webcam_motion_segmenter.py video.mp4 --cooldown-frames 20 --min-sign-frames 10
        """
    )

    parser.add_argument(
        "input_file",
        help="Input video (.mp4, .mov) or pose (.pose) file"
    )

    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Output directory for segments"
    )

    parser.add_argument(
        "--video", "-v",
        help="Video file for motion detection (when segmenting pose)"
    )

    parser.add_argument(
        "--motion-threshold",
        type=int,
        default=500000,
        help="Motion threshold (default: 500000)"
    )

    parser.add_argument(
        "--cooldown-frames",
        type=int,
        default=30,
        help="Frames of no motion to end sign (default: 30)"
    )

    parser.add_argument(
        "--min-sign-frames",
        type=int,
        default=15,
        help="Minimum frames for valid sign (default: 15)"
    )

    parser.add_argument(
        "--startup-trim",
        type=float,
        default=1.0,
        help="Seconds to trim from start (default: 1.0)"
    )

    parser.add_argument(
        "--rampdown-trim",
        type=float,
        default=1.0,
        help="Seconds to trim from end (default: 1.0)"
    )

    parser.add_argument(
        "--padding",
        type=int,
        default=5,
        help="Frames of padding before/after segments (default: 5)"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Create segmenter
    segmenter = MotionBasedSegmenter(
        motion_threshold=args.motion_threshold,
        cooldown_frames=args.cooldown_frames,
        min_sign_frames=args.min_sign_frames,
        padding_before=args.padding,
        padding_after=args.padding,
        startup_trim_seconds=args.startup_trim,
        rampdown_trim_seconds=args.rampdown_trim
    )

    input_path = Path(args.input_file)

    if input_path.suffix.lower() == '.pose':
        segment_files = segmenter.segment_pose_file(
            str(input_path),
            args.output_dir,
            video_path=args.video,
            verbose=not args.quiet
        )
    else:
        segment_files = segmenter.segment_video(
            str(input_path),
            args.output_dir,
            verbose=not args.quiet
        )

    if not args.quiet:
        print(f"\nCreated {len(segment_files)} segments in {args.output_dir}")


if __name__ == "__main__":
    main()
