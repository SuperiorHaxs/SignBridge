#!/usr/bin/env python3
"""
webcam_time_segmenter.py
Time-Based Video Segmentation for Webcam Recordings

Simple time-based segmentation assuming:
- Each sign is approximately 2-3 seconds
- Pause between signs is 1-1.5 seconds
- Startup period (1 second) is trimmed
- Ramp down period (1 second) is trimmed

This is useful for controlled webcam recordings where the signer
follows a consistent rhythm.

Usage:
    python webcam_time_segmenter.py video.mp4 --output-dir segments/
    python webcam_time_segmenter.py video.mp4 --sign-duration 2.5 --pause-duration 1.25
"""

import os
import copy
import argparse
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


class TimeBasedSegmenter:
    """
    Segment videos using fixed time intervals.

    Assumes a rhythmic signing pattern:
    [startup] [sign1] [pause] [sign2] [pause] ... [signN] [rampdown]
    """

    def __init__(
        self,
        sign_duration: float = 2.5,
        pause_duration: float = 1.25,
        startup_trim: float = 1.0,
        rampdown_trim: float = 1.0,
        padding_before: float = 0.1,
        padding_after: float = 0.1
    ):
        """
        Initialize time-based segmenter.

        Args:
            sign_duration: Expected duration of each sign in seconds (default: 2.5s)
            pause_duration: Expected pause between signs in seconds (default: 1.25s)
            startup_trim: Seconds to trim from start of video (default: 1.0s)
            rampdown_trim: Seconds to trim from end of video (default: 1.0s)
            padding_before: Extra seconds to include before each sign (default: 0.1s)
            padding_after: Extra seconds to include after each sign (default: 0.1s)
        """
        self.sign_duration = sign_duration
        self.pause_duration = pause_duration
        self.startup_trim = startup_trim
        self.rampdown_trim = rampdown_trim
        self.padding_before = padding_before
        self.padding_after = padding_after

    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata.

        Returns:
            dict with fps, frame_count, duration_seconds
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        return {
            'fps': fps,
            'frame_count': frame_count,
            'duration_seconds': frame_count / fps if fps > 0 else 0
        }

    def calculate_segments(
        self,
        total_duration: float,
        fps: float,
        num_signs: Optional[int] = None,
        verbose: bool = True
    ) -> List[Tuple[int, int]]:
        """
        Calculate segment boundaries based on time intervals.

        Args:
            total_duration: Total video duration in seconds
            fps: Frames per second
            num_signs: Number of signs expected (if None, auto-calculate)
            verbose: Print progress

        Returns:
            List of (start_frame, end_frame) tuples
        """
        # Calculate usable duration after trimming
        usable_start = self.startup_trim
        usable_end = total_duration - self.rampdown_trim
        usable_duration = usable_end - usable_start

        if usable_duration <= 0:
            if verbose:
                print(f"ERROR: Video too short. Duration: {total_duration:.2f}s, "
                      f"trims: {self.startup_trim + self.rampdown_trim:.2f}s")
            return []

        # Calculate number of signs if not specified
        if num_signs is None:
            # Each sign cycle = sign_duration + pause_duration
            # Last sign doesn't have a trailing pause
            cycle_duration = self.sign_duration + self.pause_duration

            # Estimate: (usable_duration + pause_duration) / cycle_duration
            # This accounts for the fact that last sign doesn't have trailing pause
            num_signs = int((usable_duration + self.pause_duration) / cycle_duration)
            num_signs = max(1, num_signs)

        if verbose:
            print(f"Video duration: {total_duration:.2f}s")
            print(f"Usable duration: {usable_duration:.2f}s (after {self.startup_trim}s startup, {self.rampdown_trim}s rampdown trim)")
            print(f"Expected signs: {num_signs}")
            print(f"Sign duration: {self.sign_duration}s, Pause: {self.pause_duration}s")

        segments = []
        current_time = usable_start

        for i in range(num_signs):
            # Calculate sign boundaries
            sign_start = current_time - self.padding_before
            sign_end = current_time + self.sign_duration + self.padding_after

            # Clamp to valid range
            sign_start = max(0, sign_start)
            sign_end = min(total_duration, sign_end)

            # Convert to frames
            start_frame = int(sign_start * fps)
            end_frame = int(sign_end * fps)

            if end_frame > start_frame:
                segments.append((start_frame, end_frame))

                if verbose:
                    print(f"  Sign {i+1}: {sign_start:.2f}s - {sign_end:.2f}s "
                          f"(frames {start_frame}-{end_frame})")

            # Move to next sign
            current_time += self.sign_duration + self.pause_duration

            # Stop if we've gone past usable end
            if current_time > usable_end:
                break

        return segments

    def segment_video(
        self,
        video_path: str,
        output_dir: str,
        num_signs: Optional[int] = None,
        verbose: bool = True
    ) -> List[str]:
        """
        Segment video file using time-based boundaries.

        Args:
            video_path: Path to input video
            output_dir: Directory for output segment videos
            num_signs: Expected number of signs (auto-detect if None)
            verbose: Print progress

        Returns:
            List of paths to segment video files
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required")

        os.makedirs(output_dir, exist_ok=True)

        # Get video info
        info = self.get_video_info(video_path)
        fps = info['fps']
        total_duration = info['duration_seconds']

        if verbose:
            print("=" * 60)
            print("Time-Based Video Segmentation")
            print("=" * 60)

        # Calculate segments
        segments = self.calculate_segments(total_duration, fps, num_signs, verbose)

        if not segments:
            return []

        # Extract video segments
        if verbose:
            print(f"\nExtracting {len(segments)} video segments...")

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        segment_files = []

        for i, (start_frame, end_frame) in enumerate(segments):
            output_path = os.path.join(output_dir, f"segment_{i+1:03d}.mp4")

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Write frames
            for frame_idx in range(start_frame, end_frame + 1):
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
            print(f"\nSUCCESS: Created {len(segment_files)} segment files")

        return segment_files

    def segment_pose_file(
        self,
        pose_path: str,
        output_dir: str,
        num_signs: Optional[int] = None,
        verbose: bool = True
    ) -> List[str]:
        """
        Segment pose file using time-based boundaries.

        Args:
            pose_path: Path to input .pose file
            output_dir: Directory for output segment pose files
            num_signs: Expected number of signs (auto-detect if None)
            verbose: Print progress

        Returns:
            List of paths to segment pose files
        """
        if not POSE_FORMAT_AVAILABLE:
            raise RuntimeError("pose_format library is required")

        os.makedirs(output_dir, exist_ok=True)

        # Load pose file
        with open(pose_path, "rb") as f:
            buffer = f.read()
            pose = Pose.read(buffer)

        fps = pose.body.fps
        total_frames = pose.body.data.shape[0]
        total_duration = total_frames / fps

        if verbose:
            print("=" * 60)
            print("Time-Based Pose Segmentation")
            print("=" * 60)
            print(f"Pose file: {total_frames} frames at {fps} FPS")

        # Calculate segments
        segments = self.calculate_segments(total_duration, fps, num_signs, verbose)

        if not segments:
            return []

        # Extract pose segments
        if verbose:
            print(f"\nExtracting {len(segments)} pose segments...")

        segment_files = []

        for i, (start_frame, end_frame) in enumerate(segments):
            # Clamp to valid range
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, total_frames - 1))

            # Extract segment data
            segment_data = pose.body.data[start_frame:end_frame + 1]

            # Handle confidence
            if hasattr(pose.body, 'confidence') and pose.body.confidence is not None:
                segment_confidence = pose.body.confidence[start_frame:end_frame + 1]
            else:
                segment_confidence = None

            # Create segment pose
            segment_pose = Pose(
                header=copy.deepcopy(pose.header),
                body=type(pose.body)(
                    data=segment_data,
                    confidence=segment_confidence,
                    fps=fps
                )
            )

            # Save
            output_path = os.path.join(output_dir, f"segment_{i+1:03d}.pose")
            with open(output_path, 'wb') as f:
                segment_pose.write(f)

            segment_files.append(output_path)

            if verbose:
                duration = (end_frame - start_frame + 1) / fps
                print(f"  Created: {os.path.basename(output_path)} "
                      f"(frames {start_frame}-{end_frame}, {duration:.2f}s)")

        if verbose:
            print(f"\nSUCCESS: Created {len(segment_files)} segment files")

        return segment_files


def main():
    parser = argparse.ArgumentParser(
        description="Time-based video/pose segmentation for webcam recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Segment video with default timing (2.5s signs, 1.25s pauses)
  python webcam_time_segmenter.py video.mp4 --output-dir segments/

  # Segment with custom timing
  python webcam_time_segmenter.py video.mp4 --sign-duration 3.0 --pause-duration 1.5

  # Segment pose file
  python webcam_time_segmenter.py capture.pose --output-dir segments/

  # Specify expected number of signs
  python webcam_time_segmenter.py video.mp4 --num-signs 5 --output-dir segments/
        """
    )

    parser.add_argument(
        "input_file",
        help="Input video (.mp4, .mov) or pose (.pose) file"
    )

    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Output directory for segment files"
    )

    parser.add_argument(
        "--sign-duration",
        type=float,
        default=2.5,
        help="Expected duration of each sign in seconds (default: 2.5)"
    )

    parser.add_argument(
        "--pause-duration",
        type=float,
        default=1.25,
        help="Expected pause between signs in seconds (default: 1.25)"
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
        "--num-signs", "-n",
        type=int,
        default=None,
        help="Expected number of signs (auto-detect if not specified)"
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=0.1,
        help="Seconds of padding before/after each segment (default: 0.1)"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Create segmenter
    segmenter = TimeBasedSegmenter(
        sign_duration=args.sign_duration,
        pause_duration=args.pause_duration,
        startup_trim=args.startup_trim,
        rampdown_trim=args.rampdown_trim,
        padding_before=args.padding,
        padding_after=args.padding
    )

    # Determine input type and segment
    input_path = Path(args.input_file)

    if input_path.suffix.lower() == '.pose':
        segment_files = segmenter.segment_pose_file(
            str(input_path),
            args.output_dir,
            num_signs=args.num_signs,
            verbose=not args.quiet
        )
    else:
        segment_files = segmenter.segment_video(
            str(input_path),
            args.output_dir,
            num_signs=args.num_signs,
            verbose=not args.quiet
        )

    if not args.quiet:
        print(f"\nCreated {len(segment_files)} segments in {args.output_dir}")


if __name__ == "__main__":
    main()
