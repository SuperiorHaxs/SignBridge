#!/usr/bin/env python3
"""
sentence_to_pickle.py
Maps sentence words to vocab indices, finds corresponding pickle/video files,
and concatenates videos into a combined sentence video

NEW: CLI Mode with --pose-mode flag for direct pose file concatenation
"""

import json
import os
import sys
import argparse
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime
import random
import numpy as np

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Go up: pose_utils -> project-utilities -> project root
try:
    from config import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("WARNING: config module not available. Using fallback paths.")

# Import pose-format libraries
try:
    from pose_format import Pose
    from pose_format.numpy.pose_body import NumPyPoseBody
    POSE_FORMAT_AVAILABLE = True
except ImportError:
    POSE_FORMAT_AVAILABLE = False
    print("WARNING: pose-format library not available. Pose concatenation will not work.")
    print("Install with: pip install pose-format")

def concatenate_pose_files(pose_file_list, output_path):
    """
    Concatenate multiple pose files into a single pose file.

    Args:
        pose_file_list: List of paths to pose files to concatenate
        output_path: Path to write concatenated pose file

    Returns:
        True if successful, False otherwise
    """
    if not POSE_FORMAT_AVAILABLE:
        print("ERROR: pose-format library is required for pose concatenation")
        return False

    if not pose_file_list:
        print("ERROR: No pose files provided for concatenation")
        return False

    if len(pose_file_list) == 1:
        # Single file - just copy it
        import shutil
        shutil.copy2(pose_file_list[0], output_path)
        print(f"Single pose file copied to {output_path}")
        return True

    try:
        # Load all pose files
        poses = []
        print(f"Loading {len(pose_file_list)} pose files...")
        for i, pose_file in enumerate(pose_file_list, 1):
            if not os.path.exists(pose_file):
                print(f"ERROR: Pose file not found: {pose_file}")
                return False

            with open(pose_file, 'rb') as f:
                pose = Pose.read(f.read())
                poses.append(pose)
                print(f"  {i}. Loaded {os.path.basename(pose_file)} ({pose.body.data.shape[0]} frames)")

        # Normalize and focus each pose before concatenation (ensures consistent coordinate space and origin)
        print("\nNormalizing poses to consistent coordinate space...")
        normalized_poses = []
        for i, pose in enumerate(poses, 1):
            try:
                # Remove duplicate world landmarks (keep face for better visualization)
                # Only remove POSE_WORLD_LANDMARKS which is duplicate of POSE_LANDMARKS
                filtered_pose = pose.remove_components(['POSE_WORLD_LANDMARKS'])

                # Normalize (fixed distance between reference points)
                normalized_pose = filtered_pose.normalize()

                # Focus (center at origin 0,0) - modifies in-place, returns None
                normalized_pose.focus()

                normalized_poses.append(normalized_pose)
                print(f"  {i}. Filtered, normalized and focused {os.path.basename(pose_file_list[i-1])}")
            except Exception as e:
                print(f"  WARNING: Failed to normalize pose {i}, using original: {e}")
                normalized_poses.append(pose)

        # Concatenate data and confidence arrays along time axis (axis 0 = frames)
        print("\nConcatenating pose data...")
        concatenated_data = np.ma.concatenate([p.body.data for p in normalized_poses], axis=0)
        concatenated_confidence = np.ma.concatenate([p.body.confidence for p in normalized_poses], axis=0)

        total_frames = concatenated_data.shape[0]
        print(f"Concatenated {len(poses)} poses into {total_frames} frames")

        # Create new NumPyPoseBody
        new_body = NumPyPoseBody(
            fps=normalized_poses[0].body.fps,  # Use fps from first normalized pose
            data=concatenated_data,
            confidence=concatenated_confidence
        )

        # Create new Pose with header from first NORMALIZED pose (has filtered components)
        new_pose = Pose(header=normalized_poses[0].header, body=new_body)

        # Write to output file
        print(f"Writing concatenated pose to: {output_path}")
        with open(output_path, 'wb') as f:
            new_pose.write(f)

        # Verify file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"SUCCESS: Concatenated pose file created ({file_size:,} bytes)")
            return True
        else:
            print("ERROR: Output file was not created")
            return False

    except Exception as e:
        print(f"ERROR: Failed to concatenate pose files: {e}")
        import traceback
        traceback.print_exc()
        return False

def find_pose_file_for_gloss(gloss, pose_dir, sample_strategy='first', split='all', num_glosses=50):
    """
    Find pose file for a given gloss from pose directory.

    Args:
        gloss: Gloss name (e.g., "DOG", "ACCIDENT")
        pose_dir: Base pose directory (should point to pose_files_by_gloss/ or dataset root)
        sample_strategy: 'first' or 'random' - how to pick when multiple samples exist
        split: 'train', 'test', 'val', or 'all' - which dataset split to use
        num_glosses: Number of glosses (20, 50) - used for finding split directory

    Returns:
        Full path to pose file or None if not found
    """
    # Determine the directory to search based on split
    if split != 'all':
        # Use split directory structure: dataset_splits/{num_glosses}_classes/original/pose_split_{num_glosses}_class/{split}/{gloss}/
        # Navigate from pose_dir to the split directory
        dataset_root = Path(pose_dir).parent  # Go up from pose_files_by_gloss
        split_dir = dataset_root / "dataset_splits" / f"{num_glosses}_classes" / "original" / f"pose_split_{num_glosses}_class" / split / gloss.lower()
        gloss_dir = str(split_dir)
    else:
        # Use combined directory: pose_files_by_gloss/{gloss}/
        gloss_dir = os.path.join(pose_dir, gloss.lower())

    if not os.path.exists(gloss_dir):
        return None

    if not os.path.isdir(gloss_dir):
        return None

    # Find all .pose files
    pose_files = [f for f in os.listdir(gloss_dir) if f.endswith('.pose')]

    if not pose_files:
        return None

    # Apply strategy
    if sample_strategy == 'random':
        selected = random.choice(pose_files)
    else:  # 'first'
        selected = sorted(pose_files)[0]

    return os.path.join(gloss_dir, selected)

def load_vocab(vocab_path):
    """Load vocabulary mapping from vocab.json"""
    try:
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        # Create reverse mapping: gloss -> index
        gloss_to_index = {gloss.lower(): int(idx) for idx, gloss in vocab.items()}
        return gloss_to_index, vocab
    except Exception as e:
        print(f"Error loading vocab: {e}")
        return {}, {}

def find_pickle_and_video_files(gloss, pickle_dir, video_dir):
    """Find a pickle file that contains the given gloss and corresponding video file"""
    try:
        for pkl_file in os.listdir(pickle_dir):
            if pkl_file.endswith('.pkl'):
                try:
                    import pickle
                    with open(os.path.join(pickle_dir, pkl_file), 'rb') as f:
                        data = pickle.load(f)
                        if data.get('gloss', '').lower() == gloss.lower():
                            # Found matching pickle file, now look for video file
                            base_name = pkl_file[:-4]  # Remove .pkl extension
                            video_file = find_video_file(base_name, video_dir)
                            return pkl_file, video_file
                except Exception:
                    continue
        return None, None
    except Exception as e:
        print(f"Error searching pickle files: {e}")
        return None, None

def find_video_file(base_name, video_dir):
    """Find video file with the same base name as pickle file"""
    try:
        if not os.path.exists(video_dir):
            return "VIDEO_DIR_NOT_FOUND"

        # Common video extensions
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']

        for ext in video_extensions:
            video_file = base_name + ext
            if os.path.exists(os.path.join(video_dir, video_file)):
                return video_file

        return "VIDEO_NOT_FOUND"
    except Exception as e:
        print(f"Error searching video files: {e}")
        return "VIDEO_SEARCH_ERROR"

def concatenate_videos(video_files, output_path, video_dir):
    """Concatenate videos with format standardization to fix timeline issues"""
    try:
        # Get valid video paths
        valid_videos = []
        for video_file in video_files:
            if video_file not in ["VIDEO_NOT_FOUND", "VIDEO_DIR_NOT_FOUND", "VIDEO_SEARCH_ERROR", "NOT_FOUND"]:
                video_path = os.path.join(video_dir, video_file)
                if os.path.exists(video_path):
                    valid_videos.append(video_path)

        if not valid_videos:
            print("❌ No valid video files found for concatenation")
            return False

        if len(valid_videos) == 1:
            # Single video - just copy it
            import shutil
            shutil.copy2(valid_videos[0], output_path)
            print("✅ Single video copied")
            return True

        print(f"🎬 Concatenating {len(valid_videos)} videos with format standardization...")

        # Step 1: Standardize each video to prevent timeline issues
        standardized_videos = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, video_path in enumerate(valid_videos):
                temp_video = os.path.join(temp_dir, f"std_{i}.mp4")

                # Standardize format to prevent timeline corruption
                std_cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-r', '30',           # Force consistent frame rate
                    '-s', '640x480',      # Force consistent resolution
                    '-an',                # Remove audio
                    '-t', '3',            # Limit to 3 seconds max
                    temp_video
                ]

                result = subprocess.run(std_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    standardized_videos.append(temp_video)
                    print(f"✅ Standardized video {i+1}/{len(valid_videos)}")
                else:
                    print(f"❌ Failed to standardize video {i+1}: {result.stderr}")

            if not standardized_videos:
                print("❌ No videos could be standardized")
                return False

            # Step 2: Create file list for concatenation
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for temp_video in standardized_videos:
                    escaped_path = temp_video.replace('\\', '/')
                    f.write(f"file '{escaped_path}'\n")
                file_list_path = f.name

            try:
                # Step 3: Concatenate standardized videos
                concat_cmd = [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', file_list_path,
                    '-c', 'copy',  # Copy since all videos are now standardized
                    output_path
                ]

                print("🔄 Concatenating standardized videos...")
                result = subprocess.run(concat_cmd, capture_output=True, text=True, timeout=60)

                if result.returncode == 0:
                    print("✅ Video concatenation successful!")
                    return True
                else:
                    print(f"❌ Concatenation failed: {result.stderr}")
                    return False

            finally:
                # Clean up temporary file list
                if os.path.exists(file_list_path):
                    os.unlink(file_list_path)

    except FileNotFoundError:
        print("❌ FFmpeg not found. Please install FFmpeg and add it to your PATH.")
        return False
    except Exception as e:
        print(f"❌ Error during video concatenation: {e}")
        return False

def video_to_pose(video_path, pose_output_path):
    """Convert video to pose file - exact copy from wlasl_pose_extraction.py"""
    try:
        # Skip if pose file already exists
        if os.path.exists(pose_output_path):
            print(f"✅ Pose file already exists: {pose_output_path}")
            return True

        # Run video_to_pose on single file - EXACTLY like wlasl_pose_extraction.py
        cmd = [
            'video_to_pose',
            '--format', 'mediapipe',
            '-i', str(video_path),
            '-o', pose_output_path
        ]

        print(f"🤖 Converting video to pose: {video_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0 and os.path.exists(pose_output_path):
            print(f"✅ Pose extraction successful: {pose_output_path}")
            return True
        else:
            print(f"❌ Pose extraction failed")
            if result.stdout:
                print(f"STDOUT: {result.stdout}")
            if result.stderr:
                print(f"STDERR: {result.stderr}")
            return False

    except Exception as e:
        print(f"Exception processing {os.path.basename(video_path)}: {e}")
        return False

def convert_for_mediapipe(video_path, output_path):
    """Convert video to format that MediaPipe can definitely process"""
    try:
        # Ultra-simple format that MediaPipe should handle
        cmd_convert = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', 'scale=320:240,fps=10',  # Very small, slow
            '-c:v', 'libx264',
            '-profile:v', 'baseline',       # Simplest H.264 profile
            '-level:v', '2.1',              # Low level for compatibility
            '-pix_fmt', 'yuv420p',
            '-crf', '28',                   # Lower quality for simplicity
            '-g', '10',                     # Frequent keyframes
            '-bf', '0',                     # No B-frames
            '-refs', '1',                   # Single reference frame
            '-me_method', 'dia',            # Simple motion estimation
            '-subq', '1',                   # Fast subpixel estimation
            '-trellis', '0',                # Disable trellis quantization
            '-aq-mode', '0',                # Disable adaptive quantization
            '-an',                          # No audio
            '-threads', '1',                # Single thread for consistency
            output_path
        ]

        print("🔧 Converting to ultra-simple MediaPipe format...")
        result = subprocess.run(cmd_convert, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print("✅ Ultra-simple video conversion successful")
            return True
        else:
            print(f"❌ Video conversion failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error converting video: {e}")
        return False

def visualize_pose(pose_path, video_output_path):
    """Convert pose file to visualization video using visualize_pose CLI"""
    try:
        cmd = [
            'visualize_pose',
            '-i', pose_path,
            '-o', video_output_path,
            '--normalize'
        ]

        print(f"Creating pose visualization: {pose_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0 and os.path.exists(video_output_path):
            print(f"SUCCESS: Pose visualization created: {video_output_path}")
            return True
        else:
            print(f"ERROR: Pose visualization failed: {result.stderr}")
            return False

    except FileNotFoundError:
        print("ERROR: visualize_pose not found. Please install pose-format and add to PATH.")
        print("Install with: pip install pose-format[mediapipe]")
        return False
    except subprocess.TimeoutExpired:
        print("ERROR: Pose visualization timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"ERROR: Error during pose visualization: {e}")
        return False

def process_sentence_pose_mode(sentence, pose_dir, output_dir, sample_strategy='first', visualize=False, split='all', num_glosses=50):
    """
    Process sentence in pose concatenation mode (NEW).

    Args:
        sentence: Sentence with glosses (e.g., "DOG LIKE WALK")
        pose_dir: Directory with pose_files_by_gloss structure
        output_dir: Directory to save output files
        sample_strategy: 'first' or 'random' for selecting pose samples
        visualize: Whether to create pose visualization video
        split: 'train', 'test', 'val', or 'all' - which dataset split to use (default: 'all')
        num_glosses: Number of glosses (20, 50) - used for finding split directory

    Returns:
        Tuple of (pose_file_path, metadata_file_path) or (None, None) if failed
    """
    print(f"\nProcessing sentence: '{sentence}'")
    print("=" * 70)

    # Parse sentence into glosses
    glosses = sentence.strip().upper().split()
    print(f"Glosses: {glosses}")
    print()

    # Find pose file for each gloss
    pose_files = []
    found_glosses = []
    missing_glosses = []

    for gloss in glosses:
        print(f"Finding pose for '{gloss}'...")
        pose_file = find_pose_file_for_gloss(gloss, pose_dir, sample_strategy, split, num_glosses)

        if pose_file:
            pose_files.append(pose_file)
            found_glosses.append(gloss)
            print(f"  Found: {pose_file}")
        else:
            missing_glosses.append(gloss)
            print(f"  NOT FOUND: No pose files for '{gloss}'")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"Found: {len(found_glosses)}/{len(glosses)} glosses")
    if found_glosses:
        print(f"  {', '.join(found_glosses)}")
    if missing_glosses:
        print(f"Missing: {', '.join(missing_glosses)}")

    # Check if we have any valid poses
    if not pose_files:
        print("\nERROR: No valid pose files found. Cannot create concatenated pose.")
        return None, None

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_sentence = "_".join([g.lower() for g in found_glosses])
    output_filename = f"sentence_{safe_sentence}_{timestamp}.pose"
    output_pose_path = os.path.join(output_dir, output_filename)

    # Read frame counts from each pose file to create segment metadata
    print("\n" + "=" * 70)
    print("COLLECTING SEGMENT INFORMATION:")
    segments = []
    frame_offset = 0

    for i, pose_file in enumerate(pose_files):
        try:
            with open(pose_file, 'rb') as f:
                pose = Pose.read(f.read())

            num_frames = pose.body.data.shape[0]

            segment_info = {
                "gloss": found_glosses[i],
                "start_frame": frame_offset,
                "end_frame": frame_offset + num_frames - 1,
                "num_frames": num_frames,
                "source_file": os.path.basename(pose_file)
            }
            segments.append(segment_info)

            print(f"  {i+1}. {found_glosses[i]}: frames {frame_offset}-{frame_offset + num_frames - 1} ({num_frames} frames)")

            frame_offset += num_frames
        except Exception as e:
            print(f"  ERROR: Failed to read {pose_file}: {e}")
            return None, None

    # Concatenate pose files
    print("\n" + "=" * 70)
    print("POSE CONCATENATION:")
    success = concatenate_pose_files(pose_files, output_pose_path)

    if not success:
        print("ERROR: Failed to concatenate pose files")
        return None, None

    print(f"\nFINAL OUTPUT: {output_pose_path}")

    # Save segment metadata
    metadata = {
        "pose_file": os.path.basename(output_pose_path),
        "glosses": found_glosses,
        "total_frames": frame_offset,
        "num_segments": len(segments),
        "segments": segments,
        "created_timestamp": timestamp
    }

    metadata_filename = output_filename.replace('.pose', '_segments.json')
    metadata_path = os.path.join(output_dir, metadata_filename)

    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"SEGMENT METADATA: {metadata_path}")
    except Exception as e:
        print(f"WARNING: Failed to save segment metadata: {e}")

    # Optional: Create visualization
    if visualize:
        print("\n" + "=" * 70)
        print("POSE VISUALIZATION:")
        viz_filename = output_filename.replace('.pose', '_viz.mp4')
        viz_output_path = os.path.join(output_dir, viz_filename)

        viz_success = visualize_pose(output_pose_path, viz_output_path)

        if viz_success:
            print(f"\nVisualization created: {viz_output_path}")
        else:
            print("\nWARNING: Visualization failed (optional feature)")

    print("\n" + "=" * 70)
    print("COMPLETED SUCCESSFULLY")
    print(f"Concatenated pose: {output_pose_path}")
    print(f"Segment metadata: {metadata_path}")

    return output_pose_path, metadata_path

def process_sentence(sentence, vocab_path, pickle_dir, video_dir, test_dir, create_pose_files=False):
    """Process sentence and map words to indices, pickle files, and video files"""

    # Load vocabulary
    gloss_to_index, index_to_gloss = load_vocab(vocab_path)

    if not gloss_to_index:
        print("Failed to load vocabulary")
        return

    # Split sentence into words
    words = sentence.strip().split()

    print(f"\nProcessing sentence: '{sentence}'")
    print("=" * 70)

    found_words = []
    not_found_words = []
    video_files_for_concat = []

    for word in words:
        word_lower = word.lower()

        if word_lower in gloss_to_index:
            index = gloss_to_index[word_lower]
            print(f"Word: '{word}' -> Index: {index}")

            # Find corresponding pickle and video files
            pickle_file, video_file = find_pickle_and_video_files(word, pickle_dir, video_dir)
            if pickle_file:
                print(f"  Pickle file: {pickle_file}")
                print(f"  Video file:  {video_file}")
                found_words.append((word, index, pickle_file, video_file))
                video_files_for_concat.append(video_file)
            else:
                print(f"  ⚠️  No pickle file found for '{word}'")
                found_words.append((word, index, "NOT_FOUND", "NOT_FOUND"))
        else:
            print(f"Word: '{word}' -> ❌ NOT in vocabulary")
            not_found_words.append(word)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"✅ Found in vocab: {len(found_words)} words")
    print(f"❌ Not found in vocab: {len(not_found_words)} words")

    if found_words:
        print("\nMapped words:")
        for word, index, pickle_file, video_file in found_words:
            print(f"  {word} (index {index})")
            print(f"    Pickle: {pickle_file}")
            print(f"    Video:  {video_file}")

    if not_found_words:
        print(f"\nWords not in vocabulary: {', '.join(not_found_words)}")

    # Video concatenation
    if video_files_for_concat and any(v not in ["VIDEO_NOT_FOUND", "VIDEO_DIR_NOT_FOUND", "VIDEO_SEARCH_ERROR", "NOT_FOUND"] for v in video_files_for_concat):
        print("\n" + "=" * 70)
        print("VIDEO CONCATENATION:")

        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_sentence = "".join(c if c.isalnum() or c in ' -_' else '' for c in sentence).replace(' ', '_')
        output_filename = f"sentence_{safe_sentence}_{timestamp}.mp4"
        output_path = os.path.join(test_dir, output_filename)

        # Ensure test directory exists
        os.makedirs(test_dir, exist_ok=True)

        # Concatenate videos
        success = concatenate_videos(video_files_for_concat, output_path, video_dir)

        if success:
            print(f"🎉 Combined video saved: {output_path}")

            # Create pose files if requested
            if create_pose_files:
                # Convert video to pose file
                print("\n" + "=" * 70)
                print("POSE EXTRACTION:")

                pose_filename = output_filename.replace('.mp4', '.pose')
                pose_output_path = os.path.join(test_dir, pose_filename)

                pose_success = video_to_pose(output_path, pose_output_path)

                if pose_success:
                    print(f"🎉 Pose file created: {pose_output_path}")

                    # Create pose visualization video
                    print("\n" + "=" * 70)
                    print("POSE VISUALIZATION:")

                    viz_filename = output_filename.replace('.mp4', '_pose_viz.mp4')
                    viz_output_path = os.path.join(test_dir, viz_filename)

                    viz_success = visualize_pose(pose_output_path, viz_output_path)

                    if viz_success:
                        print(f"🎉 Pose visualization created: {viz_output_path}")
                        print("\n" + "=" * 70)
                        print("FINAL OUTPUTS:")
                        print(f"📹 Original concatenated video: {output_path}")
                        print(f"🤖 Pose data file: {pose_output_path}")
                        print(f"🎨 Pose visualization video: {viz_output_path}")
                    else:
                        print("❌ Failed to create pose visualization")
                else:
                    print("❌ Failed to extract pose from video")
            else:
                # Video only output
                print("\n" + "=" * 70)
                print("FINAL OUTPUT:")
                print(f"📹 Concatenated sentence video: {output_path}")
        else:
            print("❌ Failed to create combined video")
    else:
        print("\n⚠️  No valid videos found for concatenation")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="WLASL Sentence Video & Pose Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive video mode:
    python sentence_to_pickle.py

  CLI pose concatenation mode (default - uses all data):
    python sentence_to_pickle.py --sentence "DOG LIKE WALK" --pose-mode --num-glosses 50

  CLI pose concatenation mode (test split only - for evaluation):
    python sentence_to_pickle.py --sentence "DOG LIKE WALK" --pose-mode --num-glosses 50 --split test

  CLI video mode:
    python sentence_to_pickle.py --sentence "baby arrive before" --output-dir test_outputs
        """
    )

    parser.add_argument(
        '--sentence',
        type=str,
        help='Sentence to process (enables CLI mode, e.g., "DOG LIKE WALK")'
    )
    parser.add_argument(
        '--pose-mode',
        action='store_true',
        help='Use pose file concatenation instead of video concatenation'
    )
    parser.add_argument(
        '--pose-dir',
        type=str,
        help='Directory containing pose files (default: auto-detect from config)'
    )
    parser.add_argument(
        '--pickle-dir',
        type=str,
        help='Directory containing pickle files (default: auto-detect from config)'
    )
    parser.add_argument(
        '--video-dir',
        type=str,
        help='Directory containing video files (default: auto-detect from config)'
    )
    parser.add_argument(
        '--vocab-path',
        type=str,
        help='Path to vocabulary JSON file (default: auto-detect from config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for generated files (default: test_outputs/)'
    )
    parser.add_argument(
        '--sample-strategy',
        choices=['first', 'random'],
        default='first',
        help='Strategy when multiple samples exist for a gloss (default: first)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create pose visualization video (pose mode only)'
    )
    parser.add_argument(
        '--num-glosses',
        type=int,
        choices=[20, 50],
        default=20,
        help='Number of glosses in vocabulary (20 or 50, default: 20)'
    )
    parser.add_argument(
        '--split',
        choices=['train', 'test', 'val', 'all'],
        default='all',
        help='Dataset split to use for pose files (default: all - uses all available data)'
    )

    args = parser.parse_args()

    # Get configuration
    if CONFIG_AVAILABLE:
        config = get_config()
        data_root = Path(config.data_root)
        project_root = Path(config.project_root)
    else:
        # Fallback: use relative paths from script location
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        data_root = project_root / "datasets" / "wlasl_poses_complete"

    # Default configuration paths (using config system)
    default_vocab_path = str(project_root / "models" / "openhands-modernized" / "bin" / "wlasl_20_class_model" / "class_index_mapping.json")
    default_pickle_dir = str(data_root / "pickle_files")
    default_video_dir = str(data_root / "videos") if (data_root / "videos").exists() else str(project_root / "datasets" / "videos")
    default_pose_dir = str(data_root / "pose_files_by_gloss")
    default_test_dir = str(Path(__file__).parent / "test_outputs")  # Under project-utilities folder

    # Use args or defaults
    vocab_path = args.vocab_path or default_vocab_path
    pickle_dir = args.pickle_dir or default_pickle_dir
    video_dir = args.video_dir or default_video_dir
    pose_dir = args.pose_dir or default_pose_dir
    output_dir = args.output_dir or default_test_dir

    # Update vocab path based on num_glosses if using default
    if not args.vocab_path and args.num_glosses != 20:
        # Try to find the num_glosses specific vocab file
        vocab_path_obj = Path(vocab_path)
        vocab_dir = vocab_path_obj.parent
        vocab_file_pattern = f"class_index_mapping_{args.num_glosses}.json"
        alt_vocab = vocab_dir / vocab_file_pattern
        if alt_vocab.exists():
            vocab_path = str(alt_vocab)
        else:
            print(f"WARNING: Vocab file for {args.num_glosses} glosses not found at {alt_vocab}")
            print(f"Using default: {vocab_path}")

    # CLI Mode
    if args.sentence:
        if args.pose_mode:
            print("WLASL Pose Concatenation Mode (CLI)")
            print(f"Sentence: '{args.sentence}'")
            print(f"Vocabulary: {args.num_glosses} glosses")
            print(f"Dataset split: {args.split} ({'all data' if args.split == 'all' else f'{args.split} only'})")
            print(f"Output dir: {output_dir}")
            print(f"Sample strategy: {args.sample_strategy}")
            print(f"Visualization: {'Yes' if args.visualize else 'No'}")
            print(f"Pose directory: {pose_dir}")

            # Call pose mode function
            pose_file, metadata_file = process_sentence_pose_mode(
                sentence=args.sentence,
                pose_dir=pose_dir,
                output_dir=output_dir,
                sample_strategy=args.sample_strategy,
                visualize=args.visualize,
                split=args.split,
                num_glosses=args.num_glosses
            )

            if pose_file and metadata_file:
                print(f"\nSUCCESS: Files created:")
                print(f"  Pose file: {pose_file}")
                print(f"  Metadata: {metadata_file}")
                return 0
            else:
                print("\nFAILED: Could not create pose file")
                return 1

        else:
            # Video mode (existing functionality via CLI)
            print("WLASL Video Concatenation Mode (CLI)")
            print(f"Sentence: '{args.sentence}'")
            print(f"Vocabulary: {args.num_glosses} glosses")
            print(f"Output dir: {output_dir}")
            print()

            # Call existing process_sentence function
            process_sentence(
                sentence=args.sentence,
                vocab_path=vocab_path,
                pickle_dir=pickle_dir,
                video_dir=video_dir,
                test_dir=output_dir,
                create_pose_files=args.visualize
            )
            return 0

    # Interactive Mode (existing functionality)
    print("🎯 WLASL Sentence Video & Pose Processing Pipeline")
    print("Enter a sentence using words from the WLASL vocabulary")
    print("The program will find corresponding video files for each word and concatenate them.")
    print("Example: 'baby arrive before birthday ball'")
    print("Type 'quit' to exit\n")

    # Check dependencies
    dependencies_ok = True

    # Check FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg found - video concatenation available")
        else:
            print("⚠️  FFmpeg not working properly")
            dependencies_ok = False
    except FileNotFoundError:
        print("❌ FFmpeg not found - video concatenation will not work")
        print("💡 Install FFmpeg from: https://ffmpeg.org/download.html")
        dependencies_ok = False

    # Check video_to_pose
    try:
        result = subprocess.run(['video_to_pose', '--help'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ video_to_pose found - pose extraction available")
        else:
            print("⚠️  video_to_pose not working properly")
    except FileNotFoundError:
        print("❌ video_to_pose not found - pose extraction will not work")
        print("💡 Install with: pip install pose-format[mediapipe]")
        dependencies_ok = False

    # Check visualize_pose
    try:
        result = subprocess.run(['visualize_pose', '--help'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ visualize_pose found - pose visualization available")
        else:
            print("⚠️  visualize_pose not working properly")
    except FileNotFoundError:
        print("❌ visualize_pose not found - pose visualization will not work")
        print("💡 Install with: pip install pose-format[mediapipe]")
        dependencies_ok = False

    if not dependencies_ok:
        print("\n⚠️  Some dependencies are missing. The program will continue but some features may not work.")

    print(f"📁 Test folder configured: {test_dir}\n")

    while True:
        try:
            sentence = input("Enter sentence: ").strip()

            if sentence.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not sentence:
                print("Please enter a sentence")
                continue

            # Ask user for output preferences
            print("\nOutput options:")
            print("1. Video only (concatenated sentence video)")
            print("2. Video + Pose visualization (video + pose data + pose visualization)")

            while True:
                choice = input("Choose output type (1 or 2): ").strip()
                if choice in ['1', '2']:
                    break
                print("Please enter 1 or 2")

            create_pose_files = (choice == '2')

            process_sentence(sentence, vocab_path, pickle_dir, video_dir, test_dir, create_pose_files)
            print("\n" + "-" * 70 + "\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()