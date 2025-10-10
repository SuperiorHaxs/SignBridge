#!/usr/bin/env python3
"""
sentence_to_pickle.py
Maps sentence words to vocab indices, finds corresponding pickle/video files,
and concatenates videos into a combined sentence video
"""

import json
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime

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

        print(f"🎨 Creating pose visualization: {pose_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0 and os.path.exists(video_output_path):
            print(f"✅ Pose visualization successful: {video_output_path}")
            return True
        else:
            print(f"❌ Pose visualization failed: {result.stderr}")
            return False

    except FileNotFoundError:
        print("❌ visualize_pose not found. Please install pose-format and add to PATH.")
        print("💡 Install with: pip install pose-format[mediapipe]")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Pose visualization timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error during pose visualization: {e}")
        return False

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
    # Configuration - Using 20-class experiment vocabulary
    vocab_path = "C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/wlasl_poses_complete/class_index_mapping_20.json"
    pickle_dir = "C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/wlasl_poses_complete/pickle_files"
    video_dir = "C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/videos"
    test_dir = "C:/Users/padwe/OneDrive/WLASL-proj/OpenHands-Modernized/experiments/20-classes/test"  # Test folder for combined videos

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