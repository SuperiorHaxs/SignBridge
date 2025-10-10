#!/usr/bin/env python3
"""
Generate class mapping JSON file from pose split directory.

This script creates a comprehensive mapping file that includes:
- List of all classes
- Video IDs for each class
- Reverse mapping from video ID to class
- File counts and statistics

Usage:
    python generate_class_mapping.py --split-dir <path_to_pose_split> --output <output.json>
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

def generate_class_mapping(split_dir, main_metadata_file):
    """Generate class mapping from pose split directory."""

    # Load main metadata
    with open(main_metadata_file, 'r') as f:
        main_metadata = json.load(f)

    # Get all classes from train directory
    train_dir = os.path.join(split_dir, 'train')
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    # Build class to videos mapping
    class_to_videos = {}
    video_to_class = {}
    class_file_counts = {}

    for split in ['train', 'val', 'test']:
        split_dir_path = os.path.join(split_dir, split)
        if not os.path.exists(split_dir_path):
            continue

        for class_name in classes:
            class_dir = os.path.join(split_dir_path, class_name)
            if not os.path.exists(class_dir):
                continue

            # Initialize class in mappings
            if class_name not in class_to_videos:
                class_to_videos[class_name] = []
                class_file_counts[class_name] = {'train': 0, 'val': 0, 'test': 0, 'total': 0}

            # Get all pose files
            pose_files = [f for f in os.listdir(class_dir) if f.endswith('.pose')]

            for pose_file in pose_files:
                # Extract video ID from filename
                video_id = pose_file.replace('.pose', '')

                # Add to mappings
                if video_id not in class_to_videos[class_name]:
                    class_to_videos[class_name].append(video_id)
                video_to_class[video_id] = class_name

                # Update counts
                class_file_counts[class_name][split] += 1
                class_file_counts[class_name]['total'] += 1

    # Sort video IDs for each class
    for class_name in class_to_videos:
        class_to_videos[class_name] = sorted(class_to_videos[class_name])

    # Calculate statistics
    total_videos = len(video_to_class)
    train_total = sum(counts['train'] for counts in class_file_counts.values())
    val_total = sum(counts['val'] for counts in class_file_counts.values())
    test_total = sum(counts['test'] for counts in class_file_counts.values())

    # Create mapping structure
    mapping = {
        'metadata': {
            'num_classes': len(classes),
            'total_videos': total_videos,
            'split_counts': {
                'train': train_total,
                'val': val_total,
                'test': test_total,
                'total': train_total + val_total + test_total
            },
            'source_directory': split_dir
        },
        'classes': classes,
        'class_to_videos': class_to_videos,
        'video_to_class': video_to_class,
        'class_file_counts': class_file_counts
    }

    return mapping


def main():
    parser = argparse.ArgumentParser(description='Generate class mapping JSON from pose split directory')
    parser.add_argument('--split-dir', required=True,
                       help='Path to pose split directory (contains train/val/test)')
    parser.add_argument('--output', required=True,
                       help='Output JSON file path')
    parser.add_argument('--metadata', default='C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/wlasl_poses_complete/video_to_gloss_mapping.json',
                       help='Path to main metadata file')

    args = parser.parse_args()

    print("=" * 70)
    print("GENERATING CLASS MAPPING JSON")
    print("=" * 70)
    print()
    print(f"Split directory: {args.split_dir}")
    print(f"Output file: {args.output}")
    print()

    # Generate mapping
    mapping = generate_class_mapping(args.split_dir, args.metadata)

    # Save to file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(mapping, f, indent=2)

    # Print summary
    print("Mapping generated successfully!")
    print()
    print(f"Classes: {mapping['metadata']['num_classes']}")
    print(f"Total videos: {mapping['metadata']['total_videos']}")
    print(f"Split counts:")
    print(f"  Train: {mapping['metadata']['split_counts']['train']}")
    print(f"  Val:   {mapping['metadata']['split_counts']['val']}")
    print(f"  Test:  {mapping['metadata']['split_counts']['test']}")
    print()
    print(f"Saved to: {args.output}")
    print()


if __name__ == "__main__":
    main()
