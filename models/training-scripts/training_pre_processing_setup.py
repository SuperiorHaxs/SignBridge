#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training_pre_processing_setup.py

Pre-processing and verification script for ASL training setup.
Checks and prepares dataset structure, mappings, and configurations for training.

Usage:
    python training_pre_processing_setup.py --classes 50 --verify-only
    python training_pre_processing_setup.py --classes 100 --create-splits
"""

import os
import sys
import json
import pickle
import shutil
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import configuration
from config import get_config

# Import utilities from dataset-utilities using proper package imports
# Project root is already in sys.path (line 32), so we can import directly
try:
    # Note: Python package imports use dots, filesystem uses hyphens
    # We need to handle 'dataset-utilities' directory name
    import importlib.util

    # Import splitting utilities
    split_module_path = project_root / "dataset-utilities" / "dataset-splitting" / "split_pose_files_nclass.py"
    spec = importlib.util.spec_from_file_location("split_pose_files_nclass", split_module_path)
    split_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(split_module)

    # Extract functions
    get_class_frequencies = split_module.get_class_frequencies
    select_classes_incrementally = split_module.select_classes_incrementally
    load_class_mapping_if_exists = split_module.load_class_mapping_if_exists
    save_class_mapping = split_module.save_class_mapping
    get_pose_files_by_class = split_module.get_pose_files_by_class
    split_files = split_module.split_files
    create_split_directories = split_module.create_split_directories
    copy_files_to_splits = split_module.copy_files_to_splits
    load_metadata = split_module.load_metadata

    SPLITTING_UTILS_AVAILABLE = True
except Exception as e:
    SPLITTING_UTILS_AVAILABLE = False
    print(f"WARNING: Could not import splitting utilities: {e}")

try:
    # Import conversion utilities
    converter_module_path = project_root / "dataset-utilities" / "conversion" / "pose_to_pickle_converter.py"
    spec = importlib.util.spec_from_file_location("pose_to_pickle_converter", converter_module_path)
    converter_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter_module)

    # Extract functions
    pose_to_numpy = converter_module.pose_to_numpy
    convert_pose_directory = converter_module.convert_pose_directory

    CONVERSION_UTILS_AVAILABLE = True
except Exception as e:
    CONVERSION_UTILS_AVAILABLE = False
    print(f"WARNING: Could not import conversion utilities: {e}")


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_status(message, status="INFO"):
    """Print a status message."""
    symbols = {
        "INFO": "[INFO]",
        "SUCCESS": "[OK]",
        "WARNING": "[WARN]",
        "ERROR": "[ERROR]",
        "CHECK": "[CHECK]"
    }
    symbol = symbols.get(status, "[*]")
    print(f"{symbol} {message}")


def check_directory_structure(num_classes, config):
    """Check if dataset directory structure exists for specified class count."""
    print_section(f"Step 1: Verify {num_classes}-Class Directory Structure")

    dataset_paths = config.dataset_splits.get(num_classes)
    if not dataset_paths:
        print_status(f"No configuration found for {num_classes} classes", "ERROR")
        return False

    required_dirs = {
        'train': dataset_paths['train_original'],
        'test': dataset_paths['test'],
        'val': dataset_paths.get('val')
    }

    all_exist = True
    for split_name, split_path in required_dirs.items():
        if split_path is None:
            print_status(f"{split_name.upper()}: Not configured (optional)", "WARNING")
            continue

        split_path_obj = Path(split_path)
        if split_path_obj.exists():
            # Count classes in this split
            class_count = len([d for d in split_path_obj.iterdir() if d.is_dir()])
            print_status(f"{split_name.upper()}: Found at {split_path}", "SUCCESS")
            print_status(f"  └─ Contains {class_count} class directories", "INFO")
        else:
            print_status(f"{split_name.upper()}: NOT FOUND at {split_path}", "ERROR")
            all_exist = False

    if all_exist:
        print_status(f"All required directories exist for {num_classes} classes", "SUCCESS")
    else:
        print_status(f"Missing directories for {num_classes} classes", "ERROR")

    return all_exist


def list_classes(num_classes, config):
    """List all classes in the training directory."""
    print_section(f"Step 2: List {num_classes} Classes")

    dataset_paths = config.dataset_splits.get(num_classes)
    if not dataset_paths:
        print_status(f"No configuration found for {num_classes} classes", "ERROR")
        return []

    train_path = Path(dataset_paths['train_original'])
    if not train_path.exists():
        print_status(f"Train directory not found: {train_path}", "ERROR")
        return []

    classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])

    print_status(f"Found {len(classes)} classes:", "SUCCESS")
    print()

    # Print in columns
    cols = 4
    for i in range(0, len(classes), cols):
        row = classes[i:i+cols]
        print("  " + "  ".join(f"{cls:15}" for cls in row))

    return classes


def count_samples_per_class(num_classes, config):
    """Count number of samples in each class."""
    print_section(f"Step 3: Count Samples Per Class")

    dataset_paths = config.dataset_splits.get(num_classes)
    train_path = Path(dataset_paths['train_original'])

    if not train_path.exists():
        print_status(f"Train directory not found", "ERROR")
        return {}

    sample_counts = {}
    total_samples = 0

    for class_dir in sorted(train_path.iterdir()):
        if class_dir.is_dir():
            pkl_files = list(class_dir.glob("*.pkl"))
            count = len(pkl_files)
            sample_counts[class_dir.name] = count
            total_samples += count

    # Print statistics
    if sample_counts:
        avg_samples = total_samples / len(sample_counts)
        min_samples = min(sample_counts.values())
        max_samples = max(sample_counts.values())

        print_status(f"Total samples: {total_samples}", "SUCCESS")
        print_status(f"Average per class: {avg_samples:.1f}", "INFO")
        print_status(f"Range: {min_samples} - {max_samples} samples", "INFO")
        print()

        # Show top 5 and bottom 5
        sorted_counts = sorted(sample_counts.items(), key=lambda x: x[1], reverse=True)

        print("  Classes with most samples:")
        for cls, count in sorted_counts[:5]:
            print(f"    {cls:20} {count:4} samples")

        print()
        print("  Classes with least samples:")
        for cls, count in sorted_counts[-5:]:
            print(f"    {cls:20} {count:4} samples")

    return sample_counts


def check_augmented_pool(config):
    """Check if augmented pool exists and has index."""
    print_section("Step 4: Check Augmented Pool")

    pool_path = config.augmented_pool_pickle
    index_path = config.augmented_pool_index

    if not pool_path.exists():
        print_status(f"Augmented pool NOT FOUND at {pool_path}", "ERROR")
        return False

    print_status(f"Augmented pool found at {pool_path}", "SUCCESS")

    # Count classes in pool
    class_count = len([d for d in pool_path.iterdir() if d.is_dir()])
    print_status(f"  └─ Contains {class_count} class directories", "INFO")

    # Check index
    if index_path.exists():
        print_status(f"Index file found at {index_path}", "SUCCESS")

        try:
            with open(index_path, 'r') as f:
                index = json.load(f)

            total_files = sum(len(files) for files in index.values())
            print_status(f"  └─ Index contains {len(index)} classes", "INFO")
            print_status(f"  └─ Total indexed files: {total_files}", "INFO")
        except Exception as e:
            print_status(f"Error reading index: {e}", "ERROR")
            return False
    else:
        print_status(f"Index file NOT FOUND at {index_path}", "ERROR")
        print_status("Index is required for fast augmented data loading", "WARNING")
        return False

    return True


def verify_class_coverage(num_classes, config):
    """Verify that augmented pool covers all training classes."""
    print_section("Step 5: Verify Class Coverage in Augmented Pool")

    # Get training classes
    dataset_paths = config.dataset_splits.get(num_classes)
    train_path = Path(dataset_paths['train_original'])
    train_classes = set([d.name.upper() for d in train_path.iterdir() if d.is_dir()])

    print_status(f"Training has {len(train_classes)} classes", "INFO")

    # Load augmented pool index
    index_path = config.augmented_pool_index
    if not index_path.exists():
        print_status("Cannot verify - index file missing", "ERROR")
        return False

    with open(index_path, 'r') as f:
        augmented_index = json.load(f)

    augmented_classes = set(augmented_index.keys())
    print_status(f"Augmented pool has {len(augmented_classes)} classes", "INFO")

    # Check coverage
    missing_classes = train_classes - augmented_classes
    extra_classes = augmented_classes - train_classes

    if missing_classes:
        print_status(f"Missing {len(missing_classes)} classes in augmented pool:", "ERROR")
        for cls in sorted(missing_classes):
            print(f"    - {cls}")
    else:
        print_status("All training classes found in augmented pool", "SUCCESS")

    # Count augmented samples per training class
    print()
    print("  Augmented samples per class:")
    sample_counts = {}
    for cls in sorted(train_classes):
        if cls in augmented_index:
            count = len(augmented_index[cls])
            sample_counts[cls] = count
            print(f"    {cls:20} {count:4} augmented samples")

    if sample_counts:
        avg = sum(sample_counts.values()) / len(sample_counts)
        print()
        print_status(f"Average augmented samples per class: {avg:.1f}", "INFO")

    return len(missing_classes) == 0


def create_class_mapping(num_classes, config):
    """Create or verify class mapping file."""
    print_section("Step 6: Class Mapping File")

    dataset_paths = config.dataset_splits.get(num_classes)
    mapping_path = dataset_paths.get('class_mapping')

    if not mapping_path:
        print_status("No class mapping path configured", "WARNING")
        return None

    mapping_path = Path(mapping_path)

    # Get classes from train directory
    train_path = Path(dataset_paths['train_original'])
    classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])

    if mapping_path.exists():
        print_status(f"Class mapping found at {mapping_path}", "SUCCESS")

        # Verify it matches current classes
        try:
            with open(mapping_path, 'r') as f:
                existing_mapping = json.load(f)

            existing_classes = set(existing_mapping.get('classes', []))
            current_classes = set(classes)

            if existing_classes == current_classes:
                print_status(f"  └─ Mapping is up-to-date ({len(classes)} classes)", "SUCCESS")
            else:
                print_status(f"  └─ Mapping is outdated", "WARNING")
                print_status(f"     Current: {len(current_classes)} classes", "INFO")
                print_status(f"     Mapped:  {len(existing_classes)} classes", "INFO")
        except Exception as e:
            print_status(f"Error reading mapping: {e}", "ERROR")
    else:
        print_status(f"Class mapping NOT FOUND at {mapping_path}", "WARNING")
        print_status("Mapping will be created during training if needed", "INFO")

    return classes


# ============================================================================
# ORCHESTRATION - Create complete dataset splits using imported utilities
# ============================================================================

def create_dataset_splits(num_classes, config):
    """Create complete dataset splits from .pose files using imported utilities."""
    if not SPLITTING_UTILS_AVAILABLE or not CONVERSION_UTILS_AVAILABLE:
        print_status("Required utilities not available", "ERROR")
        return False

    print_section(f"Creating {num_classes}-Class Dataset Splits")

    # Determine paths
    output_base = str(config.dataset_root / f"dataset_splits/{num_classes}_classes")
    pose_split_dir = os.path.join(output_base, "original", f"pose_split_{num_classes}_class")
    pkl_base_dir = os.path.join(output_base, "original", f"pickle_split_{num_classes}_class")
    class_mapping_path = os.path.join(output_base, "class_mapping.json")

    # Determine previous tier's class mapping path
    previous_mapping_path = None
    if num_classes == 50:
        previous_mapping_path = os.path.join(str(config.dataset_root), "dataset_splits/20_classes/class_mapping.json")
    elif num_classes == 100:
        previous_mapping_path = os.path.join(str(config.dataset_root), "dataset_splits/50_classes/class_mapping.json")

    # Select classes using imported utility
    print_status("Selecting classes using frequency-based incremental strategy", "INFO")
    target_classes = select_classes_incrementally(
        num_classes=num_classes,
        metadata_file=str(config.video_to_gloss_mapping),
        class_mapping_path=class_mapping_path,
        previous_mapping_path=previous_mapping_path
    )
    print()

    # Step 1: Split .pose files into train/val/test
    print_status("Step 1: Splitting .pose files into train/val/test", "INFO")

    pose_files_dir = str(config.pose_files_dir)
    metadata_file = str(config.video_to_gloss_mapping)

    print_status(f"Loading metadata from {metadata_file}", "INFO")
    video_to_gloss = load_metadata(target_classes)  # Uses METADATA_FILE from imported module
    print_status(f"Found {len(video_to_gloss)} video-to-gloss mappings", "INFO")

    print_status(f"Grouping pose files by class", "INFO")
    files_by_class = get_pose_files_by_class(video_to_gloss)  # Uses POSE_FILES_DIR from imported module
    total_files = sum(len(files) for files in files_by_class.values())
    print_status(f"Found {total_files} pose files across {len(files_by_class)} classes", "INFO")

    print_status("Creating split directories", "INFO")
    create_split_directories(pose_split_dir, target_classes)

    print_status("Copying files to splits", "INFO")
    stats = copy_files_to_splits(files_by_class, pose_split_dir)  # Uses POSE_FILES_DIR from imported module
    print()
    print_status(f"Train: {stats['train']} files ({stats['train']/total_files*100:.1f}%)", "INFO")
    print_status(f"Val:   {stats['val']} files ({stats['val']/total_files*100:.1f}%)", "INFO")
    print_status(f"Test:  {stats['test']} files ({stats['test']/total_files*100:.1f}%)", "INFO")
    print()

    # Step 2: Convert .pose to .pkl for each split
    print_status("Step 2: Converting .pose to .pkl format", "INFO")

    for split_name in ['train', 'val', 'test']:
        print_status(f"Converting {split_name} split", "INFO")
        pose_split_path = os.path.join(pose_split_dir, split_name)
        pkl_split_path = os.path.join(pkl_base_dir, split_name)

        if not os.path.exists(pose_split_path):
            print_status(f"  {split_name} split not found, skipping", "WARNING")
            continue

        # Use imported conversion utility
        convert_pose_directory(pose_split_path, pkl_split_path)

    print()
    print_status("Dataset splits created successfully!", "SUCCESS")
    print_status(f"Pose splits: {pose_split_dir}", "INFO")
    print_status(f"Pickle splits: {pkl_base_dir}", "INFO")
    print_status(f"Class mapping: {class_mapping_path}", "INFO")
    print()
    print_status("IMPORTANT: Update config/settings.json to point to the new pickle splits", "WARNING")
    print()

    return True


def print_summary(num_classes, all_checks_passed):
    """Print final summary."""
    print_section("Verification Summary")

    if all_checks_passed:
        print_status(f"{num_classes}-class setup is READY for training", "SUCCESS")
        print()
        print("  You can start training with:")
        print(f"    python train_asl.py --classes {num_classes} --architecture openhands --model-size small --dataset augmented")
    else:
        print_status(f"{num_classes}-class setup has ISSUES", "ERROR")
        print_status("Please resolve the errors above before training", "WARNING")

    print()


def main():
    """Main verification and setup workflow."""
    parser = argparse.ArgumentParser(
        description='Verify and prepare dataset for ASL training'
    )
    parser.add_argument(
        '--classes',
        type=int,
        required=True,
        choices=[20, 50, 100],
        help='Number of classes to verify/create (20, 50, or 100)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing splits, do not create new ones'
    )
    parser.add_argument(
        '--create-splits',
        action='store_true',
        help='Create dataset splits if they do not exist'
    )

    args = parser.parse_args()

    # Determine mode
    if args.create_splits and args.verify_only:
        print("ERROR: Cannot use both --create-splits and --verify-only")
        sys.exit(1)

    mode = "Create Splits" if args.create_splits else ("Verify Only" if args.verify_only else "Verify and Prepare")

    print_section(f"ASL Training Setup - {args.classes} Classes")
    print()
    print(f"Mode: {mode}")
    print()

    # Load configuration
    try:
        config = get_config()
        print_status(f"Configuration loaded", "SUCCESS")
        print_status(f"  └─ Data root: {config.data_root}", "INFO")
    except Exception as e:
        print_status(f"Failed to load configuration: {e}", "ERROR")
        sys.exit(1)

    # If create-splits mode, create the splits
    if args.create_splits:
        success = create_dataset_splits(args.classes, config)
        if success:
            print_status("NEXT STEP: Update config/settings.json with the new split paths", "WARNING")
            print_status("Then run verification with: --verify-only", "INFO")
            sys.exit(0)
        else:
            print_status("Failed to create dataset splits", "ERROR")
            sys.exit(1)

    # Otherwise, run verification steps
    checks_passed = []

    # Step 1: Check directory structure
    checks_passed.append(check_directory_structure(args.classes, config))

    # If verification failed and not verify-only, offer to create splits
    if not checks_passed[0] and not args.verify_only:
        print()
        print_status("Dataset splits not found", "WARNING")
        print_status("To create splits, run with --create-splits flag:", "INFO")
        print(f"  python training_pre_processing_setup.py --classes {args.classes} --create-splits")
        print()
        sys.exit(1)

    # Step 2: List classes
    classes = list_classes(args.classes, config)
    checks_passed.append(len(classes) > 0)

    # Step 3: Count samples
    sample_counts = count_samples_per_class(args.classes, config)
    checks_passed.append(len(sample_counts) > 0)

    # Step 4: Check augmented pool
    checks_passed.append(check_augmented_pool(config))

    # Step 5: Verify class coverage
    checks_passed.append(verify_class_coverage(args.classes, config))

    # Step 6: Check class mapping
    create_class_mapping(args.classes, config)

    # Final summary
    all_passed = all(checks_passed)
    print_summary(args.classes, all_passed)

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
