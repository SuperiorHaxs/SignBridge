#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training_pre_processing_setup.py

End-to-end preprocessing script for ASL training setup.
Automates dataset splitting, augmentation, and configuration for training.

Features:
    - Creates pose and pickle dataset splits (train/val/test)
    - Generates 40x augmented variants with occlusion techniques
    - Automatically updates config/settings.json
    - Detects and resumes from incomplete setups
    - Force-fresh mode for consistent regeneration

Usage:
    # Complete end-to-end setup (recommended)
    python training_pre_processing_setup.py --classes 100 --setup

    # Force fresh start (ensures all classes have 40x augmentation)
    python training_pre_processing_setup.py --classes 100 --setup --force-fresh

    # Only verify existing setup
    python training_pre_processing_setup.py --classes 50 --verify-only

    # Legacy: only create splits (no augmentation)
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
from datetime import datetime

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

try:
    # Import augmentation utilities
    augmentation_module_path = project_root / "dataset-utilities" / "augmentation" / "generate_75pt_augmented_dataset.py"
    spec = importlib.util.spec_from_file_location("generate_75pt_augmented_dataset", augmentation_module_path)
    augmentation_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(augmentation_module)

    # Extract configuration and functions
    NUM_AUGMENTATIONS = augmentation_module.NUM_AUGMENTATIONS
    AUGMENTATION_CONFIG = augmentation_module.AUGMENTATION_CONFIG
    load_class_mapping = augmentation_module.load_class_mapping
    get_classes_from_directory = augmentation_module.get_classes_from_directory
    load_and_extract_75pt = augmentation_module.load_and_extract_75pt
    apply_augmentation_pickle = augmentation_module.apply_augmentation_pickle
    save_augmented_pickle = augmentation_module.save_augmented_pickle

    AUGMENTATION_UTILS_AVAILABLE = True
except Exception as e:
    AUGMENTATION_UTILS_AVAILABLE = False
    print(f"WARNING: Could not import augmentation utilities: {e}")


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
# STATE DETECTION AND MANAGEMENT
# ============================================================================

def check_setup_state(num_classes, config):
    """
    Check the current state of setup for specified class count.

    Returns:
        dict: Status of each setup component
            {
                'pose_splits': bool,
                'pickle_splits': bool,
                'augmented_pool': bool,
                'augmented_index': bool,
                'config_updated': bool
            }
    """
    state = {
        'pose_splits': False,
        'pickle_splits': False,
        'augmented_pool': False,
        'augmented_index': False,
        'config_updated': False
    }

    # Check pose splits
    pose_split_dir = config.dataset_root / f"dataset_splits/{num_classes}_classes/original/pose_split_{num_classes}_class"
    if pose_split_dir.exists():
        train_exists = (pose_split_dir / "train").exists()
        val_exists = (pose_split_dir / "val").exists()
        test_exists = (pose_split_dir / "test").exists()
        state['pose_splits'] = train_exists and val_exists and test_exists

    # Check pickle splits
    pkl_split_dir = config.dataset_root / f"dataset_splits/{num_classes}_classes/original/pickle_split_{num_classes}_class"
    if pkl_split_dir.exists():
        train_exists = (pkl_split_dir / "train").exists()
        val_exists = (pkl_split_dir / "val").exists()
        test_exists = (pkl_split_dir / "test").exists()
        state['pickle_splits'] = train_exists and val_exists and test_exists

    # Check augmented pool (check if classes exist in pool)
    augmented_pool = config.augmented_pool_pickle
    if augmented_pool.exists():
        # Load class mapping to get expected classes
        class_mapping_path = config.dataset_root / f"dataset_splits/{num_classes}_classes/class_mapping.json"
        if class_mapping_path.exists():
            try:
                with open(class_mapping_path, 'r') as f:
                    mapping = json.load(f)
                expected_classes = mapping.get('classes', [])

                # Check if all classes exist in augmented pool
                classes_in_pool = [d.name for d in augmented_pool.iterdir() if d.is_dir()]
                state['augmented_pool'] = all(cls.upper() in [c.upper() for c in classes_in_pool] for cls in expected_classes)
            except:
                state['augmented_pool'] = False

    # Check augmented index
    index_path = config.augmented_pool_index
    state['augmented_index'] = index_path.exists()

    # Check if config is updated
    settings_path = config.project_root / "config" / "settings.json"
    if settings_path.exists():
        try:
            with open(settings_path, 'r') as f:
                settings = json.load(f)
            dataset_splits = settings.get('dataset_splits', {})
            state['config_updated'] = str(num_classes) in dataset_splits or num_classes in dataset_splits
        except:
            state['config_updated'] = False

    return state


def print_setup_state(num_classes, state):
    """Print the current setup state in a formatted way."""
    print_section(f"Setup State - {num_classes} Classes")

    status_map = {True: "✓ Complete", False: "✗ Missing"}

    print_status(f"Pose splits (train/val/test):        {status_map[state['pose_splits']]}",
                "SUCCESS" if state['pose_splits'] else "ERROR")
    print_status(f"Pickle splits (train/val/test):      {status_map[state['pickle_splits']]}",
                "SUCCESS" if state['pickle_splits'] else "ERROR")
    print_status(f"Augmented dataset (40x variants):    {status_map[state['augmented_pool']]}",
                "SUCCESS" if state['augmented_pool'] else "ERROR")
    print_status(f"Augmentation index:                  {status_map[state['augmented_index']]}",
                "SUCCESS" if state['augmented_index'] else "ERROR")
    print_status(f"Config/settings.json updated:        {status_map[state['config_updated']]}",
                "SUCCESS" if state['config_updated'] else "ERROR")

    print()

    # Determine what needs to be done
    needs_work = [k for k, v in state.items() if not v]
    if needs_work:
        print_status(f"Status: Incomplete ({len(needs_work)} steps remaining)", "WARNING")
    else:
        print_status("Status: Complete and ready for training!", "SUCCESS")


# ============================================================================
# AUGMENTATION GENERATION
# ============================================================================

def generate_augmented_dataset(num_classes, config, force=False):
    """
    Generate augmented dataset using 40x variants with occlusion.

    Args:
        num_classes: Number of classes to augment
        config: PathConfig instance
        force: If True, regenerate even if already exists

    Returns:
        bool: Success status
    """
    if not AUGMENTATION_UTILS_AVAILABLE:
        print_status("Augmentation utilities not available", "ERROR")
        return False

    print_section(f"Step 4: Generate Augmented Dataset ({num_classes} classes)")

    # Paths
    pkl_train_dir = config.dataset_root / f"dataset_splits/{num_classes}_classes/original/pickle_split_{num_classes}_class/train"
    augmented_pool = config.augmented_pool_pickle
    index_path = config.augmented_pool_index

    # Check if pickle train directory exists
    if not pkl_train_dir.exists():
        print_status(f"Pickle train directory not found: {pkl_train_dir}", "ERROR")
        return False

    # Get class list
    class_mapping_path = config.dataset_root / f"dataset_splits/{num_classes}_classes/class_mapping.json"
    try:
        with open(class_mapping_path, 'r') as f:
            mapping = json.load(f)
        target_classes = mapping.get('classes', [])
    except:
        print_status("Could not load class mapping", "ERROR")
        return False

    print_status(f"Target classes: {len(target_classes)}", "INFO")
    print_status(f"Augmentation variants: {NUM_AUGMENTATIONS} (40x)", "INFO")
    print_status(f"Output directory: {augmented_pool}", "INFO")
    print()

    # Create output directory
    augmented_pool.mkdir(parents=True, exist_ok=True)

    # Track statistics
    total_files = 0
    total_augmented = 0
    skipped_files = 0

    # Process each class
    for class_name in target_classes:
        class_upper = class_name.upper()
        print(f"Processing: {class_name}")

        input_class_dir = pkl_train_dir / class_name
        output_class_dir = augmented_pool / class_upper

        if not input_class_dir.exists():
            print(f"  WARNING: Class directory not found: {input_class_dir}")
            continue

        # Create output class directory
        output_class_dir.mkdir(parents=True, exist_ok=True)

        # Get all pickle files in this class
        input_files = sorted([f for f in input_class_dir.iterdir() if f.suffix == '.pkl'])

        if not input_files:
            print(f"  WARNING: No pickle files found")
            continue

        print(f"  Files: {len(input_files)}")

        # Process each file
        for input_file in tqdm(input_files, desc=f"  {class_name}", ncols=80):
            # Check if already augmented (unless force=True)
            base_name = input_file.stem
            first_aug_filename = f"aug_00_{base_name}.pkl"

            if not force and (output_class_dir / first_aug_filename).exists():
                # Check if it has all 40 variants
                variant_count = len(list(output_class_dir.glob(f"aug_*_{base_name}.pkl")))
                if variant_count >= NUM_AUGMENTATIONS:
                    skipped_files += 1
                    total_augmented += NUM_AUGMENTATIONS
                    total_files += 1
                    continue
                else:
                    # Has some but not all variants, regenerate
                    print(f"    Incomplete augmentation for {base_name} ({variant_count}/{NUM_AUGMENTATIONS}), regenerating...")

            try:
                # Load and extract 75-point data
                pose_data, metadata = load_and_extract_75pt(str(input_file))

                # Generate all 40 augmented variants
                for aug_id in range(NUM_AUGMENTATIONS):
                    # Apply augmentation
                    augmented_pose = apply_augmentation_pickle(pose_data, aug_id)

                    # Create output filename
                    output_filename = f"aug_{aug_id:02d}_{base_name}.pkl"
                    output_path = output_class_dir / output_filename

                    # Save augmented pickle
                    save_augmented_pickle(augmented_pose, metadata, str(output_path), aug_id)

                    total_augmented += 1

                total_files += 1

            except Exception as e:
                print(f"\n  ERROR processing {input_file.name}: {e}")
                continue

        print(f"  Completed: {len(input_files)} files -> {len(input_files) * NUM_AUGMENTATIONS} augmented")

    # Generate index file
    print()
    print_status("Generating augmentation index...", "INFO")

    index_data = {}
    for class_dir in augmented_pool.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            pkl_files = [f.name for f in class_dir.glob("*.pkl")]
            index_data[class_name] = sorted(pkl_files)

    try:
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        print_status(f"Index saved: {index_path}", "SUCCESS")
    except Exception as e:
        print_status(f"Error saving index: {e}", "ERROR")
        return False

    # Print summary
    print()
    print_status("="*70, "INFO")
    print_status(f"Augmentation complete!", "SUCCESS")
    print_status(f"Total original files: {total_files}", "INFO")
    print_status(f"Total augmented files: {total_augmented}", "INFO")
    print_status(f"Skipped (already done): {skipped_files}", "INFO")
    print_status(f"Augmentation factor: {NUM_AUGMENTATIONS}x (40 variants)", "INFO")
    print_status("="*70, "INFO")
    print()

    return True


# ============================================================================
# CONFIGURATION FILE MANAGEMENT
# ============================================================================

def update_config_file(num_classes, config):
    """
    Update config/settings.json with paths for specified class count.

    Args:
        num_classes: Number of classes
        config: PathConfig instance

    Returns:
        bool: Success status
    """
    print_section(f"Step 5: Update Configuration File")

    settings_path = config.project_root / "config" / "settings.json"

    # Create backup first
    backup_path = settings_path.with_suffix(f".json.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    try:
        shutil.copy2(settings_path, backup_path)
        print_status(f"Backup created: {backup_path.name}", "SUCCESS")
    except Exception as e:
        print_status(f"Warning: Could not create backup: {e}", "WARNING")

    # Load existing settings
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
    except Exception as e:
        print_status(f"Error reading config file: {e}", "ERROR")
        return False

    # Prepare new paths (relative to data_root)
    new_paths = {
        "train_original": f"dataset_splits/{num_classes}_classes/original/pickle_split_{num_classes}_class/train",
        "val": f"dataset_splits/{num_classes}_classes/original/pickle_split_{num_classes}_class/val",
        "test": f"dataset_splits/{num_classes}_classes/original/pickle_split_{num_classes}_class/test",
        "train_augmented": "../augmented_pool/pickle",  # Relative to data_root
        "class_mapping": f"dataset_splits/{num_classes}_classes/class_mapping.json"
    }

    # Update or create dataset_splits section
    if 'dataset_splits' not in settings:
        settings['dataset_splits'] = {}

    # Use string key for JSON compatibility
    settings['dataset_splits'][str(num_classes)] = new_paths

    # Write updated settings
    try:
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)
        print_status(f"Config updated: {settings_path}", "SUCCESS")
        print()
        print_status(f"Added paths for {num_classes} classes:", "INFO")
        for key, value in new_paths.items():
            print(f"    {key}: {value}")
        print()
        return True
    except Exception as e:
        print_status(f"Error writing config file: {e}", "ERROR")
        return False


# ============================================================================
# CLEANUP FOR FRESH START
# ============================================================================

def cleanup_for_fresh_start(num_classes, config):
    """
    Clean up existing pickle splits and augmented data for fresh regeneration.
    Preserves pose splits (source of truth).

    Args:
        num_classes: Number of classes
        config: PathConfig instance

    Returns:
        bool: Success status
    """
    print_section(f"Cleanup for Fresh Start - {num_classes} Classes")

    items_to_delete = []

    # Pickle splits
    pkl_split_dir = config.dataset_root / f"dataset_splits/{num_classes}_classes/original/pickle_split_{num_classes}_class"
    if pkl_split_dir.exists():
        items_to_delete.append(("Pickle splits", pkl_split_dir))

    # Augmented pool classes
    class_mapping_path = config.dataset_root / f"dataset_splits/{num_classes}_classes/class_mapping.json"
    if class_mapping_path.exists():
        try:
            with open(class_mapping_path, 'r') as f:
                mapping = json.load(f)
            target_classes = mapping.get('classes', [])

            augmented_pool = config.augmented_pool_pickle
            for class_name in target_classes:
                class_upper = class_name.upper()
                class_dir = augmented_pool / class_upper
                if class_dir.exists():
                    items_to_delete.append((f"Augmented class '{class_name}'", class_dir))
        except:
            pass

    # Augmented index (will be regenerated)
    index_path = config.augmented_pool_index
    if index_path.exists():
        items_to_delete.append(("Augmentation index", index_path))

    if not items_to_delete:
        print_status("Nothing to clean up (fresh state)", "SUCCESS")
        return True

    # Show what will be deleted
    print_status(f"The following items will be DELETED:", "WARNING")
    for name, path in items_to_delete:
        print(f"  - {name}: {path}")
    print()
    print_status("Pose splits will be PRESERVED (source of truth)", "INFO")
    print()

    # Confirm
    response = input("Continue with cleanup? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print_status("Cleanup cancelled", "WARNING")
        return False

    # Delete items
    print()
    deleted_count = 0
    for name, path in items_to_delete:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print_status(f"Deleted: {name}", "SUCCESS")
            deleted_count += 1
        except Exception as e:
            print_status(f"Error deleting {name}: {e}", "ERROR")

    print()
    print_status(f"Cleanup complete: {deleted_count}/{len(items_to_delete)} items deleted", "SUCCESS")
    return True


# ============================================================================
# COMPLETE SETUP ORCHESTRATION
# ============================================================================

def run_complete_setup(num_classes, config, resume=True, force_fresh=False):
    """
    Run complete end-to-end setup for specified class count.

    Steps:
        1. Class selection (from pose files)
        2. Split pose files (train/val/test)
        3. Convert pose to pickle (all 3 splits)
        4. Generate augmented dataset (40x variants)
        5. Update config/settings.json
        6. Verify complete setup

    Args:
        num_classes: Number of classes
        config: PathConfig instance
        resume: If True, skip completed steps
        force_fresh: If True, delete and restart from scratch

    Returns:
        bool: Success status
    """
    print_section(f"Complete Setup - {num_classes} Classes")
    print()
    print(f"Mode: {'Force Fresh (Delete & Restart)' if force_fresh else 'Resume (Skip completed steps)'}")
    print()

    # Step 0: Cleanup if force_fresh
    if force_fresh:
        if not cleanup_for_fresh_start(num_classes, config):
            print_status("Setup cancelled", "ERROR")
            return False

    # Check current state
    state = check_setup_state(num_classes, config)
    print_setup_state(num_classes, state)

    # Determine which steps to run
    steps_to_run = []

    if not state['pose_splits'] or not resume:
        steps_to_run.extend([1, 2])  # Class selection and pose splitting

    if not state['pickle_splits'] or not resume:
        steps_to_run.append(3)  # Pickle conversion

    if not state['augmented_pool'] or not state['augmented_index'] or not resume:
        steps_to_run.append(4)  # Augmentation

    if not state['config_updated'] or not resume:
        steps_to_run.append(5)  # Config update

    if not steps_to_run:
        print_status("All steps already complete!", "SUCCESS")
        steps_to_run = [6]  # Just verify
    else:
        print()
        print_status(f"Steps to run: {steps_to_run}", "INFO")
        if resume:
            completed_steps = [i+1 for i in range(5) if i+1 not in steps_to_run]
            if completed_steps:
                print_status(f"Steps to skip (already done): {completed_steps}", "INFO")
        print()

    # Execute steps
    success = True

    # Steps 1-2: Create pose splits (handled by create_dataset_splits)
    if 1 in steps_to_run or 2 in steps_to_run or 3 in steps_to_run:
        print_status("Running Steps 1-3: Create pose splits and convert to pickle", "INFO")
        success = create_dataset_splits(num_classes, config)
        if not success:
            print_status("Failed to create dataset splits", "ERROR")
            return False
    else:
        print_status("Steps 1-3: Skipping (already complete)", "SUCCESS")

    # Step 4: Generate augmented dataset
    if 4 in steps_to_run:
        success = generate_augmented_dataset(num_classes, config, force=force_fresh)
        if not success:
            print_status("Failed to generate augmented dataset", "ERROR")
            return False
    else:
        print_status("Step 4: Skipping augmentation (already complete)", "SUCCESS")
        print()

    # Step 5: Update config
    if 5 in steps_to_run:
        success = update_config_file(num_classes, config)
        if not success:
            print_status("Failed to update config file", "ERROR")
            return False
    else:
        print_status("Step 5: Skipping config update (already complete)", "SUCCESS")
        print()

    # Step 6: Final verification
    print_section(f"Step 6: Final Verification")

    # Re-check state
    final_state = check_setup_state(num_classes, config)
    print_setup_state(num_classes, final_state)

    all_complete = all(final_state.values())

    if all_complete:
        print_status("="*70, "SUCCESS")
        print_status(f"{num_classes}-CLASS SETUP COMPLETE!", "SUCCESS")
        print_status("="*70, "SUCCESS")
        print()
        print_status("You can now start training with:", "INFO")
        print()
        print(f"    python train_asl.py --classes {num_classes} --dataset augmented --model-size small")
        print()
        return True
    else:
        print_status("Setup incomplete - some steps failed", "ERROR")
        return False


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
        description='End-to-end preprocessing and verification for ASL training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete end-to-end setup (recommended)
  python training_pre_processing_setup.py --classes 100 --setup

  # Force fresh start (delete and regenerate)
  python training_pre_processing_setup.py --classes 100 --setup --force-fresh

  # Only verify existing setup
  python training_pre_processing_setup.py --classes 50 --verify-only

  # Only create splits (no augmentation)
  python training_pre_processing_setup.py --classes 100 --create-splits
        """
    )
    parser.add_argument(
        '--classes',
        type=int,
        required=True,
        choices=[20, 50, 100],
        help='Number of classes (20, 50, or 100)'
    )
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Run complete end-to-end setup (splits + augmentation + config)'
    )
    parser.add_argument(
        '--force-fresh',
        action='store_true',
        help='Delete existing data and start fresh (use with --setup)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing setup, do not create anything'
    )
    parser.add_argument(
        '--create-splits',
        action='store_true',
        help='Only create dataset splits (no augmentation, no config update)'
    )

    args = parser.parse_args()

    # Validate argument combinations
    mode_flags = [args.setup, args.verify_only, args.create_splits]
    if sum(mode_flags) > 1:
        print("ERROR: Can only use one of: --setup, --verify-only, --create-splits")
        sys.exit(1)

    if args.force_fresh and not args.setup:
        print("ERROR: --force-fresh can only be used with --setup")
        sys.exit(1)

    # Determine mode
    if args.setup:
        mode = "Complete Setup (with --force-fresh)" if args.force_fresh else "Complete Setup (resume mode)"
    elif args.create_splits:
        mode = "Create Splits Only"
    elif args.verify_only:
        mode = "Verify Only"
    else:
        mode = "Verify and Suggest"

    print_section(f"ASL Training Preprocessing - {args.classes} Classes")
    print()
    print(f"Mode: {mode}")
    print()

    # Load configuration
    try:
        config = get_config()
        print_status(f"Configuration loaded", "SUCCESS")
        print_status(f"  └─ Data root: {config.data_root}", "INFO")
        print()
    except Exception as e:
        print_status(f"Failed to load configuration: {e}", "ERROR")
        sys.exit(1)

    # Route to appropriate workflow
    if args.setup:
        # Complete end-to-end setup
        success = run_complete_setup(
            num_classes=args.classes,
            config=config,
            resume=(not args.force_fresh),
            force_fresh=args.force_fresh
        )
        sys.exit(0 if success else 1)

    elif args.create_splits:
        # Create splits only (legacy mode)
        success = create_dataset_splits(args.classes, config)
        if success:
            print_status("NEXT STEP: Run complete setup to add augmentation and update config", "WARNING")
            print(f"  python training_pre_processing_setup.py --classes {args.classes} --setup")
            sys.exit(0)
        else:
            print_status("Failed to create dataset splits", "ERROR")
            sys.exit(1)

    else:
        # Verify-only or suggest mode
        checks_passed = []

        # Step 1: Check directory structure
        checks_passed.append(check_directory_structure(args.classes, config))

        # If verification failed and not verify-only, suggest setup
        if not checks_passed[0] and not args.verify_only:
            print()
            print_status("Dataset not found", "WARNING")
            print_status("To set up everything, run:", "INFO")
            print(f"  python training_pre_processing_setup.py --classes {args.classes} --setup")
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
