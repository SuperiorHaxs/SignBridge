#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training_pre_processing_setup.py

Pre-processing and verification script for ASL training setup.
Checks and prepares dataset structure, mappings, and configurations for training.

Usage:
    python training_pre_processing_setup.py --classes 50
    python training_pre_processing_setup.py --classes 20 --verify-only
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import configuration
from config import get_config


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
    """Main verification workflow."""
    parser = argparse.ArgumentParser(
        description='Verify and prepare dataset for ASL training'
    )
    parser.add_argument(
        '--classes',
        type=int,
        required=True,
        choices=[20, 50, 100],
        help='Number of classes to verify (20, 50, or 100)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify, do not create missing files'
    )

    args = parser.parse_args()

    print_section(f"ASL Training Setup Verification - {args.classes} Classes")
    print()
    print(f"Mode: {'Verify Only' if args.verify_only else 'Verify and Prepare'}")

    # Load configuration
    try:
        config = get_config()
        print_status(f"Configuration loaded", "SUCCESS")
        print_status(f"  └─ Data root: {config.data_root}", "INFO")
    except Exception as e:
        print_status(f"Failed to load configuration: {e}", "ERROR")
        sys.exit(1)

    # Run verification steps
    checks_passed = []

    # Step 1: Check directory structure
    checks_passed.append(check_directory_structure(args.classes, config))

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
