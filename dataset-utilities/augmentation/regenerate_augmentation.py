#!/usr/bin/env python3
"""
Wrapper script to regenerate augmented dataset from scratch.

This script:
1. Clears existing augmented pose and pickle files
2. Generates new augmented pose files (with reduced intensity settings)
3. Converts augmented pose files to pickle format
4. Rebuilds the pickle index for fast training lookup

Usage:
    python regenerate_augmentation.py --classes 100
    python regenerate_augmentation.py --classes 100 --skip-clear  # Skip deletion step
    python regenerate_augmentation.py --classes 50 --dry-run      # Show what would be done
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config import get_config


def log(msg, level="INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {msg}")


def run_command(cmd, description, dry_run=False):
    """Run a command and handle errors."""
    log(f"{description}")
    log(f"  Command: {' '.join(cmd)}")

    if dry_run:
        log("  [DRY RUN] Skipping execution")
        return True

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        log(f"  Completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        log(f"  FAILED with exit code {e.returncode}", "ERROR")
        return False
    except Exception as e:
        log(f"  FAILED: {e}", "ERROR")
        return False


def clear_directory(dir_path, dry_run=False):
    """Clear all contents of a directory."""
    dir_path = Path(dir_path)
    if not dir_path.exists():
        log(f"  Directory does not exist: {dir_path}")
        return True

    # Count items
    items = list(dir_path.iterdir())
    log(f"  Found {len(items)} items to delete in {dir_path}")

    if dry_run:
        log("  [DRY RUN] Would delete all contents")
        return True

    for item in items:
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception as e:
            log(f"  Failed to delete {item}: {e}", "ERROR")
            return False

    log(f"  Cleared {len(items)} items")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate augmented dataset from scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python regenerate_augmentation.py --classes 100
    python regenerate_augmentation.py --classes 100 --skip-clear
    python regenerate_augmentation.py --classes 50 --dry-run
        """
    )
    parser.add_argument(
        "--classes", "-c",
        type=int,
        required=True,
        choices=[20, 50, 100],
        help="Number of classes to augment"
    )
    parser.add_argument(
        "--skip-clear",
        action="store_true",
        help="Skip clearing existing augmented data (useful for resuming)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )

    args = parser.parse_args()

    # Get paths from config
    config = get_config()
    augmented_pose_dir = config.augmented_pool_root / "pose"
    augmented_pickle_dir = config.augmented_pool_root / "pickle"

    # Script paths
    scripts_dir = Path(__file__).parent
    augment_script = scripts_dir / "generate_75pt_augmented_dataset.py"
    converter_script = scripts_dir.parent / "conversion" / "pose_to_pickle_converter.py"
    index_script = scripts_dir / "build_augmented_pool_index.py"
    python_exe = sys.executable

    print("=" * 70)
    print("AUGMENTATION REGENERATION PIPELINE")
    print("=" * 70)
    print()
    print(f"Classes: {args.classes}")
    print(f"Augmented pose dir: {augmented_pose_dir}")
    print(f"Augmented pickle dir: {augmented_pickle_dir}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Step 1: Clear existing data
    if not args.skip_clear:
        print("-" * 70)
        log("STEP 1: Clearing existing augmented data")
        print("-" * 70)

        log("Clearing pose directory...")
        if not clear_directory(augmented_pose_dir, args.dry_run):
            log("Failed to clear pose directory", "ERROR")
            return 1

        log("Clearing pickle directory...")
        if not clear_directory(augmented_pickle_dir, args.dry_run):
            log("Failed to clear pickle directory", "ERROR")
            return 1

        print()
    else:
        log("STEP 1: Skipping clear (--skip-clear specified)")
        print()

    # Step 2: Generate augmented pose files
    print("-" * 70)
    log("STEP 2: Generating augmented pose files")
    print("-" * 70)

    cmd = [
        python_exe,
        str(augment_script),
        "--input-type", "pose",
        "--classes", str(args.classes)
    ]

    if not run_command(cmd, "Running pose augmentation", args.dry_run):
        log("Augmentation failed", "ERROR")
        return 1
    print()

    # Step 3: Convert pose to pickle
    print("-" * 70)
    log("STEP 3: Converting augmented pose files to pickle")
    print("-" * 70)

    cmd = [
        python_exe,
        str(converter_script),
        "--input-dir", str(augmented_pose_dir),
        "--output-dir", str(augmented_pickle_dir)
    ]

    if not run_command(cmd, "Running pose-to-pickle conversion", args.dry_run):
        log("Conversion failed", "ERROR")
        return 1
    print()

    # Step 4: Build pickle index
    print("-" * 70)
    log("STEP 4: Building pickle index")
    print("-" * 70)

    cmd = [
        python_exe,
        str(index_script)
    ]

    if not run_command(cmd, "Building augmented pool index", args.dry_run):
        log("Index building failed", "ERROR")
        return 1
    print()

    # Done
    print("=" * 70)
    log("AUGMENTATION PIPELINE COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print(f"  1. Train the model:")
    print(f"     python models/training-scripts/train_asl.py --classes {args.classes} --dataset augmented --force-fresh")
    print()
    print(f"  2. After training, copy to production:")
    print(f"     cp -r models/training-scripts/models/wlasl_{args.classes}_class_model models/openhands-modernized/production-models/")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
