#!/usr/bin/env python3
"""
Sync local changes to Kaggle datasets using the Kaggle API.

Handles incremental updates — only re-uploads what changed.

Prerequisites:
    pip install kaggle
    # Place API key at ~/.kaggle/kaggle.json (download from kaggle.com → Settings → API)

Usage:
    python sync_to_kaggle.py --init              # First time: create both datasets
    python sync_to_kaggle.py --code              # Update code only (fast, ~150KB)
    python sync_to_kaggle.py --data              # Update data (slower, re-uploads all data)
    python sync_to_kaggle.py --data --classes 43 # Update data for 43-class only
    python sync_to_kaggle.py --all               # Update both
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STAGING_DIR = PROJECT_ROOT / "kaggle-staging"
PREPARE_SCRIPT = Path(__file__).parent / "prepare_upload.py"

# Change these to match your Kaggle username
KAGGLE_USERNAME = None  # Auto-detected from kaggle.json


def get_kaggle_username():
    """Auto-detect Kaggle username from API key."""
    global KAGGLE_USERNAME
    if KAGGLE_USERNAME:
        return KAGGLE_USERNAME

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        with open(kaggle_json) as f:
            KAGGLE_USERNAME = json.load(f).get("username")
    if not KAGGLE_USERNAME:
        print("ERROR: Could not detect Kaggle username.")
        print("  Make sure ~/.kaggle/kaggle.json exists with your API key.")
        print("  Download from: kaggle.com → Settings → API → Create New Token")
        sys.exit(1)
    return KAGGLE_USERNAME


def create_metadata(dataset_dir, slug, title):
    """Create dataset-metadata.json for Kaggle API."""
    username = get_kaggle_username()
    metadata = {
        "title": title,
        "id": f"{username}/{slug}",
        "licenses": [{"name": "CC0-1.0"}]
    }
    meta_path = dataset_dir / "dataset-metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    return meta_path


def run_kaggle_cmd(args, description=""):
    """Run a kaggle CLI command."""
    cmd = ["kaggle"] + args
    print(f"  {description}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        return False
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")
    return True


def init_datasets(classes=None):
    """First-time setup: stage files and create both datasets on Kaggle."""
    print("=== First-time Kaggle setup ===\n")

    # Stage files
    stage_args = [sys.executable, str(PREPARE_SCRIPT)]
    if classes:
        stage_args.extend(["--classes", classes])
    subprocess.run(stage_args, check=True)

    # Create metadata files
    data_dir = STAGING_DIR / "asl-data"
    code_dir = STAGING_DIR / "asl-code"

    create_metadata(data_dir, "asl-data", "ASL Training Data")
    create_metadata(code_dir, "asl-code", "ASL Training Code")

    print("\nCreating Kaggle datasets...")
    print("(This may take a while for the data upload)\n")

    run_kaggle_cmd(
        ["datasets", "create", "-p", str(code_dir), "--dir-mode", "zip"],
        "Creating asl-code"
    )
    run_kaggle_cmd(
        ["datasets", "create", "-p", str(data_dir), "--dir-mode", "zip"],
        "Creating asl-data"
    )

    print("\nDone! Both datasets created on Kaggle.")
    print("Next: Create a notebook, add both datasets, and run:")
    print("  !python /kaggle/input/asl-code/kaggle_train.py --classes 43 --dataset augmented")


def sync_code():
    """Update code dataset only."""
    print("=== Syncing code to Kaggle ===\n")
    subprocess.run([sys.executable, str(PREPARE_SCRIPT), "--code-only"], check=True)

    code_dir = STAGING_DIR / "asl-code"
    create_metadata(code_dir, "asl-code", "ASL Training Code")

    run_kaggle_cmd(
        ["datasets", "version", "-p", str(code_dir), "-m", "Code update", "--dir-mode", "zip"],
        "Updating asl-code"
    )


def sync_data(classes=None):
    """Update data dataset."""
    print("=== Syncing data to Kaggle ===\n")
    stage_args = [sys.executable, str(PREPARE_SCRIPT), "--data-only"]
    if classes:
        stage_args.extend(["--classes", classes])
    subprocess.run(stage_args, check=True)

    data_dir = STAGING_DIR / "asl-data"
    create_metadata(data_dir, "asl-data", "ASL Training Data")

    run_kaggle_cmd(
        ["datasets", "version", "-p", str(data_dir), "-m", "Data update", "--dir-mode", "zip"],
        "Updating asl-data"
    )


def main():
    parser = argparse.ArgumentParser(description="Sync ASL project to Kaggle")
    parser.add_argument("--init", action="store_true", help="First-time setup: create datasets")
    parser.add_argument("--code", action="store_true", help="Update code dataset")
    parser.add_argument("--data", action="store_true", help="Update data dataset")
    parser.add_argument("--all", action="store_true", help="Update both")
    parser.add_argument("--classes", type=str, default=None,
                        help="Comma-separated class counts (e.g., 43,100)")
    args = parser.parse_args()

    if not any([args.init, args.code, args.data, args.all]):
        parser.print_help()
        return

    if args.init:
        init_datasets(args.classes)
    else:
        if args.code or args.all:
            sync_code()
        if args.data or args.all:
            sync_data(args.classes)


if __name__ == "__main__":
    main()
