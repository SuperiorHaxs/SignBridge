#!/usr/bin/env python3
"""
Interactive Configuration Setup for ASL-v1 Project

This script helps you configure the project paths for your environment.
Run this after cloning the repository to set up dataset and project paths.
"""

import os
import sys
import json
from pathlib import Path


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).resolve().parent


def validate_path(path_str, must_exist=True):
    """Validate a path string"""
    if not path_str or path_str.strip() == "":
        return False, "Path cannot be empty"

    path = Path(path_str)

    if must_exist and not path.exists():
        return False, f"Path does not exist: {path}"

    return True, None


def prompt_for_path(prompt_text, default=None, must_exist=True):
    """Prompt user for a path with validation"""
    while True:
        if default:
            user_input = input(f"{prompt_text} [{default}]: ").strip()
            if not user_input:
                user_input = default
        else:
            user_input = input(f"{prompt_text}: ").strip()

        if not user_input:
            print("  ERROR: Path cannot be empty. Please try again.")
            continue

        # Expand ~ and environment variables
        user_input = os.path.expanduser(user_input)
        user_input = os.path.expandvars(user_input)

        valid, error = validate_path(user_input, must_exist)
        if valid:
            return Path(user_input).resolve()
        else:
            print(f"  ERROR: {error}")
            retry = input("  Try again? (y/n): ").strip().lower()
            if retry != 'y':
                return None


def detect_dataset_root():
    """Try to detect dataset root from common locations"""
    common_locations = [
        Path("C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/wlasl_poses_complete"),
        Path("E:/wlasl-data/wlasl-kaggle/wlasl_poses_complete"),
        Path("D:/wlasl-kaggle/wlasl_poses_complete"),
        Path("/mnt/external/wlasl-kaggle/wlasl_poses_complete"),
        Path.home() / "wlasl-kaggle" / "wlasl_poses_complete",
    ]

    for location in common_locations:
        if location.exists():
            return location

    return None


def main():
    """Main setup function"""
    print("="*70)
    print("ASL-v1 PROJECT CONFIGURATION SETUP")
    print("="*70)
    print()
    print("This script will help you configure paths for the ASL-v1 project.")
    print("You'll need to provide the dataset root directory location.")
    print()

    project_root = get_project_root()
    config_file = project_root / "config" / "settings.json"

    print(f"Project root detected: {project_root}")
    print(f"Config file: {config_file}")
    print()

    # Load existing config
    existing_config = {}
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                existing_config = json.load(f)
            print("Found existing configuration:")
            print(f"  Data root: {existing_config.get('data_root', 'Not set')}")
            print()

            update = input("Update existing configuration? (y/n): ").strip().lower()
            if update != 'y':
                print("Configuration unchanged. Exiting.")
                return 0
        except json.JSONDecodeError:
            print("WARNING: Existing config file is invalid. Will create new one.")

    print()
    print("-" * 70)
    print("STEP 1: Dataset Root Directory")
    print("-" * 70)
    print()
    print("This is the root directory containing your WLASL dataset files.")
    print("It should contain subdirectories like:")
    print("  - pickle_files/")
    print("  - pose_files/")
    print("  - dataset_splits/")
    print("  - class_index_mapping_XX.json files")
    print()

    # Try to detect dataset root
    detected_root = detect_dataset_root()
    if detected_root:
        print(f"Detected possible dataset location: {detected_root}")
        use_detected = input("Use this location? (y/n): ").strip().lower()
        if use_detected == 'y':
            dataset_root = detected_root
        else:
            dataset_root = prompt_for_path("Enter dataset root path", must_exist=True)
    else:
        print("Could not auto-detect dataset location.")
        dataset_root = prompt_for_path("Enter dataset root path", must_exist=True)

    if not dataset_root:
        print("\nERROR: Dataset root is required. Setup cancelled.")
        return 1

    print()
    print("-" * 70)
    print("STEP 2: Project Root Directory")
    print("-" * 70)
    print()
    print("Project root is auto-detected. You can change it if needed.")
    print(f"Current: {project_root}")
    print()

    change_project_root = input("Change project root? (y/n): ").strip().lower()
    if change_project_root == 'y':
        new_project_root = prompt_for_path("Enter project root path", default=str(project_root), must_exist=True)
        if new_project_root:
            project_root_value = str(new_project_root)
        else:
            project_root_value = "auto"
    else:
        project_root_value = "auto"

    # Verify dataset structure
    print()
    print("-" * 70)
    print("Verifying dataset structure...")
    print("-" * 70)

    expected_items = [
        ("pickle_files", "directory"),
        ("class_index_mapping_20.json", "file"),
    ]

    all_valid = True
    for item_name, item_type in expected_items:
        item_path = dataset_root / item_name
        if item_path.exists():
            print(f"  ✓ Found {item_type}: {item_name}")
        else:
            print(f"  ✗ Missing {item_type}: {item_name}")
            all_valid = False

    if not all_valid:
        print()
        print("WARNING: Some expected files/directories are missing.")
        proceed = input("Continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Setup cancelled.")
            return 1

    # Create configuration
    print()
    print("-" * 70)
    print("Creating configuration...")
    print("-" * 70)

    config = {
        "data_root": str(dataset_root),
        "project_root": project_root_value,
        "_instructions": {
            "description": "Configuration file for ASL-v1 project paths",
            "data_root": "Full path to your dataset root directory",
            "project_root": "Leave as 'auto' for automatic detection, or set to absolute path of asl-v1 directory",
            "setup": "Run 'python setup_config.py' to reconfigure"
        }
    }

    # Save configuration
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config, indent=2, fp=f)

    print(f"✓ Configuration saved to: {config_file}")
    print()
    print("="*70)
    print("CONFIGURATION COMPLETE!")
    print("="*70)
    print()
    print("Summary:")
    print(f"  Dataset Root:  {dataset_root}")
    print(f"  Project Root:  {project_root_value}")
    print()
    print("You can now run training and inference scripts.")
    print("To view configuration, run: python -m config.paths")
    print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
