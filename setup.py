#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup.py - ASL-v1 Interactive Setup Script

Interactive setup script for configuring ASL-v1 project on a new system.
Guides users through dataset setup, augmentation, and training preparation.

Usage:
    python setup.py
    python setup.py --non-interactive --data-root /path/to/data
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
import shutil

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    @staticmethod
    def disable():
        """Disable colors for non-supporting terminals."""
        Colors.HEADER = ''
        Colors.BLUE = ''
        Colors.CYAN = ''
        Colors.GREEN = ''
        Colors.YELLOW = ''
        Colors.RED = ''
        Colors.ENDC = ''
        Colors.BOLD = ''


# Disable colors on Windows by default (can be enabled with ANSI support)
if sys.platform == 'win32' and not os.environ.get('ANSICON'):
    Colors.disable()


def print_header(text):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.ENDC}")
    print("=" * 70)


def print_step(step_num, total_steps, text):
    """Print a step indicator."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step {step_num}/{total_steps}] {text}{Colors.ENDC}")


def print_success(text):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text):
    """Print an error message."""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def print_warning(text):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


def print_info(text):
    """Print an info message."""
    print(f"{Colors.CYAN}ℹ {text}{Colors.ENDC}")


def prompt_yes_no(question, default=True):
    """Prompt user for yes/no response."""
    default_str = "Y/n" if default else "y/N"
    response = input(f"{question} [{default_str}]: ").strip().lower()

    if not response:
        return default
    return response in ['y', 'yes']


def prompt_input(question, default=None):
    """Prompt user for input."""
    if default:
        response = input(f"{question} [{default}]: ").strip()
        return response if response else default
    else:
        while True:
            response = input(f"{question}: ").strip()
            if response:
                return response
            print_error("This field is required. Please enter a value.")


def check_python_version():
    """Check if Python version is compatible."""
    print_header("Checking Python Version")

    version = sys.version_info
    print_info(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error("Python 3.8 or higher is required")
        return False

    print_success("Python version is compatible")
    return True


def check_git_repo():
    """Check if we're in a git repository."""
    print_header("Verifying Git Repository")

    if not Path(".git").exists():
        print_error("Not a git repository. Please clone the repository first:")
        print_info("  git clone https://github.com/SuperiorHaxs/asl-v1.git")
        return False

    print_success("Git repository verified")
    return True


def setup_virtual_environment():
    """Setup Python virtual environment."""
    print_header("Setting Up Virtual Environment")

    venv_path = Path("venv")

    if venv_path.exists():
        print_info("Virtual environment already exists")
        if not prompt_yes_no("Recreate virtual environment?", default=False):
            print_success("Using existing virtual environment")
            return True

        print_info("Removing existing virtual environment...")
        shutil.rmtree(venv_path)

    print_info("Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print_success("Virtual environment created")

        # Get the python executable path in venv
        if sys.platform == 'win32':
            venv_python = venv_path / "Scripts" / "python.exe"
            activate_cmd = "venv\\Scripts\\activate"
        else:
            venv_python = venv_path / "bin" / "python"
            activate_cmd = "source venv/bin/activate"

        print_info(f"\nTo activate the virtual environment, run:")
        print_info(f"  {activate_cmd}")

        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment: {e}")
        return False


def install_dependencies():
    """Install Python dependencies."""
    print_header("Installing Dependencies")

    # Get venv python path
    if sys.platform == 'win32':
        venv_python = Path("venv/Scripts/python.exe")
    else:
        venv_python = Path("venv/bin/python")

    if not venv_python.exists():
        print_error("Virtual environment not found. Please run setup again.")
        return False

    print_info("Installing PyTorch...")
    print_info("Choose installation type:")
    print("  1. CPU only (works everywhere)")
    print("  2. GPU (CUDA 11.8) - requires NVIDIA GPU")
    print("  3. GPU (CUDA 12.1) - requires NVIDIA GPU")

    choice = prompt_input("Enter choice [1-3]", default="1")

    if choice == "2":
        torch_cmd = [str(venv_python), "-m", "pip", "install", "torch", "torchvision", "torchaudio",
                     "--index-url", "https://download.pytorch.org/whl/cu118"]
    elif choice == "3":
        torch_cmd = [str(venv_python), "-m", "pip", "install", "torch", "torchvision", "torchaudio",
                     "--index-url", "https://download.pytorch.org/whl/cu121"]
    else:
        torch_cmd = [str(venv_python), "-m", "pip", "install", "torch", "torchvision", "torchaudio"]

    try:
        subprocess.run(torch_cmd, check=True)
        print_success("PyTorch installed")
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install PyTorch: {e}")
        return False

    # Install other dependencies from requirements.txt
    print_info("Installing other dependencies from requirements.txt...")

    requirements_path = Path("requirements.txt")
    if requirements_path.exists():
        # Read requirements.txt and filter out torch packages
        with open(requirements_path, 'r') as f:
            requirements = [
                line.strip()
                for line in f
                if line.strip()
                and not line.strip().startswith('#')
                and not line.strip().lower().startswith('torch')
            ]

        # Create temporary requirements file without torch packages
        if requirements:
            temp_req = Path("temp_requirements.txt")
            with open(temp_req, 'w') as f:
                f.write('\n'.join(requirements))

            try:
                subprocess.run([str(venv_python), "-m", "pip", "install", "-r", str(temp_req)], check=True)
                print_success("Dependencies installed from requirements.txt")
            except subprocess.CalledProcessError as e:
                print_error(f"Failed to install dependencies: {e}")
                return False
            finally:
                if temp_req.exists():
                    temp_req.unlink()  # Clean up temp file

        return True
    else:
        # Fallback if requirements.txt doesn't exist
        print_warning("requirements.txt not found, installing minimal dependencies...")
        try:
            subprocess.run([str(venv_python), "-m", "pip", "install", "numpy", "scikit-learn"], check=True)
            print_success("Minimal dependencies installed")
            return True
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install dependencies: {e}")
            return False


def setup_config():
    """Setup configuration file."""
    print_header("Configuring Data Paths")

    config_file = Path("config/settings.json")

    print_info("The system needs to know where your datasets are located.")
    print_info("This should be a directory containing 'wlasl_poses_complete' folder.")
    print()

    # Check if config already exists
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                existing_config = json.load(f)

            current_path = existing_config.get('data_root', '')
            print_info(f"Current data_root: {current_path}")

            if not prompt_yes_no("Update data_root path?", default=False):
                print_success("Using existing configuration")
                return True
        except:
            pass

    # Suggest default path
    project_root = Path.cwd()
    suggested_path = project_root / "datasets" / "wlasl_poses_complete"

    print_info(f"Suggested path: {suggested_path}")
    print_info("Or enter your custom path (absolute or relative)")

    data_root = prompt_input("Enter data_root path", default=str(suggested_path))

    # Convert to Path and resolve
    data_root_path = Path(data_root)
    if not data_root_path.is_absolute():
        data_root_path = (project_root / data_root_path).resolve()

    # Create config
    config = {
        "data_root": str(data_root_path).replace("\\", "/"),
        "project_root": "auto"
    }

    # Save config
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print_success(f"Configuration saved to {config_file}")
    print_info(f"Data root set to: {data_root_path}")

    return True


def guide_dataset_setup():
    """Guide user through dataset setup."""
    print_header("Dataset Setup Guide")

    print_info("The ASL-v1 training requires the following dataset structure:")
    print()
    print("  datasets/")
    print("  └── wlasl_poses_complete/")
    print("      ├── dataset_splits/")
    print("      │   ├── 20_classes/")
    print("      │   │   └── original/pickle_from_pose_split_20_class/")
    print("      │   │       ├── train/  (20 class folders with .pkl files)")
    print("      │   │       ├── test/")
    print("      │   │       └── val/")
    print("      │   └── 50_classes/")
    print("      │       └── original/pickle_from_pose_split_50_class/")
    print("      │           ├── train/  (50 class folders with .pkl files)")
    print("      │           ├── test/")
    print("      │           └── val/")
    print("      └── (parent)/")
    print("          └── augmented_pool/")
    print("              └── pickle/")
    print("                  ├── accident/")
    print("                  ├── apple/")
    print("                  └── ... (augmented class folders)")
    print()

    print_warning("IMPORTANT: You need to transfer your pre-existing dataset files to this structure.")
    print_info("Dataset files are NOT included in the git repository (too large).")
    print()

    print_info("Steps to setup datasets:")
    print("  1. Copy your wlasl_poses_complete folder to the configured data_root")
    print("  2. Ensure the directory structure matches above")
    print("  3. The augmented_pool will be created in the next step if needed")
    print()

    if not prompt_yes_no("Have you copied the dataset files to the correct location?"):
        print_warning("Please copy your dataset files before continuing.")
        print_info("You can resume this setup by running: python setup.py")
        return False

    print_success("Dataset files confirmed")
    return True


def setup_augmented_pool():
    """Setup augmented dataset pool."""
    print_header("Augmented Dataset Setup")

    print_info("Augmented datasets are REQUIRED for this training system.")
    print_info("Augmentation significantly improves model performance (15x more data).")
    print()

    # Check if augmented pool already exists
    try:
        sys.path.insert(0, str(Path.cwd()))
        from config import get_config
        config = get_config()

        augmented_pool_path = config.augmented_pool_pickle
        augmented_index_path = config.augmented_pool_index

        if augmented_pool_path.exists() and augmented_index_path.exists():
            print_success(f"Augmented pool already exists at {augmented_pool_path}")

            # Check index
            with open(augmented_index_path, 'r') as f:
                index = json.load(f)

            total_files = sum(len(files) for files in index.values())
            print_info(f"Index contains {len(index)} classes with {total_files} total files")

            if not prompt_yes_no("Regenerate augmented pool?", default=False):
                print_success("Using existing augmented pool")
                return True
        else:
            print_warning("Augmented pool not found")
            print_info("You need to either:")
            print("  1. Copy pre-generated augmented pool from another system")
            print("  2. Generate augmented data (takes significant time)")
            print()

            if not prompt_yes_no("Do you have pre-generated augmented pool to copy?"):
                print_warning("Augmented data generation is not yet automated in this script.")
                print_info("Please refer to dataset-utilities/augmentation/ for augmentation scripts.")
                print_info("After generating augmented data, run this setup again.")
                return False

            copy_path = prompt_input("Enter path to pre-generated augmented_pool folder")
            copy_path = Path(copy_path)

            if not copy_path.exists():
                print_error(f"Path not found: {copy_path}")
                return False

            print_info("Copying augmented pool (this may take a while)...")
            try:
                augmented_pool_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(copy_path, augmented_pool_path, dirs_exist_ok=True)
                print_success("Augmented pool copied")
            except Exception as e:
                print_error(f"Failed to copy augmented pool: {e}")
                return False

        # Verify or create index
        if not augmented_index_path.exists():
            print_info("Creating augmented pool index...")
            if not create_augmented_index(augmented_pool_path, augmented_index_path):
                return False

        return True

    except Exception as e:
        print_error(f"Error setting up augmented pool: {e}")
        return False


def create_augmented_index(pool_path, index_path):
    """Create index file for augmented pool."""
    from collections import defaultdict

    print_info("Scanning augmented pool directory...")
    index = defaultdict(list)

    try:
        for class_dir in pool_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name.upper()
                for pkl_file in class_dir.glob("*.pkl"):
                    index[class_name].append(pkl_file.name)

        # Save index
        with open(index_path, 'w') as f:
            json.dump(dict(index), f, indent=2)

        total_files = sum(len(files) for files in index.values())
        print_success(f"Index created: {len(index)} classes, {total_files} files")
        return True

    except Exception as e:
        print_error(f"Failed to create index: {e}")
        return False


def verify_setup(num_classes):
    """Run verification for specified class count."""
    print_header(f"Verifying {num_classes}-Class Setup")

    verification_script = Path("models/training-scripts/training_pre_processing_setup.py")

    if not verification_script.exists():
        print_error("Verification script not found")
        return False

    # Get venv python path
    if sys.platform == 'win32':
        venv_python = Path("venv/Scripts/python.exe")
    else:
        venv_python = Path("venv/bin/python")

    if not venv_python.exists():
        print_warning("Virtual environment not found, using system Python")
        venv_python = sys.executable

    print_info(f"Running verification for {num_classes} classes...")
    print()

    try:
        result = subprocess.run(
            [str(venv_python), str(verification_script), "--classes", str(num_classes)],
            check=False
        )

        if result.returncode == 0:
            print_success(f"{num_classes}-class setup verified successfully")
            return True
        else:
            print_error(f"{num_classes}-class setup verification failed")
            return False

    except Exception as e:
        print_error(f"Verification failed: {e}")
        return False


def print_final_instructions():
    """Print final setup instructions."""
    print_header("Setup Complete!")

    print_success("ASL-v1 is now configured and ready for training")
    print()

    print_info("To start training, follow these steps:")
    print()

    if sys.platform == 'win32':
        activate_cmd = "venv\\Scripts\\activate"
    else:
        activate_cmd = "source venv/bin/activate"

    print(f"{Colors.BOLD}1. Activate the virtual environment:{Colors.ENDC}")
    print(f"   {activate_cmd}")
    print()

    print(f"{Colors.BOLD}2. Navigate to training scripts:{Colors.ENDC}")
    print(f"   cd models/training-scripts")
    print()

    print(f"{Colors.BOLD}3. Start training:{Colors.ENDC}")
    print()
    print("   For 20 classes:")
    print("   python train_asl.py --classes 20 --architecture openhands --model-size small --dataset augmented")
    print()
    print("   For 50 classes:")
    print("   python train_asl.py --classes 50 --architecture openhands --model-size small --dataset augmented")
    print()
    print("   With early stopping (recommended for testing):")
    print("   python train_asl.py --classes 20 --architecture openhands --model-size small --dataset augmented --early-stopping 50")
    print()

    print(f"{Colors.BOLD}Optional: Test trained model:{Colors.ENDC}")
    print("   python train_asl.py --classes 20 --architecture openhands --model-size small --test")
    print()

    print_info("For more information, see README.md or SETUP.md")
    print()


def main():
    """Main setup workflow."""
    parser = argparse.ArgumentParser(
        description='Interactive setup script for ASL-v1 project'
    )
    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Run in non-interactive mode (requires all parameters)'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        help='Path to data root directory (for non-interactive mode)'
    )
    parser.add_argument(
        '--skip-deps',
        action='store_true',
        help='Skip dependency installation'
    )

    args = parser.parse_args()

    print_header("ASL-v1 Setup Script")
    print_info("This script will guide you through setting up ASL-v1 on this system")
    print()

    total_steps = 9
    current_step = 0

    # Step 1: Check Python version
    current_step += 1
    print_step(current_step, total_steps, "Checking Python Version")
    if not check_python_version():
        sys.exit(1)

    # Step 2: Verify git repo
    current_step += 1
    print_step(current_step, total_steps, "Verifying Git Repository")
    if not check_git_repo():
        sys.exit(1)

    # Step 3: Setup virtual environment
    current_step += 1
    print_step(current_step, total_steps, "Setting Up Virtual Environment")
    if not setup_virtual_environment():
        sys.exit(1)

    # Step 4: Install dependencies
    if not args.skip_deps:
        current_step += 1
        print_step(current_step, total_steps, "Installing Dependencies")
        if not install_dependencies():
            sys.exit(1)
    else:
        print_info("Skipping dependency installation")

    # Step 5: Configure paths
    current_step += 1
    print_step(current_step, total_steps, "Configuring Data Paths")
    if not setup_config():
        sys.exit(1)

    # Step 6: Guide dataset setup
    current_step += 1
    print_step(current_step, total_steps, "Dataset Setup")
    if not guide_dataset_setup():
        print_warning("Setup paused. Resume by running: python setup.py")
        sys.exit(1)

    # Step 7: Setup augmented pool
    current_step += 1
    print_step(current_step, total_steps, "Augmented Dataset Setup")
    if not setup_augmented_pool():
        print_warning("Setup incomplete. Please resolve augmented pool setup.")
        sys.exit(1)

    # Step 8: Verify class setups
    current_step += 1
    print_step(current_step, total_steps, "Verifying Class Setups")

    print()
    print_info("Which class configurations do you want to verify?")
    verify_20 = prompt_yes_no("Verify 20-class setup?", default=True)
    verify_50 = prompt_yes_no("Verify 50-class setup?", default=True)

    verification_passed = True
    if verify_20:
        if not verify_setup(20):
            verification_passed = False

    if verify_50:
        if not verify_setup(50):
            verification_passed = False

    if not verification_passed:
        print_warning("Some verifications failed. Please check the errors above.")

    # Step 9: Final instructions
    current_step += 1
    print_step(current_step, total_steps, "Setup Complete")
    print_final_instructions()

    print_success("Setup completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print_warning("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
