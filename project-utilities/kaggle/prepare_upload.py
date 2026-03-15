#!/usr/bin/env python3
"""
Prepare local project files for Kaggle upload.

Creates two staging directories:
  1. asl-data/   — All datasets (pickles, manifests, class mappings, original splits)
  2. asl-code/   — Training scripts, model code, config, launcher notebook

Usage:
    python prepare_upload.py                    # Stage everything
    python prepare_upload.py --data-only        # Re-stage data only
    python prepare_upload.py --code-only        # Re-stage code only
    python prepare_upload.py --classes 43       # Only include 43-class data
    python prepare_upload.py --classes 43,100   # Include 43 and 100-class data
"""

import argparse
import json
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STAGING_DIR = PROJECT_ROOT / "kaggle-staging"

# ── Data layout mirrors local structure ──────────────────────────────
# Kaggle mount:  /kaggle/input/asl-data/
#   datasets/
#     augmented_pool/
#       pickle/           ← all gloss subdirs with .pkl files
#       splits/
#         43_classes/     ← manifests
#         100_classes/
#     wlasl_poses_complete/
#       dataset_splits/
#         43_classes/     ← class_mapping.json + original/ pickles
#         100_classes/
#
# ── Code layout mirrors local structure ──────────────────────────────
# Kaggle mount:  /kaggle/input/asl-code/
#   config/               ← __init__.py, paths.py, settings_kaggle.json
#   models/
#     training-scripts/   ← train_asl.py
#     openhands-modernized/src/  ← model code
#   dataset-utilities/
#     landmarks-extraction/  ← finger_features.py, landmark_config.py
#   requirements.txt
#   kaggle_train.py       ← launcher script


def stage_data(classes=None):
    """Stage dataset files."""
    data_dir = STAGING_DIR / "asl-data"
    if data_dir.exists():
        shutil.rmtree(data_dir)

    # ── Augmented pool pickles (shared across all class counts) ──
    src_pickle = PROJECT_ROOT / "datasets" / "augmented_pool" / "pickle"
    dst_pickle = data_dir / "datasets" / "augmented_pool" / "pickle"

    if src_pickle.exists():
        print(f"Copying augmented pickle pool ({sum(1 for _ in src_pickle.rglob('*.pkl'))} files)...")
        shutil.copytree(src_pickle, dst_pickle, ignore=shutil.ignore_patterns('__pycache__'))
    else:
        print(f"WARNING: {src_pickle} not found, skipping")

    # ── Augmented pool splits (per class count) ──
    src_splits = PROJECT_ROOT / "datasets" / "augmented_pool" / "splits"
    available_splits = [d.name for d in src_splits.iterdir() if d.is_dir()] if src_splits.exists() else []

    for split_name in available_splits:
        class_count = split_name.replace("_classes", "")
        if classes and class_count not in classes:
            print(f"  Skipping split {split_name} (not in --classes)")
            continue
        src = src_splits / split_name
        dst = data_dir / "datasets" / "augmented_pool" / "splits" / split_name
        print(f"Copying augmented splits: {split_name}")
        shutil.copytree(src, dst)

    # ── Original dataset splits (per class count) ──
    src_original = PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "dataset_splits"
    if src_original.exists():
        available_originals = [d.name for d in src_original.iterdir() if d.is_dir()]
        for split_name in available_originals:
            class_count = split_name.replace("_classes", "")
            if classes and class_count not in classes:
                print(f"  Skipping original split {split_name} (not in --classes)")
                continue
            src = src_original / split_name
            dst = data_dir / "datasets" / "wlasl_poses_complete" / "dataset_splits" / split_name
            print(f"Copying original splits: {split_name}")
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__'))

    # Summary
    total_size = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
    total_files = sum(1 for _ in data_dir.rglob("*") if _.is_file())
    print(f"\nasl-data staged: {total_files} files, {total_size / 1e9:.2f} GB")
    print(f"  Location: {data_dir}")


def stage_code():
    """Stage training code."""
    code_dir = STAGING_DIR / "asl-code"
    if code_dir.exists():
        shutil.rmtree(code_dir)

    def copy_file(src_rel, dst_rel=None):
        src = PROJECT_ROOT / src_rel
        dst = code_dir / (dst_rel or src_rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst

    # ── Config module ──
    copy_file("config/__init__.py")
    copy_file("config/paths.py")

    # Create Kaggle-compatible settings.json
    # Uses /kaggle/input/asl-data/ for datasets, /kaggle/working/ for outputs
    kaggle_settings = {
        "data_root": "/kaggle/input/asl-data/datasets/wlasl_poses_complete",
        "project_root": "/kaggle/working",
        "dataset_splits": {
            "43": {
                "train_original": "dataset_splits/43_classes/original/pickle_split_43_class/train",
                "pickle_pool": "../augmented_pool/pickle",
                "train_manifest": "../augmented_pool/splits/43_classes/train_manifest.json",
                "val_manifest": "../augmented_pool/splits/43_classes/val_manifest.json",
                "test_manifest": "../augmented_pool/splits/43_classes/test_manifest.json",
                "class_mapping": "dataset_splits/43_classes/class_mapping.json"
            },
            "100": {
                "train_original": "dataset_splits/100_classes/original/pickle_split_100_class/train",
                "pickle_pool": "../augmented_pool/pickle",
                "train_manifest": "../augmented_pool/splits/100_classes/train_manifest.json",
                "val_manifest": "../augmented_pool/splits/100_classes/val_manifest.json",
                "test_manifest": "../augmented_pool/splits/100_classes/test_manifest.json",
                "class_mapping": "dataset_splits/100_classes/class_mapping.json"
            },
            "125": {
                "train_original": "dataset_splits/125_classes/original/pickle_split_125_class/train",
                "pickle_pool": "../augmented_pool/pickle",
                "train_manifest": "../augmented_pool/splits/125_classes/train_manifest.json",
                "val_manifest": "../augmented_pool/splits/125_classes/val_manifest.json",
                "test_manifest": "../augmented_pool/splits/125_classes/test_manifest.json",
                "class_mapping": "dataset_splits/125_classes/class_mapping.json"
            }
        }
    }
    settings_path = code_dir / "config" / "settings.json"
    with open(settings_path, 'w') as f:
        json.dump(kaggle_settings, f, indent=2)

    # ── Training script ──
    copy_file("models/training-scripts/train_asl.py")

    # ── Model source ──
    copy_file("models/openhands-modernized/src/openhands_modernized.py")
    copy_file("models/openhands-modernized/src/util/openhands_modernized_inference.py")

    # Create __init__.py files for package imports
    for init_path in [
        "models/__init__.py",
        "models/openhands-modernized/__init__.py",
        "models/openhands-modernized/src/__init__.py",
        "models/openhands-modernized/src/util/__init__.py",
    ]:
        p = code_dir / init_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()

    # ── Finger features ──
    copy_file("dataset-utilities/landmarks-extraction/__init__.py")
    copy_file("dataset-utilities/landmarks-extraction/finger_features.py")
    copy_file("dataset-utilities/landmarks-extraction/landmark_config.py")

    # ── Requirements ──
    copy_file("requirements.txt")

    # ── Kaggle launcher script ──
    create_launcher(code_dir)

    total_files = sum(1 for _ in code_dir.rglob("*") if _.is_file())
    print(f"\nasl-code staged: {total_files} files")
    print(f"  Location: {code_dir}")


def create_launcher(code_dir):
    """Create the Kaggle launcher script."""
    launcher = code_dir / "kaggle_train.py"
    launcher.write_text('''\
#!/usr/bin/env python3
"""
Kaggle Training Launcher for ASL OpenHands-HD Model

This script sets up the environment and runs training on Kaggle.
It handles the read-only input to writable working directory copy.

Usage (in Kaggle notebook cell):
    !python /kaggle/input/asl-code/kaggle_train.py --classes 43 --dataset augmented --dropout 0.3
    !python /kaggle/input/asl-code/kaggle_train.py --classes 43 --dataset original --force-fresh
    !python /kaggle/input/asl-code/kaggle_train.py --classes 100 --dataset augmented --manifest /kaggle/input/asl-data/datasets/augmented_pool/splits/100_classes/train_manifest.json
"""

import subprocess
import sys
import os
import shutil
import json
from pathlib import Path

WORKING = Path("/kaggle/working")


def find_input_path(dataset_name):
    """Auto-detect Kaggle dataset mount path (handles both old and new mount styles)."""
    # Check new style first: /kaggle/input/datasets/<username>/<dataset>/
    # Then old style: /kaggle/input/<dataset>/
    candidates = []
    new_style = Path("/kaggle/input/datasets")
    if new_style.exists():
        for user_dir in new_style.iterdir():
            candidates.append(user_dir / dataset_name)
    candidates.append(Path(f"/kaggle/input/{dataset_name}"))

    for p in candidates:
        if p.exists() and any(p.iterdir()):  # must exist AND have content
            return p

    raise FileNotFoundError(
        f"Could not find dataset '{dataset_name}' in /kaggle/input/. "
        f"Searched: {[str(c) for c in candidates]}"
    )


def setup_environment():
    """Copy code to working directory and configure paths."""

    CODE_INPUT = find_input_path("asl-code")
    DATA_INPUT = find_input_path("asl-data")
    print(f"Code input: {CODE_INPUT}")
    print(f"Data input: {DATA_INPUT}")

    # Copy code tree to writable location (skip if already done)
    marker = WORKING / ".setup_done"
    if not marker.exists():
        print("Setting up working environment...")

        # Copy code files
        for item in ["config", "models", "dataset-utilities", "requirements.txt"]:
            src = CODE_INPUT / item
            dst = WORKING / item
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        # Update settings.json data_root to point to actual data input
        settings_path = WORKING / "config" / "settings.json"
        with open(settings_path) as f:
            settings = json.load(f)
        # Data mount may have datasets/ at top level or nested under wlasl_poses_complete
        data_root = DATA_INPUT / "datasets" / "wlasl_poses_complete"
        if not data_root.exists():
            data_root = DATA_INPUT / "wlasl_poses_complete"
        settings["data_root"] = str(data_root)
        settings["project_root"] = str(WORKING)
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)

        # Install extra deps
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pose-format"],
                      check=True)

        marker.touch()
        print("Environment ready.")
    else:
        print("Environment already set up.")


def main():
    setup_environment()

    # Build train_asl.py command from our args
    train_script = str(WORKING / "models" / "training-scripts" / "train_asl.py")

    # Pass all args through to train_asl.py, adding --architecture openhands default
    import argparse
    parser = argparse.ArgumentParser(description="Kaggle ASL Training Launcher")
    parser.add_argument("--classes", type=int, required=True, help="Number of classes (43, 100, 125)")
    parser.add_argument("--dataset", type=str, default="augmented", choices=["original", "augmented"])
    parser.add_argument("--model-size", type=str, default="small", choices=["tiny", "small", "large"])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--early-stopping", type=int, default=15)
    parser.add_argument("--manifest", type=str, default=None, help="Custom manifest path")
    parser.add_argument("--force-fresh", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable, train_script,
        "--architecture", "openhands",
        "--classes", str(args.classes),
        "--dataset", args.dataset,
        "--model-size", args.model_size,
        "--dropout", str(args.dropout),
        "--early-stopping", str(args.early_stopping),
    ]
    if args.manifest:
        cmd.extend(["--manifest", args.manifest])
    if args.force_fresh:
        cmd.append("--force-fresh")

    # Set PYTHONPATH for imports
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join([
        str(WORKING),
        str(WORKING / "models" / "openhands-modernized" / "src"),
        str(WORKING / "dataset-utilities" / "landmarks-extraction"),
    ])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
''')


def main():
    parser = argparse.ArgumentParser(description="Prepare files for Kaggle upload")
    parser.add_argument("--data-only", action="store_true", help="Only stage data")
    parser.add_argument("--code-only", action="store_true", help="Only stage code")
    parser.add_argument("--classes", type=str, default=None,
                        help="Comma-separated class counts to include (e.g., 43,100). Default: all")
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None

    STAGING_DIR.mkdir(exist_ok=True)

    if args.code_only:
        stage_code()
    elif args.data_only:
        stage_data(classes)
    else:
        stage_data(classes)
        stage_code()

    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print(f"1. Upload datasets to Kaggle:")
    print(f"   - Create dataset 'asl-data' from: {STAGING_DIR / 'asl-data'}")
    print(f"   - Create dataset 'asl-code' from: {STAGING_DIR / 'asl-code'}")
    print(f"2. In Kaggle notebook, add both datasets")
    print(f"3. Run:  !python /kaggle/input/asl-code/kaggle_train.py --classes 43 --dataset augmented")
    print()


if __name__ == "__main__":
    main()
