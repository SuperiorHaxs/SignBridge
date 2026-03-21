#!/usr/bin/env python3
"""
Prepare Kaggle upload for the Doctor Visit Scaling Experiment.

Stages two datasets:
  1. asl-data/  — Augmented pickle files for ALL 107 glosses (82 domain + 25 common)
  2. asl-code/  — Training scripts, model code, experiment runner, config

Usage:
    python prepare_kaggle_upload.py                # Stage both data + code
    python prepare_kaggle_upload.py --data-only    # Re-stage data only
    python prepare_kaggle_upload.py --code-only    # Re-stage code only

After staging, upload to Kaggle:
    cd <project_root>/kaggle-staging
    kaggle datasets create -p asl-data --dir-mode zip
    kaggle datasets create -p asl-code --dir-mode zip

Then in Kaggle notebook:
    !python /kaggle/input/asl-code/kaggle_scaling_experiment.py --domain-n 20
    !python /kaggle/input/asl-code/kaggle_scaling_experiment.py --domain-n 25
    ...
    !python /kaggle/input/asl-code/kaggle_scaling_experiment.py --analyze
"""

import argparse
import json
import os
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
EXPERIMENT_DIR = PROJECT_ROOT / "datasets" / "domain-specific" / "experiments" / "doctor_visit_scaling"
STAGING_DIR = PROJECT_ROOT / "kaggle-staging"


def stage_data():
    """Stage augmented pickle files for all 107 glosses (82 domain + 25 common)."""
    data_dir = STAGING_DIR / "asl-data"
    if data_dir.exists():
        shutil.rmtree(data_dir)

    # Load experiment config to get the full gloss list
    config_path = EXPERIMENT_DIR / "experiment_config.json"
    if not config_path.exists():
        print("ERROR: Run 'python run_scaling_experiment.py prepare' first")
        return

    with open(config_path) as f:
        config = json.load(f)

    common = config['common_words']
    domain = config['domain_candidates_fps_ordered']
    all_glosses = sorted(set(g.lower() for g in common + domain))

    print(f"Staging data for {len(all_glosses)} glosses "
          f"({len(common)} common + {len(domain)} domain)")

    # Copy only the relevant gloss subdirectories from augmented pool
    src_pool = PROJECT_ROOT / "datasets" / "augmented_pool" / "pickle"
    dst_pool = data_dir / "datasets" / "augmented_pool" / "pickle"

    total_files = 0
    for gloss in all_glosses:
        src_dir = src_pool / gloss
        if src_dir.exists():
            dst_dir = dst_pool / gloss
            shutil.copytree(src_dir, dst_dir)
            n = sum(1 for _ in dst_dir.glob('*.pkl'))
            total_files += n
        else:
            print(f"  WARNING: No augmented data for '{gloss}'")

    total_size = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
    print(f"\nasl-data staged: {total_files} pickle files, {total_size / 1e9:.2f} GB")
    print(f"  Location: {data_dir}")


def stage_code():
    """Stage code files + experiment config for Kaggle."""
    code_dir = STAGING_DIR / "asl-code"
    if code_dir.exists():
        shutil.rmtree(code_dir)

    def copy_file(src_rel, dst_rel=None):
        src = PROJECT_ROOT / src_rel
        dst = code_dir / (dst_rel or src_rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"  WARNING: {src} not found")
        return dst

    # ── Config module ──
    copy_file("config/__init__.py")
    copy_file("config/paths.py")

    kaggle_settings = {
        "data_root": "/kaggle/input/asl-data/datasets/wlasl_poses_complete",
        "project_root": "/kaggle/working",
        "dataset_splits": {}
    }
    settings_path = code_dir / "config" / "settings.json"
    with open(settings_path, 'w') as f:
        json.dump(kaggle_settings, f, indent=2)

    # ── Training script ──
    copy_file("models/training-scripts/train_asl.py")

    # ── Model source ──
    copy_file("models/openhands-modernized/src/openhands_modernized.py")
    copy_file("models/openhands-modernized/src/util/openhands_modernized_inference.py")

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

    # ── Profiling script ──
    copy_file("project-utilities/evaluation_metrics/synthetic_evaluation/profile_model_signs.py")

    # ── Requirements ──
    copy_file("requirements.txt", "config/requirements.txt")

    # ── Experiment files ──
    copy_file("dataset-utilities/scaling-experiment/kaggle_scaling_experiment.py",
              "experiment/kaggle_scaling_experiment.py")

    # experiment_config.json lives in the data experiment dir
    # Put in experiment/ subdirectory to avoid Kaggle CLI .json upload bug
    exp_config = EXPERIMENT_DIR / "experiment_config.json"
    if exp_config.exists():
        dst = code_dir / "experiment" / "experiment_config.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exp_config, dst)
    else:
        print("  WARNING: experiment_config.json not found — run prepare step first")

    total_files = sum(1 for _ in code_dir.rglob("*") if _.is_file())
    print(f"\nasl-code staged: {total_files} files")
    print(f"  Location: {code_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Kaggle upload for scaling experiment")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--code-only", action="store_true")
    args = parser.parse_args()

    STAGING_DIR.mkdir(exist_ok=True)

    if args.code_only:
        stage_code()
    elif args.data_only:
        stage_data()
    else:
        stage_data()
        stage_code()

    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print(f"1. Upload to Kaggle:")
    print(f"   kaggle datasets create -p {STAGING_DIR / 'asl-data'} --dir-mode zip")
    print(f"   kaggle datasets create -p {STAGING_DIR / 'asl-code'} --dir-mode zip")
    print(f"2. Create notebook, add both datasets, enable GPU")
    print(f"3. Run sweep points one at a time:")
    print(f"   !python /kaggle/input/datasets/nivakramuk/asl-code/experiment/kaggle_scaling_experiment.py --domain-n 20")
    print(f"   !python /kaggle/input/datasets/nivakramuk/asl-code/experiment/kaggle_scaling_experiment.py --domain-n 25")
    print(f"   ...up to --domain-n 60")
    print(f"4. Analyze:")
    print(f"   !python /kaggle/input/datasets/nivakramuk/asl-code/experiment/kaggle_scaling_experiment.py --analyze")


if __name__ == "__main__":
    main()
