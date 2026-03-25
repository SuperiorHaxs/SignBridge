#!/usr/bin/env python3
"""
Prepare Restaurant Domain Model for Kaggle Training.

End-to-end pipeline:
  1. Generate augmented data for missing restaurant glosses
  2. Create train/val/test split manifests for 60-class domain (35 restaurant + 25 common)
  3. Stage data + code for Kaggle upload
  4. Validate the staged package
  5. Upload to Kaggle

Usage:
    python prepare_restaurant_kaggle.py                    # Full pipeline
    python prepare_restaurant_kaggle.py --augment-only     # Step 1 only
    python prepare_restaurant_kaggle.py --split-only       # Step 2 only
    python prepare_restaurant_kaggle.py --stage-only       # Step 3+4 only
    python prepare_restaurant_kaggle.py --upload-only      # Step 5 only
    python prepare_restaurant_kaggle.py --dry-run          # Preview without changes
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
STAGING_DIR = PROJECT_ROOT / "kaggle-staging-restaurant"

# Domain configs
RESTAURANT_JSON = PROJECT_ROOT / "datasets" / "domain-specific" / "restaurant.json"
COMMON_WORDS_JSON = PROJECT_ROOT / "datasets" / "domain-specific" / "common_words.json"

# Data paths
AUGMENTED_POOL = PROJECT_ROOT / "datasets" / "augmented_pool" / "pickle"
POSE_BY_GLOSS = PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "pose_files_by_gloss"
SPLITS_DIR = PROJECT_ROOT / "datasets" / "augmented_pool" / "splits" / "60_restaurant"


def load_glosses():
    """Load restaurant + common word glosses."""
    with open(RESTAURANT_JSON) as f:
        restaurant = json.load(f)
    with open(COMMON_WORDS_JSON) as f:
        common = json.load(f)

    restaurant_glosses = [g.lower() for g in restaurant['classes']]
    common_glosses = [g.lower() for g in common['classes']]
    all_glosses = sorted(set(restaurant_glosses + common_glosses))

    return restaurant_glosses, common_glosses, all_glosses


def check_augmented_status(all_glosses):
    """Check which glosses have augmented data and which are missing."""
    present = []
    missing = []
    for gloss in all_glosses:
        gloss_dir = AUGMENTED_POOL / gloss
        if gloss_dir.exists():
            pkl_count = sum(1 for f in gloss_dir.iterdir() if f.suffix == '.pkl')
            if pkl_count > 0:
                present.append((gloss, pkl_count))
                continue
        missing.append(gloss)
    return present, missing


# ─── Step 1: Generate augmented data ─────────────────────────────────────────

def step_augment(dry_run=False):
    """Generate augmented data for glosses missing from the augmented pool."""
    print("\n" + "=" * 70)
    print("STEP 1: Generate augmented data for missing glosses")
    print("=" * 70)

    _, _, all_glosses = load_glosses()
    present, missing = check_augmented_status(all_glosses)

    print(f"\nGlosses with augmented data: {len(present)}")
    print(f"Glosses missing augmented data: {len(missing)}")

    if not missing:
        print("All glosses have augmented data. Skipping augmentation.")
        return True

    print(f"\nMissing glosses: {missing}")

    if dry_run:
        print("\n[DRY RUN] Would generate augmented data for these glosses.")
        return True

    # Create a temporary gloss list file for the augmentation script
    gloss_list_file = PROJECT_ROOT / "datasets" / "domain-specific" / "restaurant_all_glosses.json"
    gloss_data = {
        "classes": [g.upper() for g in missing],
        "description": "Missing restaurant + common glosses for augmentation"
    }
    with open(gloss_list_file, 'w') as f:
        json.dump(gloss_data, f, indent=2)

    print(f"\nRunning augmentation for {len(missing)} glosses...")
    print("This may take several minutes...\n")

    augment_script = PROJECT_ROOT / "dataset-utilities" / "augmentation" / "generate_augmented_dataset.py"
    cmd = [
        sys.executable, str(augment_script),
        "--gloss-file", str(gloss_list_file),
        "--landmark-config", "83pt",
    ]

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print(f"\nERROR: Augmentation failed with return code {result.returncode}")
        return False

    # Verify
    _, still_missing = check_augmented_status(all_glosses)
    if still_missing:
        print(f"\nWARNING: {len(still_missing)} glosses still missing after augmentation: {still_missing}")
        return False

    print(f"\nAugmentation complete. All {len(all_glosses)} glosses now have data.")
    return True


# ─── Step 2: Create split manifests ──────────────────────────────────────────

def step_split(dry_run=False):
    """Create train/val/test split manifests for the 60-class restaurant domain."""
    print("\n" + "=" * 70)
    print("STEP 2: Create train/val/test split manifests")
    print("=" * 70)

    _, _, all_glosses = load_glosses()

    print(f"\nTotal classes: {len(all_glosses)}")
    print(f"Split output: {SPLITS_DIR}")

    if dry_run:
        print("\n[DRY RUN] Would create splits at the above location.")
        return True

    split_script = PROJECT_ROOT / "dataset-utilities" / "dataset-splitting" / "stratified_family_split.py"

    # The split script reads from the augmented pool (which has gloss subdirectories)
    # and creates manifest files pointing to those pickle files
    cmd = [
        sys.executable, str(split_script),
        "--input-dir", str(AUGMENTED_POOL),
        "--output-dir", str(SPLITS_DIR),
        "--balance-train",
        "--train-target", "200",
        "--seed", "42",
    ]

    # We need to filter to only our 60 glosses. The split script processes
    # all subdirectories in input-dir. Let's create a temporary filtered pool.
    # Actually, the split script will process ALL gloss dirs. The manifests
    # will include all of them. We'll filter when staging for Kaggle.
    # But better: let's create a filtered input dir with symlinks.

    filtered_pool = PROJECT_ROOT / "datasets" / "augmented_pool" / "pickle_restaurant_60"
    if filtered_pool.exists():
        shutil.rmtree(filtered_pool)
    filtered_pool.mkdir(parents=True)

    # Create symlinks (or directory junctions on Windows) to the relevant gloss dirs
    linked_count = 0
    for gloss in all_glosses:
        src = AUGMENTED_POOL / gloss
        dst = filtered_pool / gloss
        if src.exists():
            # On Windows, use directory junction instead of symlink (no admin needed)
            if sys.platform == 'win32':
                subprocess.run(['cmd', '/c', 'mklink', '/J', str(dst), str(src)],
                              capture_output=True)
            else:
                dst.symlink_to(src)
            linked_count += 1
        else:
            print(f"  WARNING: No augmented data for '{gloss}'")

    print(f"Created filtered pool with {linked_count} gloss directories")

    cmd_filtered = [
        sys.executable, str(split_script),
        "--input-dir", str(filtered_pool),
        "--output-dir", str(SPLITS_DIR),
        "--balance-train",
        "--train-target", "200",
        "--seed", "42",
    ]

    result = subprocess.run(cmd_filtered, cwd=str(PROJECT_ROOT))

    # Clean up filtered pool (remove junctions/symlinks)
    if filtered_pool.exists():
        if sys.platform == 'win32':
            # Remove junctions carefully (rmtree would follow them)
            for item in filtered_pool.iterdir():
                if item.is_dir():
                    subprocess.run(['cmd', '/c', 'rmdir', str(item)], capture_output=True)
            filtered_pool.rmdir()
        else:
            shutil.rmtree(filtered_pool)

    if result.returncode != 0:
        print(f"\nERROR: Split creation failed with return code {result.returncode}")
        return False

    # Verify manifests exist
    for split in ['train_manifest.json', 'val_manifest.json', 'test_manifest.json']:
        manifest = SPLITS_DIR / split
        if not manifest.exists():
            print(f"ERROR: Expected manifest not found: {manifest}")
            return False

    print("\nSplit manifests created successfully.")
    return True


# ─── Step 3: Stage for Kaggle ────────────────────────────────────────────────

def stage_data():
    """Stage augmented pickle files for all 60 glosses."""
    data_dir = STAGING_DIR / "asl-data"
    if data_dir.exists():
        shutil.rmtree(data_dir)

    _, _, all_glosses = load_glosses()

    print(f"\nStaging data for {len(all_glosses)} glosses...")

    dst_pool = data_dir / "datasets" / "augmented_pool" / "pickle"

    total_files = 0
    for gloss in all_glosses:
        src_dir = AUGMENTED_POOL / gloss
        if src_dir.exists():
            dst_dir = dst_pool / gloss
            shutil.copytree(src_dir, dst_dir)
            n = sum(1 for _ in dst_dir.glob('*.pkl'))
            total_files += n
        else:
            print(f"  WARNING: No augmented data for '{gloss}'")

    # Copy split manifests
    if SPLITS_DIR.exists():
        splits_dst = data_dir / "splits" / "60_restaurant"
        shutil.copytree(SPLITS_DIR, splits_dst)
        print(f"  Copied split manifests to {splits_dst}")

    # Copy domain config files
    configs_dst = data_dir / "domain_config"
    configs_dst.mkdir(parents=True)
    shutil.copy2(RESTAURANT_JSON, configs_dst / "restaurant.json")
    shutil.copy2(COMMON_WORDS_JSON, configs_dst / "common_words.json")

    # Create combined class mapping for 60 classes
    restaurant_glosses, common_glosses, all_sorted = load_glosses()
    combined_mapping = {
        "domain": "restaurant",
        "scenario": "Restaurant",
        "num_classes": len(all_sorted),
        "domain_classes": len(restaurant_glosses),
        "common_classes": len(common_glosses),
        "classes": [g.upper() for g in all_sorted],
        "class_to_index": {g.upper(): i for i, g in enumerate(all_sorted)},
        "class_index_mapping": {str(i): g for i, g in enumerate(all_sorted)},
    }
    with open(configs_dst / "restaurant_60class.json", 'w') as f:
        json.dump(combined_mapping, f, indent=2)

    # Create dataset-metadata.json for Kaggle
    metadata = {
        "title": "ASL Restaurant Domain Data",
        "id": "nivakramuk/asl-restaurant-data",
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(data_dir / "dataset-metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    total_size = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
    print(f"\nasl-data staged: {total_files} pickle files, {total_size / 1e9:.2f} GB")
    print(f"  Location: {data_dir}")


def stage_code():
    """Stage code files for Kaggle."""
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

    # Kaggle-specific settings
    kaggle_settings = {
        "data_root": "/kaggle/input/asl-restaurant-data/datasets/augmented_pool/pickle",
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

    # ── Restaurant training runner ──
    # Create a simple Kaggle training script for the restaurant domain
    runner_content = '''#!/usr/bin/env python3
"""
Kaggle Training Runner for Restaurant Domain Model.

Usage (in Kaggle notebook):
    !python /kaggle/input/asl-restaurant-code/train_restaurant.py
    !python /kaggle/input/asl-restaurant-code/train_restaurant.py --epochs 800
    !python /kaggle/input/asl-restaurant-code/train_restaurant.py --resume /kaggle/working/checkpoint.pth
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# Paths on Kaggle
CODE_DIR = Path("/kaggle/input/asl-restaurant-code")
DATA_DIR = Path("/kaggle/input/asl-restaurant-data")
WORK_DIR = Path("/kaggle/working")

# Add code directory to path
sys.path.insert(0, str(CODE_DIR))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train restaurant domain ASL model")
    parser.add_argument("--epochs", type=int, default=600, help="Number of epochs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    # Load domain config
    config_path = DATA_DIR / "domain_config" / "restaurant_60class.json"
    with open(config_path) as f:
        domain_config = json.load(f)

    num_classes = domain_config["num_classes"]
    print(f"Training restaurant domain model: {num_classes} classes")
    print(f"Domain classes: {domain_config['domain_classes']}")
    print(f"Common classes: {domain_config['common_classes']}")

    # Paths
    pickle_pool = DATA_DIR / "datasets" / "augmented_pool" / "pickle"
    train_manifest = DATA_DIR / "splits" / "60_restaurant" / "train_manifest.json"
    val_manifest = DATA_DIR / "splits" / "60_restaurant" / "val_manifest.json"
    class_mapping = DATA_DIR / "domain_config" / "restaurant_60class.json"

    # Build training command
    train_script = CODE_DIR / "models" / "training-scripts" / "train_asl.py"
    cmd = [
        sys.executable, str(train_script),
        "--classes", str(num_classes),
        "--dataset", "augmented",
        "--augmented-path", str(pickle_pool),
        "--manifest-dir", str(DATA_DIR / "splits" / "60_restaurant"),
        "--architecture", "openhands",
        "--model-size", "small",
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--batch-size", str(args.batch_size),
        "--patience", "80",
        "--output-dir", str(WORK_DIR),
        "--class-mapping", str(class_mapping),
    ]

    if args.resume:
        cmd.extend(["--resume", args.resume])

    print(f"\\nCommand: {' '.join(cmd)}")
    print("=" * 70)

    result = subprocess.run(cmd, cwd=str(WORK_DIR))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
'''
    runner_path = code_dir / "train_restaurant.py"
    with open(runner_path, 'w') as f:
        f.write(runner_content)

    # Create dataset-metadata.json for Kaggle
    metadata = {
        "title": "ASL Restaurant Training Code",
        "id": "nivakramuk/asl-restaurant-code",
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(code_dir / "dataset-metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    total_files = sum(1 for _ in code_dir.rglob("*") if _.is_file())
    print(f"\nasl-code staged: {total_files} files")
    print(f"  Location: {code_dir}")


def step_stage(dry_run=False):
    """Stage data and code for Kaggle upload."""
    print("\n" + "=" * 70)
    print("STEP 3: Stage data + code for Kaggle")
    print("=" * 70)

    if dry_run:
        _, _, all_glosses = load_glosses()
        print(f"\n[DRY RUN] Would stage {len(all_glosses)} glosses + code to {STAGING_DIR}")
        return True

    STAGING_DIR.mkdir(exist_ok=True)
    stage_data()
    stage_code()
    return True


# ─── Step 4: Validate ────────────────────────────────────────────────────────

def step_validate():
    """Validate the staged Kaggle package."""
    print("\n" + "=" * 70)
    print("STEP 4: Validate staged package")
    print("=" * 70)

    errors = []
    warnings = []

    data_dir = STAGING_DIR / "asl-data"
    code_dir = STAGING_DIR / "asl-code"

    # Check directories exist
    if not data_dir.exists():
        errors.append("asl-data directory not found")
    if not code_dir.exists():
        errors.append("asl-code directory not found")

    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        return False

    # Check all glosses have data
    _, _, all_glosses = load_glosses()
    pickle_dir = data_dir / "datasets" / "augmented_pool" / "pickle"

    total_files = 0
    min_files = float('inf')
    min_gloss = ""
    missing_glosses = []

    for gloss in all_glosses:
        gloss_dir = pickle_dir / gloss
        if not gloss_dir.exists():
            missing_glosses.append(gloss)
            continue
        count = sum(1 for f in gloss_dir.glob("*.pkl"))
        total_files += count
        if count < min_files:
            min_files = count
            min_gloss = gloss
        if count < 5:
            warnings.append(f"'{gloss}' has only {count} pickle files")

    if missing_glosses:
        errors.append(f"Missing data for {len(missing_glosses)} glosses: {missing_glosses}")

    print(f"\n  Glosses: {len(all_glosses) - len(missing_glosses)}/{len(all_glosses)}")
    print(f"  Total pickle files: {total_files}")
    print(f"  Min files per gloss: {min_files} ({min_gloss})")

    # Check split manifests
    splits_dir = data_dir / "splits" / "60_restaurant"
    for split_name in ['train_manifest.json', 'val_manifest.json', 'test_manifest.json']:
        manifest_path = splits_dir / split_name
        if not manifest_path.exists():
            errors.append(f"Missing manifest: {split_name}")
        else:
            with open(manifest_path) as f:
                manifest = json.load(f)
            n_classes = manifest.get('num_classes', 0)
            n_samples = manifest.get('total_samples', 0)
            print(f"  {split_name}: {n_classes} classes, {n_samples} samples")

    # Check domain config
    config_path = data_dir / "domain_config" / "restaurant_60class.json"
    if not config_path.exists():
        errors.append("Missing restaurant_60class.json config")
    else:
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"  Domain config: {cfg['num_classes']} classes ({cfg['domain_classes']} domain + {cfg['common_classes']} common)")

    # Check code files
    essential_code = [
        "models/training-scripts/train_asl.py",
        "models/openhands-modernized/src/openhands_modernized.py",
        "config/paths.py",
        "config/settings.json",
        "train_restaurant.py",
    ]
    for code_file in essential_code:
        if not (code_dir / code_file).exists():
            errors.append(f"Missing code file: {code_file}")

    # Check metadata files
    for meta_dir in [data_dir, code_dir]:
        meta_file = meta_dir / "dataset-metadata.json"
        if not meta_file.exists():
            errors.append(f"Missing dataset-metadata.json in {meta_dir.name}")

    # Report
    total_size = sum(f.stat().st_size for f in STAGING_DIR.rglob("*") if f.is_file())

    print(f"\n  Total staged size: {total_size / 1e9:.2f} GB")

    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
        return False

    print("\n  VALIDATION PASSED")
    return True


# ─── Step 5: Upload to Kaggle ────────────────────────────────────────────────

def step_upload(dry_run=False):
    """Upload staged data and code to Kaggle."""
    print("\n" + "=" * 70)
    print("STEP 5: Upload to Kaggle")
    print("=" * 70)

    data_dir = STAGING_DIR / "asl-data"
    code_dir = STAGING_DIR / "asl-code"

    if dry_run:
        print("\n[DRY RUN] Would upload:")
        print(f"  Data: kaggle datasets create -p {data_dir} --dir-mode zip")
        print(f"  Code: kaggle datasets create -p {code_dir} --dir-mode zip")
        return True

    # Check if datasets already exist, use version if so
    for name, staged_dir in [("asl-restaurant-data", data_dir), ("asl-restaurant-code", code_dir)]:
        print(f"\nUploading {name}...")

        # Try creating new dataset first, fall back to new version
        result = subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(staged_dir), "--dir-mode", "zip"],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            if "already exists" in result.stderr.lower() or "already exists" in result.stdout.lower():
                print(f"  Dataset exists, creating new version...")
                result = subprocess.run(
                    ["kaggle", "datasets", "version", "-p", str(staged_dir),
                     "-m", "Restaurant domain update", "--dir-mode", "zip"],
                    capture_output=True, text=True
                )

            if result.returncode != 0:
                print(f"  ERROR uploading {name}:")
                print(f"  stdout: {result.stdout}")
                print(f"  stderr: {result.stderr}")
                return False

        print(f"  {result.stdout.strip()}")

    print("\nUpload complete!")
    print("\nNext steps:")
    print("1. Create a Kaggle notebook")
    print("2. Add datasets: nivakramuk/asl-restaurant-data + nivakramuk/asl-restaurant-code")
    print("3. Enable GPU (P100 or T4)")
    print("4. Run: !python /kaggle/input/asl-restaurant-code/train_restaurant.py")
    return True


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare restaurant domain model for Kaggle training")
    parser.add_argument("--augment-only", action="store_true", help="Only run augmentation")
    parser.add_argument("--split-only", action="store_true", help="Only create splits")
    parser.add_argument("--stage-only", action="store_true", help="Only stage data+code")
    parser.add_argument("--upload-only", action="store_true", help="Only upload to Kaggle")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")
    args = parser.parse_args()

    any_only = args.augment_only or args.split_only or args.stage_only or args.upload_only

    print("=" * 70)
    print("RESTAURANT DOMAIN MODEL - KAGGLE PREPARATION")
    print("=" * 70)

    restaurant_glosses, common_glosses, all_glosses = load_glosses()
    present, missing = check_augmented_status(all_glosses)

    print(f"\nDomain: Restaurant")
    print(f"  Restaurant glosses: {len(restaurant_glosses)}")
    print(f"  Common glosses: {len(common_glosses)}")
    print(f"  Total classes: {len(all_glosses)}")
    print(f"  Glosses with augmented data: {len(present)}")
    print(f"  Glosses missing augmented data: {len(missing)}")

    # Step 1: Augment
    if not any_only or args.augment_only:
        if not step_augment(dry_run=args.dry_run):
            print("\nFailed at augmentation step. Aborting.")
            return 1
        if args.augment_only:
            return 0

    # Step 2: Split
    if not any_only or args.split_only:
        if not step_split(dry_run=args.dry_run):
            print("\nFailed at split step. Aborting.")
            return 1
        if args.split_only:
            return 0

    # Step 3: Stage
    if not any_only or args.stage_only:
        if not step_stage(dry_run=args.dry_run):
            print("\nFailed at staging step. Aborting.")
            return 1

    # Step 4: Validate
    if not any_only or args.stage_only:
        if not args.dry_run and not step_validate():
            print("\nValidation failed. Fix issues before uploading.")
            return 1
        if args.stage_only:
            return 0

    # Step 5: Upload
    if not any_only or args.upload_only:
        if not step_upload(dry_run=args.dry_run):
            print("\nFailed at upload step.")
            return 1

    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
