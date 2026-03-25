#!/usr/bin/env python3
"""
Prepare ANY Domain Model for Kaggle Training.

Generic pipeline that works for any domain (restaurant, doctor_visit, etc.):
  1. Generate augmented data for missing glosses
  2. Create train/val/test split manifests
  3. Fix manifest paths for Kaggle portability
  4. Stage data + code for Kaggle upload
  5. Validate the staged package (including Kaggle-specific checks)
  6. Upload to Kaggle

Usage:
    python prepare_domain_kaggle.py restaurant              # Full pipeline
    python prepare_domain_kaggle.py doctor_visit             # Different domain
    python prepare_domain_kaggle.py restaurant --augment-only
    python prepare_domain_kaggle.py restaurant --stage-only
    python prepare_domain_kaggle.py restaurant --upload-only
    python prepare_domain_kaggle.py restaurant --dry-run

Kaggle Deployment Lessons Applied:
  - Manifest pickle_pool paths are always relative (not absolute Windows paths)
  - settings.json uses /kaggle/input/datasets/nivakramuk/ prefix (Kaggle convention)
  - Training runner only uses CLI flags that train_asl.py actually supports
  - Auto-detects Kaggle mount paths at runtime
  - Validates manifests don't contain local machine paths
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Kaggle username for dataset slugs
KAGGLE_USER = "nivakramuk"

# Data paths
AUGMENTED_POOL = PROJECT_ROOT / "datasets" / "augmented_pool" / "pickle"
POSE_BY_GLOSS = PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "pose_files_by_gloss"
COMMON_WORDS_JSON = PROJECT_ROOT / "datasets" / "domain-specific" / "common_words.json"


def get_domain_config(domain_name):
    """Load domain config and derive all paths."""
    domain_json = PROJECT_ROOT / "datasets" / "domain-specific" / f"{domain_name}.json"
    if not domain_json.exists():
        print(f"ERROR: Domain config not found: {domain_json}")
        print(f"Available domains:")
        for f in (PROJECT_ROOT / "datasets" / "domain-specific").glob("*.json"):
            if f.name != "common_words.json":
                print(f"  - {f.stem}")
        sys.exit(1)

    with open(domain_json) as f:
        domain = json.load(f)
    with open(COMMON_WORDS_JSON) as f:
        common = json.load(f)

    domain_glosses = [g.lower() for g in domain['classes']]
    common_glosses = [g.lower() for g in common['classes']]
    all_glosses = sorted(set(domain_glosses + common_glosses))
    total_classes = len(all_glosses)

    return {
        "name": domain_name,
        "scenario": domain.get("scenario", domain_name.replace("_", " ").title()),
        "domain_json": domain_json,
        "domain_glosses": domain_glosses,
        "common_glosses": common_glosses,
        "all_glosses": all_glosses,
        "total_classes": total_classes,
        "staging_dir": PROJECT_ROOT / f"kaggle-staging-{domain_name}",
        "splits_dir": PROJECT_ROOT / "datasets" / "augmented_pool" / "splits" / f"{total_classes}_{domain_name}",
        "data_slug": f"asl-{domain_name}-data",
        "code_slug": f"asl-{domain_name}-code",
    }


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

def step_augment(cfg, dry_run=False):
    """Generate augmented data for glosses missing from the augmented pool."""
    print("\n" + "=" * 70)
    print("STEP 1: Generate augmented data for missing glosses")
    print("=" * 70)

    present, missing = check_augmented_status(cfg["all_glosses"])

    print(f"\nGlosses with augmented data: {len(present)}")
    print(f"Glosses missing augmented data: {len(missing)}")

    if not missing:
        print("All glosses have augmented data. Skipping augmentation.")
        return True

    print(f"\nMissing glosses: {missing}")

    if dry_run:
        print("\n[DRY RUN] Would generate augmented data for these glosses.")
        return True

    # Create a temporary gloss list file
    gloss_list_file = PROJECT_ROOT / "datasets" / "domain-specific" / f"{cfg['name']}_missing_glosses.json"
    with open(gloss_list_file, 'w') as f:
        json.dump({"classes": [g.upper() for g in missing]}, f, indent=2)

    print(f"\nRunning augmentation for {len(missing)} glosses...")

    augment_script = PROJECT_ROOT / "dataset-utilities" / "augmentation" / "generate_augmented_dataset.py"
    result = subprocess.run(
        [sys.executable, str(augment_script), "--gloss-file", str(gloss_list_file), "--landmark-config", "83pt"],
        cwd=str(PROJECT_ROOT)
    )

    # Clean up temp file
    gloss_list_file.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"\nERROR: Augmentation failed with return code {result.returncode}")
        return False

    _, still_missing = check_augmented_status(cfg["all_glosses"])
    if still_missing:
        print(f"\nWARNING: {len(still_missing)} glosses still missing: {still_missing}")
        return False

    print(f"\nAugmentation complete. All {len(cfg['all_glosses'])} glosses now have data.")
    return True


# ─── Step 2: Create split manifests ──────────────────────────────────────────

def _fix_manifest_paths(splits_dir):
    """
    CRITICAL: Fix manifest pickle_pool paths to be relative.

    The split script writes the absolute local path of the input directory
    into the manifest's pickle_pool field. This breaks on Kaggle since those
    Windows paths don't exist there.

    Fix: Replace with a relative path that train_asl.py's --augmented-path
    will override anyway. The relative path serves as a fallback.
    """
    fixed = 0
    for manifest_name in ['train_manifest.json', 'val_manifest.json', 'test_manifest.json']:
        manifest_path = splits_dir / manifest_name
        if not manifest_path.exists():
            continue

        with open(manifest_path) as f:
            data = json.load(f)

        old_pool = data.get('pickle_pool', '')
        # Check if it contains an absolute path (Windows or Unix)
        if old_pool and (old_pool.startswith('/') or ':' in old_pool):
            data['pickle_pool'] = '../pickle'
            with open(manifest_path, 'w') as f:
                json.dump(data, f, indent=2)
            fixed += 1
            print(f"  Fixed {manifest_name}: '{old_pool}' -> '../pickle'")

    if fixed:
        print(f"  Fixed {fixed} manifest(s) with absolute paths")
    else:
        print(f"  All manifests already have portable paths")


def step_split(cfg, dry_run=False):
    """Create train/val/test split manifests."""
    print("\n" + "=" * 70)
    print("STEP 2: Create train/val/test split manifests")
    print("=" * 70)

    splits_dir = cfg["splits_dir"]
    all_glosses = cfg["all_glosses"]

    print(f"\nTotal classes: {len(all_glosses)}")
    print(f"Split output: {splits_dir}")

    if dry_run:
        print("\n[DRY RUN] Would create splits at the above location.")
        return True

    split_script = PROJECT_ROOT / "dataset-utilities" / "dataset-splitting" / "stratified_family_split.py"

    # Create a temporary filtered pool with only our domain glosses
    filtered_pool = PROJECT_ROOT / "datasets" / "augmented_pool" / f"pickle_{cfg['name']}_temp"
    if filtered_pool.exists():
        # Clean up any previous temp pool
        if sys.platform == 'win32':
            for item in filtered_pool.iterdir():
                if item.is_dir():
                    subprocess.run(['cmd', '/c', 'rmdir', str(item)], capture_output=True)
            filtered_pool.rmdir()
        else:
            shutil.rmtree(filtered_pool)
    filtered_pool.mkdir(parents=True)

    # Create junctions/symlinks to relevant gloss dirs
    linked_count = 0
    for gloss in all_glosses:
        src = AUGMENTED_POOL / gloss
        dst = filtered_pool / gloss
        if src.exists():
            if sys.platform == 'win32':
                subprocess.run(['cmd', '/c', 'mklink', '/J', str(dst), str(src)], capture_output=True)
            else:
                dst.symlink_to(src)
            linked_count += 1
        else:
            print(f"  WARNING: No augmented data for '{gloss}'")

    print(f"Created filtered pool with {linked_count} gloss directories")

    result = subprocess.run([
        sys.executable, str(split_script),
        "--input-dir", str(filtered_pool),
        "--output-dir", str(splits_dir),
        "--balance-train", "--train-target", "200", "--seed", "42",
    ], cwd=str(PROJECT_ROOT))

    # Clean up filtered pool
    if filtered_pool.exists():
        if sys.platform == 'win32':
            for item in filtered_pool.iterdir():
                if item.is_dir():
                    subprocess.run(['cmd', '/c', 'rmdir', str(item)], capture_output=True)
            filtered_pool.rmdir()
        else:
            shutil.rmtree(filtered_pool)

    if result.returncode != 0:
        print(f"\nERROR: Split creation failed")
        return False

    # Verify manifests exist
    for split in ['train_manifest.json', 'val_manifest.json', 'test_manifest.json']:
        if not (splits_dir / split).exists():
            print(f"ERROR: Expected manifest not found: {split}")
            return False

    # CRITICAL: Fix manifest paths for Kaggle portability
    _fix_manifest_paths(splits_dir)

    print("\nSplit manifests created successfully.")
    return True


# ─── Step 3: Stage for Kaggle ────────────────────────────────────────────────

def _generate_kaggle_runner(cfg):
    """
    Generate the Kaggle training runner script.

    LESSONS APPLIED:
    - Uses /kaggle/input/datasets/{user}/ path convention (not /kaggle/input/)
    - Only passes CLI flags that train_asl.py actually supports
    - Does NOT pass: --epochs, --batch-size, --patience, --output-dir, --class-mapping
    - Uses --early-stopping (not --patience)
    - Auto-detects mount paths at runtime
    """
    domain = cfg["name"]
    total_classes = cfg["total_classes"]
    data_slug = cfg["data_slug"]
    code_slug = cfg["code_slug"]
    splits_subdir = f"{total_classes}_{domain}"

    return f'''#!/usr/bin/env python3
"""
Kaggle Training Runner for {cfg["scenario"]} Domain Model.

Usage (in Kaggle notebook):
    !python /kaggle/input/datasets/{KAGGLE_USER}/{code_slug}/train_domain.py
    !python /kaggle/input/datasets/{KAGGLE_USER}/{code_slug}/train_domain.py --early-stopping 100
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# ─── Kaggle path resolution ──────────────────────────────────────────────────
# Kaggle mounts datasets at /kaggle/input/datasets/<user>/<slug>/
# This auto-detects the correct mount point at runtime.

def _resolve_kaggle_path(dataset_name):
    """Resolve Kaggle dataset path, checking known mount conventions."""
    candidates = [
        Path(f"/kaggle/input/datasets/{KAGGLE_USER}/{dataset_name}"),
        Path(f"/kaggle/input/{dataset_name}"),
    ]
    for p in candidates:
        if p.exists():
            return p
    # If neither exists, print diagnostics and fail
    input_dir = Path("/kaggle/input")
    contents = list(input_dir.iterdir()) if input_dir.exists() else []
    print(f"ERROR: Could not find dataset '{dataset_name}'")
    print(f"  Searched: {{[str(c) for c in candidates]}}")
    print(f"  /kaggle/input/ contains: {{contents}}")
    sys.exit(1)

print("Resolving Kaggle dataset paths...")
CODE_DIR = _resolve_kaggle_path("{code_slug}")
DATA_DIR = _resolve_kaggle_path("{data_slug}")
WORK_DIR = Path("/kaggle/working")
print(f"  CODE_DIR: {{CODE_DIR}}")
print(f"  DATA_DIR: {{DATA_DIR}}")

# Add code directory to path so train_asl.py can find config/ and models/
sys.path.insert(0, str(CODE_DIR))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train {cfg['scenario']} domain ASL model")
    # ONLY flags that train_asl.py actually supports:
    parser.add_argument("--early-stopping", type=int, default=80, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--model-size", choices=["tiny", "small", "large"], default="small")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--no-finger-features", action="store_true")
    args = parser.parse_args()

    # Load domain config
    config_path = DATA_DIR / "domain_config" / "{domain}_{total_classes}class.json"
    with open(config_path) as f:
        domain_config = json.load(f)

    num_classes = domain_config["num_classes"]
    print(f"Training {cfg['scenario']} domain model: {{num_classes}} classes")
    print(f"  Domain classes: {{domain_config['domain_classes']}}")
    print(f"  Common classes: {{domain_config['common_classes']}}")

    # Data paths on Kaggle
    pickle_pool = DATA_DIR / "datasets" / "augmented_pool" / "pickle"
    manifest_dir = DATA_DIR / "splits" / "{splits_subdir}"

    # Build training command — ONLY use flags train_asl.py supports
    train_script = CODE_DIR / "models" / "training-scripts" / "train_asl.py"
    cmd = [
        sys.executable, str(train_script),
        "--classes", str(num_classes),
        "--dataset", "augmented",
        "--augmented-path", str(pickle_pool),
        "--manifest-dir", str(manifest_dir),
        "--architecture", "openhands",
        "--model-size", args.model_size,
        "--early-stopping", str(args.early_stopping),
        "--lr", str(args.lr),
        "--dropout", str(args.dropout),
        "--label-smoothing", str(args.label_smoothing),
    ]

    if args.no_finger_features:
        cmd.append("--no-finger-features")

    print(f"\\nCommand: {{' '.join(cmd)}}")
    print("=" * 70)

    result = subprocess.run(cmd, cwd=str(WORK_DIR))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
'''


def stage_data(cfg):
    """Stage augmented pickle files + manifests + domain config."""
    staging_dir = cfg["staging_dir"]
    data_dir = staging_dir / "asl-data"
    if data_dir.exists():
        shutil.rmtree(data_dir)

    all_glosses = cfg["all_glosses"]
    splits_dir = cfg["splits_dir"]

    print(f"\nStaging data for {len(all_glosses)} glosses...")

    # Copy pickle data
    dst_pool = data_dir / "datasets" / "augmented_pool" / "pickle"
    total_files = 0
    for gloss in all_glosses:
        src_dir = AUGMENTED_POOL / gloss
        if src_dir.exists():
            dst_dir = dst_pool / gloss
            shutil.copytree(src_dir, dst_dir)
            total_files += sum(1 for _ in dst_dir.glob('*.pkl'))
        else:
            print(f"  WARNING: No augmented data for '{gloss}'")

    # Copy split manifests
    splits_subdir = f"{cfg['total_classes']}_{cfg['name']}"
    if splits_dir.exists():
        splits_dst = data_dir / "splits" / splits_subdir
        shutil.copytree(splits_dir, splits_dst)
        # Re-fix manifest paths in staged copy (belt and suspenders)
        _fix_manifest_paths(splits_dst)

    # Domain config
    configs_dst = data_dir / "domain_config"
    configs_dst.mkdir(parents=True)
    shutil.copy2(cfg["domain_json"], configs_dst / f"{cfg['name']}.json")
    shutil.copy2(COMMON_WORDS_JSON, configs_dst / "common_words.json")

    # Combined class mapping
    combined = {
        "domain": cfg["name"],
        "scenario": cfg["scenario"],
        "num_classes": cfg["total_classes"],
        "domain_classes": len(cfg["domain_glosses"]),
        "common_classes": len(cfg["common_glosses"]),
        "classes": [g.upper() for g in cfg["all_glosses"]],
        "class_to_index": {g.upper(): i for i, g in enumerate(cfg["all_glosses"])},
        "class_index_mapping": {str(i): g for i, g in enumerate(cfg["all_glosses"])},
    }
    with open(configs_dst / f"{cfg['name']}_{cfg['total_classes']}class.json", 'w') as f:
        json.dump(combined, f, indent=2)

    # Kaggle dataset metadata
    metadata = {
        "title": f"ASL {cfg['scenario']} Domain Data",
        "id": f"{KAGGLE_USER}/{cfg['data_slug']}",
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(data_dir / "dataset-metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    total_size = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
    print(f"\nasl-data staged: {total_files} pickle files, {total_size / 1e9:.2f} GB")


def stage_code(cfg):
    """Stage code files for Kaggle."""
    staging_dir = cfg["staging_dir"]
    code_dir = staging_dir / "asl-code"
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

    # Config module
    copy_file("config/__init__.py")
    copy_file("config/paths.py")

    # Kaggle-specific settings.json
    # LESSON: project_root must point to code dir for module resolution
    # LESSON: data_root uses /kaggle/input/datasets/{user}/ prefix
    kaggle_settings = {
        "data_root": f"/kaggle/input/datasets/{KAGGLE_USER}/{cfg['data_slug']}/datasets/augmented_pool/pickle",
        "project_root": f"/kaggle/input/datasets/{KAGGLE_USER}/{cfg['code_slug']}",
        "dataset_splits": {}
    }
    settings_path = code_dir / "config" / "settings.json"
    with open(settings_path, 'w') as f:
        json.dump(kaggle_settings, f, indent=2)

    # Training script
    copy_file("models/training-scripts/train_asl.py")

    # Model source
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

    # Finger features
    copy_file("dataset-utilities/landmarks-extraction/__init__.py")
    copy_file("dataset-utilities/landmarks-extraction/finger_features.py")
    copy_file("dataset-utilities/landmarks-extraction/landmark_config.py")

    # Profiling script
    copy_file("project-utilities/evaluation_metrics/synthetic_evaluation/profile_model_signs.py")

    # Requirements
    copy_file("requirements.txt", "config/requirements.txt")

    # Domain training runner (generated, not copied)
    runner_content = _generate_kaggle_runner(cfg)
    with open(code_dir / "train_domain.py", 'w') as f:
        f.write(runner_content)

    # Kaggle dataset metadata
    metadata = {
        "title": f"ASL {cfg['scenario']} Training Code",
        "id": f"{KAGGLE_USER}/{cfg['code_slug']}",
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(code_dir / "dataset-metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    total_files = sum(1 for _ in code_dir.rglob("*") if _.is_file())
    print(f"\nasl-code staged: {total_files} files")


def step_stage(cfg, dry_run=False):
    """Stage data and code for Kaggle upload."""
    print("\n" + "=" * 70)
    print("STEP 3: Stage data + code for Kaggle")
    print("=" * 70)

    if dry_run:
        print(f"\n[DRY RUN] Would stage to {cfg['staging_dir']}")
        return True

    cfg["staging_dir"].mkdir(exist_ok=True)
    stage_data(cfg)
    stage_code(cfg)
    return True


# ─── Step 4: Validate ────────────────────────────────────────────────────────

def step_validate(cfg):
    """Validate the staged Kaggle package with Kaggle-specific checks."""
    print("\n" + "=" * 70)
    print("STEP 4: Validate staged package")
    print("=" * 70)

    errors = []
    warnings = []
    staging_dir = cfg["staging_dir"]
    data_dir = staging_dir / "asl-data"
    code_dir = staging_dir / "asl-code"

    if not data_dir.exists():
        errors.append("asl-data directory not found")
    if not code_dir.exists():
        errors.append("asl-code directory not found")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        return False

    # Check all glosses have data
    pickle_dir = data_dir / "datasets" / "augmented_pool" / "pickle"
    total_files = 0
    min_files = float('inf')
    min_gloss = ""
    missing_glosses = []

    for gloss in cfg["all_glosses"]:
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

    print(f"\n  Glosses: {len(cfg['all_glosses']) - len(missing_glosses)}/{len(cfg['all_glosses'])}")
    print(f"  Total pickle files: {total_files}")
    if min_files < float('inf'):
        print(f"  Min files per gloss: {min_files} ({min_gloss})")

    # Check split manifests
    splits_subdir = f"{cfg['total_classes']}_{cfg['name']}"
    splits_dir = data_dir / "splits" / splits_subdir
    for split_name in ['train_manifest.json', 'val_manifest.json', 'test_manifest.json']:
        manifest_path = splits_dir / split_name
        if not manifest_path.exists():
            errors.append(f"Missing manifest: {split_name}")
        else:
            with open(manifest_path) as f:
                manifest = json.load(f)
            n_classes = manifest.get('num_classes', 0)
            n_samples = manifest.get('total_samples', 0)
            pool_path = manifest.get('pickle_pool', '')
            print(f"  {split_name}: {n_classes} classes, {n_samples} samples")

            # CRITICAL: Check for local machine paths in manifests
            if pool_path and (':' in pool_path or 'Users' in pool_path or 'home' in pool_path):
                errors.append(f"{split_name} contains local machine path in pickle_pool: '{pool_path}'")

    # Check domain config
    config_name = f"{cfg['name']}_{cfg['total_classes']}class.json"
    config_path = data_dir / "domain_config" / config_name
    if not config_path.exists():
        errors.append(f"Missing {config_name}")
    else:
        with open(config_path) as f:
            domain_cfg = json.load(f)
        print(f"  Domain config: {domain_cfg['num_classes']} classes "
              f"({domain_cfg['domain_classes']} domain + {domain_cfg['common_classes']} common)")

    # Check essential code files
    essential_code = [
        "models/training-scripts/train_asl.py",
        "models/openhands-modernized/src/openhands_modernized.py",
        "config/paths.py",
        "config/settings.json",
        "train_domain.py",
    ]
    for code_file in essential_code:
        if not (code_dir / code_file).exists():
            errors.append(f"Missing code file: {code_file}")

    # CRITICAL: Validate settings.json has correct Kaggle paths
    settings_path = code_dir / "config" / "settings.json"
    if settings_path.exists():
        with open(settings_path) as f:
            settings = json.load(f)
        project_root = settings.get("project_root", "")
        data_root = settings.get("data_root", "")

        if project_root and not project_root.startswith("/kaggle"):
            errors.append(f"settings.json project_root not a Kaggle path: '{project_root}'")
        if data_root and not data_root.startswith("/kaggle"):
            errors.append(f"settings.json data_root not a Kaggle path: '{data_root}'")

        # Check paths include the user prefix
        if KAGGLE_USER not in project_root:
            warnings.append(f"settings.json project_root missing user prefix '{KAGGLE_USER}'")
        if KAGGLE_USER not in data_root:
            warnings.append(f"settings.json data_root missing user prefix '{KAGGLE_USER}'")

    # Check metadata files
    for meta_dir in [data_dir, code_dir]:
        if not (meta_dir / "dataset-metadata.json").exists():
            errors.append(f"Missing dataset-metadata.json in {meta_dir.name}")

    # CRITICAL: Validate train_domain.py doesn't use invalid train_asl.py flags
    runner_path = code_dir / "train_domain.py"
    if runner_path.exists():
        runner_text = runner_path.read_text()
        invalid_flags = ['--epochs', '--batch-size', '--patience', '--output-dir', '--class-mapping', '--resume']
        for flag in invalid_flags:
            if f'"{flag}"' in runner_text or f"'{flag}'" in runner_text:
                errors.append(f"train_domain.py uses unsupported train_asl.py flag: {flag}")

    total_size = sum(f.stat().st_size for f in staging_dir.rglob("*") if f.is_file())
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

def step_upload(cfg, dry_run=False):
    """Upload staged data and code to Kaggle."""
    print("\n" + "=" * 70)
    print("STEP 5: Upload to Kaggle")
    print("=" * 70)

    staging_dir = cfg["staging_dir"]
    data_dir = staging_dir / "asl-data"
    code_dir = staging_dir / "asl-code"

    if dry_run:
        print("\n[DRY RUN] Would upload:")
        print(f"  Data: {cfg['data_slug']}")
        print(f"  Code: {cfg['code_slug']}")
        return True

    for slug, staged_dir in [(cfg['data_slug'], data_dir), (cfg['code_slug'], code_dir)]:
        print(f"\nUploading {slug}...")

        result = subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(staged_dir), "--dir-mode", "zip"],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            if "already exists" in (result.stderr + result.stdout).lower():
                print(f"  Dataset exists, creating new version...")
                result = subprocess.run(
                    ["kaggle", "datasets", "version", "-p", str(staged_dir),
                     "-m", f"{cfg['scenario']} domain update", "--dir-mode", "zip"],
                    capture_output=True, text=True
                )

            if result.returncode != 0:
                print(f"  ERROR uploading {slug}:")
                print(f"  stdout: {result.stdout}")
                print(f"  stderr: {result.stderr}")
                return False

        print(f"  {result.stdout.strip()}")

    code_slug = cfg['code_slug']
    print(f"\nUpload complete!")
    print(f"\nTo train on Kaggle:")
    print(f"  1. Add datasets: {KAGGLE_USER}/{cfg['data_slug']} + {KAGGLE_USER}/{code_slug}")
    print(f"  2. Enable GPU (P100 or T4)")
    print(f"  3. Run: !python /kaggle/input/datasets/{KAGGLE_USER}/{code_slug}/train_domain.py")
    return True


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare any domain model for Kaggle training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python prepare_domain_kaggle.py restaurant
    python prepare_domain_kaggle.py doctor_visit --dry-run
    python prepare_domain_kaggle.py restaurant --stage-only
        """)
    parser.add_argument("domain", help="Domain name (e.g., restaurant, doctor_visit)")
    parser.add_argument("--augment-only", action="store_true")
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-target", type=int, default=200,
                       help="Target train samples per class for balancing (default: 200)")
    args = parser.parse_args()

    cfg = get_domain_config(args.domain)
    any_only = args.augment_only or args.split_only or args.stage_only or args.upload_only

    print("=" * 70)
    print(f"{cfg['scenario'].upper()} DOMAIN MODEL - KAGGLE PREPARATION")
    print("=" * 70)

    present, missing = check_augmented_status(cfg["all_glosses"])
    print(f"\nDomain: {cfg['scenario']}")
    print(f"  Domain glosses: {len(cfg['domain_glosses'])}")
    print(f"  Common glosses: {len(cfg['common_glosses'])}")
    print(f"  Total classes: {cfg['total_classes']}")
    print(f"  Glosses with augmented data: {len(present)}")
    print(f"  Glosses missing augmented data: {len(missing)}")

    if not any_only or args.augment_only:
        if not step_augment(cfg, dry_run=args.dry_run):
            return 1
        if args.augment_only:
            return 0

    if not any_only or args.split_only:
        if not step_split(cfg, dry_run=args.dry_run):
            return 1
        if args.split_only:
            return 0

    if not any_only or args.stage_only:
        if not step_stage(cfg, dry_run=args.dry_run):
            return 1
        if not args.dry_run and not step_validate(cfg):
            return 1
        if args.stage_only:
            return 0

    if not any_only or args.upload_only:
        if not step_upload(cfg, dry_run=args.dry_run):
            return 1

    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
