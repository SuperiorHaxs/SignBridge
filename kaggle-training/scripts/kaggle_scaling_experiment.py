#!/usr/bin/env python3
"""
Kaggle Scaling Experiment Runner: Doctor Visit Domain

Single executable script for running vocabulary scaling sweep on Kaggle.
Run one sweep point at a time within the same Kaggle session.

Usage (in Kaggle notebook cells):
    # First run — sets up env, loads ALL glosses into cache, trains N=20
    !python /kaggle/input/datasets/nivakramuk/asl-code/experiment/kaggle_scaling_experiment.py --domain-n 20

    # With tuned hyperparams (recommended):
    !python /kaggle/input/datasets/nivakramuk/asl-code/experiment/kaggle_scaling_experiment.py --domain-n 20 --model-size large --dropout 0.5 --lr 5e-5

    # Re-run a point with different params (--force overwrites previous result):
    !python /kaggle/input/datasets/nivakramuk/asl-code/experiment/kaggle_scaling_experiment.py --domain-n 20 --model-size large --force

    # Continue incrementing (cache reused across runs):
    !python /kaggle/input/datasets/nivakramuk/asl-code/experiment/kaggle_scaling_experiment.py --domain-n 25 --model-size large --dropout 0.5

    # After all runs, analyze
    !python /kaggle/input/datasets/nivakramuk/asl-code/experiment/kaggle_scaling_experiment.py --analyze

Data layout on Kaggle:
    /kaggle/input/asl-data/
        datasets/augmented_pool/pickle/{gloss}/*.pkl   (all 107 glosses)
    /kaggle/input/asl-code/
        kaggle_scaling_experiment.py   (this script)
        experiment_config.json         (FPS ordering, common words)
        config/                        (paths.py, settings.json)
        models/training-scripts/       (train_asl.py)
        models/openhands-modernized/   (model code)
        dataset-utilities/             (landmarks extraction)
        project-utilities/evaluation_metrics/synthetic_evaluation/
            profile_model_signs.py     (profiling script)

Session cache:
    /kaggle/working/cache/            (precomputed .pt dataset cache)
    /kaggle/working/sweep_n{N}/       (results per sweep point)
"""

import json
import os
import sys
import shutil
import subprocess
import time
import argparse
import random
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ─── Kaggle paths ───
WORKING = Path("/kaggle/working")
CACHE_DIR = WORKING / "cache"
RESULTS_DIR = WORKING / "scaling_results"

# These get set during setup
CODE_INPUT = None
DATA_INPUT = None
PICKLE_POOL = None

# ─── Experiment config ───
TOP1_THRESHOLD = 70.0
TOP3_THRESHOLD = 85.0
COMMON_COUNT = 25


def find_input_path(dataset_name):
    """Auto-detect Kaggle dataset mount path."""
    candidates = []
    new_style = Path("/kaggle/input/datasets")
    if new_style.exists():
        for user_dir in new_style.iterdir():
            candidates.append(user_dir / dataset_name)
    candidates.append(Path(f"/kaggle/input/{dataset_name}"))

    for p in candidates:
        if p.exists() and any(p.iterdir()):
            return p

    raise FileNotFoundError(
        f"Could not find dataset '{dataset_name}' in /kaggle/input/. "
        f"Searched: {[str(c) for c in candidates]}"
    )


def setup_environment():
    """One-time Kaggle environment setup. Idempotent — skips if already done."""
    global CODE_INPUT, DATA_INPUT, PICKLE_POOL

    CODE_INPUT = find_input_path("asl-code")
    DATA_INPUT = find_input_path("asl-data")
    print(f"Code input: {CODE_INPUT}")
    print(f"Data input: {DATA_INPUT}")

    # Detect pickle pool location
    pool_candidate = DATA_INPUT / "datasets" / "augmented_pool" / "pickle"
    if not pool_candidate.exists():
        pool_candidate = DATA_INPUT / "augmented_pool" / "pickle"
    PICKLE_POOL = pool_candidate
    print(f"Pickle pool: {PICKLE_POOL}")

    marker = WORKING / ".scaling_setup_done"
    if not marker.exists():
        print("Setting up working environment...")

        # Copy code files to writable location
        for item in ["config", "models", "dataset-utilities",
                     "project-utilities", "experiment"]:
            src = CODE_INPUT / item
            dst = WORKING / item
            if src.exists():
                if src.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

        # Update settings.json for Kaggle paths
        settings_path = WORKING / "config" / "settings.json"
        if settings_path.exists():
            with open(settings_path) as f:
                settings = json.load(f)
            data_root = DATA_INPUT / "datasets" / "wlasl_poses_complete"
            if not data_root.exists():
                data_root = DATA_INPUT / "wlasl_poses_complete"
            settings["data_root"] = str(data_root)
            settings["project_root"] = str(WORKING)
            with open(settings_path, "w") as f:
                json.dump(settings, f, indent=2)

        # Install deps
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                       "pose-format"], check=False)

        # Create dirs
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        marker.touch()
        print("Environment ready.")
    else:
        print("Environment already set up.")

    return PICKLE_POOL


def load_experiment_config():
    """Load FPS ordering and common words from experiment_config.json."""
    # Search multiple possible locations
    search_paths = [
        Path(__file__).parent / "experiment_config.json",
        Path(__file__).parent / "experiment" / "experiment_config.json",
    ]
    if CODE_INPUT:
        search_paths.extend([
            CODE_INPUT / "experiment_config.json",
            CODE_INPUT / "experiment" / "experiment_config.json",
        ])

    for cfg_path in search_paths:
        if cfg_path.exists():
            with open(cfg_path) as f:
                return json.load(f)

    raise FileNotFoundError(
        "experiment_config.json not found. "
        "Run prepare step locally first."
    )


def create_manifests(common_glosses, domain_glosses, sweep_dir, pickle_pool):
    """Create train/val/test split manifests for a sweep point.

    Deterministic splitting (seed=42) ensures reproducibility across runs.
    """
    all_glosses = sorted(set(g.lower() for g in common_glosses + domain_glosses))
    random.seed(42)

    manifest_data = {
        "train": {"split": "train", "pickle_pool": str(pickle_pool),
                  "total_samples": 0, "classes": {}},
        "val": {"split": "val", "pickle_pool": str(pickle_pool),
                "total_samples": 0, "classes": {}},
        "test": {"split": "test", "pickle_pool": str(pickle_pool),
                 "total_samples": 0, "classes": {}},
    }

    for gloss in all_glosses:
        gloss_dir = pickle_pool / gloss
        if not gloss_dir.exists():
            print(f"  WARNING: No pickle dir for {gloss}, skipping")
            continue

        # Find all original videos and their augmented families
        all_files = sorted(gloss_dir.glob('*.pkl'))
        originals = [f for f in all_files if 'aug' not in f.name]
        augmented = [f for f in all_files if 'aug' in f.name]

        # Group augmented by original video id
        families = {}
        for orig in originals:
            vid_id = orig.stem
            family_augs = [f for f in augmented
                          if f.name.startswith(vid_id + '_aug_')]
            families[vid_id] = {
                "original": orig.name,
                "augmented": [f.name for f in family_augs],
            }

        # Split families: 70/15/15 (deterministic with seed=42)
        family_ids = list(families.keys())
        random.shuffle(family_ids)
        n = len(family_ids)

        if n >= 3:
            n_val = max(1, round(n * 0.15))
            n_test = max(1, round(n * 0.15))
            n_train = n - n_val - n_test
        elif n == 2:
            n_train, n_val, n_test = 1, 1, 0
        else:
            n_train, n_val, n_test = 1, 0, 0

        split_assignment = {}
        for i, fid in enumerate(family_ids):
            if i < n_train:
                split_assignment[fid] = "train"
            elif i < n_train + n_val:
                split_assignment[fid] = "val"
            else:
                split_assignment[fid] = "test"

        for fid, split_name in split_assignment.items():
            fam = families[fid]
            all_fam_files = [fam["original"]] + fam["augmented"]

            if gloss not in manifest_data[split_name]["classes"]:
                manifest_data[split_name]["classes"][gloss] = []

            manifest_data[split_name]["classes"][gloss].append({
                "video_id": fid,
                "files": all_fam_files,
            })
            manifest_data[split_name]["total_samples"] += len(all_fam_files)

    # Save manifests
    sweep_dir.mkdir(parents=True, exist_ok=True)
    for split_name, data in manifest_data.items():
        manifest_path = sweep_dir / f"{split_name}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(data, f, indent=2)

    print(f"  Splits: train={manifest_data['train']['total_samples']}, "
          f"val={manifest_data['val']['total_samples']}, "
          f"test={manifest_data['test']['total_samples']}")

    return manifest_data


def profile_model(model_dir, val_manifest, output_dir):
    """Profile all signs in the model using profile_model_signs.py."""
    profile_script = WORKING / "project-utilities" / "evaluation_metrics" / \
        "synthetic_evaluation" / "profile_model_signs.py"

    if not profile_script.exists():
        print(f"  ERROR: Profile script not found at {profile_script}")
        print("  Falling back to inline profiling...")
        return profile_model_inline(model_dir, val_manifest, output_dir)

    cmd = [
        sys.executable, str(profile_script),
        str(model_dir), str(val_manifest),
        "--output-dir", str(output_dir),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join([
        str(WORKING),
        str(WORKING / "models" / "openhands-modernized" / "src"),
        str(WORKING / "dataset-utilities" / "landmarks-extraction"),
    ])

    print(f"  Running profiler...")
    result = subprocess.run(cmd, env=env, cwd=str(WORKING))
    return result.returncode == 0


def profile_model_inline(model_dir, val_manifest_path, output_dir):
    """Inline profiling when external script is unavailable."""
    sys.path.insert(0, str(WORKING))
    sys.path.insert(0, str(WORKING / "models" / "openhands-modernized" / "src"))
    sys.path.insert(0, str(WORKING / "dataset-utilities" / "landmarks-extraction"))

    import torch
    from openhands_modernized import OpenHandsModel, OpenHandsConfig, WLASLPoseProcessor

    # Load model
    with open(model_dir / "config.json") as f:
        cfg_dict = json.load(f)
    with open(model_dir / "class_index_mapping.json") as f:
        class_mapping = json.load(f)

    config = OpenHandsConfig(**cfg_dict)
    model = OpenHandsModel(config)
    state_dict = torch.load(model_dir / "pytorch_model.bin",
                           map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    processor = WLASLPoseProcessor()
    num_classes = len(class_mapping)

    # Load val manifest
    with open(val_manifest_path) as f:
        manifest = json.load(f)

    pickle_pool = Path(manifest["pickle_pool"])
    classes = manifest["classes"]
    profiles = {}

    for gloss, families in classes.items():
        all_files = []
        for fam in families:
            for fname in fam["files"]:
                fpath = pickle_pool / gloss / fname
                if fpath.exists():
                    all_files.append(fpath)

        if not all_files:
            continue

        top1_correct = 0
        top3_hits = 0
        top1_confs = []
        correct_in_top3_confs = []

        for fpath in all_files:
            try:
                with open(fpath, "rb") as f:
                    data = pickle.load(f)
                kp = data["keypoints"]
                result = processor.process_keypoints(kp)
                tensor = torch.tensor(result["pose"], dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    logits = model(tensor)
                probs = torch.softmax(logits, dim=-1)[0]
                top5_vals, top5_idx = torch.topk(probs, min(5, num_classes))

                top5_glosses = [class_mapping[str(i.item())] for i in top5_idx]
                top5_confs = [v.item() * 100 for v in top5_vals]

                top1_confs.append(top5_confs[0])
                if top5_glosses[0] == gloss:
                    top1_correct += 1

                if gloss in top5_glosses[:3]:
                    top3_hits += 1
                    idx_in_top3 = top5_glosses[:3].index(gloss)
                    correct_in_top3_confs.append(top5_confs[idx_in_top3])
            except Exception:
                continue

        n = len(all_files)
        if n == 0:
            continue

        profiles[gloss] = {
            "n_pickles": n,
            "top1_accuracy": top1_correct / n * 100,
            "top3_hit_rate": top3_hits / n * 100,
            "avg_top1_confidence": np.mean(top1_confs) if top1_confs else 0,
            "std_top1_confidence": np.std(top1_confs) if top1_confs else 0,
            "avg_correct_in_top3_confidence":
                np.mean(correct_in_top3_confs) if correct_in_top3_confs else 0,
            "std_correct_in_top3_confidence":
                np.std(correct_in_top3_confs) if correct_in_top3_confs else 0,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "per_sign_profiles.json"
    with open(profile_path, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"  Profiles saved: {profile_path} ({len(profiles)} signs)")
    return True


def compute_pass_rate(profiles):
    """Compute pass rate from sign profiles."""
    n_signs = len(profiles)
    n_pass = sum(1 for s in profiles.values()
                 if s['top3_hit_rate'] >= TOP3_THRESHOLD
                 and s['avg_top1_confidence'] >= TOP1_THRESHOLD)
    pass_rate = n_pass / n_signs * 100 if n_signs > 0 else 0
    return n_pass, n_signs, pass_rate


def curate_vocabulary(profiles, common_glosses, domain_glosses, confusion_threshold=0.5):
    """Cheap curation: identify conflict pairs and drop the less important sign.

    A 'conflict pair' is when a failing sign sends >confusion_threshold of its
    errors to one other sign in the vocabulary.

    Priority for keeping: domain signs > common signs. Within same category,
    keep the sign with better top3_hit_rate.

    Returns dict with curated stats (no re-training needed).
    """
    common_set = set(g.upper() for g in common_glosses)
    domain_set = set(g.upper() for g in domain_glosses)
    all_signs = set(profiles.keys())

    # Find failing signs
    failing = {s for s, p in profiles.items()
               if not (p['top3_hit_rate'] >= TOP3_THRESHOLD
                       and p['avg_top1_confidence'] >= TOP1_THRESHOLD)}

    # Build conflict pairs from failing signs
    conflicts = []  # (sign_to_drop, sign_kept, confusion_pct)
    for sign in failing:
        p = profiles[sign]
        confusions = p.get('confusions', {})
        if not confusions:
            continue

        total_errors = sum(confusions.values())
        if total_errors == 0:
            continue

        # Find dominant confusion target
        top_target, top_count = max(confusions.items(), key=lambda x: x[1])
        confusion_pct = top_count / total_errors

        if confusion_pct >= confusion_threshold and top_target in all_signs:
            conflicts.append((sign, top_target, confusion_pct, top_count))

    # Decide which sign to drop from each conflict
    drops = set()
    conflict_details = []

    for sign, target, conf_pct, conf_count in conflicts:
        # sign is already failing — check if target is passing
        if target not in failing:
            # Target passes, sign fails → drop the failing sign
            drops.add(sign)
            conflict_details.append({
                "dropped": sign,
                "kept": target,
                "reason": f"{sign} fails, confused with passing {target} ({conf_pct:.0%})",
            })
        else:
            # Both fail — drop the less important one
            sign_is_domain = sign in domain_set
            target_is_domain = target in domain_set

            if sign_is_domain and not target_is_domain:
                drops.add(target)
                conflict_details.append({
                    "dropped": target,
                    "kept": sign,
                    "reason": f"both fail, {sign} is domain, {target} is common",
                })
            elif target_is_domain and not sign_is_domain:
                drops.add(sign)
                conflict_details.append({
                    "dropped": sign,
                    "kept": target,
                    "reason": f"both fail, {target} is domain, {sign} is common",
                })
            else:
                # Same category — drop the one with worse top3
                if profiles[sign]['top3_hit_rate'] <= profiles[target]['top3_hit_rate']:
                    drops.add(sign)
                    conflict_details.append({
                        "dropped": sign,
                        "kept": target,
                        "reason": f"both fail same category, {sign} T3={profiles[sign]['top3_hit_rate']:.1f}% worse",
                    })
                else:
                    drops.add(target)
                    conflict_details.append({
                        "dropped": target,
                        "kept": sign,
                        "reason": f"both fail same category, {target} T3={profiles[target]['top3_hit_rate']:.1f}% worse",
                    })

    # Compute curated stats
    curated_signs = all_signs - drops
    curated_profiles = {s: p for s, p in profiles.items() if s in curated_signs}
    n_pass_curated, n_curated, curated_rate = compute_pass_rate(curated_profiles)

    # Count how many domain vs common were dropped
    domain_dropped = sorted(drops & domain_set)
    common_dropped = sorted(drops & common_set)

    return {
        "n_original": len(all_signs),
        "n_dropped": len(drops),
        "n_curated": len(curated_signs),
        "n_pass_curated": n_pass_curated,
        "curated_pass_rate_pct": round(curated_rate, 1),
        "dropped_signs": sorted(drops),
        "domain_dropped": domain_dropped,
        "common_dropped": common_dropped,
        "conflict_details": conflict_details,
    }


def write_masked_classes(model_dir, dropped_signs, class_index_mapping):
    """Write masked_classes.json for production inference.

    Looks up class IDs for dropped signs and writes the mask file
    in the same format used by openhands_modernized_inference.py.
    """
    # class_index_mapping is {str(id): gloss_lower}
    # Reverse it to {GLOSS_UPPER: id}
    gloss_to_id = {v.upper(): int(k) for k, v in class_index_mapping.items()}

    masked_ids = []
    masked_names = []
    for sign in sorted(dropped_signs):
        if sign in gloss_to_id:
            masked_ids.append(gloss_to_id[sign])
            masked_names.append(sign)
        else:
            print(f"  WARNING: {sign} not found in class_index_mapping, skipping")

    total_classes = len(class_index_mapping)
    mask_config = {
        "description": "Classes to mask at inference time due to confusion patterns (auto-generated by curation)",
        "threshold_used": f"confusion_threshold=0.5, quality={TOP1_THRESHOLD}/{TOP3_THRESHOLD}",
        "masked_class_ids": sorted(masked_ids),
        "masked_class_names": masked_names,
        "effective_classes": total_classes - len(masked_ids),
    }

    mask_path = Path(model_dir) / "masked_classes.json"
    with open(mask_path, 'w') as f:
        json.dump(mask_config, f, indent=2)

    print(f"  Wrote {mask_path} ({len(masked_ids)} masked, "
          f"{total_classes - len(masked_ids)} effective)")
    return mask_path


def run_sweep_point(domain_n, config, pickle_pool, train_args=None):
    """Run a single sweep point: create manifests, train, profile, save results."""
    train_args = train_args or {}
    total_n = domain_n + COMMON_COUNT
    common = config['common_words']
    fps_ordered = config['domain_candidates_fps_ordered']

    if domain_n > len(fps_ordered):
        print(f"ERROR: Only {len(fps_ordered)} domain candidates, requested {domain_n}")
        return None

    domain_subset = fps_ordered[:domain_n]
    sweep_dir = RESULTS_DIR / f"sweep_n{domain_n}"

    # Check if already completed
    result_file = sweep_dir / "sweep_result.json"
    force = train_args.get('force', False)
    if result_file.exists() and not force:
        print(f"Sweep N={domain_n} already completed. Loading existing results.")
        print(f"  (Use --force to re-run with different params)")
        with open(result_file) as f:
            return json.load(f)

    print(f"\n{'='*70}")
    print(f"SWEEP POINT: {domain_n} domain + {COMMON_COUNT} common = {total_n} total")
    print(f"{'='*70}")
    print(f"Domain glosses (FPS order): {domain_subset}")
    print()

    sweep_dir.mkdir(parents=True, exist_ok=True)
    model_dir = sweep_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Create gloss list & class mapping ──
    print("Step 1: Creating gloss list and manifests...")
    all_glosses = sorted(set(g.upper() for g in common + domain_subset))
    class_index_mapping = {str(i): g.lower() for i, g in enumerate(all_glosses)}

    gloss_config = {
        "num_classes": len(all_glosses),
        "common_count": COMMON_COUNT,
        "domain_count": domain_n,
        "classes": all_glosses,
        "class_index_mapping": class_index_mapping,
    }
    with open(sweep_dir / "gloss_list.json", 'w') as f:
        json.dump(gloss_config, f, indent=2)

    # Save class_index_mapping for the model
    with open(model_dir / "class_index_mapping.json", 'w') as f:
        json.dump(class_index_mapping, f, indent=2)

    # ── Step 2: Create train/val/test splits ──
    create_manifests(common, domain_subset, sweep_dir, pickle_pool)

    # ── Step 3: Train model ──
    print(f"\nStep 2: Training {total_n}-class model...")
    train_script = WORKING / "models" / "training-scripts" / "train_asl.py"

    # Use shared cache dir — files from previous sweep points that
    # overlap are reused automatically (keyed by file path)
    model_size = train_args.get('model_size', 'small')
    dropout = train_args.get('dropout', 0.4)
    early_stopping = train_args.get('early_stopping', 100)
    label_smoothing = train_args.get('label_smoothing', 0.1)

    train_cmd = [
        sys.executable, str(train_script),
        "--classes", str(total_n),
        "--dataset", "augmented",
        "--architecture", "openhands",
        "--model-size", model_size,
        "--dropout", str(dropout),
        "--label-smoothing", str(label_smoothing),
        "--manifest-dir", str(sweep_dir),
        "--early-stopping", str(early_stopping),
        "--force-fresh",
        "--cache-dir", str(CACHE_DIR),
    ]

    # Optional args (only added if explicitly set)
    if train_args.get('hidden_size') is not None:
        train_cmd.extend(["--hidden-size", str(train_args['hidden_size'])])
    if train_args.get('num_layers') is not None:
        train_cmd.extend(["--num-layers", str(train_args['num_layers'])])
    if train_args.get('lr') is not None:
        train_cmd.extend(["--lr", str(train_args['lr'])])
    if train_args.get('warmup_epochs') is not None:
        train_cmd.extend(["--warmup-epochs", str(train_args['warmup_epochs'])])
    if train_args.get('weight_decay') is not None:
        train_cmd.extend(["--weight-decay", str(train_args['weight_decay'])])

    print(f"\n  Training config: model={model_size}, dropout={dropout}, "
          f"lr={train_args.get('lr', 'auto')}, early_stop={early_stopping}")

    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join([
        str(WORKING),
        str(WORKING / "models" / "openhands-modernized" / "src"),
        str(WORKING / "dataset-utilities" / "landmarks-extraction"),
    ])

    print(f"  Command: {' '.join(train_cmd)}")
    t0 = time.time()
    result = subprocess.run(train_cmd, env=env, cwd=str(WORKING))
    train_time = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR: Training failed (return code {result.returncode})")
        return None

    print(f"  Training completed in {train_time/60:.1f} minutes")

    # Find and copy model to sweep dir
    # train_asl.py saves to WORKING/models/wlasl_{N}_class_model/
    default_model_dir = WORKING / "models" / f"wlasl_{total_n}_class_model"
    if default_model_dir.exists():
        for fname in ['pytorch_model.bin', 'config.json', 'checkpoint.pth']:
            src = default_model_dir / fname
            if src.exists():
                shutil.copy2(src, model_dir / fname)
        print(f"  Model saved to {model_dir}")
    else:
        print(f"  WARNING: Expected model at {default_model_dir} not found")
        # Check if it saved to a different location
        for candidate in WORKING.glob("models/wlasl_*_class_model"):
            if (candidate / "pytorch_model.bin").exists():
                print(f"  Found model at {candidate}")
                for fname in ['pytorch_model.bin', 'config.json']:
                    src = candidate / fname
                    if src.exists():
                        shutil.copy2(src, model_dir / fname)
                break

    # Ensure class_index_mapping is present
    if not (model_dir / "class_index_mapping.json").exists():
        with open(model_dir / "class_index_mapping.json", 'w') as f:
            json.dump(class_index_mapping, f, indent=2)

    # ── Step 4: Profile model ──
    print(f"\nStep 3: Profiling model...")
    if not (model_dir / "pytorch_model.bin").exists():
        print("  ERROR: No trained model found, skipping profiling")
        return None

    val_manifest = sweep_dir / "val_manifest.json"
    t0 = time.time()
    success = profile_model(model_dir, val_manifest, sweep_dir)
    profile_time = time.time() - t0

    if not success:
        print("  ERROR: Profiling failed")
        return None

    print(f"  Profiling completed in {profile_time/60:.1f} minutes")

    # ── Step 5: Compute pass rate ──
    print(f"\nStep 4: Computing pass rate...")
    profile_file = None
    for candidate in [
        sweep_dir / "per_sign_profiles.json",
        sweep_dir / "per_sign_profiles_model.json",
    ]:
        if candidate.exists():
            profile_file = candidate
            break

    if not profile_file:
        candidates = list(sweep_dir.glob("per_sign_profiles*.json"))
        if candidates:
            profile_file = candidates[0]

    if not profile_file:
        print("  ERROR: No profile file found")
        return None

    with open(profile_file) as f:
        profiles = json.load(f)

    n_pass, n_signs, pass_rate = compute_pass_rate(profiles)

    # Build result
    passing_signs = sorted([s for s, p in profiles.items()
                            if p['top3_hit_rate'] >= TOP3_THRESHOLD
                            and p['avg_top1_confidence'] >= TOP1_THRESHOLD])
    failing_signs = sorted([s for s, p in profiles.items()
                            if not (p['top3_hit_rate'] >= TOP3_THRESHOLD
                                    and p['avg_top1_confidence'] >= TOP1_THRESHOLD)])

    # ── Step 5b: Cheap curation ──
    print(f"\nStep 4b: Curating vocabulary (conflict-pair analysis)...")
    curation = curate_vocabulary(profiles, common, domain_subset)

    for detail in curation['conflict_details']:
        print(f"  DROP {detail['dropped']} (keep {detail['kept']}): {detail['reason']}")
    print(f"  Dropped {curation['n_dropped']} signs: {curation['dropped_signs']}")
    print(f"  Curated: {curation['n_pass_curated']}/{curation['n_curated']} pass "
          f"({curation['curated_pass_rate_pct']}%)")

    # Write masked_classes.json for production inference
    if curation['n_dropped'] > 0:
        mapping_file = model_dir / "class_index_mapping.json"
        if mapping_file.exists():
            with open(mapping_file) as f:
                class_index_mapping = json.load(f)
            write_masked_classes(model_dir, curation['dropped_signs'], class_index_mapping)
        else:
            print(f"  WARNING: {mapping_file} not found, skipping masked_classes.json")

    result_data = {
        "domain_n": domain_n,
        "total_n": total_n,
        "common_n": COMMON_COUNT,
        "n_signs_profiled": n_signs,
        "n_pass": n_pass,
        "pass_rate_pct": round(pass_rate, 1),
        "threshold_top1": TOP1_THRESHOLD,
        "threshold_top3": TOP3_THRESHOLD,
        "train_time_minutes": round(train_time / 60, 1),
        "profile_time_minutes": round(profile_time / 60, 1),
        "timestamp": datetime.now().isoformat(),
        "passing_signs": passing_signs,
        "failing_signs": failing_signs,
        "curation": curation,
        "per_sign_summary": {
            s: {
                "top1_acc": round(p['top1_accuracy'], 1),
                "top3_hit": round(p['top3_hit_rate'], 1),
                "top1_conf": round(p['avg_top1_confidence'], 1),
            }
            for s, p in sorted(profiles.items())
        },
    }

    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  RESULT: {n_pass}/{n_signs} signs pass 70/85 threshold ({pass_rate:.1f}%)")
    print(f"  CURATED: {curation['n_pass_curated']}/{curation['n_curated']} pass "
          f"({curation['curated_pass_rate_pct']}%) after dropping {curation['n_dropped']} conflicts")
    print(f"  Passing: {passing_signs[:10]}{'...' if len(passing_signs) > 10 else ''}")
    print(f"  Failing: {failing_signs[:10]}{'...' if len(failing_signs) > 10 else ''}")
    print(f"  Saved to: {result_file}")
    print(f"{'='*70}")

    return result_data


def analyze_results():
    """Analyze all completed sweep results."""
    print(f"\n{'='*70}")
    print("SCALING ANALYSIS: Doctor Visit Domain")
    print(f"{'='*70}\n")

    results = []
    for sweep_dir in sorted(RESULTS_DIR.glob("sweep_n*")):
        result_file = sweep_dir / "sweep_result.json"
        if result_file.exists():
            with open(result_file) as f:
                results.append(json.load(f))

    if not results:
        print("No results found yet.")
        return

    results.sort(key=lambda r: r['domain_n'])

    # Print scaling curve
    print(f"{'Domain N':>9} {'Total N':>8} {'Pass':>5} {'Total':>6} "
          f"{'Rate':>7} {'Train':>8} {'Status':>10}")
    print("-" * 65)
    for r in results:
        status = "PASS" if r['pass_rate_pct'] >= 80 else (
            "BORDER" if r['pass_rate_pct'] >= 70 else "FAIL")
        train_t = f"{r.get('train_time_minutes', '?')}m"
        print(f"{r['domain_n']:>9} {r['total_n']:>8} {r['n_pass']:>5} "
              f"{r['n_signs_profiled']:>6} {r['pass_rate_pct']:>6.1f}% "
              f"{train_t:>8} {status:>10}")

    # Find crossover points
    print(f"\n--- Crossover Analysis ---")
    for target in [85, 80, 75, 70]:
        above = [r for r in results if r['pass_rate_pct'] >= target]
        below = [r for r in results if r['pass_rate_pct'] < target]
        if above and below:
            max_above = max(above, key=lambda r: r['domain_n'])
            min_below = min(below, key=lambda r: r['domain_n'])
            print(f"  {target}% gate: passes at N<={max_above['domain_n']} "
                  f"(total={max_above['total_n']}), "
                  f"fails at N>={min_below['domain_n']} "
                  f"(total={min_below['total_n']})")
            if min_below['domain_n'] - max_above['domain_n'] > 5:
                print(f"    --> Bisect: try N={
                    (max_above['domain_n'] + min_below['domain_n']) // 2}")
        elif above:
            print(f"  {target}% gate: ALL points pass "
                  f"(max tested: N={results[-1]['domain_n']})")
        else:
            print(f"  {target}% gate: NO points pass")

    # Chronic failures
    print(f"\n--- Signs That Consistently Fail ---")
    fail_counts = defaultdict(int)
    total_appearances = defaultdict(int)
    for r in results:
        for s in r.get('failing_signs', []):
            fail_counts[s] += 1
        for s in r.get('passing_signs', []) + r.get('failing_signs', []):
            total_appearances[s] += 1

    chronic = [(s, fail_counts[s], total_appearances[s])
               for s in fail_counts if fail_counts[s] >= 2]
    chronic.sort(key=lambda x: (-x[1], x[0]))

    if chronic:
        for s, n_fail, n_total in chronic[:20]:
            print(f"  {s:<20} fails {n_fail}/{n_total} times")
    else:
        print("  (not enough data points yet)")

    # Signs that degrade as N increases
    if len(results) >= 2:
        print(f"\n--- Degradation Analysis ---")
        # Compare first vs last sweep point
        first = results[0]
        last = results[-1]
        first_passing = set(first.get('passing_signs', []))
        last_passing = set(last.get('passing_signs', []))
        degraded = first_passing - last_passing
        if degraded:
            print(f"  Signs that pass at N={first['domain_n']} "
                  f"but fail at N={last['domain_n']}:")
            for s in sorted(degraded):
                print(f"    {s}")
        else:
            print(f"  No signs degraded between N={first['domain_n']} "
                  f"and N={last['domain_n']}")

    # Save analysis
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "scaling_curve": [{
            "domain_n": r['domain_n'],
            "total_n": r['total_n'],
            "pass_rate": r['pass_rate_pct'],
            "n_pass": r['n_pass'],
            "n_total": r['n_signs_profiled'],
            "train_minutes": r.get('train_time_minutes'),
        } for r in results],
    }

    analysis_path = RESULTS_DIR / "scaling_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to: {analysis_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Kaggle Scaling Experiment: Doctor Visit Domain")
    parser.add_argument('--domain-n', type=int, default=None,
                       help='Number of domain glosses for this sweep point '
                            '(total classes = N + 25 common)')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze all completed sweep results')
    # Training hyperparameters (forwarded to train_asl.py)
    parser.add_argument('--model-size', choices=['tiny', 'small', 'large'],
                       default='small', help='Model size (default: small)')
    parser.add_argument('--hidden-size', type=int, default=None,
                       help='Custom hidden size (overrides model-size)')
    parser.add_argument('--num-layers', type=int, default=None,
                       help='Custom number of layers (overrides model-size)')
    parser.add_argument('--dropout', type=float, default=0.4,
                       help='Dropout rate (default: 0.4)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (default: auto based on architecture)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing (default: 0.1)')
    parser.add_argument('--early-stopping', type=int, default=100,
                       help='Early stopping patience (default: 100)')
    parser.add_argument('--warmup-epochs', type=int, default=None,
                       help='Warmup epochs (default: 10%% of total)')
    parser.add_argument('--weight-decay', type=float, default=None,
                       help='Weight decay (default: auto)')
    parser.add_argument('--force', action='store_true',
                       help='Re-run even if results exist for this sweep point')

    args = parser.parse_args()

    if not args.domain_n and not args.analyze:
        parser.print_help()
        print("\nExample usage:")
        print("  python kaggle_scaling_experiment.py --domain-n 20")
        print("  python kaggle_scaling_experiment.py --domain-n 25")
        print("  python kaggle_scaling_experiment.py --analyze")
        return

    # Setup
    pickle_pool = setup_environment()
    config = load_experiment_config()

    if args.analyze:
        analyze_results()
        return

    # Collect training args (including force flag)
    train_args = {
        'force': args.force,
        'model_size': args.model_size,
        'dropout': args.dropout,
        'early_stopping': args.early_stopping,
        'label_smoothing': args.label_smoothing,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'lr': args.lr,
        'warmup_epochs': args.warmup_epochs,
        'weight_decay': args.weight_decay,
    }

    # Run sweep point
    result = run_sweep_point(args.domain_n, config, pickle_pool, train_args)

    if result:
        # Show running analysis after each point
        print("\n")
        analyze_results()


if __name__ == "__main__":
    main()
