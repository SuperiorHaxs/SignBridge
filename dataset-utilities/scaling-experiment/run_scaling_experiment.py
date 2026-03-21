#!/usr/bin/env python3
"""
Vocabulary Scaling Experiment: Doctor Visit Domain

Research question:
  What is the maximum number of domain glosses N such that >=X% of all glosses
  (N domain + 25 common) meet the 70% Top-1 confidence + 85% Top-3 hit rate
  threshold for statistically significant LLM improvement?

Experiment design:
  - Independent variable: N (number of domain glosses, swept from 20 to 60)
  - Dependent variable: pass rate (% of signs meeting 70/85 threshold)
  - Controlled: architecture, hyperparameters, augmentation pipeline,
    gloss selection method (FPS ordering), common words (fixed 25)

Pipeline per sweep point:
  1. Select top-N domain glosses via farthest-point sampling ordering
  2. Generate augmented pickles for any missing glosses (if needed)
  3. Create train/val/test split manifests
  4. Train model
  5. Profile all signs with profile_model_signs.py
  6. Record pass rate and per-sign results

Usage:
  # Step 1: Generate FPS ordering and check data availability
  python run_scaling_experiment.py prepare

  # Step 2: Generate augmented data for missing glosses
  python run_scaling_experiment.py augment

  # Step 3: Run a single sweep point (e.g., N=20 domain glosses -> 45 total)
  python run_scaling_experiment.py run --domain-n 20

  # Step 4: Run the full sweep
  python run_scaling_experiment.py sweep

  # Step 5: Analyze results
  python run_scaling_experiment.py analyze
"""

import json
import sys
import argparse
import shutil
import subprocess
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ─── Project paths ───
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
EXPERIMENT_DIR = PROJECT_ROOT / "datasets" / "domain-specific" / "experiments" / "doctor_visit_scaling"

# Pipeline tool paths
AUGMENTATION_SCRIPT = PROJECT_ROOT / "dataset-utilities" / "augmentation" / "generate_augmented_dataset.py"
SPLIT_SCRIPT = PROJECT_ROOT / "dataset-utilities" / "dataset-splitting" / "stratified_family_split.py"
TRAIN_SCRIPT = PROJECT_ROOT / "models" / "training-scripts" / "train_asl.py"
PROFILE_SCRIPT = PROJECT_ROOT / "project-utilities" / "evaluation_metrics" / "synthetic_evaluation" / "profile_model_signs.py"

# Data paths
PICKLE_POOL = PROJECT_ROOT / "datasets" / "augmented_pool" / "pickle"
POSE_DIR = PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "pose_files_by_gloss"
FLAT_PICKLE_DIR = PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "pickle_files"
VIDEO_TO_GLOSS = PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "video_to_gloss_mapping.json"
WLASL_MAPPING = PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "splits" / "2000_classes" / "class_index_mapping.json"
COMMON_WORDS_FILE = PROJECT_ROOT / "datasets" / "domain-specific" / "common_words.json"

# FPS tools
sys.path.insert(0, str(PROJECT_ROOT / "dataset-utilities" / "domain-gloss-lists"))

# Experiment config
SWEEP_POINTS = [20, 25, 30, 35, 40, 45, 50, 55, 60]  # domain glosses (total = N + 25)
COMMON_COUNT = 25
TOP1_THRESHOLD = 70.0
TOP3_THRESHOLD = 85.0


def load_common_words():
    """Load the fixed 25 common words."""
    with open(COMMON_WORDS_FILE) as f:
        cw = json.load(f)
    return [g.upper() for g in cw['classes']]


def load_domain_candidates():
    """Load the full Doctor Visit candidate pool from domain_config."""
    from domain_config import DOMAIN_CANDIDATES

    with open(WLASL_MAPPING) as f:
        wlasl = set(json.load(f).keys())

    common_set = set(load_common_words())
    candidates = [g for g in DOMAIN_CANDIDATES['Doctor Visit']
                  if g in wlasl and g not in common_set]
    return candidates


def compute_fps_ordering(candidates):
    """Compute farthest-point sampling ordering for domain candidates.

    Note: We implement FPS ordering directly here rather than calling
    farthest_point_sampling() because that function short-circuits when
    target_count >= n (returns input order unchanged). We need the actual
    greedy FPS ordering of ALL candidates to determine which are most
    discriminable.
    """
    from pose_distance import load_gloss_centroids, compute_distance_matrix
    import numpy as np

    print(f"Computing FPS ordering for {len(candidates)} candidates...")
    centroids = load_gloss_centroids(candidates, FLAT_PICKLE_DIR, VIDEO_TO_GLOSS)
    available = [g for g in candidates if g in centroids]
    missing = [g for g in candidates if g not in centroids]

    if missing:
        print(f"  WARNING: {len(missing)} candidates have no centroid data: {missing}")

    dist_matrix, available = compute_distance_matrix(available, centroids)
    n = len(available)

    # Manual FPS: greedily pick the most diverse ordering
    avg_distances = np.mean(dist_matrix, axis=1)
    seed_idx = int(np.argmin(avg_distances))

    selected_indices = [seed_idx]
    min_dist_to_selected = dist_matrix[seed_idx].copy()

    for _ in range(n - 1):
        min_dist_to_selected[selected_indices] = -1.0
        next_idx = int(np.argmax(min_dist_to_selected))
        selected_indices.append(next_idx)
        min_dist_to_selected = np.minimum(
            min_dist_to_selected, dist_matrix[next_idx])

    ordered = [available[i] for i in selected_indices]

    # Compute diversity metrics
    idx = np.array(selected_indices)
    sub_matrix = dist_matrix[np.ix_(idx, idx)]
    triu = sub_matrix[np.triu_indices_from(sub_matrix, k=1)]
    metrics = {
        "avg_pairwise_distance": float(np.mean(triu)),
        "min_pairwise_distance": float(np.min(triu)),
        "max_pairwise_distance": float(np.max(triu)),
    }

    print(f"  FPS ordering computed for {len(ordered)} glosses")
    print(f"  Seed: {ordered[0]} (most central)")
    print(f"  Avg pairwise distance: {metrics['avg_pairwise_distance']:.6f}")

    return ordered, missing, metrics


def check_data_availability(glosses):
    """Check which glosses have augmented data vs need generation."""
    has_aug = []
    needs_aug = []

    for g in glosses:
        g_lower = g.lower()
        aug_dir = PICKLE_POOL / g_lower
        if aug_dir.exists() and len(list(aug_dir.glob('*aug*.pkl'))) > 0:
            n_files = len(list(aug_dir.glob('*.pkl')))
            has_aug.append((g, n_files))
        else:
            pose_dir = POSE_DIR / g_lower
            n_pose = len(list(pose_dir.glob('*.pose'))) if pose_dir.exists() else 0
            needs_aug.append((g, n_pose))

    return has_aug, needs_aug


def cmd_prepare(args):
    """Prepare: compute FPS ordering, check data, save experiment config."""
    common = load_common_words()
    candidates = load_domain_candidates()

    print(f"Common words: {len(common)}")
    print(f"Domain candidates: {len(candidates)}")
    print()

    # Compute FPS ordering
    fps_ordered, fps_missing, fps_metrics = compute_fps_ordering(candidates)

    # Check data availability for everything
    all_glosses = common + fps_ordered
    has_aug, needs_aug = check_data_availability(all_glosses)

    print(f"\nData availability:")
    print(f"  Already augmented: {len(has_aug)}")
    print(f"  Need augmentation: {len(needs_aug)}")

    if needs_aug:
        print(f"\n  Glosses needing augmentation:")
        for g, n_pose in needs_aug:
            src = "CW" if g in common else "DV"
            print(f"    {g:<20} {n_pose} pose files  [{src}]")

    # Save experiment config
    config = {
        "experiment": "doctor_visit_vocabulary_scaling",
        "created_at": datetime.now().isoformat(),
        "research_question": "Maximum N domain glosses where >=X% meet 70/85 threshold",
        "common_words": common,
        "domain_candidates_fps_ordered": fps_ordered,
        "domain_candidates_not_in_centroids": fps_missing,
        "fps_metrics": fps_metrics,
        "sweep_points": SWEEP_POINTS,
        "thresholds": {
            "top1_confidence": TOP1_THRESHOLD,
            "top3_hit_rate": TOP3_THRESHOLD,
        },
        "controlled_variables": {
            "architecture": "openhands",
            "model_size": "small",
            "hidden_size": 64,
            "num_layers": 3,
            "augmentation": "60_per_original",
            "gloss_selection": "farthest_point_sampling",
        },
        "data_availability": {
            "augmented": len(has_aug),
            "needs_augmentation": len(needs_aug),
            "glosses_needing_augmentation": [g for g, _ in needs_aug],
        },
    }

    config_path = EXPERIMENT_DIR / "experiment_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nExperiment config saved to: {config_path}")
    print(f"\nFPS ordering (first 20 = most discriminable):")
    for i, g in enumerate(fps_ordered[:20]):
        print(f"  {i+1:>2}. {g}")
    if len(fps_ordered) > 20:
        print(f"  ... ({len(fps_ordered)} total)")

    print(f"\nSweep points: {SWEEP_POINTS}")
    print(f"Total models to train: {len(SWEEP_POINTS)}")
    print(f"Total glosses per model: {[n + COMMON_COUNT for n in SWEEP_POINTS]}")

    if needs_aug:
        print(f"\n>>> Next step: run 'python {Path(__file__).name} augment' to generate missing data")
    else:
        print(f"\n>>> All data ready. Run 'python {Path(__file__).name} sweep' to start.")


def cmd_augment(args):
    """Generate augmented pickles for all missing glosses."""
    config_path = EXPERIMENT_DIR / "experiment_config.json"
    if not config_path.exists():
        print("ERROR: Run 'prepare' first to generate experiment_config.json")
        return

    with open(config_path) as f:
        config = json.load(f)

    glosses_needed = config['data_availability']['glosses_needing_augmentation']
    if not glosses_needed:
        print("All glosses already have augmented data!")
        return

    print(f"Generating augmented data for {len(glosses_needed)} glosses...")

    # Write a temporary gloss file for the augmentation script
    gloss_file = EXPERIMENT_DIR / "glosses_to_augment.json"
    with open(gloss_file, 'w') as f:
        json.dump(glosses_needed, f)

    cmd = [
        sys.executable, str(AUGMENTATION_SCRIPT),
        "--gloss-file", str(gloss_file),
        "--landmark-config", "83pt",
        "--target", "200",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print(f"ERROR: Augmentation failed with return code {result.returncode}")
        return

    # Verify
    still_missing = []
    for g in glosses_needed:
        aug_dir = PICKLE_POOL / g.lower()
        if not aug_dir.exists() or len(list(aug_dir.glob('*aug*.pkl'))) == 0:
            still_missing.append(g)

    if still_missing:
        print(f"\nWARNING: {len(still_missing)} glosses still missing: {still_missing}")
    else:
        print(f"\nAll {len(glosses_needed)} glosses now have augmented data!")
        print(f">>> Next step: run 'python {Path(__file__).name} sweep' to start training.")


def create_gloss_list_json(common, domain_subset, output_path):
    """Create a gloss list JSON for a sweep point."""
    all_glosses = sorted(set(common + domain_subset))
    gloss_config = {
        "domain": "healthcare",
        "scenario": "Doctor Visit (scaling experiment)",
        "num_classes": len(all_glosses),
        "common_count": len(common),
        "domain_count": len(domain_subset),
        "classes": all_glosses,
        "class_to_index": {g: i for i, g in enumerate(all_glosses)},
        "class_index_mapping": {str(i): g.lower() for i, g in enumerate(all_glosses)},
    }
    with open(output_path, 'w') as f:
        json.dump(gloss_config, f, indent=2)
    return gloss_config


def create_split_for_sweep(common, domain_subset, sweep_dir):
    """Create train/val/test split manifests for a sweep point."""
    all_glosses = sorted(set(g.lower() for g in common + domain_subset))

    # Use the stratified family split script
    # It expects an input-dir with class folders — that's our pickle pool
    # But it processes ALL classes in the input dir. We need to filter.

    # Alternative: create a filtered input directory with symlinks
    # Or: create manifests directly by filtering the pickle pool

    # Let's create manifests directly since we understand the format
    import random
    random.seed(42)

    manifest_data = {
        "train": {"split": "train", "pickle_pool": str(PICKLE_POOL),
                  "total_samples": 0, "classes": {}},
        "val": {"split": "val", "pickle_pool": str(PICKLE_POOL),
                "total_samples": 0, "classes": {}},
        "test": {"split": "test", "pickle_pool": str(PICKLE_POOL),
                 "total_samples": 0, "classes": {}},
    }

    for gloss in all_glosses:
        gloss_dir = PICKLE_POOL / gloss
        if not gloss_dir.exists():
            print(f"  WARNING: No pickle dir for {gloss}")
            continue

        # Find all original videos and their augmented families
        all_files = sorted(gloss_dir.glob('*.pkl'))
        originals = [f for f in all_files if 'aug' not in f.name]
        augmented = [f for f in all_files if 'aug' in f.name]

        # Group augmented by original video id
        families = {}
        for orig in originals:
            vid_id = orig.stem  # e.g., "01024"
            family_augs = [f for f in augmented if f.name.startswith(vid_id + '_aug_')]
            families[vid_id] = {
                "original": orig.name,
                "augmented": [f.name for f in family_augs],
            }

        # Split families: 70/15/15
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

        for fid, split in split_assignment.items():
            fam = families[fid]
            all_fam_files = [fam["original"]] + fam["augmented"]

            if gloss not in manifest_data[split]["classes"]:
                manifest_data[split]["classes"][gloss] = []

            manifest_data[split]["classes"][gloss].append({
                "video_id": fid,
                "files": all_fam_files,
            })
            manifest_data[split]["total_samples"] += len(all_fam_files)

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


def cmd_run(args):
    """Run a single sweep point."""
    domain_n = args.domain_n
    total_n = domain_n + COMMON_COUNT

    config_path = EXPERIMENT_DIR / "experiment_config.json"
    if not config_path.exists():
        print("ERROR: Run 'prepare' first")
        return

    with open(config_path) as f:
        config = json.load(f)

    common = config['common_words']
    fps_ordered = config['domain_candidates_fps_ordered']

    if domain_n > len(fps_ordered):
        print(f"ERROR: Only {len(fps_ordered)} domain candidates available, requested {domain_n}")
        return

    domain_subset = fps_ordered[:domain_n]
    print(f"=== Sweep point: {domain_n} domain + {COMMON_COUNT} common = {total_n} total ===")
    print(f"Domain glosses: {domain_subset}")
    print()

    sweep_dir = EXPERIMENT_DIR / f"sweep_n{domain_n}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create gloss list
    print("Step 1: Creating gloss list...")
    gloss_config = create_gloss_list_json(common, domain_subset,
                                          sweep_dir / "gloss_list.json")
    print(f"  {gloss_config['num_classes']} classes")

    # Step 2: Create splits
    print("\nStep 2: Creating train/val/test splits...")
    create_split_for_sweep(common, domain_subset, sweep_dir)

    # Step 3: Train model
    print("\nStep 3: Training model...")
    model_dir = sweep_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save class_index_mapping for the model
    with open(model_dir / "class_index_mapping.json", 'w') as f:
        json.dump(gloss_config['class_index_mapping'], f, indent=2)

    train_cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--classes", str(total_n),
        "--dataset", "augmented",
        "--architecture", "openhands",
        "--model-size", "small",
        "--dropout", "0.4",
        "--manifest-dir", str(sweep_dir),
        "--early-stopping", "100",
        "--force-fresh",
    ]

    print(f"  Command: {' '.join(train_cmd)}")
    t0 = time.time()
    result = subprocess.run(train_cmd, cwd=str(PROJECT_ROOT))
    train_time = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR: Training failed (return code {result.returncode})")
        return

    print(f"  Training completed in {train_time/60:.1f} minutes")

    # Find the saved model (train_asl.py saves to models/wlasl_{N}_class_model/)
    default_model_dir = PROJECT_ROOT / "models" / f"wlasl_{total_n}_class_model"
    if default_model_dir.exists():
        # Copy model files to our sweep directory
        for f in ['pytorch_model.bin', 'config.json', 'checkpoint.pth']:
            src = default_model_dir / f
            if src.exists():
                shutil.copy2(src, model_dir / f)
        print(f"  Model copied to {model_dir}")
    else:
        print(f"  WARNING: Expected model at {default_model_dir} not found")
        # Check if train script saved elsewhere
        print(f"  Checking for model in sweep dir...")

    # Ensure class_index_mapping is in model dir
    if not (model_dir / "class_index_mapping.json").exists():
        with open(model_dir / "class_index_mapping.json", 'w') as f:
            json.dump(gloss_config['class_index_mapping'], f, indent=2)

    # Step 4: Profile model
    print("\nStep 4: Profiling model...")
    val_manifest = sweep_dir / "val_manifest.json"

    if not (model_dir / "pytorch_model.bin").exists():
        print("  ERROR: No trained model found, skipping profiling")
        return

    profile_cmd = [
        sys.executable, str(PROFILE_SCRIPT),
        str(model_dir),
        str(val_manifest),
        "--output-dir", str(sweep_dir),
    ]

    print(f"  Command: {' '.join(profile_cmd)}")
    t0 = time.time()
    result = subprocess.run(profile_cmd, cwd=str(PROJECT_ROOT))
    profile_time = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR: Profiling failed (return code {result.returncode})")
        return

    print(f"  Profiling completed in {profile_time/60:.1f} minutes")

    # Step 5: Compute pass rate
    print("\nStep 5: Computing pass rate...")
    profile_file = sweep_dir / f"per_sign_profiles_model.json"
    if not profile_file.exists():
        # Try the auto-generated name
        candidates = list(sweep_dir.glob("per_sign_profiles_*.json"))
        if candidates:
            profile_file = candidates[0]
        else:
            print("  ERROR: No profile file found")
            return

    with open(profile_file) as f:
        profiles = json.load(f)

    n_signs = len(profiles)
    n_pass = sum(1 for s in profiles.values()
                 if s['top3_hit_rate'] >= TOP3_THRESHOLD
                 and s['avg_top1_confidence'] >= TOP1_THRESHOLD)
    pass_rate = n_pass / n_signs * 100 if n_signs > 0 else 0

    # Save result
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
        "passing_signs": sorted([s for s, p in profiles.items()
                                  if p['top3_hit_rate'] >= TOP3_THRESHOLD
                                  and p['avg_top1_confidence'] >= TOP1_THRESHOLD]),
        "failing_signs": sorted([s for s, p in profiles.items()
                                  if not (p['top3_hit_rate'] >= TOP3_THRESHOLD
                                          and p['avg_top1_confidence'] >= TOP1_THRESHOLD)]),
    }

    result_path = sweep_dir / "sweep_result.json"
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2)

    print(f"\n  RESULT: {n_pass}/{n_signs} signs pass ({pass_rate:.1f}%)")
    print(f"  Saved to: {result_path}")


def cmd_sweep(args):
    """Run the full sweep across all N values."""
    config_path = EXPERIMENT_DIR / "experiment_config.json"
    if not config_path.exists():
        print("ERROR: Run 'prepare' first")
        return

    # Check if augmentation is needed
    with open(config_path) as f:
        config = json.load(f)

    needs_aug = config['data_availability']['glosses_needing_augmentation']
    if needs_aug:
        # Verify they're actually still missing
        still_missing = []
        for g in needs_aug:
            aug_dir = PICKLE_POOL / g.lower()
            if not aug_dir.exists() or len(list(aug_dir.glob('*aug*.pkl'))) == 0:
                still_missing.append(g)
        if still_missing:
            print(f"ERROR: {len(still_missing)} glosses still need augmentation: {still_missing[:10]}...")
            print(f"Run 'python {Path(__file__).name} augment' first.")
            return

    points = args.points if args.points else SWEEP_POINTS
    print(f"Running sweep across {len(points)} points: {points}")
    print(f"Total models to train: {len(points)}")
    print()

    for domain_n in points:
        print(f"\n{'='*80}")
        print(f"SWEEP POINT: domain_n={domain_n} (total={domain_n + COMMON_COUNT})")
        print(f"{'='*80}")

        # Create a mock args object for cmd_run
        run_args = argparse.Namespace(domain_n=domain_n)
        cmd_run(run_args)

    # Run analysis after sweep
    cmd_analyze(args)


def cmd_analyze(args):
    """Analyze results from all completed sweep points."""
    print(f"\n{'='*80}")
    print("SCALING ANALYSIS: Doctor Visit Domain")
    print(f"{'='*80}\n")

    results = []
    for sweep_dir in sorted(EXPERIMENT_DIR.glob("sweep_n*")):
        result_file = sweep_dir / "sweep_result.json"
        if result_file.exists():
            with open(result_file) as f:
                results.append(json.load(f))

    if not results:
        print("No results found. Run 'sweep' first.")
        return

    results.sort(key=lambda r: r['domain_n'])

    # Print scaling curve
    print(f"{'Domain N':>9} {'Total N':>8} {'Pass':>5} {'Total':>6} {'Rate':>7} {'Status':>10}")
    print("-" * 55)
    for r in results:
        status = "PASS" if r['pass_rate_pct'] >= 80 else ("BORDER" if r['pass_rate_pct'] >= 70 else "FAIL")
        print(f"{r['domain_n']:>9} {r['total_n']:>8} {r['n_pass']:>5} "
              f"{r['n_signs_profiled']:>6} {r['pass_rate_pct']:>6.1f}% {status:>10}")

    # Find crossover point
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
        elif above:
            print(f"  {target}% gate: ALL points pass (max tested: N={results[-1]['domain_n']})")
        else:
            print(f"  {target}% gate: NO points pass")

    # Signs that consistently fail
    print(f"\n--- Consistently Failing Signs ---")
    fail_counts = defaultdict(int)
    total_appearances = defaultdict(int)
    for r in results:
        for s in r.get('failing_signs', []):
            fail_counts[s] += 1
        for s in r.get('passing_signs', []) + r.get('failing_signs', []):
            total_appearances[s] += 1

    chronic_fails = [(s, fail_counts[s], total_appearances[s])
                     for s in fail_counts
                     if fail_counts[s] >= 2]
    chronic_fails.sort(key=lambda x: -x[1])

    if chronic_fails:
        for s, n_fail, n_total in chronic_fails:
            print(f"  {s:<20} fails {n_fail}/{n_total} times")

    # Save analysis
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "scaling_curve": [{
            "domain_n": r['domain_n'],
            "total_n": r['total_n'],
            "pass_rate": r['pass_rate_pct'],
            "n_pass": r['n_pass'],
            "n_total": r['n_signs_profiled'],
        } for r in results],
        "chronic_failures": [{
            "sign": s, "fail_count": n_fail, "total_appearances": n_total
        } for s, n_fail, n_total in chronic_fails] if chronic_fails else [],
    }

    analysis_path = EXPERIMENT_DIR / "scaling_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to: {analysis_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Doctor Visit Vocabulary Scaling Experiment")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # prepare
    subparsers.add_parser('prepare', help='Compute FPS ordering and check data')

    # augment
    subparsers.add_parser('augment', help='Generate augmented data for missing glosses')

    # run
    run_parser = subparsers.add_parser('run', help='Run a single sweep point')
    run_parser.add_argument('--domain-n', type=int, required=True,
                           help='Number of domain glosses (total = N + 25)')

    # sweep
    sweep_parser = subparsers.add_parser('sweep', help='Run full sweep')
    sweep_parser.add_argument('--points', type=int, nargs='+', default=None,
                             help=f'Custom sweep points (default: {SWEEP_POINTS})')

    # analyze
    subparsers.add_parser('analyze', help='Analyze completed sweep results')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        'prepare': cmd_prepare,
        'augment': cmd_augment,
        'run': cmd_run,
        'sweep': cmd_sweep,
        'analyze': cmd_analyze,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
