#!/usr/bin/env python3
"""
Design Experiment: Find the minimum Top-3 confidence below which
LLM does NOT add statistically significant CTQI v3 improvement.

Combines three strategies:
  1. MULTI-PICKLE PROFILING: Profile ALL 43 signs using ALL val pickle files
     (original + augmented) to get per-sign confidence distributions with
     natural variance — not just one sample per sign.
  2. UNSEEN SIGN DISCOVERY: Profile the 11 signs not in the original 53-sentence
     study to discover new hard/medium signs for low-confidence bands.
  3. VARIABLE-LENGTH SENTENCES: Use 3-5 gloss sentences to control Top-3
     accuracy bands (more hard signs in longer sentences = lower Top-3 %).

Output:
  - Per-sign confidence profile report (all pickles)
  - Stratified dataset JSON ready for evaluate_synthetic_dataset.py
  - Power analysis and experiment design doc

Usage:
  python design_threshold_experiment.py [--output-dir DIR] [--dry-run] [--profile-only]
"""

import json
import random
import math
import argparse
import sys
from pathlib import Path
from collections import defaultdict

# ─── Paths ───
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
EVAL_RESULTS = SCRIPT_DIR / "evaluation_results_gemini_t1_n53_v4" / "evaluation_results.json"
EXISTING_DATASET = PROJECT_ROOT / "datasets" / "synthetic_sentences" / "synthetic_gloss_to_sentence_llm_dataset_43_glosses_50.json"
VAL_MANIFEST = PROJECT_ROOT / "datasets" / "augmented_pool" / "splits" / "43_classes" / "val_manifest.json"
PRODUCTION_MODEL_DIR = PROJECT_ROOT / "models" / "openhands-modernized" / "production-models" / "wlasl_43_class_50s_model"

VOCAB_43 = [
    "africa", "apple", "basketball", "before", "birthday", "bowling", "but",
    "chair", "change", "computer", "cow", "dance", "deaf", "doctor", "drink",
    "enjoy", "family", "fine", "full", "give", "help", "hot", "jacket",
    "later", "man", "need", "no", "orange", "paper", "play", "son", "study",
    "table", "tall", "tell", "thanksgiving", "thursday", "time", "walk",
    "water", "what", "wrong", "year"
]


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: MULTI-PICKLE PROFILING
# ═══════════════════════════════════════════════════════════════════════

def load_model():
    """Load the 43-class production model for inference."""
    openhands_src = PROJECT_ROOT / "models" / "openhands-modernized" / "src"
    sys.path.insert(0, str(openhands_src))

    import torch
    from openhands_modernized import OpenHandsModel, OpenHandsConfig, WLASLPoseProcessor

    model_dir = PRODUCTION_MODEL_DIR

    # Load config
    with open(model_dir / "config.json", 'r') as f:
        config_dict = json.load(f)

    model_config = OpenHandsConfig(
        num_pose_keypoints=config_dict.get('num_pose_keypoints', 83),
        pose_channels=config_dict.get('pose_channels', 3),
        pose_features=config_dict.get('pose_features', 279),
        use_finger_features=config_dict.get('use_finger_features', True),
        finger_features=config_dict.get('finger_features', 30),
        hidden_size=config_dict.get('hidden_size', 256),
        num_hidden_layers=config_dict.get('num_hidden_layers', 6),
        num_attention_heads=config_dict.get('num_attention_heads', 16),
        intermediate_size=config_dict.get('intermediate_size', 1024),
        max_position_embeddings=config_dict.get('max_position_embeddings', 257),
        dropout_prob=config_dict.get('dropout_prob', 0.2),
        vocab_size=config_dict.get('vocab_size', 43),
        use_cls_token=config_dict.get('use_cls_token', True)
    )

    # Load class mapping
    with open(model_dir / "class_index_mapping.json", 'r') as f:
        id_to_gloss = json.load(f)

    # Load masked classes (optional)
    masked_class_ids = []
    masked_file = model_dir / "masked_classes.json"
    if masked_file.exists():
        with open(masked_file, 'r') as f:
            masked_config = json.load(f)
        masked_class_ids = masked_config.get('masked_class_ids', [])
        print(f"  Masking {len(masked_class_ids)} classes")

    # Create model
    model = OpenHandsModel(model_config)
    weights_path = model_dir / "pytorch_model.bin"
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    processor = WLASLPoseProcessor()

    print(f"  Model loaded: {model_config.vocab_size} classes, {model_config.hidden_size} hidden")
    return model, model_config, id_to_gloss, masked_class_ids, processor


def predict_pickle(pickle_path, model, model_config, id_to_gloss, masked_class_ids, processor):
    """Run prediction on a single pickle file. Returns top-3 predictions."""
    import torch
    import numpy as np

    pose_sequence = processor.load_pickle_pose(str(pickle_path))
    pose_sequence = processor.preprocess_pose_sequence(pose_sequence, augment=False)

    # Finger features
    finger_features = None
    if model_config.use_finger_features:
        finger_features = processor.extract_finger_features(pose_sequence)

    pose_sequence, attention_mask = processor.pad_or_truncate_sequence(pose_sequence, 256)

    if finger_features is not None:
        seq_len = len(finger_features)
        if seq_len > 256:
            finger_features = finger_features[:256]
        elif seq_len < 256:
            padding = np.zeros((256 - seq_len, 30), dtype=np.float32)
            finger_features = np.vstack([finger_features, padding])

    pose_tensor = torch.from_numpy(pose_sequence).float().unsqueeze(0)
    attention_tensor = torch.from_numpy(attention_mask).long().unsqueeze(0)
    finger_tensor = None
    if finger_features is not None:
        finger_tensor = torch.from_numpy(finger_features).float().unsqueeze(0)

    with torch.no_grad():
        outputs = model(pose_tensor, attention_tensor, finger_tensor)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs

        if masked_class_ids:
            for cid in masked_class_ids:
                logits[:, cid] = float('-inf')

        probs = torch.softmax(logits, dim=-1)

    top_probs, top_indices = torch.topk(probs[0], 3)
    result = {
        'top_prediction': id_to_gloss[str(top_indices[0].item())].lower(),
        'confidence': top_probs[0].item(),
        'top_k': []
    }
    for i in range(3):
        idx = top_indices[i].item()
        result['top_k'].append({
            'gloss': id_to_gloss[str(idx)].lower(),
            'confidence': top_probs[i].item()
        })
    return result


def profile_all_signs(model, model_config, id_to_gloss, masked_class_ids, processor,
                      use_augmented=True, max_pickles_per_sign=None):
    """
    Profile ALL 43 signs using ALL val pickle files.

    Args:
        use_augmented: If True, use augmented pickles too (not just originals)
        max_pickles_per_sign: Limit per sign (None = use all)

    Returns:
        dict: gloss -> profile with detailed stats
    """
    with open(VAL_MANIFEST, 'r') as f:
        manifest = json.load(f)

    pickle_pool = Path(manifest['pickle_pool'])

    profiles = {}
    total_predictions = 0

    for gloss_lower in sorted(manifest['classes'].keys()):
        gloss_upper = gloss_lower.upper()
        families = manifest['classes'][gloss_lower]

        all_files = []
        for family in families:
            for fname in family['files']:
                if not use_augmented and 'aug' in fname:
                    continue
                fpath = pickle_pool / gloss_lower / fname
                if fpath.exists():
                    all_files.append((fname, fpath))

        if max_pickles_per_sign and len(all_files) > max_pickles_per_sign:
            # Always include originals, sample the rest
            originals = [(f, p) for f, p in all_files if 'aug' not in f]
            augmented = [(f, p) for f, p in all_files if 'aug' in f]
            random.shuffle(augmented)
            all_files = originals + augmented[:max_pickles_per_sign - len(originals)]

        top1_correct = 0
        top3_hits = 0
        top1_confidences = []
        correct_in_top3_confidences = []
        confusions = defaultdict(int)
        per_pickle_results = []

        for fname, fpath in all_files:
            try:
                pred = predict_pickle(fpath, model, model_config, id_to_gloss,
                                      masked_class_ids, processor)
                total_predictions += 1

                top1 = pred['top_prediction'].upper()
                top1_conf = pred['confidence']
                top1_confidences.append(top1_conf)

                if top1 == gloss_upper:
                    top1_correct += 1
                else:
                    confusions[top1] += 1

                top3_glosses = [t['gloss'].upper() for t in pred['top_k']]
                in_top3 = gloss_upper in top3_glosses
                if in_top3:
                    top3_hits += 1
                    for t in pred['top_k']:
                        if t['gloss'].upper() == gloss_upper:
                            correct_in_top3_confidences.append(t['confidence'])
                            break

                per_pickle_results.append({
                    'file': fname,
                    'is_augmented': 'aug' in fname,
                    'top1': top1,
                    'top1_conf': top1_conf,
                    'in_top3': in_top3,
                    'correct_conf': correct_in_top3_confidences[-1] if in_top3 else 0,
                })

            except Exception as e:
                print(f"  ERROR {gloss_lower}/{fname}: {e}")
                continue

        n = len(per_pickle_results)
        if n == 0:
            continue

        import numpy as np
        top1_acc = top1_correct / n * 100
        top3_rate = top3_hits / n * 100
        avg_top1_conf = np.mean(top1_confidences) * 100
        std_top1_conf = np.std(top1_confidences) * 100
        avg_correct_conf = (
            np.mean(correct_in_top3_confidences) * 100
            if correct_in_top3_confidences else 0
        )
        std_correct_conf = (
            np.std(correct_in_top3_confidences) * 100
            if len(correct_in_top3_confidences) > 1 else 0
        )

        # Classify difficulty based on multi-pickle stats
        if top3_rate < 30:
            difficulty = 'HARD'
        elif top3_rate < 60:
            difficulty = 'MEDIUM-HARD'
        elif top3_rate < 85:
            difficulty = 'MEDIUM'
        elif avg_correct_conf < 50:
            difficulty = 'MEDIUM-EASY'
        else:
            difficulty = 'EASY'

        profiles[gloss_upper] = {
            'n_pickles': n,
            'n_originals': sum(1 for r in per_pickle_results if not r['is_augmented']),
            'top1_accuracy': round(top1_acc, 1),
            'top3_hit_rate': round(top3_rate, 1),
            'avg_top1_confidence': round(avg_top1_conf, 1),
            'std_top1_confidence': round(std_top1_conf, 1),
            'avg_correct_in_top3_confidence': round(avg_correct_conf, 1),
            'std_correct_in_top3_confidence': round(std_correct_conf, 1),
            'difficulty': difficulty,
            'confusions': dict(confusions),
            'per_pickle': per_pickle_results,
        }

        # Progress
        print(f"  {gloss_upper:<14} n={n:>2} Top1={top1_acc:>5.1f}% Top3={top3_rate:>5.1f}% "
              f"AvgConf={avg_top1_conf:>5.1f}% -> {difficulty}")

    print(f"\nTotal predictions run: {total_predictions}")
    return profiles


def print_profiles(profiles):
    """Print readable profile table."""
    import numpy as np

    print()
    print("=" * 130)
    print("PER-SIGN CONFIDENCE PROFILES (multi-pickle, all val data)")
    print("=" * 130)
    print(f"{'Sign':<14} {'N':>3} {'Top1Acc%':>9} {'Top3Hit%':>9} "
          f"{'AvgTop1%':>9} {'StdTop1':>8} {'AvgCorr%':>9} {'StdCorr':>8} "
          f"{'Difficulty':<12} {'Top Confusions'}")
    print("-" * 130)

    diff_order = {'HARD': 0, 'MEDIUM-HARD': 1, 'MEDIUM': 2, 'MEDIUM-EASY': 3, 'EASY': 4}
    sorted_profiles = sorted(profiles.items(),
                             key=lambda x: (diff_order.get(x[1]['difficulty'], 5), x[0]))

    for gloss, p in sorted_profiles:
        conf_str = ""
        if p['confusions']:
            top_conf = sorted(p['confusions'].items(), key=lambda x: -x[1])[:3]
            conf_str = ", ".join(f"{k}({v})" for k, v in top_conf)

        print(f"{gloss:<14} {p['n_pickles']:>3} {p['top1_accuracy']:>9.1f} "
              f"{p['top3_hit_rate']:>9.1f} {p['avg_top1_confidence']:>9.1f} "
              f"{p['std_top1_confidence']:>8.1f} {p['avg_correct_in_top3_confidence']:>9.1f} "
              f"{p['std_correct_in_top3_confidence']:>8.1f} {p['difficulty']:<12} {conf_str}")

    print()
    counts = defaultdict(int)
    for p in profiles.values():
        counts[p['difficulty']] += 1
    print("Difficulty distribution:")
    for diff in ['HARD', 'MEDIUM-HARD', 'MEDIUM', 'MEDIUM-EASY', 'EASY']:
        signs = [g for g, p in profiles.items() if p['difficulty'] == diff]
        print(f"  {diff:<14}: {counts[diff]:>2} signs  {', '.join(sorted(signs))}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: POWER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def compute_sample_sizes():
    """Calculate required sample sizes using power analysis."""
    from scipy import stats as scipy_stats

    print()
    print("=" * 100)
    print("POWER ANALYSIS -- REQUIRED SAMPLE SIZES")
    print("=" * 100)

    alpha = 0.05
    z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)

    print(f"\nSignificance level: alpha = {alpha} (two-tailed)")
    print()
    print(f"{'Power':>6} {'Min Delta':>10} {'Std Dev':>9} {'N required':>11} {'Description'}")
    print("-" * 70)

    scenarios = [
        (0.80, 5, 10, "Small effect, low variance"),
        (0.80, 5, 15, "Small effect, medium variance"),
        (0.80, 5, 20, "Small effect, high variance"),
        (0.80, 8, 15, "Practical effect, medium variance"),
        (0.80, 10, 15, "Medium effect, medium variance"),
        (0.80, 10, 20, "Medium effect, high variance"),
        (0.90, 5, 15, "Small effect, med var, 90% power"),
        (0.90, 8, 15, "Practical, med var, 90% power"),
        (0.90, 10, 20, "Medium effect, high var, 90% power"),
    ]

    for power, delta, sigma, desc in scenarios:
        z_beta = scipy_stats.norm.ppf(power)
        n = math.ceil(((z_alpha + z_beta) * sigma / delta) ** 2)
        print(f"{power:>6.0%} {delta:>10} {sigma:>9} {n:>11} {desc}")

    print()
    print("RECOMMENDATION: 5 bands x 30 sentences = 150 total")
    print("  Detects delta >= 8 CTQI with 80% power at std=15")


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: STRATIFIED SENTENCE GENERATION
# ═══════════════════════════════════════════════════════════════════════

def design_dataset(profiles, n_per_band=30):
    """
    Generate stratified sentence dataset using variable-length sentences.

    For a sentence with N glosses where H are hard (top3_hit_rate < X):
      Expected Top-3 accuracy ~ (N - H) / N * 100

    Variable lengths (3-5 glosses) let us hit more target bands even with
    limited hard signs.
    """
    print()
    print("=" * 100)
    print("SENTENCE GENERATION -- STRATIFIED BY TOP-3 ACCURACY BAND")
    print("=" * 100)

    # Classify signs by difficulty
    hard = [g for g, p in profiles.items() if p['difficulty'] in ('HARD', 'MEDIUM-HARD')]
    medium = [g for g, p in profiles.items() if p['difficulty'] in ('MEDIUM', 'MEDIUM-EASY')]
    easy = [g for g, p in profiles.items() if p['difficulty'] == 'EASY']

    print(f"\nSign pools:")
    print(f"  HARD+MEDIUM-HARD: {len(hard):>2} signs: {', '.join(sorted(hard))}")
    print(f"  MEDIUM+MEDIUM-EASY: {len(medium):>2} signs: {', '.join(sorted(medium))}")
    print(f"  EASY:             {len(easy):>2} signs: {', '.join(sorted(easy))}")

    # Band definitions with multiple recipes (length, n_hard, n_medium, n_easy)
    # Each recipe produces sentences with a specific expected Top-3 accuracy
    bands = [
        {
            'name': 'Band A: 0-33% Top-3',
            'target': '0-33%',
            'recipes': [
                # (length, n_hard, n_med, n_easy) -> expected top3 acc
                (3, 3, 0, 0),  # 0/3 = 0%
                (3, 2, 1, 0),  # ~0-33%
                (4, 3, 1, 0),  # ~0-25%
                (4, 4, 0, 0),  # 0%
                (5, 4, 1, 0),  # ~0-20%
                (5, 5, 0, 0),  # 0%
                (5, 3, 2, 0),  # ~0-40% (borderline)
            ],
        },
        {
            'name': 'Band B: 33-50% Top-3',
            'target': '33-50%',
            'recipes': [
                (3, 2, 0, 1),  # ~33%
                (3, 2, 1, 0),  # ~33%
                (4, 2, 1, 1),  # ~25-50%
                (4, 3, 0, 1),  # ~25%
                (5, 3, 0, 2),  # ~40%
                (5, 3, 1, 1),  # ~20-40%
                (5, 2, 2, 1),  # ~40-60%
            ],
        },
        {
            'name': 'Band C: 50-67% Top-3',
            'target': '50-67%',
            'recipes': [
                (3, 1, 0, 2),  # ~67%
                (3, 1, 1, 1),  # ~33-67%
                (4, 1, 1, 2),  # ~50-75%
                (4, 2, 0, 2),  # ~50%
                (5, 2, 0, 3),  # ~60%
                (5, 1, 2, 2),  # ~60-80%
                (5, 2, 1, 2),  # ~40-60%
            ],
        },
        {
            'name': 'Band D: 67-83% Top-3',
            'target': '67-83%',
            'recipes': [
                (3, 0, 1, 2),  # ~67-100%
                (3, 1, 0, 2),  # ~67%
                (4, 1, 0, 3),  # ~75%
                (4, 0, 1, 3),  # ~75-100%
                (5, 1, 0, 4),  # ~80%
                (5, 0, 1, 4),  # ~80-100%
                (5, 1, 1, 3),  # ~60-80%
            ],
        },
        {
            'name': 'Band E: 83-100% Top-3',
            'target': '83-100%',
            'recipes': [
                (3, 0, 0, 3),  # ~100%
                (4, 0, 0, 4),  # ~100%
                (5, 0, 0, 5),  # ~100%
                (5, 0, 1, 4),  # ~80-100%
                (4, 0, 1, 3),  # ~75-100%
            ],
        },
    ]

    existing_combos = set()
    if EXISTING_DATASET.exists():
        with open(EXISTING_DATASET, 'r') as f:
            existing = json.load(f)
        for entry in existing:
            key = tuple(sorted(g.upper() for g in entry['glosses']))
            existing_combos.add(key)

    dataset = []
    entry_id = 1000

    for band in bands:
        band_sentences = []
        attempts = 0
        max_attempts = n_per_band * 200

        while len(band_sentences) < n_per_band and attempts < max_attempts:
            attempts += 1

            # Pick a random recipe
            recipe = random.choice(band['recipes'])
            length, n_h, n_m, n_e = recipe

            # Try to fill the recipe
            try:
                chosen = []

                if n_h > 0:
                    if len(hard) < n_h:
                        continue
                    chosen.extend(random.sample(hard, n_h))

                if n_m > 0:
                    available = [s for s in medium if s not in chosen]
                    if len(available) < n_m:
                        continue
                    chosen.extend(random.sample(available, n_m))

                if n_e > 0:
                    available = [s for s in easy if s not in chosen]
                    if len(available) < n_e:
                        continue
                    chosen.extend(random.sample(available, n_e))

            except ValueError:
                continue

            if len(chosen) != length:
                continue

            combo_key = tuple(sorted(chosen))
            if combo_key in existing_combos:
                continue
            existing_combos.add(combo_key)

            random.shuffle(chosen)

            # Auto-generate placeholder sentence
            sentence = " ".join(g.lower() for g in chosen).capitalize() + "."

            # Calculate expected top-3 accuracy based on sign profiles
            top3_probs = []
            for g in chosen:
                if g in profiles:
                    top3_probs.append(profiles[g]['top3_hit_rate'] / 100)
                else:
                    top3_probs.append(0.5)  # unknown
            expected_top3 = sum(top3_probs) / len(top3_probs) * 100

            band_sentences.append({
                'glosses': chosen,
                'sentence': sentence,
                'id': entry_id,
                'band': band['target'],
                'n_glosses': length,
                'recipe': f"{n_h}H+{n_m}M+{n_e}E",
                'expected_top3_acc': round(expected_top3, 1),
                'note': f"Threshold exp - {band['target']} ({n_h}H+{n_m}M+{n_e}E, len={length})",
            })
            entry_id += 1

        print(f"\n  {band['name']}: {len(band_sentences)}/{n_per_band} "
              f"({attempts} attempts)")
        if band_sentences:
            recipes_used = defaultdict(int)
            for s in band_sentences:
                recipes_used[s['recipe']] += 1
            for recipe, count in sorted(recipes_used.items()):
                print(f"    {recipe}: {count} sentences")

        dataset.extend(band_sentences)

    return dataset


def print_dataset_summary(dataset):
    """Print summary of generated dataset."""
    import numpy as np

    print()
    print("=" * 100)
    print("GENERATED DATASET SUMMARY")
    print("=" * 100)

    band_counts = defaultdict(list)
    for entry in dataset:
        band_counts[entry['band']].append(entry)

    print(f"\nTotal sentences: {len(dataset)}")

    for band_key in sorted(band_counts.keys()):
        entries = band_counts[band_key]
        lengths = [e['n_glosses'] for e in entries]
        exp_top3 = [e['expected_top3_acc'] for e in entries]
        unique_glosses = set()
        for e in entries:
            for g in e['glosses']:
                unique_glosses.add(g)

        import numpy as np
        print(f"\n  {band_key}: {len(entries)} sentences")
        print(f"    Gloss lengths: {min(lengths)}-{max(lengths)} (avg {np.mean(lengths):.1f})")
        print(f"    Expected Top-3 acc: {np.mean(exp_top3):.1f}% "
              f"(range {min(exp_top3):.0f}-{max(exp_top3):.0f}%)")
        print(f"    Unique glosses: {len(unique_glosses)}")
        for e in entries[:3]:
            print(f"    Ex: {' '.join(e['glosses'])} [{e['recipe']}] "
                  f"(exp_top3={e['expected_top3_acc']:.0f}%)")

    print()
    print("NOTE: Reference sentences are placeholder gloss-joins.")
    print("      Rewrite to natural English before running evaluation.")


def save_outputs(dataset, profiles, output_dir):
    """Save all outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset for evaluation
    eval_dataset = [{
        'glosses': e['glosses'],
        'sentence': e['sentence'],
        'id': e['id'],
        'note': e.get('note', ''),
        'target_band': e.get('band', ''),
        'expected_top3_acc': e.get('expected_top3_acc', 0),
    } for e in dataset]

    dataset_path = output_dir / "threshold_experiment_dataset.json"
    with open(dataset_path, 'w') as f:
        json.dump(eval_dataset, f, indent=2)
    print(f"\nSaved dataset: {dataset_path} ({len(eval_dataset)} sentences)")

    # Profiles (without per_pickle details for readability)
    profiles_summary = {}
    for gloss, p in profiles.items():
        profiles_summary[gloss] = {k: v for k, v in p.items() if k != 'per_pickle'}
    profiles_path = output_dir / "per_sign_profiles.json"
    with open(profiles_path, 'w') as f:
        json.dump(profiles_summary, f, indent=2)
    print(f"Saved profiles: {profiles_path}")

    # Full profiles with per-pickle data
    full_profiles_path = output_dir / "per_sign_profiles_full.json"
    with open(full_profiles_path, 'w') as f:
        json.dump(profiles, f, indent=2)
    print(f"Saved full profiles: {full_profiles_path}")

    # Experiment design doc
    design_path = output_dir / "experiment_design.md"
    with open(design_path, 'w') as f:
        f.write("# Threshold Experiment Design\n\n")
        f.write("## Objective\n")
        f.write("Find the minimum Top-3 confidence below which LLM sentence construction\n")
        f.write("does NOT add statistically significant CTQI v3 improvement.\n\n")

        f.write("## Methodology\n")
        f.write("### 1. Multi-Pickle Profiling\n")
        f.write("Profiled all 43 signs using all val pickle files (original + augmented)\n")
        f.write("to get robust per-sign confidence distributions.\n\n")
        f.write("### 2. Variable-Length Sentences\n")
        f.write("Used 3-5 gloss sentences mixing hard/medium/easy signs to target\n")
        f.write("specific Top-3 accuracy bands.\n\n")
        f.write("### 3. Stratified Design\n")
        f.write("5 bands x 30 sentences = 150 total, targeting:\n")
        f.write("- Band A: 0-33% Top-3 accuracy\n")
        f.write("- Band B: 33-50%\n")
        f.write("- Band C: 50-67%\n")
        f.write("- Band D: 67-83%\n")
        f.write("- Band E: 83-100%\n\n")

        f.write("## Sign Difficulty Classification\n")
        f.write("| Difficulty | Signs |\n")
        f.write("|---|---|\n")
        for diff in ['HARD', 'MEDIUM-HARD', 'MEDIUM', 'MEDIUM-EASY', 'EASY']:
            signs = sorted([g for g, p in profiles.items() if p['difficulty'] == diff])
            if signs:
                f.write(f"| {diff} | {', '.join(signs)} |\n")
        f.write("\n")

        f.write("## How to Run\n")
        f.write("### Step 1: Rewrite reference sentences\n")
        f.write("Auto-generated sentences are gloss-joins. Rewrite to natural English.\n\n")
        f.write("### Step 2: Run evaluation\n")
        f.write("```bash\n")
        f.write("python evaluate_synthetic_dataset.py \\\n")
        f.write(f"  --dataset {dataset_path} \\\n")
        f.write("  --use-manifest --use-llm --use-gemini-plausibility \\\n")
        f.write(f"  --output-dir {output_dir}/results\n")
        f.write("```\n\n")
        f.write("### Step 3: Analyze results\n")
        f.write("Run top3_threshold_analysis.py on the results directory.\n")

    print(f"Saved design doc: {design_path}")
    return dataset_path


def main():
    parser = argparse.ArgumentParser(description="Design threshold experiment")
    parser.add_argument("--output-dir", type=str,
                        default=str(SCRIPT_DIR / "threshold_experiment"),
                        help="Output directory")
    parser.add_argument("--n-per-band", type=int, default=30,
                        help="Sentences per band (default: 30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print analysis only, don't generate dataset")
    parser.add_argument("--profile-only", action="store_true",
                        help="Only run sign profiling, skip sentence generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max-pickles", type=int, default=None,
                        help="Max pickles per sign for profiling (None=all)")
    parser.add_argument("--originals-only", action="store_true",
                        help="Only use original (non-augmented) pickles for profiling")
    args = parser.parse_args()

    random.seed(args.seed)

    # Step 1: Profile all signs
    print("=" * 100)
    print("STEP 1: MULTI-PICKLE SIGN PROFILING")
    print("=" * 100)
    print(f"\nLoading model from {PRODUCTION_MODEL_DIR}...")

    model, model_config, id_to_gloss, masked_class_ids, processor = load_model()

    print(f"\nProfiling all {len(VOCAB_43)} signs using "
          f"{'original only' if args.originals_only else 'all'} val pickles...")

    profiles = profile_all_signs(
        model, model_config, id_to_gloss, masked_class_ids, processor,
        use_augmented=not args.originals_only,
        max_pickles_per_sign=args.max_pickles,
    )
    print_profiles(profiles)

    if args.profile_only:
        # Save just profiles
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        profiles_path = output_dir / "per_sign_profiles.json"
        profiles_summary = {g: {k: v for k, v in p.items() if k != 'per_pickle'}
                            for g, p in profiles.items()}
        with open(profiles_path, 'w') as f:
            json.dump(profiles_summary, f, indent=2)
        print(f"\nSaved profiles: {profiles_path}")
        return

    # Step 2: Power analysis
    compute_sample_sizes()

    if args.dry_run:
        print("\n[DRY RUN] Skipping dataset generation.")
        return

    # Step 3: Generate stratified dataset
    dataset = design_dataset(profiles, n_per_band=args.n_per_band)
    print_dataset_summary(dataset)

    # Save
    save_outputs(dataset, profiles, args.output_dir)

    print()
    print("=" * 100)
    print("NEXT STEPS")
    print("=" * 100)
    print("\n1. REVIEW per_sign_profiles.json — check new hard/medium sign discoveries")
    print("2. REWRITE reference sentences in threshold_experiment_dataset.json")
    print("3. RUN evaluation pipeline (see experiment_design.md)")
    print("4. ANALYZE with top3_threshold_analysis.py")


if __name__ == "__main__":
    main()
