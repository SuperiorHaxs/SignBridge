#!/usr/bin/env python3
"""
Profile all signs in a given production model using val manifest pickle files.
Reports per-sign Top-3 hit rate, confidence, and difficulty classification.

Usage:
  python profile_model_signs.py <model_dir> <val_manifest> [--output-dir DIR]

Example:
  python profile_model_signs.py \
    models/openhands-modernized/production-models/wlasl_38_class_healthcare_model \
    datasets/augmented_pool/splits/38_healthcare/val_manifest.json
"""

import json
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent


def load_model(model_dir):
    """Load a production model for inference."""
    openhands_src = PROJECT_ROOT / "models" / "openhands-modernized" / "src"
    sys.path.insert(0, str(openhands_src))

    import torch
    from openhands_modernized import OpenHandsModel, OpenHandsConfig, WLASLPoseProcessor

    model_dir = Path(model_dir)

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

    with open(model_dir / "class_index_mapping.json", 'r') as f:
        id_to_gloss = json.load(f)

    masked_class_ids = []
    masked_file = model_dir / "masked_classes.json"
    if masked_file.exists():
        with open(masked_file, 'r') as f:
            masked_config = json.load(f)
        masked_class_ids = masked_config.get('masked_class_ids', [])
        print(f"  Masking {len(masked_class_ids)} classes")

    model = OpenHandsModel(model_config)
    weights_path = model_dir / "pytorch_model.bin"
    import torch
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    processor = WLASLPoseProcessor()

    print(f"  Model loaded: {model_config.vocab_size} classes, {model_config.hidden_size} hidden")
    return model, model_config, id_to_gloss, masked_class_ids, processor


def predict_pickle(pickle_path, model, model_config, id_to_gloss, masked_class_ids, processor):
    """Run prediction on a single pickle file. Returns top-3 predictions."""
    import torch

    pose_sequence = processor.load_pickle_pose(str(pickle_path))
    pose_sequence = processor.preprocess_pose_sequence(pose_sequence, augment=False)

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

    k = min(5, probs.shape[-1])
    top_probs, top_indices = torch.topk(probs[0], k)
    result = {
        'top_prediction': id_to_gloss[str(top_indices[0].item())].lower(),
        'confidence': top_probs[0].item(),
        'top_k': []
    }
    for i in range(k):
        idx = top_indices[i].item()
        result['top_k'].append({
            'gloss': id_to_gloss[str(idx)].lower(),
            'confidence': top_probs[i].item()
        })
    return result


def profile_all_signs(model, model_config, id_to_gloss, masked_class_ids, processor,
                      val_manifest_path, max_pickles_per_sign=None):
    """Profile all signs using all val pickle files."""
    import random

    with open(val_manifest_path, 'r') as f:
        manifest = json.load(f)

    pickle_pool = Path(manifest['pickle_pool'])

    profiles = {}
    total_predictions = 0

    for gloss_lower in sorted(manifest['classes'].keys()):
        gloss_upper = gloss_lower.upper()
        class_data = manifest['classes'][gloss_lower]

        # Handle both manifest formats:
        # Format A (43-class): list of family dicts with 'families' key
        # Format B (38-healthcare): list of video dicts with 'files' key
        all_files = []
        if isinstance(class_data, list) and len(class_data) > 0:
            first = class_data[0]
            if 'families' in first:
                # Format A: families structure
                for family in class_data:
                    for fam_data in family['families'].values():
                        for fname in fam_data['samples']:
                            fpath = pickle_pool / gloss_lower / fname
                            if fpath.exists():
                                all_files.append((fname, fpath))
            elif 'files' in first:
                # Format B: flat list of video entries
                for entry in class_data:
                    for fname in entry['files']:
                        fpath = pickle_pool / gloss_lower / fname
                        if fpath.exists():
                            all_files.append((fname, fpath))
        elif isinstance(class_data, dict) and 'families' in class_data:
            for fam_id, fam_data in class_data['families'].items():
                for fname in fam_data['samples']:
                    fpath = pickle_pool / gloss_lower / fname
                    if fpath.exists():
                        all_files.append((fname, fpath))

        if not all_files:
            print(f"  WARNING: No pickle files found for {gloss_lower}")
            continue

        if max_pickles_per_sign and len(all_files) > max_pickles_per_sign:
            originals = [(f, p) for f, p in all_files if 'aug' not in f]
            augmented = [(f, p) for f, p in all_files if 'aug' in f]
            random.shuffle(augmented)
            all_files = originals + augmented[:max_pickles_per_sign - len(originals)]

        top1_correct = 0
        top3_hits = 0
        top5_hits = 0
        top1_confidences = []
        correct_in_top3_confidences = []
        correct_in_top5_confidences = []
        confusions = defaultdict(int)
        n_originals = 0

        for fname, fpath in all_files:
            try:
                pred = predict_pickle(fpath, model, model_config, id_to_gloss,
                                      masked_class_ids, processor)
                total_predictions += 1

                if 'aug' not in fname:
                    n_originals += 1

                top1 = pred['top_prediction'].upper()
                top1_conf = pred['confidence']
                top1_confidences.append(top1_conf)

                if top1 == gloss_upper:
                    top1_correct += 1
                else:
                    confusions[top1] += 1

                top_k_glosses = [t['gloss'].upper() for t in pred['top_k']]
                top3_glosses = top_k_glosses[:3]
                top5_glosses = top_k_glosses[:5]

                in_top3 = gloss_upper in top3_glosses
                in_top5 = gloss_upper in top5_glosses

                if in_top3:
                    top3_hits += 1
                    for t in pred['top_k'][:3]:
                        if t['gloss'].upper() == gloss_upper:
                            correct_in_top3_confidences.append(t['confidence'])
                            break

                if in_top5:
                    top5_hits += 1
                    for t in pred['top_k'][:5]:
                        if t['gloss'].upper() == gloss_upper:
                            correct_in_top5_confidences.append(t['confidence'])
                            break

            except Exception as e:
                print(f"  ERROR {gloss_lower}/{fname}: {e}")
                continue

        n = len(top1_confidences)
        if n == 0:
            continue

        top1_acc = top1_correct / n * 100
        top3_rate = top3_hits / n * 100
        top5_rate = top5_hits / n * 100
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
        avg_correct_top5_conf = (
            np.mean(correct_in_top5_confidences) * 100
            if correct_in_top5_confidences else 0
        )
        std_correct_top5_conf = (
            np.std(correct_in_top5_confidences) * 100
            if len(correct_in_top5_confidences) > 1 else 0
        )

        # Classify difficulty
        if top3_rate < 30:
            difficulty = "HARD"
        elif top3_rate < 85:
            difficulty = "MEDIUM"
        elif avg_correct_conf < 50:
            difficulty = "MEDIUM-EASY"
        else:
            difficulty = "EASY"

        profiles[gloss_upper] = {
            'n_pickles': n,
            'n_originals': n_originals,
            'top1_accuracy': round(top1_acc, 1),
            'top3_hit_rate': round(top3_rate, 1),
            'top5_hit_rate': round(top5_rate, 1),
            'avg_top1_confidence': round(avg_top1_conf, 1),
            'std_top1_confidence': round(std_top1_conf, 1),
            'avg_correct_in_top3_confidence': round(avg_correct_conf, 1),
            'std_correct_in_top3_confidence': round(std_correct_conf, 1),
            'avg_correct_in_top5_confidence': round(avg_correct_top5_conf, 1),
            'std_correct_in_top5_confidence': round(std_correct_top5_conf, 1),
            'difficulty': difficulty,
            'confusions': dict(sorted(confusions.items(), key=lambda x: -x[1])),
        }

        # Progress
        sys.stdout.write(f"\r  Profiled {len(profiles)}/{len(manifest['classes'])} signs ({total_predictions} predictions)")
        sys.stdout.flush()

    print()
    return profiles


def print_report(profiles):
    """Print a summary report of all sign profiles."""
    print("\n" + "=" * 100)
    print("SIGN PROFILE REPORT")
    print("=" * 100)

    # Sort by difficulty then confidence
    difficulty_order = {'HARD': 0, 'MEDIUM': 1, 'MEDIUM-EASY': 2, 'EASY': 3}
    sorted_signs = sorted(profiles.items(),
                          key=lambda x: (difficulty_order.get(x[1]['difficulty'], 4),
                                         x[1]['avg_correct_in_top3_confidence']))

    hdr = f"{'Sign':<16} {'Diff':<10} {'Top1%':>6} {'Top3%':>6} {'Top5%':>6} {'T1Conf%':>8} {'T3Conf%':>8} {'T5Conf%':>8} {'nPkl':>5} {'Top Confusion':<18}"
    print(hdr)
    print("-" * 110)

    for sign, p in sorted_signs:
        top_conf = ""
        if p['confusions']:
            top_c = list(p['confusions'].items())[0]
            top_conf = f"{top_c[0]} ({top_c[1]}x)"
        print(f"{sign:<16} {p['difficulty']:<10} {p['top1_accuracy']:>6.1f} {p['top3_hit_rate']:>6.1f} "
              f"{p.get('top5_hit_rate', 0):>6.1f} {p['avg_top1_confidence']:>8.1f} "
              f"{p['avg_correct_in_top3_confidence']:>8.1f} {p.get('avg_correct_in_top5_confidence', 0):>8.1f} "
              f"{p['n_pickles']:>5} {top_conf:<18}")

    # Summary stats
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    by_diff = defaultdict(list)
    for sign, p in profiles.items():
        by_diff[p['difficulty']].append(sign)

    for diff in ['HARD', 'MEDIUM', 'MEDIUM-EASY', 'EASY']:
        signs = by_diff.get(diff, [])
        print(f"  {diff}: {len(signs)} signs — {', '.join(sorted(signs))}")

    # Threshold analysis
    print("\n" + "=" * 100)
    print("LLM THRESHOLD ANALYSIS")
    print("=" * 100)
    print("\nCriteria: 100% Top-3 hit rate AND >=85% correct-gloss confidence")
    print("(Based on 150-sentence threshold experiment finding)\n")

    meets_criteria = []
    close = []
    fails = []

    for sign, p in sorted(profiles.items()):
        if p['top3_hit_rate'] == 100.0 and p['avg_correct_in_top3_confidence'] >= 85.0:
            meets_criteria.append((sign, p))
        elif p['top3_hit_rate'] >= 90.0:
            close.append((sign, p))
        else:
            fails.append((sign, p))

    print(f"MEETS CRITERIA ({len(meets_criteria)}/{len(profiles)} signs):")
    for sign, p in meets_criteria:
        print(f"  [Y] {sign:<20} Top3={p['top3_hit_rate']:.0f}%  Conf={p['avg_correct_in_top3_confidence']:.1f}%")

    print(f"\nCLOSE (Top-3 >=90% but doesn't fully meet criteria) ({len(close)} signs):")
    for sign, p in close:
        print(f"  [~] {sign:<20} Top3={p['top3_hit_rate']:.0f}%  Conf={p['avg_correct_in_top3_confidence']:.1f}%")

    print(f"\nFAILS (Top-3 <90%) ({len(fails)} signs):")
    for sign, p in fails:
        print(f"  [X] {sign:<20} Top3={p['top3_hit_rate']:.0f}%  Conf={p['avg_correct_in_top3_confidence']:.1f}%")

    pct = len(meets_criteria) / len(profiles) * 100
    print(f"\n=> {len(meets_criteria)}/{len(profiles)} signs ({pct:.0f}%) meet the LLM-effective threshold")


def main():
    parser = argparse.ArgumentParser(description="Profile model signs for LLM threshold analysis")
    parser.add_argument('model_dir', help="Path to production model directory")
    parser.add_argument('val_manifest', help="Path to val_manifest.json")
    parser.add_argument('--output-dir', default=None, help="Directory to save profiles JSON")
    parser.add_argument('--max-pickles', type=int, default=None, help="Max pickles per sign")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir

    val_manifest = Path(args.val_manifest)
    if not val_manifest.is_absolute():
        val_manifest = PROJECT_ROOT / val_manifest

    print(f"Model: {model_dir.name}")
    print(f"Manifest: {val_manifest}")
    print(f"Loading model...")
    model, model_config, id_to_gloss, masked_class_ids, processor = load_model(model_dir)

    print(f"Profiling signs...")
    profiles = profile_all_signs(
        model, model_config, id_to_gloss, masked_class_ids, processor,
        val_manifest, max_pickles_per_sign=args.max_pickles
    )

    print_report(profiles)

    # Save profiles
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = SCRIPT_DIR / "profiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"per_sign_profiles_{model_dir.name}.json"
    with open(out_file, 'w') as f:
        json.dump(profiles, f, indent=2)
    print(f"\nProfiles saved to: {out_file}")


if __name__ == "__main__":
    main()
