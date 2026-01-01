#!/usr/bin/env python3
"""
smart_gloss_selector.py

Smart 3-stage pipeline for selecting next glosses to add to a trained model.
Much faster than computing embeddings for all 2000 glosses.

Stage 1: Quick Filtering
  - Filter by sample count (>= min_samples)
  - Exclude glosses that current model confuses WITH
  - Result: ~100-200 candidates

Stage 2: Zero-Shot Distinctiveness Test
  - Run candidate samples through existing model
  - Measure prediction entropy (high = distinct = good)
  - Rank candidates by distinctiveness

Stage 3: Pre-validation / Confusion Prediction
  - Show which existing classes candidates would confuse with
  - Recommend YES/NO for each candidate

Usage:
    python smart_gloss_selector.py --model-dir <path> --keep-classes <gloss_list.json> --num-to-select 10
"""

import os
os.environ['PYTORCH_DISABLE_ONNX_METADATA'] = '1'
os.environ['TORCH_DYNAMO_DISABLE'] = '1'

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from config import get_config

# Add openhands model path
config = get_config()
openhands_src = config.openhands_dir / "src"
sys.path.insert(0, str(openhands_src))

from openhands_modernized import (
    OpenHandsModel, OpenHandsConfig, WLASLOpenHandsDataset
)


def load_model_and_classes(model_dir: Path):
    """Load trained model and its class mapping."""
    model_dir = Path(model_dir)

    # Load config
    config_path = model_dir / "config.json"
    with open(config_path, 'r') as f:
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
        vocab_size=config_dict.get('vocab_size', 100),
        use_cls_token=config_dict.get('use_cls_token', True)
    )

    # Load class mapping
    mapping_path = model_dir / "class_index_mapping.json"
    with open(mapping_path, 'r') as f:
        id_to_gloss = json.load(f)

    gloss_to_id = {g.upper(): int(i) for i, g in id_to_gloss.items()}

    # Create and load model
    model = OpenHandsModel(model_config)
    weights_path = model_dir / "pytorch_model.bin"
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    return model, id_to_gloss, gloss_to_id, model_config


def get_all_available_glosses(data_pool: Path, mapping_file: Path = None):
    """Get all glosses and their sample counts from data pool.

    Supports two modes:
    1. Directory-based: data_pool contains gloss subdirectories with .pkl/.pose files
    2. Flat with mapping: data_pool contains flat .pkl files, mapping_file maps video_id -> gloss
    """
    gloss_samples = {}

    # Check if this is a flat structure with a mapping file
    if mapping_file and mapping_file.exists():
        # Flat structure: use mapping to group files by gloss
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)

        # Group by gloss
        from collections import defaultdict
        gloss_files = defaultdict(list)

        for video_id, info in mapping.items():
            gloss = info['gloss'].upper()
            pkl_file = data_pool / f"{video_id}.pkl"
            if pkl_file.exists():
                gloss_files[gloss].append(str(pkl_file))

        for gloss, files in gloss_files.items():
            gloss_samples[gloss] = {
                'count': len(files),
                'files': files
            }
    else:
        # Directory-based structure
        for gloss_dir in data_pool.iterdir():
            if gloss_dir.is_dir():
                gloss_name = gloss_dir.name.upper()
                # Look for .pkl files
                data_files = list(gloss_dir.glob("*.pkl"))
                if data_files:
                    gloss_samples[gloss_name] = {
                        'count': len(data_files),
                        'files': [str(f) for f in data_files]
                    }

    return gloss_samples


def load_accuracy_report(report_path: Path):
    """Load accuracy report to get confusion information."""
    if not report_path.exists():
        return None

    with open(report_path, 'r') as f:
        return json.load(f)


def stage1_quick_filter(
    all_glosses: dict,
    keep_classes: set,
    accuracy_report: dict = None,
    min_samples: int = 3
):
    """
    Stage 1: Quick filtering.

    Args:
        all_glosses: Dict of gloss -> {count, files}
        keep_classes: Set of classes already in model (to keep)
        accuracy_report: Optional accuracy report for confusion target exclusion
        min_samples: Minimum samples needed for inference testing (default: 3)

    Returns:
        List of candidate glosses that pass filtering
    """
    print("\n" + "="*70)
    print("STAGE 1: Filtering")
    print("="*70)

    # Get confusion targets from accuracy report
    # These are glosses that existing classes get MISPREDICTED as
    confusion_targets = set()
    if accuracy_report:
        for gloss, stats in accuracy_report.get('per_class_stats', {}).items():
            for confusion in stats.get('top_confusions', []):
                conf_gloss = confusion['gloss'].upper()
                # If model confuses X with Y at high rate, Y is a "confusion target"
                if confusion['percent'] >= 25:  # 25%+ confusion rate
                    confusion_targets.add(conf_gloss)
        print(f"  Found {len(confusion_targets)} confusion targets to exclude")

    candidates = []
    excluded_count = 0
    confusion_excluded = 0
    too_few = 0

    for gloss, data in all_glosses.items():
        # Skip if already in keep classes
        if gloss in keep_classes:
            excluded_count += 1
            continue

        # Need at least a few samples to run inference test
        if data['count'] < min_samples:
            too_few += 1
            continue

        # Skip confusion targets (model already mispredicts other signs as this)
        if gloss in confusion_targets:
            confusion_excluded += 1
            continue

        candidates.append({
            'gloss': gloss,
            'sample_count': data['count'],
            'files': data['files']
        })

    # Sort by sample count (more samples = better diversity after augmentation)
    candidates.sort(key=lambda x: x['sample_count'], reverse=True)

    print(f"\n  Total glosses in pool: {len(all_glosses)}")
    print(f"  Already in model: {excluded_count}")
    print(f"  Too few samples (<{min_samples}): {too_few}")
    print(f"  Confusion targets excluded: {confusion_excluded}")
    print(f"  Candidates for Stage 2: {len(candidates)}")

    return candidates


def stage2_distinctiveness_test(
    model,
    candidates: list,
    gloss_to_id: dict,
    id_to_gloss: dict,
    baseline_classes: set,
    device: torch.device,
    max_samples_per_gloss: int = 20,
    max_seq_length: int = 256
):
    """
    Stage 2: Zero-shot distinctiveness test.

    Run candidate samples through existing model and measure prediction entropy
    ONLY against the baseline classes (not all classes the model knows).

    High entropy = predictions spread across baseline classes = gloss is distinct = good
    Low entropy = model confidently predicts one baseline class = too similar = bad

    Args:
        baseline_classes: Set of gloss names (uppercase) to measure distinctiveness against

    Returns:
        List of candidates with distinctiveness scores and confusion predictions
    """
    print("\n" + "="*70)
    print("STAGE 2: Zero-Shot Distinctiveness Test")
    print("="*70)
    print(f"  Measuring distinctiveness against {len(baseline_classes)} baseline classes")

    model.eval()
    results = []

    # Get indices of baseline classes in the model's output
    baseline_indices = []
    baseline_id_to_gloss = {}
    for class_id, gloss in id_to_gloss.items():
        if gloss.upper() in baseline_classes:
            baseline_indices.append(int(class_id))
            baseline_id_to_gloss[int(class_id)] = gloss

    baseline_indices = torch.tensor(baseline_indices, device=device)
    num_baseline = len(baseline_indices)
    print(f"  Found {num_baseline} baseline classes in model")

    for i, candidate in enumerate(candidates):
        gloss = candidate['gloss']
        files = candidate['files'][:max_samples_per_gloss]

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Processing {i+1}/{len(candidates)}: {gloss}")

        # Create dataset with dummy label (we just want predictions)
        # Use first class as dummy label
        first_class = list(gloss_to_id.keys())[0]
        labels = [first_class] * len(files)

        try:
            dataset = WLASLOpenHandsDataset(
                files, labels, gloss_to_id,
                max_seq_length, augment=False, use_finger_features=True
            )
        except Exception as e:
            print(f"    Warning: Failed to create dataset for {gloss}: {e}")
            continue

        if len(dataset) == 0:
            continue

        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        all_baseline_probs = []  # Probabilities over baseline classes only
        all_predictions = []     # Predicted baseline class
        all_confidences = []     # Confidence in predicted baseline class

        with torch.no_grad():
            for batch in loader:
                pose_sequences = batch['pose_sequence'].to(device)
                attention_masks = batch['attention_mask'].to(device)
                finger_features = batch.get('finger_features')
                if finger_features is not None:
                    finger_features = finger_features.to(device)

                logits = model(pose_sequences, attention_masks, finger_features)

                # Extract only baseline class logits and re-normalize
                baseline_logits = logits[:, baseline_indices]
                baseline_probs = torch.softmax(baseline_logits, dim=-1)

                # Get top prediction among baseline classes
                confidences, pred_indices = torch.max(baseline_probs, dim=-1)

                # Map back to original class IDs
                predictions = baseline_indices[pred_indices]

                all_baseline_probs.append(baseline_probs.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())

        if not all_predictions:
            continue

        # Compute distinctiveness metrics against baseline only
        predictions = np.array(all_predictions)
        confidences = np.array(all_confidences)
        all_probs = np.concatenate(all_baseline_probs, axis=0)

        # Count predictions per baseline class
        pred_counts = defaultdict(int)
        for pred in predictions:
            pred_gloss = baseline_id_to_gloss.get(int(pred), f"CLASS_{pred}")
            pred_counts[pred_gloss] += 1

        # Compute entropy of prediction distribution over baseline classes
        # Use average probability distribution across all samples
        avg_probs = np.mean(all_probs, axis=0)
        entropy = float(-np.sum(avg_probs * np.log(avg_probs + 1e-10)))
        max_entropy = float(np.log(num_baseline))  # Max entropy over baseline classes
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Average confidence in predicted baseline class
        avg_confidence = float(np.mean(confidences))

        # Distinctiveness score: high entropy + low confidence = distinct from baseline
        distinctiveness = 0.6 * normalized_entropy + 0.4 * (1 - avg_confidence)

        # Top predicted classes (potential confusions)
        sorted_preds = sorted(pred_counts.items(), key=lambda x: -x[1])
        top_confusions = [
            {'gloss': g, 'percent': round(c / len(predictions) * 100, 1)}
            for g, c in sorted_preds[:3]
        ]

        results.append({
            'gloss': gloss,
            'sample_count': candidate['sample_count'],
            'distinctiveness': round(distinctiveness, 4),
            'entropy': round(normalized_entropy, 4),
            'avg_confidence': round(avg_confidence, 4),
            'top_confusions': top_confusions,
            'n_samples_tested': len(predictions)
        })

    # Sort by distinctiveness (higher = better)
    results.sort(key=lambda x: x['distinctiveness'], reverse=True)

    print(f"\n  Tested {len(results)} candidates")

    return results


def stage3_prevalidation(
    results: list,
    num_to_select: int = 10,
    min_distinctiveness: float = 0.3
):
    """
    Stage 3: Pre-validation and recommendations.

    Analyze results and provide recommendations.
    """
    print("\n" + "="*70)
    print("STAGE 3: Pre-validation & Recommendations")
    print("="*70)

    # Categorize candidates
    recommended = []
    maybe = []
    not_recommended = []

    for r in results:
        if r['distinctiveness'] >= 0.5:
            r['recommendation'] = 'YES'
            r['reason'] = 'High distinctiveness - predictions spread across classes'
            recommended.append(r)
        elif r['distinctiveness'] >= min_distinctiveness:
            r['recommendation'] = 'MAYBE'
            r['reason'] = 'Moderate distinctiveness - some confusion risk'
            maybe.append(r)
        else:
            r['recommendation'] = 'NO'
            top_conf = r['top_confusions'][0] if r['top_confusions'] else {'gloss': '?', 'percent': 0}
            r['reason'] = f"Low distinctiveness - {top_conf['percent']}% confused with {top_conf['gloss']}"
            not_recommended.append(r)

    print(f"\n  Recommended (distinctiveness >= 0.5): {len(recommended)}")
    print(f"  Maybe (distinctiveness >= {min_distinctiveness}): {len(maybe)}")
    print(f"  Not recommended: {len(not_recommended)}")

    # Select top candidates
    selected = recommended[:num_to_select]
    if len(selected) < num_to_select:
        # Add from maybe list if needed
        remaining = num_to_select - len(selected)
        selected.extend(maybe[:remaining])

    return {
        'selected': selected,
        'recommended': recommended,
        'maybe': maybe,
        'not_recommended': not_recommended,
        'all_results': results
    }


def print_results(stage3_results: dict, keep_classes: set, num_to_select: int):
    """Print human-readable results."""

    print("\n" + "="*70)
    print("SELECTED GLOSSES TO ADD")
    print("="*70)

    selected = stage3_results['selected'][:num_to_select]

    for i, r in enumerate(selected, 1):
        print(f"\n  {i}. {r['gloss']}")
        print(f"     Distinctiveness: {r['distinctiveness']:.3f} | Samples: {r['sample_count']}")
        print(f"     Recommendation: {r['recommendation']} - {r['reason']}")
        confusions = ", ".join([f"{c['gloss']}({c['percent']}%)" for c in r['top_confusions'][:2]])
        print(f"     Predicted confusions: {confusions}")

    print("\n" + "="*70)
    print("TOP 20 OTHER CANDIDATES (for reference)")
    print("="*70)

    # Show more candidates
    shown = set(r['gloss'] for r in selected)
    others = [r for r in stage3_results['all_results'] if r['gloss'] not in shown][:20]

    for r in others:
        conf = r['top_confusions'][0] if r['top_confusions'] else {'gloss': '?', 'percent': 0}
        print(f"  {r['gloss']:<20} dist={r['distinctiveness']:.3f}  "
              f"samples={r['sample_count']:>3}  "
              f"confused_with={conf['gloss']}({conf['percent']}%)")

    print("\n" + "="*70)
    print("NEW GLOSS LIST")
    print("="*70)

    new_glosses = sorted(keep_classes) + [r['gloss'].lower() for r in selected]
    print(f"\nTotal classes: {len(new_glosses)}")
    print(f"Existing: {len(keep_classes)}, New: {len(selected)}")
    print(f"\nGlosses: {new_glosses}")


def save_results(stage3_results: dict, keep_classes: set, output_dir: Path, num_to_select: int):
    """Save results to files."""

    selected = stage3_results['selected'][:num_to_select]

    # Save detailed report
    report = {
        'timestamp': datetime.now().isoformat(),
        'num_selected': len(selected),
        'num_existing': len(keep_classes),
        'total_new_classes': len(keep_classes) + len(selected),
        'selected_glosses': selected,
        'all_candidates_ranked': stage3_results['all_results'],
        'summary': {
            'recommended': len(stage3_results['recommended']),
            'maybe': len(stage3_results['maybe']),
            'not_recommended': len(stage3_results['not_recommended'])
        }
    }

    report_path = output_dir / "smart_selection_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Save new gloss list for training
    new_glosses = sorted([g.lower() for g in keep_classes]) + [r['gloss'].lower() for r in selected]
    new_count = len(new_glosses)

    gloss_list_path = output_dir / f"gloss_list_{new_count}_class.json"
    with open(gloss_list_path, 'w') as f:
        json.dump(new_glosses, f)

    print(f"\n  Saved report to: {report_path}")
    print(f"  Saved gloss list to: {gloss_list_path}")

    return gloss_list_path


def main():
    parser = argparse.ArgumentParser(
        description="Smart 3-stage gloss selection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--model-dir", "-m", type=Path, required=True,
                        help="Path to trained model directory")
    parser.add_argument("--keep-classes", "-k", type=Path, required=True,
                        help="Path to JSON file with classes to keep")
    parser.add_argument("--accuracy-report", "-a", type=Path, default=None,
                        help="Path to accuracy report (for confusion target exclusion)")
    parser.add_argument("--num-to-select", "-n", type=int, default=10,
                        help="Number of new glosses to select (default: 10)")
    parser.add_argument("--max-candidates", type=int, default=0,
                        help="Maximum candidates to test in Stage 2 (0=all, default: 0)")
    parser.add_argument("--data-pool", "-d", type=Path, default=None,
                        help="Path to data pool with all glosses (default: pose_files_by_gloss)")
    parser.add_argument("--output-dir", "-o", type=Path, default=None,
                        help="Output directory (default: script directory)")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = script_dir

    print("="*70)
    print("SMART GLOSS SELECTOR - 3-Stage Pipeline")
    print("="*70)
    print(f"Model: {args.model_dir}")
    print(f"Keep classes: {args.keep_classes}")
    print(f"Num to select: {args.num_to_select}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load keep classes
    with open(args.keep_classes, 'r') as f:
        keep_list = json.load(f)
    keep_classes = set(g.upper() for g in keep_list)
    print(f"Keep classes: {len(keep_classes)}")

    # Load model
    print("\nLoading model...")
    model, id_to_gloss, gloss_to_id, model_config = load_model_and_classes(args.model_dir)
    model = model.to(device)

    # Get data pool path (all available glosses)
    proj_config = get_config()
    mapping_file = None

    if args.data_pool:
        data_pool = args.data_pool
    else:
        # Default to complete pickle_files with mapping (all 2000 glosses)
        complete_pool = proj_config.dataset_root / "pickle_files"
        complete_mapping = proj_config.dataset_root / "video_to_gloss_mapping.json"

        if complete_pool.exists() and complete_mapping.exists():
            data_pool = complete_pool
            mapping_file = complete_mapping
        else:
            # Fallback to augmented pickle pool
            data_pool = None
            for num_classes in [100, 125, 43]:
                if num_classes in proj_config.dataset_splits:
                    splits = proj_config.dataset_splits[num_classes]
                    if 'pickle_pool' in splits:
                        data_pool = splits['pickle_pool']
                        break
            if data_pool is None:
                data_pool = proj_config.dataset_root.parent / "augmented_pool" / "pickle"

    data_pool = Path(data_pool)
    print(f"Data pool: {data_pool}")
    if mapping_file:
        print(f"Using mapping: {mapping_file}")

    # Get all available glosses
    print("\nScanning available glosses...")
    all_glosses = get_all_available_glosses(data_pool, mapping_file)
    print(f"Found {len(all_glosses)} glosses in pool")

    # Load accuracy report if provided (for confusion target exclusion)
    accuracy_report = None
    if args.accuracy_report:
        accuracy_report = load_accuracy_report(args.accuracy_report)

    # Stage 1: Quick filtering
    candidates = stage1_quick_filter(all_glosses, keep_classes, accuracy_report)

    # Optionally limit candidates for Stage 2
    if args.max_candidates > 0:
        candidates = candidates[:args.max_candidates]
        print(f"\n  Limited to top {len(candidates)} candidates for Stage 2")
    else:
        print(f"\n  Testing all {len(candidates)} candidates in Stage 2")

    # Stage 2: Distinctiveness test (against baseline classes only)
    results = stage2_distinctiveness_test(
        model, candidates, gloss_to_id, id_to_gloss, keep_classes, device
    )

    # Stage 3: Pre-validation
    stage3_results = stage3_prevalidation(results, args.num_to_select)

    # Print and save results
    print_results(stage3_results, keep_classes, args.num_to_select)
    gloss_list_path = save_results(stage3_results, keep_classes, args.output_dir, args.num_to_select)

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print(f"""
1. Review the selected glosses above

2. If satisfied, run pre-processing:
   python training_pre_processing_setup.py --gloss-file "{gloss_list_path}" --setup --force-fresh

3. Train the new model:
   python train_asl.py --num-classes {len(keep_classes) + args.num_to_select} --model-size small --dropout 0.3
""")

    return 0


if __name__ == "__main__":
    exit(main())
