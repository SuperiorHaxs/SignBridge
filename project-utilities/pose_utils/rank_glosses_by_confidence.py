#!/usr/bin/env python3
"""
rank_glosses_by_confidence.py

Ranks glosses by their prediction accuracy and confidence on dataset samples.
Runs model predictions on all pickle files in a split and ranks glosses by:
  1) Top-1 accuracy (% of times gloss is correct in top-1)
  2) Top-3 accuracy (% of times gloss is correct in top-3)
  3) Average confidence when present in top-5

Usage:
    python rank_glosses_by_confidence.py --dataset-path PATH --classes 100
    python rank_glosses_by_confidence.py --dataset-path PATH --classes 100 --split val
    python rank_glosses_by_confidence.py --dataset-path PATH --classes 100 --checkpoint PATH
    python rank_glosses_by_confidence.py --dataset-path PATH --classes 100 --output results.json

Example:
    python rank_glosses_by_confidence.py \
        --dataset-path "C:/Users/ashwi/Projects/WLASL-proj/asl-v1/datasets/wlasl_poses_complete/dataset_splits" \
        --classes 100 --split val
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Add paths for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Go up two levels: pose_utils -> project-utilities -> project root
MODELS_DIR = PROJECT_ROOT / "models"

sys.path.insert(0, str(MODELS_DIR / "openhands-modernized" / "src"))
sys.path.insert(0, str(MODELS_DIR / "openhands-modernized" / "src" / "util"))


def main():
    parser = argparse.ArgumentParser(
        description="Rank glosses by prediction confidence on test set"
    )
    parser.add_argument(
        "--dataset-path", "-d",
        required=True,
        help="Path to dataset_splits folder"
    )
    parser.add_argument(
        "--classes", "-c",
        type=int,
        required=True,
        choices=[20, 50, 100, 125],
        help="Number of classes (20, 50, or 100)"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to model checkpoint (default: auto-detect based on classes)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file for detailed results"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of glosses to process (for testing)"
    )
    parser.add_argument(
        "--split", "-s",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: test)"
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"ERROR: Dataset path not found: {dataset_path}")
        sys.exit(1)

    # Build path to test pickle files
    # Structure varies by class count:
    #   20/50 classes: dataset_splits/{N}_classes/original/pickle_from_pose_split_{N}_class/test/{gloss}/*.pkl
    #   100 classes: dataset_splits/{N}_classes/original/pickle_split_{N}_class/test/{gloss}/*.pkl
    classes_dir = dataset_path / f"{args.classes}_classes" / "original"

    # Try different folder naming patterns
    possible_pickle_dirs = [
        classes_dir / f"pickle_from_pose_split_{args.classes}_class",
        classes_dir / f"pickle_split_{args.classes}_class",
    ]

    split_path = None
    for pickle_dir in possible_pickle_dirs:
        candidate = pickle_dir / args.split
        if candidate.exists():
            split_path = candidate
            break

    if split_path is None:
        print(f"ERROR: {args.split} path not found. Tried:")
        for p in possible_pickle_dirs:
            print(f"  - {p / args.split}")
        sys.exit(1)

    # Auto-detect checkpoint if not provided
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = MODELS_DIR / "openhands-modernized" / "production-models" / f"wlasl_{args.classes}_class_model"

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print("=" * 70)
    print("Gloss Confidence Ranking")
    print("=" * 70)
    print(f"Dataset path: {dataset_path}")
    print(f"Classes: {args.classes}")
    print(f"Split: {args.split}")
    print(f"Split path: {split_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print()

    # Load model
    print("Loading model...")
    from openhands_modernized_inference import load_model_from_checkpoint, predict_pose_file
    model, tokenizer = load_model_from_checkpoint(str(checkpoint_path))
    print()

    # Get all gloss folders
    gloss_folders = sorted([d for d in split_path.iterdir() if d.is_dir()])
    total_glosses = len(gloss_folders)
    print(f"Found {total_glosses} gloss folders")

    if args.limit:
        gloss_folders = gloss_folders[:args.limit]
        print(f"Limiting to first {args.limit} glosses")
    print()

    # Process each gloss
    gloss_results = {}

    for i, gloss_folder in enumerate(gloss_folders, 1):
        gloss_name = gloss_folder.name.upper()
        pickle_files = list(gloss_folder.glob("*.pkl"))

        if not pickle_files:
            print(f"[{i}/{len(gloss_folders)}] {gloss_name}: No pickle files found")
            continue

        top1_correct = 0
        top3_correct = 0
        top5_confidences = []  # Confidence when present in top-5
        incorrect_predictions = []
        total_samples = len(pickle_files)

        for pkl_file in pickle_files:
            try:
                result = predict_pose_file(str(pkl_file), model=model, tokenizer=tokenizer)
                top_k_predictions = result['top_k_predictions'][:5]  # Top-5

                # Check Top-1
                if top_k_predictions and top_k_predictions[0]['gloss'].upper() == gloss_name:
                    top1_correct += 1

                # Check Top-3 and Top-5
                found_in_top3 = False
                found_in_top5 = False
                for rank, pred in enumerate(top_k_predictions):
                    if pred['gloss'].upper() == gloss_name:
                        if rank < 3:
                            found_in_top3 = True
                        found_in_top5 = True
                        top5_confidences.append(pred['confidence'])
                        break

                if found_in_top3:
                    top3_correct += 1

                if not found_in_top5:
                    incorrect_predictions.append({
                        'file': pkl_file.name,
                        'top_5': [(p['gloss'], p['confidence']) for p in top_k_predictions]
                    })
            except Exception as e:
                print(f"  Error processing {pkl_file.name}: {e}")

        # Calculate metrics
        top1_accuracy = top1_correct / total_samples if total_samples > 0 else 0
        top3_accuracy = top3_correct / total_samples if total_samples > 0 else 0
        avg_top5_conf = sum(top5_confidences) / len(top5_confidences) if top5_confidences else 0

        gloss_results[gloss_name] = {
            'gloss': gloss_name,
            'total_samples': total_samples,
            'top1_correct': top1_correct,
            'top3_correct': top3_correct,
            'top1_accuracy': top1_accuracy,
            'top3_accuracy': top3_accuracy,
            'avg_top5_confidence': avg_top5_conf,
            'top5_confidences': top5_confidences,
            'incorrect_predictions': incorrect_predictions
        }

        # Progress output
        conf_str = f"{avg_top5_conf:.1%}" if top5_confidences else "N/A"
        print(f"[{i}/{len(gloss_folders)}] {gloss_name:15s} | Top-1: {top1_accuracy:.0%} | Top-3: {top3_accuracy:.0%} | Avg Conf: {conf_str}")

    print()
    print("=" * 80)
    print("RANKING BY: 1) Top-1 Accuracy, 2) Top-3 Accuracy, 3) Avg Top-5 Confidence")
    print("=" * 80)
    print()

    # Sort by: 1) top1_accuracy, 2) top3_accuracy, 3) avg_top5_confidence
    sorted_glosses = sorted(
        gloss_results.values(),
        key=lambda x: (x['top1_accuracy'], x['top3_accuracy'], x['avg_top5_confidence']),
        reverse=True
    )

    print(f"{'Rank':<5} {'Gloss':<15} {'Top-1 Acc':<12} {'Top-3 Acc':<12} {'Avg Conf':<12} {'Samples':<10}")
    print("-" * 80)

    for rank, result in enumerate(sorted_glosses, 1):
        top1_str = f"{result['top1_accuracy']:.0%} ({result['top1_correct']}/{result['total_samples']})"
        top3_str = f"{result['top3_accuracy']:.0%}"
        conf_str = f"{result['avg_top5_confidence']:.1%}" if result['top5_confidences'] else "N/A"
        print(f"{rank:<5} {result['gloss']:<15} {top1_str:<12} {top3_str:<12} {conf_str:<12} {result['total_samples']:<10}")

    # Summary statistics
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_samples = sum(r['total_samples'] for r in gloss_results.values())
    total_top1_correct = sum(r['top1_correct'] for r in gloss_results.values())
    total_top3_correct = sum(r['top3_correct'] for r in gloss_results.values())
    overall_top1_accuracy = total_top1_correct / total_samples if total_samples > 0 else 0
    overall_top3_accuracy = total_top3_correct / total_samples if total_samples > 0 else 0

    all_confidences = []
    for r in gloss_results.values():
        all_confidences.extend(r['top5_confidences'])

    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

    print(f"Total glosses: {len(gloss_results)}")
    print(f"Total samples: {total_samples}")
    print(f"Overall Top-1 accuracy: {overall_top1_accuracy:.1%} ({total_top1_correct}/{total_samples})")
    print(f"Overall Top-3 accuracy: {overall_top3_accuracy:.1%} ({total_top3_correct}/{total_samples})")
    print(f"Average confidence (when in Top-5): {avg_confidence:.1%}")

    # Top 10 and Bottom 10
    print()
    print("Top 10 Best Glosses (by Top-1 accuracy):")
    for i, r in enumerate(sorted_glosses[:10], 1):
        top1_str = f"{r['top1_accuracy']:.0%}"
        top3_str = f"{r['top3_accuracy']:.0%}"
        print(f"  {i}. {r['gloss']} (Top-1: {top1_str}, Top-3: {top3_str})")

    print()
    print("Bottom 10 Worst Glosses (by Top-1 accuracy):")
    for i, r in enumerate(sorted_glosses[-10:], 1):
        top1_str = f"{r['top1_accuracy']:.0%}"
        top3_str = f"{r['top3_accuracy']:.0%}"
        print(f"  {i}. {r['gloss']} (Top-1: {top1_str}, Top-3: {top3_str})")

    # Save detailed results if output specified
    if args.output:
        output_path = Path(args.output)

        # Prepare JSON-serializable output
        output_data = {
            'summary': {
                'classes': args.classes,
                'split': args.split,
                'total_glosses': len(gloss_results),
                'total_samples': total_samples,
                'total_top1_correct': total_top1_correct,
                'total_top3_correct': total_top3_correct,
                'overall_top1_accuracy': overall_top1_accuracy,
                'overall_top3_accuracy': overall_top3_accuracy,
                'avg_top5_confidence': avg_confidence
            },
            'ranking': [
                {
                    'rank': i,
                    'gloss': r['gloss'],
                    'top1_accuracy': r['top1_accuracy'],
                    'top3_accuracy': r['top3_accuracy'],
                    'top1_correct': r['top1_correct'],
                    'top3_correct': r['top3_correct'],
                    'total_samples': r['total_samples'],
                    'avg_top5_confidence': r['avg_top5_confidence']
                }
                for i, r in enumerate(sorted_glosses, 1)
            ],
            'detailed_results': {
                gloss: {
                    'gloss': data['gloss'],
                    'top1_accuracy': data['top1_accuracy'],
                    'top3_accuracy': data['top3_accuracy'],
                    'top1_correct': data['top1_correct'],
                    'top3_correct': data['top3_correct'],
                    'total_samples': data['total_samples'],
                    'avg_top5_confidence': data['avg_top5_confidence'],
                    'incorrect_predictions': data['incorrect_predictions']
                }
                for gloss, data in gloss_results.items()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print()
        print(f"Detailed results saved to: {output_path}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
