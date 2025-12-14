#!/usr/bin/env python3
"""
rank_glosses_by_confidence.py

Ranks glosses by their prediction confidence on test set samples.
Runs model predictions on all test pickle files and ranks glosses by
average confidence when the correct gloss appears in Top-3 predictions.

Usage:
    python rank_glosses_by_confidence.py --dataset-path PATH --classes 100
    python rank_glosses_by_confidence.py --dataset-path PATH --classes 100 --checkpoint PATH
    python rank_glosses_by_confidence.py --dataset-path PATH --classes 100 --output results.json

Example:
    python rank_glosses_by_confidence.py \
        --dataset-path "C:/Users/ashwi/Projects/WLASL-proj/asl-v1/datasets/wlasl_poses_complete/dataset_splits" \
        --classes 100
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Add paths for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
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
        choices=[20, 50, 100],
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

    test_path = None
    for pickle_dir in possible_pickle_dirs:
        candidate = pickle_dir / "test"
        if candidate.exists():
            test_path = candidate
            break

    if test_path is None:
        print(f"ERROR: Test path not found. Tried:")
        for p in possible_pickle_dirs:
            print(f"  - {p / 'test'}")
        sys.exit(1)

    # Auto-detect checkpoint if not provided
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = MODELS_DIR / "training-scripts" / "models" / f"wlasl_{args.classes}_class_model"

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print("=" * 70)
    print("Gloss Confidence Ranking")
    print("=" * 70)
    print(f"Dataset path: {dataset_path}")
    print(f"Classes: {args.classes}")
    print(f"Test path: {test_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print()

    # Load model
    print("Loading model...")
    from openhands_modernized_inference import load_model_from_checkpoint, predict_pose_file
    model, tokenizer = load_model_from_checkpoint(str(checkpoint_path))
    print()

    # Get all gloss folders
    gloss_folders = sorted([d for d in test_path.iterdir() if d.is_dir()])
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

        correct_confidences = []
        incorrect_predictions = []
        total_correct = 0
        total_samples = len(pickle_files)

        for pkl_file in pickle_files:
            try:
                result = predict_pose_file(str(pkl_file), model=model, tokenizer=tokenizer)
                top_k_predictions = result['top_k_predictions'][:3]  # Top-3

                # Check if correct gloss is in Top-3
                found_in_top_k = False
                for pred in top_k_predictions:
                    if pred['gloss'].upper() == gloss_name:
                        correct_confidences.append(pred['confidence'])
                        total_correct += 1
                        found_in_top_k = True
                        break

                if not found_in_top_k:
                    incorrect_predictions.append({
                        'file': pkl_file.name,
                        'top_3': [(p['gloss'], p['confidence']) for p in top_k_predictions]
                    })
            except Exception as e:
                print(f"  Error processing {pkl_file.name}: {e}")

        # Calculate metrics
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_correct_conf = sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0
        weighted_score = accuracy * avg_correct_conf  # Weighted average of accuracy and confidence

        gloss_results[gloss_name] = {
            'gloss': gloss_name,
            'total_samples': total_samples,
            'correct': total_correct,
            'accuracy': accuracy,
            'avg_correct_confidence': avg_correct_conf,
            'weighted_score': weighted_score,
            'correct_confidences': correct_confidences,
            'incorrect_predictions': incorrect_predictions
        }

        # Progress output
        conf_str = f"{avg_correct_conf:.1%}" if correct_confidences else "N/A"
        print(f"[{i}/{len(gloss_folders)}] {gloss_name:15s} | Acc: {accuracy:.0%} ({total_correct}/{total_samples}) | Avg Conf: {conf_str}")

    print()
    print("=" * 70)
    print("RANKING BY WEIGHTED SCORE (Accuracy x Confidence)")
    print("=" * 70)
    print()

    # Sort by weighted score (accuracy * confidence)
    sorted_glosses = sorted(
        gloss_results.values(),
        key=lambda x: x['weighted_score'],
        reverse=True
    )

    print(f"{'Rank':<5} {'Gloss':<15} {'Top-3 Acc':<12} {'Avg Conf':<12} {'Weighted':<12} {'Samples':<10}")
    print("-" * 70)

    for rank, result in enumerate(sorted_glosses, 1):
        acc_str = f"{result['accuracy']:.0%} ({result['correct']}/{result['total_samples']})"
        conf_str = f"{result['avg_correct_confidence']:.1%}" if result['correct'] > 0 else "N/A"
        weighted_str = f"{result['weighted_score']:.1%}" if result['correct'] > 0 else "N/A"
        print(f"{rank:<5} {result['gloss']:<15} {acc_str:<12} {conf_str:<12} {weighted_str:<12} {result['total_samples']:<10}")

    # Summary statistics
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_samples = sum(r['total_samples'] for r in gloss_results.values())
    total_correct = sum(r['correct'] for r in gloss_results.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

    all_confidences = []
    for r in gloss_results.values():
        all_confidences.extend(r['correct_confidences'])

    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

    print(f"Total glosses: {len(gloss_results)}")
    print(f"Total samples: {total_samples}")
    print(f"Overall Top-3 accuracy: {overall_accuracy:.1%} ({total_correct}/{total_samples})")
    print(f"Average confidence (when in Top-3): {avg_confidence:.1%}")

    # Top 10 and Bottom 10
    print()
    print("Top 10 Best Glosses (by weighted score):")
    for i, r in enumerate(sorted_glosses[:10], 1):
        weighted_str = f"{r['weighted_score']:.1%}" if r['correct'] > 0 else "N/A"
        print(f"  {i}. {r['gloss']} (weighted: {weighted_str})")

    print()
    print("Bottom 10 Worst Glosses (by weighted score):")
    for i, r in enumerate(sorted_glosses[-10:], 1):
        weighted_str = f"{r['weighted_score']:.1%}" if r['correct'] > 0 else "N/A (0 correct)"
        print(f"  {i}. {r['gloss']} (weighted: {weighted_str})")

    # Save detailed results if output specified
    if args.output:
        output_path = Path(args.output)

        # Prepare JSON-serializable output
        output_data = {
            'summary': {
                'classes': args.classes,
                'total_glosses': len(gloss_results),
                'total_samples': total_samples,
                'total_correct': total_correct,
                'overall_accuracy': overall_accuracy,
                'avg_correct_confidence': avg_confidence
            },
            'ranking': [
                {
                    'rank': i,
                    'gloss': r['gloss'],
                    'accuracy': r['accuracy'],
                    'correct': r['correct'],
                    'total_samples': r['total_samples'],
                    'avg_correct_confidence': r['avg_correct_confidence'],
                    'weighted_score': r['weighted_score']
                }
                for i, r in enumerate(sorted_glosses, 1)
            ],
            'detailed_results': {
                gloss: {
                    'gloss': data['gloss'],
                    'accuracy': data['accuracy'],
                    'correct': data['correct'],
                    'total_samples': data['total_samples'],
                    'avg_correct_confidence': data['avg_correct_confidence'],
                    'weighted_score': data['weighted_score'],
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
