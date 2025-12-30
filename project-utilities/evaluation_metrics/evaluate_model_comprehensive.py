#!/usr/bin/env python3
"""
evaluate_model_comprehensive.py

Comprehensive evaluation metrics for isolated sign recognition model.
Generates all metrics needed for research paper.

Usage:
    python evaluate_model_comprehensive.py --dataset-path PATH --classes 100
    python evaluate_model_comprehensive.py --dataset-path PATH --classes 100 --split test --output results/
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add paths for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Go up: evaluation_metrics -> project-utilities -> project root
MODELS_DIR = PROJECT_ROOT / "models"

sys.path.insert(0, str(MODELS_DIR / "openhands-modernized" / "src"))
sys.path.insert(0, str(MODELS_DIR / "openhands-modernized" / "src" / "util"))


def calculate_metrics(y_true, y_pred, y_pred_proba, class_names):
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: Ground truth labels (list of class indices)
        y_pred: Predicted labels (list of class indices)
        y_pred_proba: Prediction probabilities (list of prob arrays)
        class_names: List of class names

    Returns:
        Dictionary with all metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, top_k_accuracy_score
    )

    n_classes = len(class_names)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)

    metrics = {}

    # ===== Basic Accuracy =====
    metrics['top1_accuracy'] = accuracy_score(y_true, y_pred)

    # Top-K Accuracy
    labels = list(range(n_classes))
    for k in [3, 5, 10]:
        if k <= n_classes:
            metrics[f'top{k}_accuracy'] = top_k_accuracy_score(y_true, y_pred_proba, k=k, labels=labels)

    # ===== Precision, Recall, F1 =====
    # Macro (unweighted mean across classes)
    metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Weighted (weighted by support)
    metrics['weighted_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['weighted_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Micro (global)
    metrics['micro_precision'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['micro_recall'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)

    # ===== Per-Class Metrics =====
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Count samples per class
    class_support = np.bincount(y_true, minlength=n_classes)

    metrics['per_class'] = {}
    for i, name in enumerate(class_names):
        metrics['per_class'][name] = {
            'precision': float(per_class_precision[i]) if i < len(per_class_precision) else 0.0,
            'recall': float(per_class_recall[i]) if i < len(per_class_recall) else 0.0,
            'f1': float(per_class_f1[i]) if i < len(per_class_f1) else 0.0,
            'support': int(class_support[i]) if i < len(class_support) else 0
        }

    # ===== Confusion Matrix =====
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    metrics['confusion_matrix'] = cm.tolist()

    # Most confused pairs
    confused_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    'true': class_names[i],
                    'predicted': class_names[j],
                    'count': int(cm[i, j])
                })
    confused_pairs.sort(key=lambda x: x['count'], reverse=True)
    metrics['top_confused_pairs'] = confused_pairs[:20]

    # ===== Confidence Statistics =====
    correct_confidences = []
    incorrect_confidences = []
    for i, (true_label, pred_proba) in enumerate(zip(y_true, y_pred_proba)):
        pred_label = np.argmax(pred_proba)
        confidence = pred_proba[pred_label]
        if pred_label == true_label:
            correct_confidences.append(confidence)
        else:
            incorrect_confidences.append(confidence)

    metrics['confidence_stats'] = {
        'correct_mean': float(np.mean(correct_confidences)) if correct_confidences else 0.0,
        'correct_std': float(np.std(correct_confidences)) if correct_confidences else 0.0,
        'incorrect_mean': float(np.mean(incorrect_confidences)) if incorrect_confidences else 0.0,
        'incorrect_std': float(np.std(incorrect_confidences)) if incorrect_confidences else 0.0,
    }

    # ===== Baseline Comparisons =====
    metrics['baselines'] = {
        'random_guess': 1.0 / n_classes,
        'improvement_over_random': metrics['top1_accuracy'] / (1.0 / n_classes)
    }

    return metrics


def run_evaluation(dataset_path, n_classes, checkpoint_path, split='test', limit=None):
    """
    Run comprehensive evaluation on dataset.
    """
    from openhands_modernized_inference import load_model_from_checkpoint, predict_pose_file

    dataset_path = Path(dataset_path)

    # Find split path
    classes_dir = dataset_path / f"{n_classes}_classes" / "original"
    possible_pickle_dirs = [
        classes_dir / f"pickle_from_pose_split_{n_classes}_class",
        classes_dir / f"pickle_split_{n_classes}_class",
    ]

    split_path = None
    for pickle_dir in possible_pickle_dirs:
        candidate = pickle_dir / split
        if candidate.exists():
            split_path = candidate
            break

    if split_path is None:
        raise FileNotFoundError(f"Split path not found for {split}")

    print(f"Evaluating on: {split_path}")

    # Load model
    print("Loading model...")
    model, tokenizer = load_model_from_checkpoint(str(checkpoint_path))

    # Get class names from tokenizer (id -> gloss mapping)
    class_names = [tokenizer.get(str(i), f"CLASS_{i}") for i in range(n_classes)]

    # Reverse mapping for ground truth
    gloss_to_id = {v.upper(): int(k) for k, v in tokenizer.items()}

    # Collect predictions
    y_true = []
    y_pred = []
    y_pred_proba = []

    gloss_folders = sorted([d for d in split_path.iterdir() if d.is_dir()])

    if limit:
        gloss_folders = gloss_folders[:limit]

    total_samples = 0
    for gloss_folder in gloss_folders:
        gloss_name = gloss_folder.name.upper()
        true_label = gloss_to_id.get(gloss_name)

        if true_label is None:
            print(f"  Warning: {gloss_name} not in vocabulary, skipping")
            continue

        pickle_files = list(gloss_folder.glob("*.pkl"))

        for pkl_file in pickle_files:
            try:
                result = predict_pose_file(str(pkl_file), model=model, tokenizer=tokenizer)

                # Get predicted class
                pred_gloss = result['gloss'].upper()
                pred_label = gloss_to_id.get(pred_gloss, -1)

                # Build probability array from top-k
                proba = np.zeros(n_classes)
                for pred in result['top_k_predictions']:
                    pred_id = gloss_to_id.get(pred['gloss'].upper())
                    if pred_id is not None:
                        proba[pred_id] = pred['confidence']

                y_true.append(true_label)
                y_pred.append(pred_label)
                y_pred_proba.append(proba)
                total_samples += 1

            except Exception as e:
                print(f"  Error processing {pkl_file.name}: {e}")

    print(f"Evaluated {total_samples} samples across {len(gloss_folders)} classes")

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba, class_names)
    metrics['evaluation_info'] = {
        'dataset_path': str(dataset_path),
        'split': split,
        'n_classes': n_classes,
        'n_samples': total_samples,
        'checkpoint': str(checkpoint_path),
        'timestamp': datetime.now().isoformat()
    }

    return metrics


def print_report(metrics):
    """Print formatted evaluation report."""

    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION REPORT")
    print("=" * 80)

    info = metrics['evaluation_info']
    print(f"\nDataset: {info['dataset_path']}")
    print(f"Split: {info['split']}")
    print(f"Classes: {info['n_classes']}")
    print(f"Samples: {info['n_samples']}")

    print("\n" + "-" * 80)
    print("ACCURACY METRICS")
    print("-" * 80)
    print(f"  Top-1 Accuracy:  {metrics['top1_accuracy']:.2%}")
    if 'top3_accuracy' in metrics:
        print(f"  Top-3 Accuracy:  {metrics['top3_accuracy']:.2%}")
    if 'top5_accuracy' in metrics:
        print(f"  Top-5 Accuracy:  {metrics['top5_accuracy']:.2%}")
    if 'top10_accuracy' in metrics:
        print(f"  Top-10 Accuracy: {metrics['top10_accuracy']:.2%}")

    print("\n" + "-" * 80)
    print("PRECISION / RECALL / F1-SCORE")
    print("-" * 80)
    print(f"  {'Averaging':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"  {'Macro':<12} {metrics['macro_precision']:<12.2%} {metrics['macro_recall']:<12.2%} {metrics['macro_f1']:<12.2%}")
    print(f"  {'Weighted':<12} {metrics['weighted_precision']:<12.2%} {metrics['weighted_recall']:<12.2%} {metrics['weighted_f1']:<12.2%}")
    print(f"  {'Micro':<12} {metrics['micro_precision']:<12.2%} {metrics['micro_recall']:<12.2%} {metrics['micro_f1']:<12.2%}")

    print("\n" + "-" * 80)
    print("CONFIDENCE STATISTICS")
    print("-" * 80)
    conf = metrics['confidence_stats']
    print(f"  Correct predictions:   {conf['correct_mean']:.2%} +/- {conf['correct_std']:.2%}")
    print(f"  Incorrect predictions: {conf['incorrect_mean']:.2%} +/- {conf['incorrect_std']:.2%}")

    print("\n" + "-" * 80)
    print("BASELINE COMPARISON")
    print("-" * 80)
    base = metrics['baselines']
    print(f"  Random guess baseline: {base['random_guess']:.2%}")
    print(f"  Improvement over random: {base['improvement_over_random']:.1f}x")

    print("\n" + "-" * 80)
    print("TOP 10 BEST CLASSES (by F1-Score)")
    print("-" * 80)
    sorted_classes = sorted(
        metrics['per_class'].items(),
        key=lambda x: x[1]['f1'],
        reverse=True
    )
    print(f"  {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    for name, m in sorted_classes[:10]:
        print(f"  {name:<15} {m['precision']:<12.2%} {m['recall']:<12.2%} {m['f1']:<12.2%} {m['support']:<10}")

    print("\n" + "-" * 80)
    print("TOP 10 WORST CLASSES (by F1-Score)")
    print("-" * 80)
    print(f"  {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    for name, m in sorted_classes[-10:]:
        print(f"  {name:<15} {m['precision']:<12.2%} {m['recall']:<12.2%} {m['f1']:<12.2%} {m['support']:<10}")

    print("\n" + "-" * 80)
    print("TOP 10 MOST CONFUSED PAIRS")
    print("-" * 80)
    print(f"  {'True Label':<15} {'Predicted As':<15} {'Count':<10}")
    for pair in metrics['top_confused_pairs'][:10]:
        print(f"  {pair['true']:<15} {pair['predicted']:<15} {pair['count']:<10}")

    print("\n" + "=" * 80)


def save_results(metrics, output_dir):
    """Save results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_classes = metrics['evaluation_info']['n_classes']

    # Save full JSON
    json_path = output_dir / f"evaluation_{n_classes}class_{timestamp}.json"
    with open(json_path, 'w') as f:
        # Convert numpy types to native Python types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(metrics, f, indent=2, default=convert)

    print(f"\nResults saved to: {json_path}")

    # Save summary CSV
    csv_path = output_dir / f"per_class_metrics_{n_classes}class_{timestamp}.csv"
    with open(csv_path, 'w') as f:
        f.write("class,precision,recall,f1,support\n")
        for name, m in sorted(metrics['per_class'].items()):
            f.write(f"{name},{m['precision']:.4f},{m['recall']:.4f},{m['f1']:.4f},{m['support']}\n")

    print(f"Per-class CSV saved to: {csv_path}")

    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive model evaluation for research paper"
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
        help="Number of classes"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to model checkpoint (default: auto-detect)"
    )
    parser.add_argument(
        "--split", "-s",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: test)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of classes (for testing)"
    )

    args = parser.parse_args()

    # Auto-detect checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = MODELS_DIR / "openhands-modernized" / "production-models" / f"wlasl_{args.classes}_class_model"

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Run evaluation
    metrics = run_evaluation(
        args.dataset_path,
        args.classes,
        checkpoint_path,
        split=args.split,
        limit=args.limit
    )

    # Print report
    print_report(metrics)

    # Save results
    if args.output:
        save_results(metrics, args.output)


if __name__ == "__main__":
    main()
