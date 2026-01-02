#!/usr/bin/env python3
"""
analyze_per_class_accuracy.py

Analyze a trained model to compute per-class validation accuracy.
Identifies high-accuracy classes to keep and low-accuracy classes to drop.

Usage:
    python analyze_per_class_accuracy.py --model-dir <path_to_model> --num-classes 100
    python analyze_per_class_accuracy.py --model-dir ./models/wlasl_100_class_model -n 100 --threshold 70

Output:
    - accuracy_report.json with per-class breakdown
    - Prints recommendations for pruning
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
import torch
from torch.utils.data import DataLoader

# Add project root to path
script_dir = Path(__file__).resolve().parent
# project-utilities/training_utils/incremental_training -> go up 3 levels to project root
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import configuration
from config import get_config

# Add openhands model path
config = get_config()
openhands_src = config.openhands_dir / "src"
sys.path.insert(0, str(openhands_src))

from openhands_modernized import (
    OpenHandsModel, OpenHandsConfig, WLASLOpenHandsDataset
)


def load_model_from_dir(model_dir: Path):
    """
    Load trained model from directory.

    Args:
        model_dir: Directory containing config.json, pytorch_model.bin, class_index_mapping.json

    Returns:
        Tuple of (model, id_to_gloss, config)
    """
    model_dir = Path(model_dir)

    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

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
    if not mapping_path.exists():
        raise FileNotFoundError(f"Class mapping not found: {mapping_path}")

    with open(mapping_path, 'r') as f:
        id_to_gloss = json.load(f)

    # Create model
    model = OpenHandsModel(model_config)

    # Load weights
    weights_path = model_dir / "pytorch_model.bin"
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model from {model_dir}")
    print(f"  Classes: {model_config.vocab_size}")
    print(f"  Hidden size: {model_config.hidden_size}")
    print(f"  Layers: {model_config.num_hidden_layers}")

    return model, id_to_gloss, model_config


def load_validation_dataset(num_classes: int, model_id_to_gloss: dict, max_seq_length: int = 256):
    """
    Load validation dataset for the specified number of classes.

    Args:
        num_classes: Number of classes the model was trained on
        model_id_to_gloss: Mapping from class ID to gloss name from the trained model
        max_seq_length: Maximum sequence length

    Returns:
        Tuple of (val_dataset, gloss_to_id, id_to_gloss)
    """
    config = get_config()

    # Create gloss_to_id from model's mapping (ensure we use model's class ordering)
    gloss_to_id = {gloss.upper(): int(id_) for id_, gloss in model_id_to_gloss.items()}
    id_to_gloss = {int(id_): gloss.upper() for id_, gloss in model_id_to_gloss.items()}

    if num_classes not in config.dataset_splits:
        raise ValueError(f"No dataset config for {num_classes} classes. "
                        f"Available: {list(config.dataset_splits.keys())}")

    splits = config.dataset_splits[num_classes]

    file_paths = []
    labels = []

    # Load validation data - handle both manifest and legacy formats
    if 'val_manifest' in splits:
        # Manifest-based format
        manifest_path = splits['val_manifest']
        pickle_pool = splits['pickle_pool']

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        pickle_pool = Path(pickle_pool)

        for gloss, families in manifest['classes'].items():
            gloss_dir = pickle_pool / gloss.lower()
            for family in families:
                for filename in family['files']:
                    file_path = gloss_dir / filename
                    if file_path.exists():
                        file_paths.append(str(file_path))
                        labels.append(gloss.upper())

        print(f"Loaded {len(file_paths)} validation samples from manifest")

    elif 'val' in splits:
        # Legacy directory-based format
        val_dir = Path(splits['val'])
        if not val_dir.exists():
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")

        # Each subdirectory is a class
        for gloss_dir in val_dir.iterdir():
            if gloss_dir.is_dir():
                gloss = gloss_dir.name.upper()
                for pickle_file in gloss_dir.glob("*.pkl"):
                    file_paths.append(str(pickle_file))
                    labels.append(gloss)

        print(f"Loaded {len(file_paths)} validation samples from directory: {val_dir}")

    else:
        raise ValueError("No validation data found (need 'val_manifest' or 'val' in config)")

    # Filter to only include samples for classes in the model
    filtered_paths = []
    filtered_labels = []
    for path, label in zip(file_paths, labels):
        if label in gloss_to_id:
            filtered_paths.append(path)
            filtered_labels.append(label)
        else:
            print(f"  Warning: Skipping sample with unknown gloss '{label}'")

    if len(filtered_paths) < len(file_paths):
        print(f"  Filtered {len(file_paths)} samples to {len(filtered_paths)} (removed classes not in model)")

    file_paths = filtered_paths
    labels = filtered_labels

    # Create dataset
    val_dataset = WLASLOpenHandsDataset(
        file_paths, labels, gloss_to_id,
        max_seq_length, augment=False, use_finger_features=True
    )

    return val_dataset, gloss_to_id, id_to_gloss


def compute_per_class_accuracy(model, val_loader, id_to_gloss, device):
    """
    Compute per-class accuracy on validation set.

    Args:
        model: Trained model
        val_loader: Validation DataLoader
        id_to_gloss: Mapping from class ID to gloss name
        device: Torch device

    Returns:
        Dict with per-class statistics
    """
    model.eval()

    # Track predictions per class
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_top3_correct = defaultdict(int)

    # Also track confusion data
    predictions_per_class = defaultdict(list)  # true_class -> [predicted_classes]

    with torch.no_grad():
        for batch in val_loader:
            pose_sequences = batch['pose_sequence'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            finger_features = batch.get('finger_features')
            if finger_features is not None:
                finger_features = finger_features.to(device)

            # Forward pass
            logits = model(pose_sequences, attention_masks, finger_features)

            # Get predictions
            _, predicted = torch.max(logits, 1)
            _, top3_predicted = torch.topk(logits, min(3, logits.size(1)), dim=1)

            # Track per-class accuracy
            for i in range(labels.size(0)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()

                class_total[true_label] += 1
                predictions_per_class[true_label].append(pred_label)

                if pred_label == true_label:
                    class_correct[true_label] += 1

                if true_label in top3_predicted[i].tolist():
                    class_top3_correct[true_label] += 1

    # Compute per-class accuracy
    per_class_stats = {}
    for class_id in class_total.keys():
        gloss = id_to_gloss.get(str(class_id), id_to_gloss.get(class_id, f"CLASS_{class_id}"))
        total = class_total[class_id]
        correct = class_correct[class_id]
        top3_correct = class_top3_correct[class_id]

        # Find most common confusion
        predictions = predictions_per_class[class_id]
        confusion_counts = defaultdict(int)
        for pred in predictions:
            if pred != class_id:
                confusion_counts[pred] += 1

        top_confusions = sorted(confusion_counts.items(), key=lambda x: -x[1])[:3]
        confusion_info = [
            {
                'gloss': id_to_gloss.get(str(pred_id), id_to_gloss.get(pred_id, f"CLASS_{pred_id}")),
                'count': count,
                'percent': round(count / total * 100, 1)
            }
            for pred_id, count in top_confusions
        ]

        per_class_stats[gloss] = {
            'class_id': class_id,
            'total_samples': total,
            'correct': correct,
            'accuracy': round(correct / total * 100, 2) if total > 0 else 0,
            'top3_correct': top3_correct,
            'top3_accuracy': round(top3_correct / total * 100, 2) if total > 0 else 0,
            'top_confusions': confusion_info
        }

    return per_class_stats


def generate_report(per_class_stats, threshold: float, top3_threshold: float, output_path: Path):
    """
    Generate accuracy report with recommendations.

    Args:
        per_class_stats: Per-class accuracy statistics
        threshold: Top-1 accuracy threshold for keeping classes (e.g., 80%)
        top3_threshold: Top-3 accuracy threshold for keeping classes (e.g., 100%)
        output_path: Path to save JSON report
    """
    # Sort by accuracy
    sorted_classes = sorted(
        per_class_stats.items(),
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )

    # Categorize using combined criteria: top-3 >= top3_threshold AND top-1 >= threshold
    high_accuracy = [(g, s) for g, s in sorted_classes
                     if s['top3_accuracy'] >= top3_threshold and s['accuracy'] >= threshold]
    low_accuracy = [(g, s) for g, s in sorted_classes
                    if s['top3_accuracy'] < top3_threshold or s['accuracy'] < threshold]

    # Compute overall stats
    total_samples = sum(s['total_samples'] for s in per_class_stats.values())
    total_correct = sum(s['correct'] for s in per_class_stats.values())
    overall_accuracy = round(total_correct / total_samples * 100, 2) if total_samples > 0 else 0

    total_top3_correct = sum(s['top3_correct'] for s in per_class_stats.values())
    overall_top3_accuracy = round(total_top3_correct / total_samples * 100, 2) if total_samples > 0 else 0

    report = {
        'timestamp': datetime.now().isoformat(),
        'threshold': threshold,
        'top3_threshold': top3_threshold,
        'criteria': f'top1 >= {threshold}% AND top3 >= {top3_threshold}%',
        'summary': {
            'total_classes': len(per_class_stats),
            'total_samples': total_samples,
            'overall_accuracy': overall_accuracy,
            'overall_top3_accuracy': overall_top3_accuracy,
            'classes_above_threshold': len(high_accuracy),
            'classes_below_threshold': len(low_accuracy),
        },
        'recommendations': {
            'keep_classes': [g for g, s in high_accuracy],
            'drop_classes': [g for g, s in low_accuracy],
        },
        'per_class_stats': {
            g: s for g, s in sorted_classes
        },
        'classes_by_accuracy': {
            'high_accuracy': [
                {'gloss': g, 'accuracy': s['accuracy'], 'top3_accuracy': s['top3_accuracy']}
                for g, s in high_accuracy
            ],
            'low_accuracy': [
                {'gloss': g, 'accuracy': s['accuracy'], 'top3_accuracy': s['top3_accuracy'],
                 'top_confusions': s['top_confusions']}
                for g, s in low_accuracy
            ],
        }
    }

    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    return report


def print_summary(report):
    """Print human-readable summary."""
    print("\n" + "="*70)
    print("PER-CLASS ACCURACY REPORT")
    print("="*70)

    summary = report['summary']
    print(f"\nOverall Statistics:")
    print(f"  Total classes: {summary['total_classes']}")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Overall accuracy: {summary['overall_accuracy']}%")
    print(f"  Overall top-3 accuracy: {summary['overall_top3_accuracy']}%")

    criteria = report.get('criteria', f"top1 >= {report['threshold']}%")
    print(f"\nSelection Criteria: {criteria}")
    print(f"  Classes meeting criteria (KEEP): {summary['classes_above_threshold']}")
    print(f"  Classes not meeting criteria (DROP): {summary['classes_below_threshold']}")

    print(f"\n{'='*70}")
    print("HIGH ACCURACY CLASSES (KEEP)")
    print("="*70)
    for item in report['classes_by_accuracy']['high_accuracy'][:20]:
        print(f"  {item['gloss']:<20} {item['accuracy']:>6.1f}% (top-3: {item['top3_accuracy']:.1f}%)")
    if len(report['classes_by_accuracy']['high_accuracy']) > 20:
        print(f"  ... and {len(report['classes_by_accuracy']['high_accuracy']) - 20} more")

    print(f"\n{'='*70}")
    print("LOW ACCURACY CLASSES (DROP)")
    print("="*70)
    for item in report['classes_by_accuracy']['low_accuracy']:
        confusions = ", ".join([f"{c['gloss']}({c['percent']}%)" for c in item['top_confusions'][:2]])
        print(f"  {item['gloss']:<20} top1:{item['accuracy']:>5.1f}% top3:{item['top3_accuracy']:>5.1f}% | confused with: {confusions}")

    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print("="*70)
    keep = report['recommendations']['keep_classes']
    drop = report['recommendations']['drop_classes']
    print(f"\nKeep {len(keep)} classes: {', '.join(keep[:10])}{'...' if len(keep) > 10 else ''}")
    print(f"\nDrop {len(drop)} classes: {', '.join(drop)}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-class accuracy of a trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--model-dir", "-m", type=Path, required=True,
                        help="Path to model directory (contains config.json, pytorch_model.bin)")
    parser.add_argument("--num-classes", "-n", type=int, required=True,
                        help="Number of classes the model was trained on")
    parser.add_argument("--threshold", "-t", type=float, default=80.0,
                        help="Top-1 accuracy threshold for keeping classes (default: 80%%)")
    parser.add_argument("--top3-threshold", type=float, default=100.0,
                        help="Top-3 accuracy threshold for keeping classes (default: 100%%)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output path for JSON report (default: accuracy_report.json)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for inference (default: 32)")
    parser.add_argument("--max-seq-length", type=int, default=256,
                        help="Maximum sequence length (default: 256)")

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        args.output = Path(__file__).parent / "accuracy_report.json"

    print(f"Analyzing model: {args.model_dir}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Selection criteria: top1 >= {args.threshold}% AND top3 >= {args.top3_threshold}%")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model, id_to_gloss_from_model, model_config = load_model_from_dir(args.model_dir)
    model = model.to(device)

    # Load validation dataset using model's class mapping
    val_dataset, gloss_to_id, id_to_gloss = load_validation_dataset(
        args.num_classes, id_to_gloss_from_model, args.max_seq_length
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0
    )

    print(f"\nValidation set: {len(val_dataset)} samples")

    # Compute per-class accuracy
    print("\nComputing per-class accuracy...")
    per_class_stats = compute_per_class_accuracy(model, val_loader, id_to_gloss, device)

    # Generate report
    report = generate_report(per_class_stats, args.threshold, args.top3_threshold, args.output)

    # Print summary
    print_summary(report)

    print(f"\nReport saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
