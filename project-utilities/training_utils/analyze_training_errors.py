#!/usr/bin/env python3
"""
Comprehensive Error Analysis for ASL Sign Recognition

This script performs multi-dimensional error analysis to identify:
1. Which glosses have the most errors
2. Root causes grouped by themes
3. Actionable recommendations for each theme

Themes analyzed:
- THEME 1: Per-Gloss Performance (which signs are hardest)
- THEME 2: Sign Confusion (which signs get confused with each other)
- THEME 3: Signer Variation (errors by video ID clusters)
- THEME 4: Temporal Issues (short vs long videos)
- THEME 5: Augmentation Quality (original vs augmented samples)
- THEME 6: Motion Complexity (static vs dynamic signs)

Usage:
    # From a predictions JSON file
    python analyze_training_errors.py --predictions-file results/predictions.json

    # Directly from a model checkpoint (can run while training continues)
    python analyze_training_errors.py --model-path models/wlasl_125_class_model --classes 125

    # With output saved
    python analyze_training_errors.py --model-path models/wlasl_125_class_model --classes 125 --output analysis.json
"""

import sys
import json
import argparse
import re
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Add pose_utils to path
pose_utils_dir = project_root / "project-utilities" / "pose_utils"
sys.path.insert(0, str(pose_utils_dir))

# Add training scripts to path for model imports
training_scripts_dir = project_root / "models" / "training-scripts"
sys.path.insert(0, str(training_scripts_dir))

# Add openhands model to path
openhands_dir = project_root / "models" / "openhands-modernized" / "src"
sys.path.insert(0, str(openhands_dir))


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PredictionRecord:
    """Single prediction record with metadata."""
    video_id: str
    true_label: str
    predicted_label: str
    correct: bool
    confidence: float = 0.0
    is_augmented: bool = False
    frame_count: int = 0
    file_path: str = ""


@dataclass
class GlossStats:
    """Statistics for a single gloss."""
    gloss: str
    total: int
    correct: int
    errors: int
    accuracy: float
    confused_with: Dict[str, int]  # predicted_label -> count
    error_video_ids: List[str]


@dataclass
class ThemeAnalysis:
    """Analysis results for a single theme."""
    theme_id: str
    theme_name: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    summary: str
    metrics: Dict[str, Any]
    affected_glosses: List[str]
    action_items: List[str]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_video_id_number(video_id: str) -> Optional[int]:
    """Extract numeric portion from video ID for clustering."""
    if '_aug' in video_id:
        video_id = video_id.split('_aug')[0]
    if '_balance' in video_id:
        video_id = video_id.split('_balance')[0]

    numbers = re.findall(r'\d+', video_id)
    if numbers:
        return max(int(n) for n in numbers)
    return None


def is_augmented_sample(video_id: str) -> bool:
    """Check if this is an augmented sample."""
    return '_aug' in video_id or '_balance' in video_id


def get_original_video_id(video_id: str) -> str:
    """Get the original video ID without augmentation suffix."""
    if '_aug' in video_id:
        return video_id.split('_aug')[0]
    if '_balance' in video_id:
        return video_id.split('_balance')[0]
    return video_id


def severity_level(value: float, thresholds: Tuple[float, float, float]) -> str:
    """Determine severity level based on thresholds."""
    if value < thresholds[0]:
        return 'low'
    elif value < thresholds[1]:
        return 'medium'
    elif value < thresholds[2]:
        return 'high'
    return 'critical'


# =============================================================================
# MODEL PREDICTION GENERATION
# =============================================================================

def generate_predictions_from_checkpoint(
    checkpoint_path: Path,
    num_classes: int,
    split: str = 'val',
    config_path: Optional[Path] = None,
) -> Tuple[List[PredictionRecord], Path]:
    """
    Generate predictions from a model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint.pth or pytorch_model.bin
        num_classes: Number of classes
        split: Which split to evaluate ('val' or 'test')
        config_path: Optional path to config.json (auto-detected if not provided)

    Returns:
        Tuple of (list of PredictionRecords, pickle_pool_path)
    """
    import torch
    from torch.utils.data import DataLoader

    # Import model classes
    from openhands_modernized import OpenHandsConfig, OpenHandsModel, WLASLOpenHandsDataset
    from config import get_config

    print(f"\n{'=' * 60}")
    print("GENERATING PREDICTIONS FROM CHECKPOINT")
    print('=' * 60)

    # Determine paths
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        # It's a model directory
        model_dir = checkpoint_path
        checkpoint_file = model_dir / "checkpoint.pth"
        if not checkpoint_file.exists():
            checkpoint_file = model_dir / "pytorch_model.bin"
    else:
        checkpoint_file = checkpoint_path
        model_dir = checkpoint_file.parent

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    print(f"Checkpoint: {checkpoint_file}")

    # Find config
    if config_path is None:
        config_path = model_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    print(f"Config: {config_path}")

    # Load config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    config = OpenHandsConfig(**config_dict)
    print(f"Model: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")

    # Create model
    model = OpenHandsModel(config)

    # Load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    checkpoint = torch.load(checkpoint_file, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        best_acc = checkpoint.get('best_val_acc', 0)
        print(f"Loaded from epoch {epoch}, best val acc: {best_acc:.2f}%")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights (no epoch info)")

    model.to(device)
    model.eval()

    # Load dataset paths
    app_config = get_config()

    # Get dataset paths from config
    dataset_splits = app_config.dataset_splits.get(num_classes)
    if not dataset_splits:
        raise ValueError(f"No dataset configuration for {num_classes} classes")

    # Load manifest
    if split == 'val':
        manifest_path = Path(str(dataset_splits.get('val_manifest', '')))
    else:
        manifest_path = Path(str(dataset_splits.get('test_manifest', '')))

    pickle_pool = Path(str(dataset_splits.get('pickle_pool', '')))
    class_mapping_path = Path(str(dataset_splits.get('class_mapping', '')))

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    print(f"\nLoading {split} data from: {manifest_path}")

    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Load class mapping
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)

    classes = class_mapping.get('classes', [])
    gloss_to_id = {g.upper(): i for i, g in enumerate(classes)}
    id_to_gloss = {i: g.upper() for i, g in enumerate(classes)}

    # Build file list from manifest
    file_paths = []
    labels = []

    for gloss, families in manifest['classes'].items():
        gloss_dir = pickle_pool / gloss.lower()
        for family in families:
            for filename in family['files']:
                file_path = gloss_dir / filename
                file_paths.append(str(file_path))
                labels.append(gloss.upper())

    print(f"Found {len(file_paths)} samples in {split} split")

    # Create dataset
    MAX_SEQ_LENGTH = 128  # Match training
    dataset = WLASLOpenHandsDataset(
        file_paths, labels, gloss_to_id,
        MAX_SEQ_LENGTH, augment=False
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = []
    all_file_paths = file_paths  # Keep reference

    sample_idx = 0
    with torch.no_grad():
        for batch in loader:
            pose_sequences = batch['pose_sequence'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            batch_labels = batch['label']

            # Forward pass
            outputs = model(pose_sequences, attention_masks)

            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            pred_indices = outputs.argmax(dim=1).cpu().numpy()
            confidences = probs.max(dim=1).values.cpu().numpy()

            # Record predictions
            for i in range(len(batch_labels)):
                true_idx = batch_labels[i].item()
                pred_idx = pred_indices[i]

                video_id = Path(all_file_paths[sample_idx]).stem
                true_label = id_to_gloss[true_idx]
                pred_label = id_to_gloss[pred_idx]

                predictions.append(PredictionRecord(
                    video_id=video_id,
                    true_label=true_label,
                    predicted_label=pred_label,
                    correct=(true_idx == pred_idx),
                    confidence=float(confidences[i]),
                    file_path=all_file_paths[sample_idx],
                ))

                sample_idx += 1

    # Summary
    correct = sum(1 for p in predictions if p.correct)
    accuracy = correct / len(predictions)
    print(f"\nGenerated {len(predictions)} predictions")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(predictions)})")

    return predictions, pickle_pool


# =============================================================================
# THEME 1: PER-GLOSS PERFORMANCE
# =============================================================================

def analyze_per_gloss_performance(predictions: List[PredictionRecord]) -> ThemeAnalysis:
    """
    Analyze which glosses have the most errors.

    Returns detailed breakdown of performance by gloss.
    """
    # Group by true label
    gloss_data = defaultdict(lambda: {
        'total': 0, 'correct': 0, 'confused_with': defaultdict(int), 'error_ids': []
    })

    for pred in predictions:
        g = gloss_data[pred.true_label]
        g['total'] += 1
        if pred.correct:
            g['correct'] += 1
        else:
            g['confused_with'][pred.predicted_label] += 1
            g['error_ids'].append(pred.video_id)

    # Calculate stats
    gloss_stats = []
    for gloss, data in gloss_data.items():
        accuracy = data['correct'] / data['total'] if data['total'] > 0 else 0
        gloss_stats.append(GlossStats(
            gloss=gloss,
            total=data['total'],
            correct=data['correct'],
            errors=data['total'] - data['correct'],
            accuracy=accuracy,
            confused_with=dict(data['confused_with']),
            error_video_ids=data['error_ids'][:10],  # Keep first 10
        ))

    # Sort by error count (worst first)
    gloss_stats.sort(key=lambda x: x.errors, reverse=True)

    # Identify worst performers
    worst_glosses = [g for g in gloss_stats if g.accuracy < 0.5 and g.total >= 5]
    overall_acc = sum(p.correct for p in predictions) / len(predictions)

    # Determine severity
    pct_below_50 = len(worst_glosses) / len(gloss_stats) if gloss_stats else 0
    severity = severity_level(pct_below_50, (0.1, 0.2, 0.3))

    # Build action items
    action_items = []
    if worst_glosses:
        action_items.append(f"Focus on {len(worst_glosses)} glosses with <50% accuracy")
        top_5_worst = [g.gloss for g in worst_glosses[:5]]
        action_items.append(f"Priority glosses: {', '.join(top_5_worst)}")

        # Check if worst glosses have few samples
        low_sample_worst = [g for g in worst_glosses if g.total < 50]
        if low_sample_worst:
            action_items.append(f"Consider more augmentation for {len(low_sample_worst)} low-sample glosses")

    return ThemeAnalysis(
        theme_id='per_gloss',
        theme_name='THEME 1: Per-Gloss Performance',
        severity=severity,
        summary=f"{len(worst_glosses)} glosses have <50% accuracy (overall: {overall_acc:.1%})",
        metrics={
            'overall_accuracy': overall_acc,
            'total_glosses': len(gloss_stats),
            'glosses_below_50pct': len(worst_glosses),
            'worst_10_glosses': [asdict(g) for g in gloss_stats[:10]],
            'best_10_glosses': [asdict(g) for g in gloss_stats[-10:]],
            'accuracy_distribution': {
                '0-25%': len([g for g in gloss_stats if g.accuracy < 0.25]),
                '25-50%': len([g for g in gloss_stats if 0.25 <= g.accuracy < 0.5]),
                '50-75%': len([g for g in gloss_stats if 0.5 <= g.accuracy < 0.75]),
                '75-100%': len([g for g in gloss_stats if g.accuracy >= 0.75]),
            }
        },
        affected_glosses=[g.gloss for g in worst_glosses],
        action_items=action_items,
    )


# =============================================================================
# THEME 2: SIGN CONFUSION ANALYSIS
# =============================================================================

def analyze_confusion_patterns(predictions: List[PredictionRecord]) -> ThemeAnalysis:
    """
    Analyze which signs get confused with each other.

    Identifies confusion pairs and potential sign similarity issues.
    """
    # Build confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    for pred in predictions:
        confusion[pred.true_label][pred.predicted_label] += 1

    # Find confusion pairs (both directions)
    confusion_pairs = []
    seen_pairs = set()

    for true_label in confusion:
        for pred_label, count in confusion[true_label].items():
            if true_label == pred_label:
                continue

            pair = tuple(sorted([true_label, pred_label]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            # Get reverse confusion
            reverse_count = confusion[pred_label].get(true_label, 0)
            total_confusion = count + reverse_count

            # Get total samples for both classes
            total_a = sum(confusion[true_label].values())
            total_b = sum(confusion[pred_label].values())

            if total_confusion >= 2:  # At least 2 confusions
                confusion_pairs.append({
                    'gloss_a': pair[0],
                    'gloss_b': pair[1],
                    'a_to_b': confusion[pair[0]].get(pair[1], 0),
                    'b_to_a': confusion[pair[1]].get(pair[0], 0),
                    'total_confusion': total_confusion,
                    'pct_of_a': count / total_a if total_a > 0 else 0,
                    'pct_of_b': reverse_count / total_b if total_b > 0 else 0,
                })

    # Sort by total confusion
    confusion_pairs.sort(key=lambda x: x['total_confusion'], reverse=True)

    # Identify systematic confusion (>20% of a class confused with another)
    systematic_pairs = [p for p in confusion_pairs
                        if p['pct_of_a'] > 0.2 or p['pct_of_b'] > 0.2]

    # Severity based on systematic confusion
    severity = severity_level(len(systematic_pairs), (3, 8, 15))

    # Action items
    action_items = []
    if systematic_pairs:
        action_items.append(f"Investigate {len(systematic_pairs)} sign pairs with >20% confusion rate")

        top_pairs = systematic_pairs[:3]
        for p in top_pairs:
            action_items.append(
                f"  - {p['gloss_a']} <-> {p['gloss_b']}: "
                f"Consider adding distinguishing features or more training data"
            )

        # Check for one-sided confusion (easier to fix)
        one_sided = [p for p in systematic_pairs
                     if p['a_to_b'] > 3 * max(1, p['b_to_a']) or
                        p['b_to_a'] > 3 * max(1, p['a_to_b'])]
        if one_sided:
            action_items.append(f"{len(one_sided)} pairs have one-sided confusion (easier to fix)")

    affected_glosses = list(set(
        [p['gloss_a'] for p in systematic_pairs] +
        [p['gloss_b'] for p in systematic_pairs]
    ))

    return ThemeAnalysis(
        theme_id='confusion',
        theme_name='THEME 2: Sign Confusion Patterns',
        severity=severity,
        summary=f"{len(confusion_pairs)} confusion pairs, {len(systematic_pairs)} systematic (>20%)",
        metrics={
            'total_confusion_pairs': len(confusion_pairs),
            'systematic_pairs': len(systematic_pairs),
            'top_20_pairs': confusion_pairs[:20],
            'one_sided_confusion': len([p for p in confusion_pairs
                                        if p['a_to_b'] > 3 * max(1, p['b_to_a']) or
                                           p['b_to_a'] > 3 * max(1, p['a_to_b'])]),
        },
        affected_glosses=affected_glosses,
        action_items=action_items,
    )


# =============================================================================
# THEME 3: SIGNER VARIATION (Video ID Clusters)
# =============================================================================

def analyze_signer_variation(predictions: List[PredictionRecord],
                              n_clusters: int = 10) -> ThemeAnalysis:
    """
    Analyze if errors cluster by video ID ranges (pseudo-signers).
    """
    # Extract numeric IDs and cluster
    video_ids = [p.video_id for p in predictions]
    id_to_num = {}
    for vid in video_ids:
        num = extract_video_id_number(vid)
        if num is not None:
            id_to_num[vid] = num

    if not id_to_num:
        return ThemeAnalysis(
            theme_id='signer_variation',
            theme_name='THEME 3: Signer Variation',
            severity='low',
            summary='Could not extract video ID numbers for clustering',
            metrics={},
            affected_glosses=[],
            action_items=['Ensure video IDs contain numeric identifiers'],
        )

    # Cluster by ID range
    min_id = min(id_to_num.values())
    max_id = max(id_to_num.values())
    range_size = (max_id - min_id + 1) / n_clusters

    clusters = {}
    for vid, num in id_to_num.items():
        cluster = int((num - min_id) / range_size)
        cluster = min(cluster, n_clusters - 1)
        clusters[vid] = cluster

    # Analyze per cluster
    cluster_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'id_min': float('inf'), 'id_max': 0})

    for pred in predictions:
        vid = pred.video_id
        if vid not in clusters:
            continue

        c = cluster_stats[clusters[vid]]
        c['total'] += 1
        if pred.correct:
            c['correct'] += 1

        num = id_to_num.get(vid)
        if num:
            c['id_min'] = min(c['id_min'], num)
            c['id_max'] = max(c['id_max'], num)

    # Calculate accuracies
    cluster_accuracies = []
    for cid, stats in cluster_stats.items():
        if stats['total'] >= 5:
            acc = stats['correct'] / stats['total']
            cluster_accuracies.append({
                'cluster': cid,
                'id_range': f"{stats['id_min']}-{stats['id_max']}",
                'total': stats['total'],
                'accuracy': acc,
            })

    cluster_accuracies.sort(key=lambda x: x['accuracy'])

    # Calculate variance
    if cluster_accuracies:
        accs = [c['accuracy'] for c in cluster_accuracies]
        acc_std = np.std(accs)
        acc_range = max(accs) - min(accs)
        overall_acc = sum(p.correct for p in predictions) / len(predictions)

        # Problem clusters (>10% below overall)
        problem_clusters = [c for c in cluster_accuracies if c['accuracy'] < overall_acc - 0.1]
    else:
        acc_std = 0
        acc_range = 0
        problem_clusters = []
        overall_acc = 0

    # Severity based on variance
    severity = severity_level(acc_range, (0.15, 0.25, 0.35))

    # Action items
    action_items = []
    if acc_range > 0.2:
        action_items.append("HIGH variance across video ID clusters detected!")
        action_items.append("This suggests different signers have different recognition rates")
        action_items.append("Consider: Signer-stratified splitting to ensure all 'signers' in all splits")
        action_items.append("Consider: More aggressive augmentation for signer variation (scale, speed)")

        if problem_clusters:
            worst = problem_clusters[0]
            action_items.append(
                f"Worst cluster: ID range {worst['id_range']} has {worst['accuracy']:.1%} accuracy"
            )

    return ThemeAnalysis(
        theme_id='signer_variation',
        theme_name='THEME 3: Signer Variation (Video ID Clusters)',
        severity=severity,
        summary=f"Accuracy range across clusters: {acc_range:.1%} (std: {acc_std:.1%})",
        metrics={
            'n_clusters': n_clusters,
            'accuracy_std': acc_std,
            'accuracy_range': acc_range,
            'problem_clusters': len(problem_clusters),
            'cluster_details': cluster_accuracies,
        },
        affected_glosses=[],  # Affects all glosses
        action_items=action_items,
    )


# =============================================================================
# THEME 4: TEMPORAL ISSUES (Video Length)
# =============================================================================

def analyze_temporal_issues(predictions: List[PredictionRecord],
                            pickle_pool_path: Optional[Path] = None) -> ThemeAnalysis:
    """
    Analyze if video length correlates with errors.
    """
    # Try to get frame counts from pickle files
    frame_counts = {}

    if pickle_pool_path and pickle_pool_path.exists():
        for pred in predictions:
            # Construct path to pickle file
            gloss_dir = pickle_pool_path / pred.true_label.lower()
            pkl_path = gloss_dir / f"{pred.video_id}.pkl"

            if pkl_path.exists():
                try:
                    with open(pkl_path, 'rb') as f:
                        data = pickle.load(f)
                    frame_counts[pred.video_id] = data['keypoints'].shape[0]
                except:
                    pass

    # If we have frame counts, analyze by length buckets
    if frame_counts:
        # Define buckets
        buckets = {
            'very_short': (0, 30),
            'short': (30, 50),
            'medium': (50, 80),
            'long': (80, 120),
            'very_long': (120, float('inf')),
        }

        bucket_stats = {name: {'total': 0, 'correct': 0} for name in buckets}

        for pred in predictions:
            frames = frame_counts.get(pred.video_id, 0)
            if frames == 0:
                continue

            for bucket_name, (min_f, max_f) in buckets.items():
                if min_f <= frames < max_f:
                    bucket_stats[bucket_name]['total'] += 1
                    if pred.correct:
                        bucket_stats[bucket_name]['correct'] += 1
                    break

        # Calculate accuracies
        bucket_accs = {}
        for name, stats in bucket_stats.items():
            if stats['total'] >= 10:
                bucket_accs[name] = {
                    'total': stats['total'],
                    'accuracy': stats['correct'] / stats['total'],
                }

        if bucket_accs:
            accs = [b['accuracy'] for b in bucket_accs.values()]
            acc_range = max(accs) - min(accs) if accs else 0

            # Find worst bucket
            worst_bucket = min(bucket_accs.items(), key=lambda x: x[1]['accuracy'])
            best_bucket = max(bucket_accs.items(), key=lambda x: x[1]['accuracy'])
        else:
            acc_range = 0
            worst_bucket = ('unknown', {'accuracy': 0})
            best_bucket = ('unknown', {'accuracy': 0})

        severity = severity_level(acc_range, (0.1, 0.2, 0.3))

        action_items = []
        if acc_range > 0.15:
            action_items.append(f"Video length affects accuracy by {acc_range:.1%}")
            action_items.append(f"Worst: {worst_bucket[0]} videos ({worst_bucket[1]['accuracy']:.1%})")
            action_items.append(f"Best: {best_bucket[0]} videos ({best_bucket[1]['accuracy']:.1%})")

            if worst_bucket[0] in ['very_short', 'short']:
                action_items.append("Short videos may lack temporal context - consider:")
                action_items.append("  - Frame padding/interpolation during preprocessing")
                action_items.append("  - Speed augmentation to create longer sequences")
            elif worst_bucket[0] in ['long', 'very_long']:
                action_items.append("Long videos may have alignment issues - consider:")
                action_items.append("  - Better temporal segmentation")
                action_items.append("  - Attention mechanisms to focus on key frames")

        metrics = {
            'videos_with_frame_info': len(frame_counts),
            'bucket_accuracies': bucket_accs,
            'accuracy_range': acc_range,
        }
    else:
        severity = 'low'
        action_items = ['Frame count data not available - provide pickle_pool path for analysis']
        metrics = {'videos_with_frame_info': 0}

    return ThemeAnalysis(
        theme_id='temporal',
        theme_name='THEME 4: Temporal Issues (Video Length)',
        severity=severity,
        summary=f"Analyzed {len(frame_counts)} videos with frame data" if frame_counts else "No frame data",
        metrics=metrics,
        affected_glosses=[],
        action_items=action_items,
    )


# =============================================================================
# THEME 5: AUGMENTATION QUALITY
# =============================================================================

def analyze_augmentation_quality(predictions: List[PredictionRecord]) -> ThemeAnalysis:
    """
    Compare error rates between original and augmented samples.
    """
    original_stats = {'total': 0, 'correct': 0}
    augmented_stats = {'total': 0, 'correct': 0}

    # Per-gloss breakdown
    gloss_original = defaultdict(lambda: {'total': 0, 'correct': 0})
    gloss_augmented = defaultdict(lambda: {'total': 0, 'correct': 0})

    for pred in predictions:
        is_aug = is_augmented_sample(pred.video_id)

        if is_aug:
            augmented_stats['total'] += 1
            if pred.correct:
                augmented_stats['correct'] += 1
            gloss_augmented[pred.true_label]['total'] += 1
            if pred.correct:
                gloss_augmented[pred.true_label]['correct'] += 1
        else:
            original_stats['total'] += 1
            if pred.correct:
                original_stats['correct'] += 1
            gloss_original[pred.true_label]['total'] += 1
            if pred.correct:
                gloss_original[pred.true_label]['correct'] += 1

    # Calculate accuracies
    orig_acc = original_stats['correct'] / original_stats['total'] if original_stats['total'] > 0 else 0
    aug_acc = augmented_stats['correct'] / augmented_stats['total'] if augmented_stats['total'] > 0 else 0

    acc_diff = orig_acc - aug_acc  # Positive means originals are better

    # Find glosses where augmentation hurts most
    aug_hurts = []
    for gloss in set(gloss_original.keys()) | set(gloss_augmented.keys()):
        orig_g = gloss_original[gloss]
        aug_g = gloss_augmented[gloss]

        if orig_g['total'] >= 3 and aug_g['total'] >= 3:
            orig_g_acc = orig_g['correct'] / orig_g['total']
            aug_g_acc = aug_g['correct'] / aug_g['total']

            if orig_g_acc - aug_g_acc > 0.2:  # 20% worse on augmented
                aug_hurts.append({
                    'gloss': gloss,
                    'original_acc': orig_g_acc,
                    'augmented_acc': aug_g_acc,
                    'difference': orig_g_acc - aug_g_acc,
                })

    aug_hurts.sort(key=lambda x: x['difference'], reverse=True)

    # Severity
    severity = severity_level(abs(acc_diff), (0.05, 0.1, 0.15))

    action_items = []
    if acc_diff > 0.05:
        action_items.append(f"Augmented samples are {acc_diff:.1%} less accurate than originals")
        action_items.append("This may indicate augmentation is too aggressive or introducing artifacts")
        action_items.append("Consider: Reducing augmentation intensity (scale, noise levels)")
        action_items.append("Consider: Reviewing augmentation parameters for realism")
    elif acc_diff < -0.05:
        action_items.append(f"Augmented samples are {-acc_diff:.1%} MORE accurate than originals")
        action_items.append("This is unusual - augmentations may be 'easier' variants")
        action_items.append("Consider: Adding more challenging augmentations")
    else:
        action_items.append("Augmentation quality is good (similar accuracy to originals)")

    if aug_hurts:
        action_items.append(f"\n{len(aug_hurts)} glosses where augmentation hurts significantly:")
        for g in aug_hurts[:5]:
            action_items.append(f"  - {g['gloss']}: {g['original_acc']:.1%} -> {g['augmented_acc']:.1%}")

    return ThemeAnalysis(
        theme_id='augmentation',
        theme_name='THEME 5: Augmentation Quality',
        severity=severity,
        summary=f"Original: {orig_acc:.1%}, Augmented: {aug_acc:.1%} (diff: {acc_diff:+.1%})",
        metrics={
            'original_total': original_stats['total'],
            'original_accuracy': orig_acc,
            'augmented_total': augmented_stats['total'],
            'augmented_accuracy': aug_acc,
            'accuracy_difference': acc_diff,
            'glosses_hurt_by_augmentation': len(aug_hurts),
            'worst_augmented_glosses': aug_hurts[:10],
        },
        affected_glosses=[g['gloss'] for g in aug_hurts],
        action_items=action_items,
    )


# =============================================================================
# THEME 6: MOTION COMPLEXITY
# =============================================================================

def analyze_motion_complexity(predictions: List[PredictionRecord],
                               pickle_pool_path: Optional[Path] = None) -> ThemeAnalysis:
    """
    Analyze errors by sign motion complexity (static vs dynamic).

    Uses velocity variance as a proxy for motion complexity.
    """
    if not pickle_pool_path or not pickle_pool_path.exists():
        return ThemeAnalysis(
            theme_id='motion',
            theme_name='THEME 6: Motion Complexity',
            severity='low',
            summary='Pickle pool path required for motion analysis',
            metrics={},
            affected_glosses=[],
            action_items=['Provide --pickle-pool path to enable motion analysis'],
        )

    # Calculate motion complexity per video
    motion_stats = {}

    for pred in predictions[:500]:  # Sample for efficiency
        gloss_dir = pickle_pool_path / pred.true_label.lower()
        pkl_path = gloss_dir / f"{pred.video_id}.pkl"

        if not pkl_path.exists():
            continue

        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            keypoints = data['keypoints']

            if len(keypoints) < 2:
                continue

            # Calculate frame-to-frame velocity
            velocities = np.diff(keypoints, axis=0)
            velocity_magnitude = np.linalg.norm(velocities, axis=(1, 2))
            motion_complexity = np.std(velocity_magnitude)

            motion_stats[pred.video_id] = {
                'complexity': motion_complexity,
                'correct': pred.correct,
                'gloss': pred.true_label,
            }
        except:
            continue

    if len(motion_stats) < 50:
        return ThemeAnalysis(
            theme_id='motion',
            theme_name='THEME 6: Motion Complexity',
            severity='low',
            summary=f'Insufficient data ({len(motion_stats)} samples)',
            metrics={},
            affected_glosses=[],
            action_items=['Need more valid pickle files for motion analysis'],
        )

    # Bin by complexity
    complexities = [s['complexity'] for s in motion_stats.values()]
    p33, p66 = np.percentile(complexities, [33, 66])

    bins = {
        'static': {'total': 0, 'correct': 0},
        'moderate': {'total': 0, 'correct': 0},
        'dynamic': {'total': 0, 'correct': 0},
    }

    for vid, stats in motion_stats.items():
        if stats['complexity'] < p33:
            bin_name = 'static'
        elif stats['complexity'] < p66:
            bin_name = 'moderate'
        else:
            bin_name = 'dynamic'

        bins[bin_name]['total'] += 1
        if stats['correct']:
            bins[bin_name]['correct'] += 1

    # Calculate accuracies
    bin_accs = {}
    for name, data in bins.items():
        if data['total'] > 0:
            bin_accs[name] = {
                'total': data['total'],
                'accuracy': data['correct'] / data['total'],
            }

    accs = [b['accuracy'] for b in bin_accs.values() if b['total'] >= 10]
    acc_range = max(accs) - min(accs) if len(accs) >= 2 else 0

    severity = severity_level(acc_range, (0.1, 0.15, 0.25))

    action_items = []
    if acc_range > 0.1:
        worst_bin = min(bin_accs.items(), key=lambda x: x[1]['accuracy'] if x[1]['total'] >= 10 else 1)
        action_items.append(f"Motion complexity affects accuracy by {acc_range:.1%}")
        action_items.append(f"'{worst_bin[0]}' signs are hardest ({worst_bin[1]['accuracy']:.1%})")

        if worst_bin[0] == 'static':
            action_items.append("Static signs may rely more on hand shape - ensure hand keypoints are accurate")
        elif worst_bin[0] == 'dynamic':
            action_items.append("Dynamic signs need better temporal modeling - consider more LSTM layers or attention")

    return ThemeAnalysis(
        theme_id='motion',
        theme_name='THEME 6: Motion Complexity',
        severity=severity,
        summary=f"Analyzed {len(motion_stats)} videos for motion patterns",
        metrics={
            'videos_analyzed': len(motion_stats),
            'complexity_thresholds': {'static': f'<{p33:.3f}', 'dynamic': f'>{p66:.3f}'},
            'bin_accuracies': bin_accs,
            'accuracy_range': acc_range,
        },
        affected_glosses=[],
        action_items=action_items,
    )


# =============================================================================
# MAIN ANALYSIS & REPORTING
# =============================================================================

def run_comprehensive_analysis(
    predictions: List[PredictionRecord],
    pickle_pool_path: Optional[Path] = None,
) -> Dict[str, ThemeAnalysis]:
    """Run all theme analyses."""

    print("\n" + "=" * 80)
    print("COMPREHENSIVE ERROR ANALYSIS")
    print("=" * 80)
    print(f"\nTotal predictions: {len(predictions)}")
    print(f"Overall accuracy: {sum(p.correct for p in predictions) / len(predictions):.1%}")

    analyses = {}

    # Theme 1: Per-Gloss
    print("\n[1/6] Analyzing per-gloss performance...")
    analyses['per_gloss'] = analyze_per_gloss_performance(predictions)

    # Theme 2: Confusion
    print("[2/6] Analyzing confusion patterns...")
    analyses['confusion'] = analyze_confusion_patterns(predictions)

    # Theme 3: Signer Variation
    print("[3/6] Analyzing signer variation...")
    analyses['signer_variation'] = analyze_signer_variation(predictions)

    # Theme 4: Temporal
    print("[4/6] Analyzing temporal issues...")
    analyses['temporal'] = analyze_temporal_issues(predictions, pickle_pool_path)

    # Theme 5: Augmentation
    print("[5/6] Analyzing augmentation quality...")
    analyses['augmentation'] = analyze_augmentation_quality(predictions)

    # Theme 6: Motion
    print("[6/6] Analyzing motion complexity...")
    analyses['motion'] = analyze_motion_complexity(predictions, pickle_pool_path)

    return analyses


def print_analysis_report(analyses: Dict[str, ThemeAnalysis]) -> None:
    """Print formatted analysis report."""

    # Severity colors/symbols
    severity_symbols = {
        'low': '[OK]',
        'medium': '[~]',
        'high': '[!]',
        'critical': '[!!]',
    }

    print("\n")
    print("=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    # Summary table
    print(f"\n{'Theme':<45} {'Severity':<10} {'Summary'}")
    print("-" * 80)

    for theme_id, analysis in analyses.items():
        sym = severity_symbols.get(analysis.severity, '[?]')
        print(f"{analysis.theme_name:<45} {sym:<10} {analysis.summary[:40]}")

    # Detailed reports by severity
    for severity in ['critical', 'high', 'medium', 'low']:
        themes_at_severity = [a for a in analyses.values() if a.severity == severity]

        if not themes_at_severity:
            continue

        print(f"\n\n{'=' * 80}")
        print(f"DETAILED ANALYSIS - {severity.upper()} SEVERITY")
        print("=" * 80)

        for analysis in themes_at_severity:
            print(f"\n{'-' * 80}")
            print(f"{analysis.theme_name}")
            print(f"{'-' * 80}")
            print(f"Summary: {analysis.summary}")

            if analysis.affected_glosses:
                print(f"\nAffected glosses ({len(analysis.affected_glosses)}):")
                print(f"  {', '.join(analysis.affected_glosses[:15])}")
                if len(analysis.affected_glosses) > 15:
                    print(f"  ... and {len(analysis.affected_glosses) - 15} more")

            if analysis.action_items:
                print(f"\nACTION ITEMS:")
                for item in analysis.action_items:
                    print(f"  {item}")

    # Priority action items
    print("\n\n" + "=" * 80)
    print("PRIORITY ACTION ITEMS (by severity)")
    print("=" * 80)

    priority_num = 1
    for severity in ['critical', 'high', 'medium']:
        for analysis in analyses.values():
            if analysis.severity != severity:
                continue
            if not analysis.action_items:
                continue

            print(f"\n{priority_num}. [{severity.upper()}] {analysis.theme_name}")
            for item in analysis.action_items[:3]:  # Top 3 per theme
                print(f"   - {item}")
            priority_num += 1


def load_predictions_from_file(path: Path) -> List[PredictionRecord]:
    """Load predictions from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        preds = data
    else:
        preds = data.get('predictions', [])

    return [
        PredictionRecord(
            video_id=p.get('video_id', ''),
            true_label=p.get('true_label', p.get('true', '')),
            predicted_label=p.get('predicted_label', p.get('predicted', '')),
            correct=p.get('correct', False),
            confidence=p.get('confidence', 0.0),
        )
        for p in preds
    ]


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive ASL Recognition Error Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--predictions-file', type=Path,
        help='Path to JSON file with predictions'
    )
    parser.add_argument(
        '--model-path', type=Path,
        help='Path to trained model (will generate predictions)'
    )
    parser.add_argument(
        '--classes', type=int, default=125,
        help='Number of classes'
    )
    parser.add_argument(
        '--split', choices=['val', 'test'], default='val',
        help='Which split to evaluate'
    )
    parser.add_argument(
        '--pickle-pool', type=Path,
        help='Path to pickle pool for detailed analysis'
    )
    parser.add_argument(
        '--output', type=Path,
        help='Save full analysis to JSON file'
    )

    args = parser.parse_args()

    # Load predictions
    pickle_pool_path = args.pickle_pool

    if args.predictions_file:
        print(f"Loading predictions from {args.predictions_file}...")
        predictions = load_predictions_from_file(args.predictions_file)
        print(f"Loaded {len(predictions)} predictions")

    elif args.model_path:
        print(f"Generating predictions from checkpoint: {args.model_path}")
        predictions, auto_pickle_pool = generate_predictions_from_checkpoint(
            checkpoint_path=args.model_path,
            num_classes=args.classes,
            split=args.split,
        )
        # Use auto-detected pickle pool if not provided
        if pickle_pool_path is None:
            pickle_pool_path = auto_pickle_pool

    else:
        print("Error: Must provide --predictions-file or --model-path")
        sys.exit(1)

    # Run analysis
    analyses = run_comprehensive_analysis(predictions, pickle_pool_path)

    # Print report
    print_analysis_report(analyses)

    # Save results
    if args.output:
        output_data = {
            'summary': {
                'total_predictions': len(predictions),
                'overall_accuracy': sum(p.correct for p in predictions) / len(predictions),
            },
            'themes': {
                theme_id: {
                    'theme_name': a.theme_name,
                    'severity': a.severity,
                    'summary': a.summary,
                    'metrics': a.metrics,
                    'affected_glosses': a.affected_glosses,
                    'action_items': a.action_items,
                }
                for theme_id, a in analyses.items()
            }
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nFull analysis saved to {args.output}")


if __name__ == "__main__":
    main()
