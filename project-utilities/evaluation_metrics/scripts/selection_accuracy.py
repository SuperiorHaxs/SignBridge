"""
Selection Accuracy Metrics for Two-Stage LLM Pipeline

This module provides metrics to evaluate the effectiveness of LLM-based
gloss selection compared to model top-1 predictions.

Key metrics:
- Model Top-1 Accuracy: How often the model's top prediction matches ground truth
- Top-K Recall: How often ground truth appears in top-k predictions
- LLM Selection Accuracy: How often LLM's selection matches ground truth
- LLM Lift: Improvement of LLM selection over model top-1
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict


def calculate_selection_accuracy(
    predictions: List[Dict[str, Any]],
    ground_truth_glosses: List[str],
    llm_selections: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate accuracy metrics for gloss selection.

    Args:
        predictions: Model predictions with top-k for each position.
            Each item: {'gloss': str, 'confidence': float, 'top_k': [...]}
        ground_truth_glosses: The actual glosses that were signed.
        llm_selections: Optional LLM-selected glosses (Stage 1 output).

    Returns:
        {
            'model_top1_accuracy': float,      # % where top-1 = ground truth
            'model_top3_recall': float,        # % where ground truth is in top-3
            'llm_selection_accuracy': float,   # % where LLM selection = ground truth
            'llm_lift': float,                 # LLM accuracy - top-1 accuracy
            'llm_lift_percentage': float,      # Relative improvement
            'per_position_details': [...]      # Detailed breakdown
        }
    """
    if len(predictions) != len(ground_truth_glosses):
        raise ValueError(
            f"Mismatch: {len(predictions)} predictions vs {len(ground_truth_glosses)} ground truth"
        )

    total = len(predictions)
    if total == 0:
        return {
            'model_top1_accuracy': 0.0,
            'model_top3_recall': 0.0,
            'llm_selection_accuracy': 0.0,
            'llm_lift': 0.0,
            'llm_lift_percentage': 0.0,
            'per_position_details': []
        }

    model_top1_correct = 0
    top3_has_gt = 0
    llm_correct = 0
    per_position_details = []

    for i, (pred, gt) in enumerate(zip(predictions, ground_truth_glosses)):
        # Normalize ground truth
        gt_normalized = gt.upper().strip()

        # Get top-k predictions
        top_k = pred.get('top_k', [])
        if not top_k:
            # Single prediction, create pseudo top-k
            top_k = [{'gloss': pred.get('gloss', 'UNKNOWN'), 'confidence': pred.get('confidence', 0)}]

        # Model top-1
        model_top1 = top_k[0].get('gloss', '').upper().strip() if top_k else ''
        model_top1_conf = top_k[0].get('confidence', 0) if top_k else 0
        is_top1_correct = (model_top1 == gt_normalized)
        if is_top1_correct:
            model_top1_correct += 1

        # Top-3 recall
        top3_glosses = [p.get('gloss', '').upper().strip() for p in top_k[:3]]
        gt_in_top3 = gt_normalized in top3_glosses
        if gt_in_top3:
            top3_has_gt += 1

        # LLM selection (if provided)
        llm_selection = None
        is_llm_correct = None
        if llm_selections and i < len(llm_selections):
            llm_selection = llm_selections[i].upper().strip()
            is_llm_correct = (llm_selection == gt_normalized)
            if is_llm_correct:
                llm_correct += 1

        # Record details
        per_position_details.append({
            'position': i + 1,
            'ground_truth': gt_normalized,
            'model_top1': model_top1,
            'model_top1_confidence': model_top1_conf,
            'model_top1_correct': is_top1_correct,
            'ground_truth_in_top3': gt_in_top3,
            'top3_options': top3_glosses,
            'llm_selection': llm_selection,
            'llm_correct': is_llm_correct,
            'llm_changed_from_top1': llm_selection != model_top1 if llm_selection else None
        })

    # Calculate aggregate metrics
    model_top1_accuracy = model_top1_correct / total
    top3_recall = top3_has_gt / total

    result = {
        'model_top1_accuracy': model_top1_accuracy,
        'model_top1_correct': model_top1_correct,
        'model_top3_recall': top3_recall,
        'top3_has_ground_truth': top3_has_gt,
        'total_positions': total,
        'per_position_details': per_position_details
    }

    # Add LLM metrics if selections provided
    if llm_selections:
        llm_accuracy = llm_correct / total
        llm_lift = llm_accuracy - model_top1_accuracy
        llm_lift_pct = (llm_lift / model_top1_accuracy * 100) if model_top1_accuracy > 0 else 0

        result.update({
            'llm_selection_accuracy': llm_accuracy,
            'llm_correct': llm_correct,
            'llm_lift': llm_lift,
            'llm_lift_percentage': llm_lift_pct
        })
    else:
        result.update({
            'llm_selection_accuracy': None,
            'llm_correct': None,
            'llm_lift': None,
            'llm_lift_percentage': None
        })

    return result


def calculate_selection_accuracy_by_confidence(
    predictions: List[Dict[str, Any]],
    ground_truth_glosses: List[str],
    llm_selections: List[str],
    confidence_buckets: List[tuple] = None
) -> Dict[str, Any]:
    """
    Calculate selection accuracy broken down by confidence level.

    Args:
        predictions: Model predictions with top-k
        ground_truth_glosses: Ground truth glosses
        llm_selections: LLM-selected glosses
        confidence_buckets: List of (min, max, label) tuples.
            Default: [(0, 0.5, 'low'), (0.5, 0.8, 'medium'), (0.8, 1.0, 'high')]

    Returns:
        Dict with accuracy metrics per confidence bucket
    """
    if confidence_buckets is None:
        confidence_buckets = [
            (0.0, 0.5, 'low'),
            (0.5, 0.8, 'medium'),
            (0.8, 1.0, 'high')
        ]

    # Initialize buckets
    bucket_stats = {}
    for min_conf, max_conf, label in confidence_buckets:
        bucket_stats[label] = {
            'range': f"{min_conf*100:.0f}%-{max_conf*100:.0f}%",
            'total': 0,
            'model_top1_correct': 0,
            'llm_correct': 0
        }

    # Categorize each position
    for i, (pred, gt, llm_sel) in enumerate(zip(predictions, ground_truth_glosses, llm_selections)):
        gt_norm = gt.upper().strip()
        llm_norm = llm_sel.upper().strip()

        top_k = pred.get('top_k', [])
        if not top_k:
            top_k = [{'gloss': pred.get('gloss'), 'confidence': pred.get('confidence', 0)}]

        model_top1 = top_k[0].get('gloss', '').upper().strip()
        conf = top_k[0].get('confidence', 0)

        # Find bucket
        for min_conf, max_conf, label in confidence_buckets:
            if min_conf <= conf < max_conf or (max_conf == 1.0 and conf == 1.0):
                bucket_stats[label]['total'] += 1
                if model_top1 == gt_norm:
                    bucket_stats[label]['model_top1_correct'] += 1
                if llm_norm == gt_norm:
                    bucket_stats[label]['llm_correct'] += 1
                break

    # Calculate per-bucket accuracy
    for label, stats in bucket_stats.items():
        total = stats['total']
        if total > 0:
            stats['model_top1_accuracy'] = stats['model_top1_correct'] / total
            stats['llm_accuracy'] = stats['llm_correct'] / total
            stats['llm_lift'] = stats['llm_accuracy'] - stats['model_top1_accuracy']
        else:
            stats['model_top1_accuracy'] = 0
            stats['llm_accuracy'] = 0
            stats['llm_lift'] = 0

    return bucket_stats


def generate_selection_report(
    predictions: List[Dict[str, Any]],
    ground_truth_glosses: List[str],
    llm_selections: List[str]
) -> str:
    """
    Generate a human-readable report of selection accuracy.

    Args:
        predictions: Model predictions with top-k
        ground_truth_glosses: Ground truth glosses
        llm_selections: LLM-selected glosses

    Returns:
        Formatted string report
    """
    # Calculate metrics
    metrics = calculate_selection_accuracy(predictions, ground_truth_glosses, llm_selections)
    by_conf = calculate_selection_accuracy_by_confidence(
        predictions, ground_truth_glosses, llm_selections
    )

    # Build report
    lines = [
        "=" * 60,
        "TWO-STAGE SELECTION ACCURACY REPORT",
        "=" * 60,
        "",
        f"Total Positions: {metrics['total_positions']}",
        "",
        "BASELINE (Model Top-1):",
        f"  Accuracy: {metrics['model_top1_accuracy']*100:.1f}% ({metrics['model_top1_correct']}/{metrics['total_positions']})",
        "",
        "TOP-3 RECALL (ceiling for LLM):",
        f"  Ground truth in top-3: {metrics['model_top3_recall']*100:.1f}% ({metrics['top3_has_ground_truth']}/{metrics['total_positions']})",
        "",
        "STAGE 1 (LLM Selection):",
        f"  Accuracy: {metrics['llm_selection_accuracy']*100:.1f}% ({metrics['llm_correct']}/{metrics['total_positions']})",
        "",
        "LLM LIFT:",
        f"  Absolute: {metrics['llm_lift']*100:+.1f}%",
        f"  Relative: {metrics['llm_lift_percentage']:+.1f}% improvement",
        "",
        "BREAKDOWN BY CONFIDENCE:",
    ]

    for label in ['high', 'medium', 'low']:
        if label in by_conf:
            stats = by_conf[label]
            if stats['total'] > 0:
                lines.append(
                    f"  {label.capitalize()} ({stats['range']}): "
                    f"Top-1: {stats['model_top1_accuracy']*100:.1f}%, "
                    f"LLM: {stats['llm_accuracy']*100:.1f}% "
                    f"({stats['llm_lift']*100:+.1f}%)"
                )

    # Add examples where LLM improved
    lines.append("")
    lines.append("EXAMPLES WHERE LLM IMPROVED:")

    improvements = [
        d for d in metrics['per_position_details']
        if d['llm_correct'] and not d['model_top1_correct']
    ][:5]

    if improvements:
        for d in improvements:
            lines.append(
                f"  Position {d['position']}: Model predicted '{d['model_top1']}' "
                f"({d['model_top1_confidence']*100:.0f}%), "
                f"LLM selected '{d['llm_selection']}' = correct (GT: '{d['ground_truth']}')"
            )
    else:
        lines.append("  (None found)")

    # Add examples where LLM made it worse
    lines.append("")
    lines.append("EXAMPLES WHERE LLM GOT IT WRONG:")

    regressions = [
        d for d in metrics['per_position_details']
        if not d['llm_correct'] and d['model_top1_correct']
    ][:5]

    if regressions:
        for d in regressions:
            lines.append(
                f"  Position {d['position']}: Model had correct '{d['model_top1']}', "
                f"but LLM selected '{d['llm_selection']}' (GT: '{d['ground_truth']}')"
            )
    else:
        lines.append("  (None found)")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


class SelectionAccuracyTracker:
    """
    Tracks selection accuracy across multiple samples/batches.

    Use this during inference to accumulate metrics over time.
    """

    def __init__(self):
        self.all_predictions: List[Dict] = []
        self.all_ground_truth: List[str] = []
        self.all_llm_selections: List[str] = []
        self.sample_count = 0

    def add_sample(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth_glosses: List[str],
        llm_selections: List[str]
    ):
        """Add a sample (one window of glosses) to the tracker."""
        self.all_predictions.extend(predictions)
        self.all_ground_truth.extend(ground_truth_glosses)
        self.all_llm_selections.extend(llm_selections)
        self.sample_count += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics across all samples."""
        if not self.all_predictions:
            return {'error': 'No samples recorded'}

        metrics = calculate_selection_accuracy(
            self.all_predictions,
            self.all_ground_truth,
            self.all_llm_selections
        )
        metrics['sample_count'] = self.sample_count
        return metrics

    def get_report(self) -> str:
        """Generate human-readable report."""
        if not self.all_predictions:
            return "No samples recorded."

        return generate_selection_report(
            self.all_predictions,
            self.all_ground_truth,
            self.all_llm_selections
        )

    def reset(self):
        """Clear all recorded data."""
        self.all_predictions = []
        self.all_ground_truth = []
        self.all_llm_selections = []
        self.sample_count = 0
