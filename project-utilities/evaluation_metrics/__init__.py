"""
Evaluation Metrics Library for ASL Translation

This package provides modular metric calculation functions for evaluating
ASL gloss prediction and sentence translation quality.

Available Metrics:
- calculate_gloss_accuracy: Model Gloss Accuracy (Top-1 or Effective)
- calculate_perfect_translation_rate: Perfect Translation Rate (PTR) - binary metric
- calculate_bleu_score: BLEU Score for translation quality
- calculate_bert_score: BERTScore for semantic similarity
- calculate_quality_score: Reference-free sentence quality (GPT-2 perplexity)
- calculate_composite_score: Composite Translation Quality Index (CTQI)
- calculate_coverage: Content word alignment metrics

CTQI (Composite Translation Quality Index) weights:
- Gloss Accuracy: 40% - Sign recognition accuracy (continuous)
- Quality Score: 40% - Reference-free grammaticality/fluency (continuous)
- Perfect Translation Rate: 20% - Bonus for complete correctness (binary)
"""

from .metrics import (
    # Core metric functions
    calculate_gloss_accuracy,
    calculate_perfect_translation_rate,
    calculate_bleu_score,
    calculate_bert_score,
    calculate_quality_score,
    calculate_composite_score,
    calculate_coverage,
    calculate_all_metrics,

    # Quality scorer class (for reuse across multiple calls)
    QualityScorer,

    # Constants
    METRIC_WEIGHTS,

    # Availability flags
    BLEU_AVAILABLE,
    BERTSCORE_AVAILABLE,
    QUALITY_SCORING_AVAILABLE,
)

__all__ = [
    'calculate_gloss_accuracy',
    'calculate_perfect_translation_rate',
    'calculate_bleu_score',
    'calculate_bert_score',
    'calculate_quality_score',
    'calculate_composite_score',
    'calculate_coverage',
    'calculate_all_metrics',
    'QualityScorer',
    'METRIC_WEIGHTS',
    'BLEU_AVAILABLE',
    'BERTSCORE_AVAILABLE',
    'QUALITY_SCORING_AVAILABLE',
]
