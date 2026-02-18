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
- calculate_composite_score: Composite Translation Quality Index (CTQI v1)
- calculate_composite_score_v2: CTQI v2 (geometric mean with Coverage F1)
- calculate_coverage: Content word alignment metrics

CTQI v1 (Composite Translation Quality Index) weights:
- Gloss Accuracy: 40% - Sign recognition accuracy (continuous)
- Quality Score: 40% - Reference-free grammaticality/fluency (continuous)
- Perfect Translation Rate: 20% - Bonus for complete correctness (binary)

CTQI v2 (Prerequisite Chain):
- CTQI = (GA/100) * (CF1/100) * (0.5 + 0.5 * P/100) * 100
- Gloss Accuracy (GA): Sign recognition accuracy
- Coverage F1 (CF1): Semantic content coverage (improved with lemmatization)
- Plausibility (P): LLM-assessed grammar + semantics + naturalness

CTQI v3 (Human-Validated Formula):
- CTQI = (GA/100) * (CF1/100) * (0.5 + 0.5 * P/100 * GA/100) * 100
- Key improvement: Plausibility's contribution scales with GA
- Prevents high plausibility from masking translation errors
- Human correlation: r=0.9427 (highest among all formulas tested)
"""

from .metrics import (
    # Core metric functions
    calculate_gloss_accuracy,
    calculate_perfect_translation_rate,
    calculate_bleu_score,
    calculate_bert_score,
    calculate_quality_score,
    calculate_composite_score,
    calculate_composite_score_v2,
    calculate_composite_score_v2_chain,
    calculate_composite_score_v3,
    calculate_coverage,
    calculate_coverage_v2,
    calculate_all_metrics,

    # Quality scorer class (for reuse across multiple calls)
    QualityScorer,

    # Constants
    METRIC_WEIGHTS,
    METRIC_WEIGHTS_V2,
    CTQI_V2_EPSILON,

    # Availability flags
    BLEU_AVAILABLE,
    BERTSCORE_AVAILABLE,
    QUALITY_SCORING_AVAILABLE,
    NLTK_AVAILABLE,
)

from .ctqi_v2 import (
    # CTQI v2 - Prerequisite Chain
    calculate_ctqi_v2,
    calculate_coverage_v2,
    calculate_plausibility,
    calculate_all_metrics_v2,
    PlausibilityScorer,
    NLTK_AVAILABLE,
    GEMINI_AVAILABLE,
)

__all__ = [
    # v1 metrics
    'calculate_gloss_accuracy',
    'calculate_perfect_translation_rate',
    'calculate_bleu_score',
    'calculate_bert_score',
    'calculate_quality_score',
    'calculate_composite_score',
    'calculate_composite_score_v2',
    'calculate_coverage',
    'calculate_all_metrics',
    'QualityScorer',
    'METRIC_WEIGHTS',
    'METRIC_WEIGHTS_V2',
    'CTQI_V2_EPSILON',
    'BLEU_AVAILABLE',
    'BERTSCORE_AVAILABLE',
    'QUALITY_SCORING_AVAILABLE',
    'NLTK_AVAILABLE',
    # v2 metrics from metrics.py
    'calculate_composite_score_v2_chain',
    'calculate_coverage_v2',
    # v3 metric (human-validated)
    'calculate_composite_score_v3',
    # v2 metrics from ctqi_v2.py (Prerequisite Chain standalone)
    'calculate_ctqi_v2',
    'calculate_plausibility',
    'calculate_all_metrics_v2',
    'PlausibilityScorer',
    'NLTK_AVAILABLE',
    'GEMINI_AVAILABLE',
]
