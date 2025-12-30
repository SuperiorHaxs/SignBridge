#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics.py - Modular evaluation metrics for ASL translation

This module provides standalone functions for computing evaluation metrics:
- Model Gloss Accuracy (Top-1)
- Effective Gloss Accuracy (LLM-selected from Top-K)
- BLEU Score
- BERTScore
- Sentence Quality Score (GPT-2 perplexity-based)
- Composite Translation Quality Index (CTQI)
- Coverage metrics (recall, precision, F1)

Each function can be called independently from external scripts.
"""

import os
import string
from typing import List, Dict, Optional, Tuple, Any

# Suppress HuggingFace Hub warnings
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# ============================================================================
# AVAILABILITY FLAGS - Set based on installed packages
# ============================================================================

# Try to import sacrebleu for BLEU scoring
try:
    import sacrebleu
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

# Try to import BERTScore
try:
    from bert_score import score as bert_score_func
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

# Try to import transformers for quality scoring
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    QUALITY_SCORING_AVAILABLE = True
except ImportError:
    QUALITY_SCORING_AVAILABLE = False


# ============================================================================
# CONFIGURABLE METRIC WEIGHTS FOR COMPOSITE SCORE
# ============================================================================
# Adjust these weights to change how the composite score is calculated
# All weights should sum to 1.0

METRIC_WEIGHTS = {
    'quality': 0.4,        # Reference-free grammaticality/fluency (0-100)
    'coverage_f1': 0.25,   # Content word alignment with reference (0-100)
    'bertscore': 0.2,      # Semantic similarity to reference (0-100)
    'bleu': 0.15           # Lexical n-gram overlap with reference (0-100)
}

# Verify weights sum to 1.0
assert abs(sum(METRIC_WEIGHTS.values()) - 1.0) < 0.001, \
    f"Metric weights must sum to 1.0, got {sum(METRIC_WEIGHTS.values())}"


# ============================================================================
# GLOSS ACCURACY
# ============================================================================

def calculate_gloss_accuracy(
    predicted_glosses: List[str],
    original_glosses: List[str]
) -> Dict[str, Any]:
    """
    Calculate gloss prediction accuracy by comparing predicted glosses to original input glosses.

    This function is used for both:
    - Model Gloss Accuracy (Top-1): When predicted_glosses are the top-1 model predictions
    - Effective Gloss Accuracy: When predicted_glosses are the LLM-selected glosses from top-k

    Args:
        predicted_glosses: List of predicted/selected glosses (strings)
        original_glosses: List of original input glosses (ground truth)

    Returns:
        dict with:
            - 'accuracy': Accuracy percentage (0-100)
            - 'correct': Number of correct predictions
            - 'total': Total number of comparisons
            - 'mismatches': List of mismatch details
    """
    if not predicted_glosses or not original_glosses:
        return {
            'accuracy': 0.0,
            'correct': 0,
            'total': 0,
            'mismatches': []
        }

    # Normalize for comparison
    correct = 0
    total = min(len(predicted_glosses), len(original_glosses))
    mismatches = []

    for i in range(total):
        pred = predicted_glosses[i].upper().strip()
        orig = original_glosses[i].upper().strip()

        if pred == orig:
            correct += 1
        else:
            mismatches.append({
                'position': i,
                'predicted': predicted_glosses[i],
                'original': original_glosses[i]
            })

    # If lengths differ, add missing/extra glosses as mismatches
    if len(predicted_glosses) != len(original_glosses):
        if len(predicted_glosses) > len(original_glosses):
            for i in range(len(original_glosses), len(predicted_glosses)):
                mismatches.append({
                    'position': i,
                    'predicted': predicted_glosses[i],
                    'original': '[NONE]'
                })
        else:
            for i in range(len(predicted_glosses), len(original_glosses)):
                mismatches.append({
                    'position': i,
                    'predicted': '[NONE]',
                    'original': original_glosses[i]
                })

    accuracy = (correct / total * 100) if total > 0 else 0.0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'mismatches': mismatches
    }


# ============================================================================
# BLEU SCORE
# ============================================================================

def calculate_bleu_score(
    hypothesis: str,
    reference: str,
    normalize: bool = True
) -> Optional[float]:
    """
    Calculate BLEU score between hypothesis and reference sentences.

    Args:
        hypothesis: Generated/predicted sentence
        reference: Reference (ground truth) sentence
        normalize: If True, normalize sentences (lowercase, remove punctuation)

    Returns:
        BLEU score (0-100) or None if BLEU calculation is not available
    """
    if not BLEU_AVAILABLE:
        return None

    try:
        if normalize:
            # Normalize both sentences (case-insensitive, punctuation-insensitive)
            translator = str.maketrans('', '', string.punctuation)
            hypothesis_normalized = hypothesis.translate(translator).strip().lower()
            reference_normalized = reference.translate(translator).strip().lower()
        else:
            hypothesis_normalized = hypothesis
            reference_normalized = reference

        # Calculate BLEU score using sacrebleu
        bleu = sacrebleu.sentence_bleu(
            hypothesis_normalized,
            [reference_normalized]
        )

        return bleu.score

    except Exception as e:
        print(f"ERROR: BLEU calculation failed: {e}")
        return None


# ============================================================================
# BERT SCORE
# ============================================================================

def calculate_bert_score(
    hypothesis: str,
    reference: str,
    lang: str = "en"
) -> Optional[float]:
    """
    Calculate BERTScore (F1) between hypothesis and reference sentences.

    BERTScore measures semantic similarity using BERT embeddings.

    Args:
        hypothesis: Generated/predicted sentence
        reference: Reference (ground truth) sentence
        lang: Language code (default: "en" for English)

    Returns:
        BERTScore F1 (0-100 scale) or None if BERTScore is not available
    """
    if not BERTSCORE_AVAILABLE:
        return None

    try:
        P, R, F1 = bert_score_func(
            [hypothesis],
            [reference],
            lang=lang,
            verbose=False
        )
        # Convert to 0-100 scale (BERTScore returns 0-1)
        return F1.item() * 100

    except Exception as e:
        print(f"ERROR: BERTScore calculation failed: {e}")
        return None


# ============================================================================
# QUALITY SCORE (GPT-2 Perplexity-based)
# ============================================================================

class QualityScorer:
    """
    Quality scorer using GPT-2 perplexity for reference-free sentence quality evaluation.

    This class manages the GPT-2 model for calculating perplexity-based quality scores.
    Use this class when you need to calculate quality scores for multiple sentences
    (avoids reloading the model each time).
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the quality scorer.

        Args:
            verbose: If True, print loading status messages
        """
        self.model = None
        self.tokenizer = None
        self.verbose = verbose
        self._initialized = False

        if QUALITY_SCORING_AVAILABLE:
            self._init_model()

    def _init_model(self):
        """Initialize GPT-2 model for quality scoring."""
        try:
            if self.verbose:
                print("Loading GPT-2 model for quality scoring...")

            try:
                # Try to load from cache first (faster, no download)
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'gpt2',
                    local_files_only=True
                )
                self.model = GPT2LMHeadModel.from_pretrained(
                    'gpt2',
                    local_files_only=True
                )
                if self.verbose:
                    print("SUCCESS: Loaded GPT-2 from cache")
            except Exception:
                # Not in cache, need to download
                if self.verbose:
                    print("Cache not found, downloading GPT-2 model...")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'gpt2',
                    force_download=False,
                    resume_download=True
                )
                self.model = GPT2LMHeadModel.from_pretrained(
                    'gpt2',
                    force_download=False,
                    resume_download=True
                )
                if self.verbose:
                    print("SUCCESS: GPT-2 downloaded and cached")

            self.model.eval()  # Set to evaluation mode
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self._initialized = True

            if self.verbose:
                print("SUCCESS: Quality scorer is ready")

        except Exception as e:
            print(f"ERROR: Failed to initialize quality scorer: {e}")
            self.model = None
            self.tokenizer = None
            self._initialized = False

    def calculate(self, sentence: str, debug: bool = False) -> Optional[float]:
        """
        Calculate reference-free quality score for a sentence.

        Args:
            sentence: The sentence to evaluate
            debug: If True, print debug information

        Returns:
            Quality score (0-100, higher is better) or None if unavailable
        """
        if not self._initialized or self.model is None:
            return None

        try:
            # Tokenize input
            inputs = self.tokenizer(
                sentence,
                return_tensors='pt',
                truncation=True,
                max_length=512
            )

            # Calculate perplexity
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()

            if debug:
                print(f"    [DEBUG] Sentence: '{sentence[:50]}...' | Perplexity: {perplexity:.2f}")

            # Convert perplexity to 0-100 score using logarithmic scale
            # Lower perplexity = better (more natural)
            score = self._perplexity_to_score(perplexity)

            if debug:
                print(f"    [DEBUG] Quality Score: {score:.2f}")

            return max(0, min(100, score))

        except Exception as e:
            print(f"ERROR: Quality score calculation failed: {e}")
            return None

    def _perplexity_to_score(self, perplexity: float) -> float:
        """
        Convert perplexity to a 0-100 quality score.

        Observed perplexity ranges:
        - perplexity 50-100: Excellent (90-100)
        - perplexity 100-500: Good (60-90)
        - perplexity 500-2000: Fair (30-60)
        - perplexity 2000-5000: Poor (10-30)
        - perplexity 5000+: Very poor (0-10)
        """
        if perplexity < 50:  # Excellent
            score = 100
        elif perplexity < 100:  # Very Good
            score = 90 + (100 - perplexity) / 50 * 10  # 100 down to 90
        elif perplexity < 500:  # Good
            score = 60 + (500 - perplexity) / 400 * 30  # 90 down to 60
        elif perplexity < 2000:  # Fair
            score = 30 + (2000 - perplexity) / 1500 * 30  # 60 down to 30
        elif perplexity < 5000:  # Poor
            score = 10 + (5000 - perplexity) / 3000 * 20  # 30 down to 10
        else:  # Very Poor
            score = max(0, 10 - (perplexity - 5000) / 1000)  # 10 down to 0

        return score

    @property
    def is_available(self) -> bool:
        """Check if quality scoring is available."""
        return self._initialized and self.model is not None


# Global quality scorer instance (lazy initialization)
_global_quality_scorer: Optional[QualityScorer] = None


def calculate_quality_score(
    sentence: str,
    scorer: Optional[QualityScorer] = None,
    debug: bool = False
) -> Optional[float]:
    """
    Calculate reference-free quality score based on fluency/grammaticality.

    Uses GPT-2 perplexity as a proxy for naturalness.
    Lower perplexity = more natural = higher score.

    Args:
        sentence: The sentence to evaluate
        scorer: Optional QualityScorer instance (reuse for multiple calls)
        debug: If True, print debug information

    Returns:
        Quality score (0-100, higher is better) or None if unavailable
    """
    global _global_quality_scorer

    if not QUALITY_SCORING_AVAILABLE:
        return None

    # Use provided scorer or get/create global instance
    if scorer is not None:
        return scorer.calculate(sentence, debug=debug)

    # Lazy initialization of global scorer
    if _global_quality_scorer is None:
        _global_quality_scorer = QualityScorer(verbose=True)

    return _global_quality_scorer.calculate(sentence, debug=debug)


# ============================================================================
# COVERAGE METRICS
# ============================================================================

def calculate_coverage(
    reference_sentence: str,
    output_sentence: str
) -> Dict[str, Any]:
    """
    Calculate semantic coverage metrics (content word alignment).

    Measures how well the output sentence covers the reference sentence content:
    - Recall: % of reference content words that appear in output (detects omissions)
    - Precision: % of output content words that appear in reference (detects hallucinations)
    - F1: Harmonic mean of precision and recall

    Args:
        reference_sentence: Reference sentence (ground truth)
        output_sentence: Generated/predicted sentence

    Returns:
        dict with:
            - 'recall': Recall percentage (0-100)
            - 'precision': Precision percentage (0-100)
            - 'f1': F1 score (0-100)
            - 'missing_words': Words in reference but not in output
            - 'hallucinated_words': Words in output but not in reference
    """
    try:
        # Function words to exclude (articles, prepositions, auxiliaries, etc.)
        function_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'of', 'to', 'in', 'on', 'at',
            'for', 'with', 'from', 'by', 'about', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'over', 'and',
            'or', 'but', 'if', 'then', 'so', 'than', 'that', 'this', 'these', 'those'
        }

        # Normalize and tokenize sentences
        translator = str.maketrans('', '', string.punctuation)

        reference_normalized = reference_sentence.translate(translator).strip().lower()
        reference_words = [w for w in reference_normalized.split() if w not in function_words]

        output_normalized = output_sentence.translate(translator).strip().lower()
        output_words = [w for w in output_normalized.split() if w not in function_words]

        # Calculate recall: how many reference content words appear in output?
        matched_reference_words = []
        missing_words = []

        for ref_word in reference_words:
            # Check if reference word or its stem appears in output
            # Handle verb forms (e.g., "walk" matches "walking", "walked")
            found = False
            for out_word in output_words:
                # Bidirectional substring matching + prefix matching
                if (ref_word in out_word or out_word in ref_word or
                    (out_word.startswith(ref_word[:4]) and len(ref_word) >= 4)):
                    found = True
                    break

            if found:
                matched_reference_words.append(ref_word)
            else:
                missing_words.append(ref_word)

        recall = len(matched_reference_words) / len(reference_words) if reference_words else 0

        # Calculate precision: how many output content words appear in reference?
        matched_output_words = []
        hallucinated_words = []

        for out_word in output_words:
            # Check if output word matches any reference word
            found = False
            for ref_word in reference_words:
                # Bidirectional substring matching + prefix matching
                if (ref_word in out_word or out_word in ref_word or
                    (out_word.startswith(ref_word[:4]) and len(ref_word) >= 4)):
                    found = True
                    break

            if found:
                matched_output_words.append(out_word)
            else:
                hallucinated_words.append(out_word)

        precision = len(matched_output_words) / len(output_words) if output_words else 0

        # Calculate F1
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0

        return {
            'recall': recall * 100,  # Convert to percentage
            'precision': precision * 100,
            'f1': f1 * 100,
            'missing_words': missing_words,
            'hallucinated_words': hallucinated_words
        }

    except Exception as e:
        print(f"ERROR: Coverage calculation failed: {e}")
        return {
            'recall': None,
            'precision': None,
            'f1': None,
            'missing_words': [],
            'hallucinated_words': []
        }


# ============================================================================
# COMPOSITE SCORE (CTQI - Composite Translation Quality Index)
# ============================================================================

def calculate_composite_score(
    bleu: Optional[float] = None,
    bertscore: Optional[float] = None,
    quality: Optional[float] = None,
    coverage_f1: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None
) -> Optional[float]:
    """
    Calculate Composite Translation Quality Index (CTQI).

    This is a weighted average of available metrics, normalized to handle
    missing metrics gracefully.

    Default weights (must sum to 1.0):
    - quality: 0.4 (40%) - Reference-free grammaticality/fluency
    - coverage_f1: 0.25 (25%) - Content word alignment
    - bertscore: 0.2 (20%) - Semantic similarity
    - bleu: 0.15 (15%) - Lexical n-gram overlap

    Args:
        bleu: BLEU score (0-100) or None
        bertscore: BERTScore F1 (0-100) or None
        quality: Quality score (0-100) or None
        coverage_f1: Coverage F1 score (0-100) or None
        weights: Optional custom weights dict (keys: 'bleu', 'bertscore', 'quality', 'coverage_f1')

    Returns:
        Composite score (0-100) or None if no metrics available
    """
    # Use provided weights or defaults
    metric_weights = weights if weights is not None else METRIC_WEIGHTS

    scores = {}
    weights_sum = 0

    # Add available scores
    if bleu is not None:
        scores['bleu'] = bleu
        weights_sum += metric_weights['bleu']

    if bertscore is not None:
        scores['bertscore'] = bertscore
        weights_sum += metric_weights['bertscore']

    if quality is not None:
        scores['quality'] = quality
        weights_sum += metric_weights['quality']

    if coverage_f1 is not None:
        scores['coverage_f1'] = coverage_f1
        weights_sum += metric_weights['coverage_f1']

    # Need at least one score
    if not scores or weights_sum == 0:
        return None

    # Calculate weighted average (renormalize weights if some metrics missing)
    composite = sum(
        scores[key] * metric_weights[key] / weights_sum
        for key in scores
    )

    return composite


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def calculate_all_metrics(
    hypothesis: str,
    reference: str,
    predicted_glosses: Optional[List[str]] = None,
    original_glosses: Optional[List[str]] = None,
    quality_scorer: Optional[QualityScorer] = None
) -> Dict[str, Any]:
    """
    Calculate all available metrics for a hypothesis-reference pair.

    This is a convenience function that calculates all metrics at once.

    Args:
        hypothesis: Generated/predicted sentence
        reference: Reference (ground truth) sentence
        predicted_glosses: Optional list of predicted glosses (for accuracy)
        original_glosses: Optional list of original glosses (for accuracy)
        quality_scorer: Optional QualityScorer instance (reuse for efficiency)

    Returns:
        dict with all computed metrics
    """
    results = {
        'bleu': calculate_bleu_score(hypothesis, reference),
        'bertscore': calculate_bert_score(hypothesis, reference),
        'quality': calculate_quality_score(hypothesis, scorer=quality_scorer),
    }

    # Calculate coverage
    coverage = calculate_coverage(reference, hypothesis)
    results['coverage_recall'] = coverage['recall']
    results['coverage_precision'] = coverage['precision']
    results['coverage_f1'] = coverage['f1']
    results['missing_words'] = coverage['missing_words']
    results['hallucinated_words'] = coverage['hallucinated_words']

    # Calculate composite score
    results['composite'] = calculate_composite_score(
        bleu=results['bleu'],
        bertscore=results['bertscore'],
        quality=results['quality'],
        coverage_f1=results['coverage_f1']
    )

    # Calculate gloss accuracy if glosses provided
    if predicted_glosses is not None and original_glosses is not None:
        accuracy_result = calculate_gloss_accuracy(predicted_glosses, original_glosses)
        results['gloss_accuracy'] = accuracy_result['accuracy']
        results['gloss_correct'] = accuracy_result['correct']
        results['gloss_total'] = accuracy_result['total']
        results['gloss_mismatches'] = accuracy_result['mismatches']

    return results


# ============================================================================
# MODULE TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EVALUATION METRICS LIBRARY TEST")
    print("=" * 70)

    # Test availability
    print("\nAvailability:")
    print(f"  BLEU: {'Available' if BLEU_AVAILABLE else 'Not available'}")
    print(f"  BERTScore: {'Available' if BERTSCORE_AVAILABLE else 'Not available'}")
    print(f"  Quality Scoring: {'Available' if QUALITY_SCORING_AVAILABLE else 'Not available'}")

    # Test data
    reference = "The man is walking now."
    hypothesis = "The man will walk now."
    predicted_glosses = ["MAN", "WALK", "NOW"]
    original_glosses = ["MAN", "WALK", "NOW"]

    print(f"\nTest Data:")
    print(f"  Reference: '{reference}'")
    print(f"  Hypothesis: '{hypothesis}'")
    print(f"  Predicted Glosses: {predicted_glosses}")
    print(f"  Original Glosses: {original_glosses}")

    # Test gloss accuracy
    print("\n--- Gloss Accuracy ---")
    accuracy = calculate_gloss_accuracy(predicted_glosses, original_glosses)
    print(f"  Accuracy: {accuracy['accuracy']:.1f}%")
    print(f"  Correct: {accuracy['correct']}/{accuracy['total']}")

    # Test with mismatch
    predicted_mismatch = ["MAN", "RUN", "NOW"]
    accuracy_mismatch = calculate_gloss_accuracy(predicted_mismatch, original_glosses)
    print(f"\n  With mismatch (RUN vs WALK):")
    print(f"  Accuracy: {accuracy_mismatch['accuracy']:.1f}%")
    print(f"  Mismatches: {accuracy_mismatch['mismatches']}")

    # Test BLEU
    print("\n--- BLEU Score ---")
    bleu = calculate_bleu_score(hypothesis, reference)
    print(f"  BLEU: {bleu:.2f}" if bleu is not None else "  BLEU: Not available")

    # Test BERTScore
    print("\n--- BERTScore ---")
    bert = calculate_bert_score(hypothesis, reference)
    print(f"  BERTScore F1: {bert:.2f}" if bert is not None else "  BERTScore: Not available")

    # Test Quality Score
    print("\n--- Quality Score ---")
    quality = calculate_quality_score(hypothesis)
    print(f"  Quality: {quality:.2f}" if quality is not None else "  Quality: Not available")

    # Test Coverage
    print("\n--- Coverage ---")
    coverage = calculate_coverage(reference, hypothesis)
    print(f"  Recall: {coverage['recall']:.1f}%")
    print(f"  Precision: {coverage['precision']:.1f}%")
    print(f"  F1: {coverage['f1']:.1f}%")

    # Test Composite Score
    print("\n--- Composite Score (CTQI) ---")
    composite = calculate_composite_score(
        bleu=bleu,
        bertscore=bert,
        quality=quality,
        coverage_f1=coverage['f1']
    )
    print(f"  Composite: {composite:.2f}" if composite is not None else "  Composite: Not available")

    # Test all metrics at once
    print("\n--- All Metrics (convenience function) ---")
    all_metrics = calculate_all_metrics(
        hypothesis, reference,
        predicted_glosses, original_glosses
    )
    for key, value in all_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        elif isinstance(value, list) and len(value) == 0:
            print(f"  {key}: []")
        elif value is not None:
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
