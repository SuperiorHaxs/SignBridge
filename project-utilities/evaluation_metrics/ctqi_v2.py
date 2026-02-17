#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ctqi_v2.py - CTQI v2: Prerequisite Chain Metric for ASL Translation

Redesigned Composite Translation Quality Index using a prerequisite chain
instead of weighted averages. The formula:

    CTQI = (GA/100) * (CF1/100) * (0.5 + 0.5 * P/100) * 100

Components:
- Gloss Accuracy (GA): Did we recognize the signs? (imported from metrics.py)
- Coverage F1 (CF1): Did we preserve the meaning? (improved with lemmatization)
- Plausibility (P): Does the output make sense? (LLM-based, replaces GPT-2 perplexity)

Design rationale:
- GA and CF1 multiply: both must be high or the score collapses.
- Plausibility is a modifier (0.5x to 1.0x): bad output halves the score,
  but correct-and-meaningful content is never zeroed out by poor grammar.
- No arbitrary weights, no thresholds, no epsilon floors.
"""

import json
import re
import string
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

# ============================================================================
# AVAILABILITY FLAGS
# ============================================================================

# NLTK for lemmatization (improves Coverage F1 word matching)
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# LLM interface for plausibility scoring
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from llm_interface import create_llm_provider
    GEMINI_AVAILABLE = True
except (ImportError, Exception):
    GEMINI_AVAILABLE = False

# Fallback: GPT-2 quality scorer from existing metrics
try:
    from .metrics import QualityScorer, QUALITY_SCORING_AVAILABLE
except ImportError:
    try:
        from metrics import QualityScorer, QUALITY_SCORING_AVAILABLE
    except ImportError:
        QUALITY_SCORING_AVAILABLE = False

# Import gloss accuracy from existing metrics
try:
    from .metrics import calculate_gloss_accuracy
except ImportError:
    try:
        from metrics import calculate_gloss_accuracy
    except ImportError:
        calculate_gloss_accuracy = None


# ============================================================================
# SECTION 1: IMPROVED COVERAGE F1 (with lemmatization)
# ============================================================================

# Function words to exclude from content matching
# (articles, prepositions, auxiliaries, conjunctions, determiners)
FUNCTION_WORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'can', 'of', 'to', 'in', 'on', 'at',
    'for', 'with', 'from', 'by', 'about', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'over', 'and',
    'or', 'but', 'if', 'then', 'so', 'than', 'that', 'this', 'these', 'those',
    'i', 'me', 'my', 'you', 'your', 'he', 'she', 'it', 'we', 'they',
    'him', 'her', 'its', 'our', 'their', 'not', 'no', 'yes',
}


def _lemmatize_word(word: str) -> str:
    """
    Lemmatize a word by trying all POS tags and returning the shortest result.

    'giving' -> 'give', 'candies' -> 'candy', 'liked' -> 'like'

    Falls back to the original word if NLTK is unavailable.
    """
    if not NLTK_AVAILABLE:
        return word

    lemmatizer = WordNetLemmatizer()
    # Try noun, verb, adjective, adverb — pick the shortest (most reduced)
    candidates = set()
    for pos in ('n', 'v', 'a', 'r'):
        candidates.add(lemmatizer.lemmatize(word, pos=pos))

    # Return shortest candidate (most reduced form)
    return min(candidates, key=len)


def _extract_content_words(sentence: str) -> List[str]:
    """
    Extract and lemmatize content words from a sentence.

    Removes punctuation, lowercases, filters function words, and lemmatizes.
    """
    translator = str.maketrans('', '', string.punctuation)
    normalized = sentence.translate(translator).strip().lower()
    words = [w for w in normalized.split() if w not in FUNCTION_WORDS]
    return [_lemmatize_word(w) for w in words]


def _words_match(word_a: str, word_b: str) -> bool:
    """
    Check if two words match using exact match on lemmatized forms,
    plus bidirectional substring and 4-char prefix as fallback.
    """
    if word_a == word_b:
        return True
    if word_a in word_b or word_b in word_a:
        return True
    if len(word_a) >= 4 and word_b.startswith(word_a[:4]):
        return True
    if len(word_b) >= 4 and word_a.startswith(word_b[:4]):
        return True
    return False


def calculate_coverage_v2(
    reference_sentence: str,
    output_sentence: str
) -> Dict[str, Any]:
    """
    Calculate semantic coverage metrics with lemmatization-improved matching.

    Improvement over v1: lemmatizes content words before matching, so
    'giving' matches 'give', 'candies' matches 'candy', etc.

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
        reference_words = _extract_content_words(reference_sentence)
        output_words = _extract_content_words(output_sentence)

        # Recall: how many reference content words appear in output?
        matched_reference = []
        missing_words = []

        for ref_word in reference_words:
            found = any(_words_match(ref_word, out_word) for out_word in output_words)
            if found:
                matched_reference.append(ref_word)
            else:
                missing_words.append(ref_word)

        recall = len(matched_reference) / len(reference_words) if reference_words else 0

        # Precision: how many output content words appear in reference?
        matched_output = []
        hallucinated_words = []

        for out_word in output_words:
            found = any(_words_match(out_word, ref_word) for ref_word in reference_words)
            if found:
                matched_output.append(out_word)
            else:
                hallucinated_words.append(out_word)

        precision = len(matched_output) / len(output_words) if output_words else 0

        # F1
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0

        return {
            'recall': recall * 100,
            'precision': precision * 100,
            'f1': f1 * 100,
            'missing_words': missing_words,
            'hallucinated_words': hallucinated_words,
        }

    except Exception as e:
        print(f"ERROR: Coverage v2 calculation failed: {e}")
        return {
            'recall': None,
            'precision': None,
            'f1': None,
            'missing_words': [],
            'hallucinated_words': [],
        }


# ============================================================================
# SECTION 2: PLAUSIBILITY SCORER (LLM-based)
# ============================================================================

PLAUSIBILITY_PROMPT = """You are evaluating the quality of an English sentence produced by an ASL-to-English translation system.

Rate the following sentence on a scale of 0-100 for overall plausibility.

Sentence: "{sentence}"

Consider these three factors:
1. GRAMMATICAL (0-100): Is it grammatically correct? Proper subject-verb agreement, tense, articles?
2. SEMANTIC (0-100): Does it describe something that could plausibly happen in the real world?
3. NATURALNESS (0-100): Does it sound like something a fluent English speaker would say?

Scoring guide:
- 90-100: Perfect or near-perfect English
- 70-89: Minor issues but clearly understandable
- 50-69: Noticeable errors but meaning is recoverable
- 30-49: Significant problems, meaning is unclear
- 0-29: Largely incomprehensible or nonsensical

Respond with ONLY this JSON, no other text:
{{"grammatical": <int>, "semantic": <int>, "naturalness": <int>, "overall": <int>, "justification": "<one sentence>"}}

The overall score should reflect the geometric mean of the three sub-scores."""


class PlausibilityScorer:
    """
    LLM-based plausibility scorer for translation output.

    Evaluates grammaticality, semantic plausibility, and naturalness using
    Gemini. Falls back to GPT-2 perplexity (QualityScorer) if unavailable.

    Rate limiting: Uses a minimum delay between API calls to stay within
    Google AI Studio free tier limits (15 requests/minute).
    """

    # Minimum delay between API calls (seconds) to respect rate limits
    # Free tier: 15 requests/minute = 4 seconds between requests
    MIN_REQUEST_DELAY = 4.5  # Slightly higher for safety margin

    def __init__(self, verbose: bool = True, provider_name: str = 'googleaistudio',
                 api_key: Optional[str] = None, rate_limit_delay: Optional[float] = None):
        self._provider = None
        self._fallback_scorer = None
        self._initialized = False
        self._fallback_mode = False
        self.verbose = verbose
        self._provider_name = provider_name
        self._api_key = api_key
        self._last_request_time = 0.0
        self._rate_limit_delay = rate_limit_delay if rate_limit_delay is not None else self.MIN_REQUEST_DELAY

    def _init_provider(self) -> None:
        """Initialize the LLM provider (Gemini)."""
        if self._initialized:
            return

        # Try Gemini first
        if GEMINI_AVAILABLE:
            try:
                kwargs = {
                    'provider': self._provider_name,
                    'temperature': 0.15,
                    'max_tokens': 200,
                }
                if self._api_key:
                    kwargs['api_key'] = self._api_key
                self._provider = create_llm_provider(**kwargs)
                self._initialized = True
                if self.verbose:
                    print("PlausibilityScorer: Initialized with Gemini")
                return
            except Exception as e:
                if self.verbose:
                    print(f"PlausibilityScorer: Gemini init failed ({e}), trying fallback...")

        # Fallback to GPT-2 QualityScorer
        if QUALITY_SCORING_AVAILABLE:
            try:
                self._fallback_scorer = QualityScorer(verbose=self.verbose)
                self._fallback_mode = True
                self._initialized = True
                if self.verbose:
                    print("PlausibilityScorer: Initialized with GPT-2 fallback")
                return
            except Exception as e:
                if self.verbose:
                    print(f"PlausibilityScorer: GPT-2 fallback also failed ({e})")

        self._initialized = True  # Mark as attempted
        if self.verbose:
            print("PlausibilityScorer: No scorer available")

    def _parse_llm_response(self, response_text: str) -> Tuple[Optional[float], str]:
        """
        Parse LLM JSON response into a score and justification.

        Handles markdown code fences and falls back to regex extraction.
        """
        # Strip markdown code fences
        text = response_text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

        try:
            data = json.loads(text)
            score = float(data.get('overall', 0))
            score = max(0.0, min(100.0, score))
            justification = data.get('justification', '')
            return score, justification
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Regex fallback: look for "overall": <number>
        match = re.search(r'"overall"\s*:\s*(\d+(?:\.\d+)?)', response_text)
        if match:
            score = max(0.0, min(100.0, float(match.group(1))))
            return score, "Parsed from partial response"

        return None, "Failed to parse response"

    def _apply_rate_limit(self):
        """Apply rate limiting delay if needed."""
        if self._rate_limit_delay > 0:
            elapsed = time.time() - self._last_request_time
            if elapsed < self._rate_limit_delay:
                wait_time = self._rate_limit_delay - elapsed
                if self.verbose:
                    print(f"  [Rate limit] Waiting {wait_time:.1f}s before next Gemini request...")
                time.sleep(wait_time)
            self._last_request_time = time.time()

    def calculate(self, sentence: str, debug: bool = False) -> Optional[float]:
        """
        Calculate plausibility score for a sentence.

        Args:
            sentence: The English sentence to evaluate
            debug: If True, print sub-scores and justification

        Returns:
            Plausibility score (0-100) or None if unavailable
        """
        self._init_provider()

        if self._fallback_mode and self._fallback_scorer:
            score = self._fallback_scorer.calculate(sentence, debug=debug)
            if debug and score is not None:
                print(f"  [Fallback] GPT-2 Quality Score: {score:.1f}")
            return score

        if self._provider is None:
            return None

        try:
            # Apply rate limiting for Gemini API
            self._apply_rate_limit()

            prompt = PLAUSIBILITY_PROMPT.format(sentence=sentence)
            response = self._provider.generate(prompt, temperature=0.15, max_tokens=200)
            score, justification = self._parse_llm_response(response)

            if debug:
                print(f"  Plausibility: {score}")
                print(f"  Justification: {justification}")
                print(f"  Raw response: {response[:200]}")

            if score is not None:
                return score

            # LLM returned unparseable response — try fallback
            if self._fallback_scorer is None and QUALITY_SCORING_AVAILABLE:
                self._fallback_scorer = QualityScorer(verbose=False)
            if self._fallback_scorer:
                return self._fallback_scorer.calculate(sentence)

            return None

        except Exception as e:
            if debug:
                print(f"  Plausibility error: {e}")
            # Try fallback on error
            if self._fallback_scorer is None and QUALITY_SCORING_AVAILABLE:
                self._fallback_scorer = QualityScorer(verbose=False)
            if self._fallback_scorer:
                return self._fallback_scorer.calculate(sentence)
            return None

    @property
    def is_available(self) -> bool:
        """Check if any scoring method is available."""
        self._init_provider()
        return self._provider is not None or self._fallback_scorer is not None


# Module-level convenience
_global_plausibility_scorer: Optional[PlausibilityScorer] = None


def calculate_plausibility(
    sentence: str,
    scorer: Optional[PlausibilityScorer] = None,
    debug: bool = False
) -> Optional[float]:
    """
    Calculate plausibility score for a sentence (module-level convenience).

    Args:
        sentence: The English sentence to evaluate
        scorer: Optional pre-initialized PlausibilityScorer
        debug: If True, print debug information

    Returns:
        Plausibility score (0-100) or None if unavailable
    """
    global _global_plausibility_scorer
    if scorer is not None:
        return scorer.calculate(sentence, debug=debug)
    if _global_plausibility_scorer is None:
        _global_plausibility_scorer = PlausibilityScorer(verbose=False)
    return _global_plausibility_scorer.calculate(sentence, debug=debug)


# ============================================================================
# SECTION 3: CTQI V2 — PREREQUISITE CHAIN FORMULA
# ============================================================================

def calculate_ctqi_v2(
    gloss_accuracy: float,
    coverage_f1: float,
    plausibility: float,
) -> float:
    """
    CTQI v2: Prerequisite Chain formula.

        CTQI = (GA/100) * (CF1/100) * (0.5 + 0.5 * P/100) * 100

    Design:
    - GA gates everything: wrong signs -> score collapses toward 0
    - CF1 gates plausibility: missing meaning -> score collapses
    - Plausibility modifies (0.5x to 1.0x): bad grammar halves score,
      but correct content is never zeroed out by poor fluency alone

    Args:
        gloss_accuracy: Sign recognition accuracy (0-100)
        coverage_f1: Semantic coverage F1 score (0-100)
        plausibility: LLM-assessed sentence plausibility (0-100)

    Returns:
        CTQI v2 score (0-100)
    """
    ga = max(0.0, min(100.0, gloss_accuracy)) / 100.0
    cf1 = max(0.0, min(100.0, coverage_f1)) / 100.0
    p = max(0.0, min(100.0, plausibility)) / 100.0

    plausibility_modifier = 0.5 + 0.5 * p

    return ga * cf1 * plausibility_modifier * 100.0


# ============================================================================
# SECTION 4: CONVENIENCE FUNCTION
# ============================================================================

def calculate_all_metrics_v2(
    hypothesis: str,
    reference: str,
    predicted_glosses: Optional[List[str]] = None,
    original_glosses: Optional[List[str]] = None,
    plausibility_scorer: Optional[PlausibilityScorer] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Calculate all CTQI v2 metrics in one call.

    Args:
        hypothesis: Generated/predicted sentence
        reference: Reference sentence (ground truth)
        predicted_glosses: List of predicted glosses (for GA)
        original_glosses: List of original glosses (for GA)
        plausibility_scorer: Optional pre-initialized PlausibilityScorer
        debug: If True, print debug information

    Returns:
        dict with all component scores and final CTQI v2
    """
    result = {}

    # Gloss Accuracy
    if predicted_glosses and original_glosses and calculate_gloss_accuracy:
        ga_result = calculate_gloss_accuracy(predicted_glosses, original_glosses)
        result['gloss_accuracy'] = ga_result['accuracy']
        result['gloss_correct'] = ga_result['correct']
        result['gloss_total'] = ga_result['total']
        result['gloss_mismatches'] = ga_result['mismatches']
    else:
        result['gloss_accuracy'] = None
        result['gloss_correct'] = 0
        result['gloss_total'] = 0
        result['gloss_mismatches'] = []

    # Coverage F1 (improved with lemmatization)
    coverage = calculate_coverage_v2(reference, hypothesis)
    result['coverage_recall_v2'] = coverage['recall']
    result['coverage_precision_v2'] = coverage['precision']
    result['coverage_f1_v2'] = coverage['f1']
    result['missing_words_v2'] = coverage['missing_words']
    result['hallucinated_words_v2'] = coverage['hallucinated_words']

    # Plausibility
    result['plausibility'] = calculate_plausibility(
        hypothesis, scorer=plausibility_scorer, debug=debug
    )

    # CTQI v2
    if (result['gloss_accuracy'] is not None and
        result['coverage_f1_v2'] is not None and
        result['plausibility'] is not None):
        result['ctqi_v2'] = calculate_ctqi_v2(
            result['gloss_accuracy'],
            result['coverage_f1_v2'],
            result['plausibility']
        )
    else:
        result['ctqi_v2'] = None

    return result


# ============================================================================
# SECTION 5: MODULE TEST
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("CTQI v2 - Prerequisite Chain Metric Test")
    print("=" * 70)

    # Availability
    print(f"\nNLTK (lemmatization): {'Available' if NLTK_AVAILABLE else 'NOT available'}")
    print(f"Gemini (plausibility): {'Available' if GEMINI_AVAILABLE else 'NOT available'}")
    print(f"GPT-2 (fallback):     {'Available' if QUALITY_SCORING_AVAILABLE else 'NOT available'}")

    # --- Test 1: Lemmatization improvement ---
    print("\n" + "=" * 70)
    print("TEST 1: Coverage F1 — Lemmatization vs Original")
    print("=" * 70)

    # Import original for comparison
    try:
        from metrics import calculate_coverage as calculate_coverage_v1
        has_v1 = True
    except ImportError:
        has_v1 = False

    test_pairs = [
        ("The doctor is giving an apple.", "The doctor will give an apple.",
         "giving vs give — verb form"),
        ("The cousin likes many candies.", "My cousin likes many candy.",
         "candies vs candy — plural form"),
        ("The mother is giving an apple now.", "Mother, give the apple now.",
         "giving vs give — different tense"),
        ("The cool man likes all computers.", "The cool man like computer all.",
         "likes vs like, computers vs computer"),
        ("The man is walking now.", "The man will walk now.",
         "walking vs walk — progressive vs base"),
    ]

    for ref, out, desc in test_pairs:
        v2_result = calculate_coverage_v2(ref, out)
        print(f"\n  [{desc}]")
        print(f"  Ref: {ref}")
        print(f"  Out: {out}")
        if has_v1:
            v1_result = calculate_coverage_v1(ref, out)
            print(f"  CF1 v1: {v1_result['f1']:.1f}  |  CF1 v2: {v2_result['f1']:.1f}  "
                  f"({'improved' if v2_result['f1'] > v1_result['f1'] else 'same'})")
        else:
            print(f"  CF1 v2: {v2_result['f1']:.1f}")
        if v2_result['missing_words']:
            print(f"  Missing: {v2_result['missing_words']}")
        if v2_result['hallucinated_words']:
            print(f"  Hallucinated: {v2_result['hallucinated_words']}")

    # --- Test 2: CTQI v2 formula edge cases ---
    print("\n" + "=" * 70)
    print("TEST 2: CTQI v2 Formula — Edge Cases")
    print("=" * 70)

    edge_cases = [
        (100, 100, 100, "Perfect translation"),
        (100, 100, 0,   "Correct but unreadable"),
        (0,   100, 100, "Wrong signs, fluent English"),
        (100, 0,   100, "Right signs, no meaning preserved"),
        (100, 100, 50,  "Correct but mediocre grammar"),
        (50,  50,  50,  "Everything mediocre"),
        (0,   0,   0,   "Complete failure"),
    ]

    print(f"\n  {'GA':>5} {'CF1':>5} {'P':>5} | {'CTQI v2':>8}  Description")
    print(f"  {'-'*5} {'-'*5} {'-'*5} | {'-'*8}  {'-'*30}")
    for ga, cf1, p, desc in edge_cases:
        score = calculate_ctqi_v2(ga, cf1, p)
        print(f"  {ga:5.1f} {cf1:5.1f} {p:5.1f} | {score:8.1f}  {desc}")

    # --- Test 3: Plausibility scoring ---
    print("\n" + "=" * 70)
    print("TEST 3: Plausibility Scorer")
    print("=" * 70)

    scorer = PlausibilityScorer(verbose=True)

    plausibility_tests = [
        ("The dog likes a bath.", "Good — grammatical and plausible"),
        ("The man gave a kiss to the woman on Thanksgiving.", "Good — natural sentence"),
        ("Apple plays basketball.", "Bad — semantically nonsensical"),
        ("The cool man like computer all.", "Bad — broken grammar"),
        ("Son basketball plays my.", "Bad — word salad"),
        ("The woman is wrong about the apple.", "OK — grammatical but odd meaning"),
    ]

    for sentence, desc in plausibility_tests:
        score = scorer.calculate(sentence)
        score_str = f"{score:.1f}" if score is not None else "N/A"
        print(f"  [{score_str:>5}] {sentence:<55} ({desc})")

    print("\n" + "=" * 70)
    print("Tests complete.")
    print("=" * 70)
