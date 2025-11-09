#!/usr/bin/env python3
"""
BLEU Score Calculator for ASL Sentence Construction

Calculates BLEU score by comparing a predicted sentence against reference sentences
from the synthetic sentence dataset. Looks up reference sentences by matching glosses.

Usage:
    # With custom predicted sentence
    python calculate_sent_bleu.py --glosses "I,WANT,BOOK" --sentence "I want to read a book" --num-glosses 20

    # Using concatenated glosses as sentence (adds period automatically)
    python calculate_sent_bleu.py --glosses "MANY,HOT,DOG" --num-glosses 50
    # Uses "MANY HOT DOG." as the predicted sentence
"""

import os
import sys
import json
import argparse
from pathlib import Path
from sacrebleu import sentence_bleu

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config

# ============================================================================
# CONFIGURATION
# ============================================================================
REFERENCE_DATASET_DIR = "datasets/synthetic_sentences"  # Relative to project root
REFERENCE_FILENAME_TEMPLATE = "synthetic_gloss_to_sentence_llm_dataset_{num_glosses}_glosses.json"
IGNORE_PUNCTUATION = True  # If True, strips trailing punctuation before comparison
IGNORE_CASE = True  # If True, normalizes both sentences to lowercase before comparison


def load_reference_dataset(num_glosses, config):
    """
    Load reference sentence dataset.

    Args:
        num_glosses: Number of glosses in the dataset
        config: Config object with project_root

    Returns:
        List of dicts with 'glosses' and 'sentence' keys
    """
    dataset_dir = config.project_root / REFERENCE_DATASET_DIR
    filename = REFERENCE_FILENAME_TEMPLATE.format(num_glosses=num_glosses)
    dataset_file = dataset_dir / filename

    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: {dataset_file}\n"
            f"Make sure you've generated the dataset with:\n"
            f"  python dataset-utilities/sentence-construction/synthetic-sentence-generator.py --num-glosses {num_glosses}"
        )

    with open(dataset_file, 'r') as f:
        return json.load(f)


def find_reference_sentence(glosses, reference_data):
    """
    Find reference sentence by matching glosses.

    Args:
        glosses: List of gloss strings to match
        reference_data: List of reference sentence dicts

    Returns:
        Reference sentence string or None if not found
    """
    # Normalize input glosses to uppercase
    glosses_normalized = [g.upper().strip() for g in glosses]

    for item in reference_data:
        # Normalize reference glosses to uppercase
        ref_glosses = [g.upper().strip() for g in item['glosses']]

        # Check if glosses match
        if ref_glosses == glosses_normalized:
            return item['sentence']

    return None


def calculate_bleu_score(predicted_sentence, reference_sentence):
    """
    Calculate BLEU score between predicted and reference sentence.

    Args:
        predicted_sentence: Predicted sentence string
        reference_sentence: Reference sentence string

    Returns:
        BLEU score (0-100)
    """
    # Normalize sentences based on configuration
    predicted_normalized = predicted_sentence
    reference_normalized = reference_sentence

    if IGNORE_PUNCTUATION:
        import string
        # Remove ALL punctuation from both sentences
        translator = str.maketrans('', '', string.punctuation)
        predicted_normalized = predicted_normalized.translate(translator).strip()
        reference_normalized = reference_normalized.translate(translator).strip()

    if IGNORE_CASE:
        # Normalize to lowercase for case-insensitive comparison
        predicted_normalized = predicted_normalized.lower()
        reference_normalized = reference_normalized.lower()

    # sacrebleu expects reference as a list (can have multiple references)
    bleu = sentence_bleu(predicted_normalized, [reference_normalized])
    return bleu.score


def calculate_bleu_from_glosses(glosses, predicted_sentence, num_glosses, verbose=False):
    """
    Calculate BLEU score for a predicted sentence.

    Args:
        glosses: List of gloss strings (e.g., ["I", "WANT", "BOOK"])
        predicted_sentence: Predicted sentence string
        num_glosses: Number of glosses in reference dataset (20, 50, etc.)
        verbose: Print detailed information

    Returns:
        dict with:
            - 'bleu_score': float (0-100)
            - 'reference': str (reference sentence)
            - 'found': bool (whether reference was found)

    Raises:
        FileNotFoundError: If reference dataset doesn't exist
    """
    config = get_config()

    if verbose:
        print(f"Input glosses: {glosses}")
        print(f"Predicted sentence: {predicted_sentence}")
        print(f"Loading reference dataset ({num_glosses} glosses)...")

    # Load reference dataset
    reference_data = load_reference_dataset(num_glosses, config)

    if verbose:
        print(f"Loaded {len(reference_data)} reference sentences")

    # Find reference sentence
    reference_sentence = find_reference_sentence(glosses, reference_data)

    if reference_sentence is None:
        return {
            'bleu_score': 0.0,
            'reference': None,
            'found': False
        }

    if verbose:
        print(f"Reference sentence: {reference_sentence}")

    # Calculate BLEU score
    bleu_score = calculate_bleu_score(predicted_sentence, reference_sentence)

    if verbose:
        print(f"\n{'='*70}")
        print(f"BLEU Score Results")
        print(f"{'='*70}")
        print(f"Glosses:            {', '.join(glosses)}")
        print(f"Predicted:          {predicted_sentence}")
        print(f"Reference:          {reference_sentence}")
        print(f"BLEU Score:         {bleu_score:.2f}")
        print(f"{'='*70}")

    return {
        'bleu_score': bleu_score,
        'reference': reference_sentence,
        'found': True
    }


def main():
    parser = argparse.ArgumentParser(
        description='Calculate BLEU score for predicted ASL sentence'
    )
    parser.add_argument(
        '--glosses',
        type=str,
        required=True,
        help='Comma-separated list of glosses (e.g., "I,WANT,BOOK")'
    )
    parser.add_argument(
        '--sentence',
        type=str,
        required=False,
        default=None,
        help='Predicted sentence to evaluate (if not provided, uses concatenated glosses with period)'
    )
    parser.add_argument(
        '--num-glosses',
        type=int,
        required=True,
        help='Number of glosses in the reference dataset (e.g., 20, 50)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information'
    )

    args = parser.parse_args()

    # Get config
    config = get_config()

    # Parse glosses
    glosses = [g.strip() for g in args.glosses.split(',')]

    # If no sentence provided, use concatenated glosses with period
    if args.sentence is None:
        predicted_sentence = ' '.join(glosses) + '.'
        if args.verbose:
            print(f"No sentence provided, using concatenated glosses: {predicted_sentence}")
    else:
        predicted_sentence = args.sentence

    if args.verbose:
        print(f"Input glosses: {glosses}")
        print(f"Predicted sentence: {predicted_sentence}")
        print(f"Loading reference dataset ({args.num_glosses} glosses)...")

    # Load reference dataset
    try:
        reference_data = load_reference_dataset(args.num_glosses, config)
        if args.verbose:
            print(f"Loaded {len(reference_data)} reference sentences")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Find reference sentence
    reference_sentence = find_reference_sentence(glosses, reference_data)

    if reference_sentence is None:
        print(f"Error: No reference sentence found for glosses: {glosses}", file=sys.stderr)
        print(f"These glosses may not exist in the reference dataset.", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Reference sentence: {reference_sentence}")

    # Calculate BLEU score
    bleu_score = calculate_bleu_score(predicted_sentence, reference_sentence)

    # Output results
    if args.verbose:
        print(f"\n{'='*70}")
        print(f"BLEU Score Results")
        print(f"{'='*70}")
        print(f"Glosses:            {', '.join(glosses)}")
        print(f"Predicted:          {predicted_sentence}")
        print(f"Reference:          {reference_sentence}")
        print(f"BLEU Score:         {bleu_score:.2f}")
        print(f"{'='*70}")
    else:
        # Simple output for programmatic use
        print(f"{bleu_score:.2f}")

    return bleu_score


if __name__ == "__main__":
    main()
