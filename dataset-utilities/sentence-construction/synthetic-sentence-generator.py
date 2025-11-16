#!/usr/bin/env python3
"""
Synthetic Sentence Generator for ASL Testing

Generates synthetic sentences using glosses from WLASL dataset and configured LLM.
Uses batch generation for reliability and speed.

Configuration:
    All LLM settings (model, API key, temperature, etc.) are configured in:
    project-utilities/llm-interface/.env

    To change the model, edit LLM_MODEL in that .env file.

Usage:
    # Generate 100 sentences (default) with 50 glosses
    python synthetic-sentence-generator.py --num-glosses 50

    # Generate 200 sentences with custom batch size
    python synthetic-sentence-generator.py --num-glosses 50 --num-sentences 200 --batch-size 25

    # Quick test with just 20 sentences
    python synthetic-sentence-generator.py --num-glosses 50 --num-sentences 20

Configuration:
    To change the gloss distribution (e.g., 3-8 vs 5-10 glosses per sentence),
    edit the GLOSS_DISTRIBUTION dictionary at the top of this file.
    Three preset options are provided: Option A (realistic ASL), Option B (complex),
    and Option C (balanced). Simply comment/uncomment the desired option.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "project-utilities"))

from config import get_config

# Import LLM interface (now IDE-resolvable via package __init__.py)
try:
    from llm_interface import create_llm_provider, get_current_model_info, LLMError
except ImportError as e:
    print(f"ERROR: Failed to import LLM interface: {e}")
    print("Make sure you're running from the project root and llm-interface is set up")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================
PROMPT_TEMPLATE_FILE = "syn_sentence_gen_prompt.txt"
OUTPUT_FILENAME_TEMPLATE = "synthetic_gloss_to_sentence_llm_dataset_{class_count}_glosses.json"
OUTPUT_REF_SENTENCE_DATASET_DIR = "datasets/synthetic_sentences"  # Relative to project root

# Gloss Distribution Configuration
# Defines how many glosses per sentence and their distribution
# Format: {num_glosses: percentage}
# Percentages should sum to 1.0 (100%)

# Option A: Realistic ASL Distribution (CURRENT)
# Reflects actual ASL usage patterns - most sentences use 3-5 glosses
GLOSS_DISTRIBUTION = {
    3: 0.30,  # 30% - very common in everyday ASL
    4: 0.30,  # 30% - common
    5: 0.20,  # 20% - typical
    6: 0.10,  # 10% - less common
    7: 0.05,  # 5%  - rare
    8: 0.05,  # 5%  - very rare
}

# Option B: Model Testing Focus (complex translation challenges)
# GLOSS_DISTRIBUTION = {
#     5: 0.20,  # 20%
#     6: 0.20,  # 20%
#     7: 0.20,  # 20%
#     8: 0.20,  # 20%
#     9: 0.10,  # 10%
#     10: 0.10, # 10%
# }

# Option C: Balanced Approach (realistic + challenging)
# GLOSS_DISTRIBUTION = {
#     3: 0.20,  # 20%
#     4: 0.20,  # 20%
#     5: 0.20,  # 20%
#     6: 0.20,  # 20%
#     7: 0.10,  # 10%
#     8: 0.10,  # 10%
# }

# Minimum and maximum glosses (derived from distribution)
MIN_GLOSSES = min(GLOSS_DISTRIBUTION.keys())
MAX_GLOSSES = max(GLOSS_DISTRIBUTION.keys())


def find_closest_dataset(num_glosses, dataset_root):
    """
    Find the dataset with the closest number of classes <= num_glosses.

    Args:
        num_glosses: Target number of glosses
        dataset_root: Path to dataset_splits directory

    Returns:
        Tuple of (class_count, dataset_path) or (None, None) if not found
    """
    dataset_root = Path(dataset_root)
    available_datasets = []

    # Scan for available class folders
    for class_dir in dataset_root.iterdir():
        if class_dir.is_dir() and "_classes" in class_dir.name:
            try:
                class_count = int(class_dir.name.split("_")[0])
                if class_count <= num_glosses:
                    # Check if train directory exists
                    train_path = class_dir / "original" / f"pickle_from_pose_split_{class_count}_class" / "train"
                    if train_path.exists():
                        available_datasets.append((class_count, train_path))
            except ValueError:
                continue

    if not available_datasets:
        return None, None

    # Sort by class count descending and return the largest that's <= num_glosses
    available_datasets.sort(reverse=True)
    return available_datasets[0]


def extract_glosses(dataset_path):
    """
    Extract gloss names from dataset directory.

    Args:
        dataset_path: Path to train directory containing gloss folders

    Returns:
        List of gloss names (uppercase)
    """
    dataset_path = Path(dataset_path)
    glosses = []

    for item in dataset_path.iterdir():
        if item.is_dir():
            glosses.append(item.name.upper())

    return sorted(glosses)


def generate_sentences_batch(glosses, batch_num_sentences, llm, prompt_template):
    """
    Generate a single batch of sentences.

    Args:
        glosses: List of gloss strings
        batch_num_sentences: Number of sentences to generate in this batch
        llm: LLM provider instance
        prompt_template: Prompt template string

    Returns:
        List of dicts with 'glosses' and 'sentence' keys
    """
    # Format glosses as a numbered list for better visibility
    gloss_list = "\n".join([f"{i+1}. {gloss}" for i, gloss in enumerate(glosses)])

    # Create distribution description for the prompt
    dist_desc = ", ".join([f"{count} glosses ({int(pct*100)}%)" for count, pct in sorted(GLOSS_DISTRIBUTION.items())])

    # Calculate exact sentence counts for this batch based on distribution
    exact_breakdown = []
    for count, pct in sorted(GLOSS_DISTRIBUTION.items()):
        num_for_this_count = max(1, int(batch_num_sentences * pct))  # At least 1
        exact_breakdown.append(f"- Generate {num_for_this_count} sentences with EXACTLY {count} words")
    exact_breakdown_str = "\n".join(exact_breakdown)

    prompt = prompt_template.format(
        num_sentences=batch_num_sentences,
        num_glosses=len(glosses),
        gloss_list=gloss_list,
        min_glosses=MIN_GLOSSES,
        max_glosses=MAX_GLOSSES,
        distribution_desc=dist_desc,
        exact_breakdown=exact_breakdown_str
    )

    response_text = llm.generate(prompt)

    # Parse response
    # Extract JSON from response (handle markdown code blocks if present)
    response_text = response_text.strip()

    # Remove markdown code blocks
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        # Remove first and last lines (the ``` markers)
        response_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        response_text = response_text.strip()

    # Remove "json" language identifier if present
    if response_text.startswith("json"):
        response_text = response_text[4:].strip()

    try:
        sentences_data = json.loads(response_text)
    except json.JSONDecodeError as e:
        # Print partial response for debugging
        print(f"  JSON Parse Error at position {e.pos}")
        print(f"  Response preview (first 500 chars):")
        print(f"  {response_text[:500]}")
        print(f"  Response preview (last 200 chars):")
        print(f"  ...{response_text[-200:]}")
        raise

    # Validate and normalize
    validated_sentences = []
    gloss_set = set(g.upper() for g in glosses)

    for item in sentences_data:
        # Normalize glosses to uppercase
        item_glosses = [g.upper() for g in item['glosses']]

        # Validate gloss count range
        if len(item_glosses) < MIN_GLOSSES:
            print(f"  Warning: Skipping sentence - only {len(item_glosses)} glosses (minimum {MIN_GLOSSES} required)")
            continue
        if len(item_glosses) > MAX_GLOSSES:
            print(f"  Warning: Skipping sentence - {len(item_glosses)} glosses (maximum {MAX_GLOSSES} allowed)")
            continue

        # Validate glosses are in our set
        invalid_glosses = [g for g in item_glosses if g not in gloss_set]
        if not invalid_glosses:
            validated_sentences.append({
                'glosses': item_glosses,
                'sentence': item['sentence']
            })
        else:
            print(f"  Warning: Skipping sentence - invalid glosses {invalid_glosses} not in allowed set")

    return validated_sentences


def generate_sentences(glosses, num_sentences=100, batch_size=20):
    """
    Generate sentences using configured LLM in batches.

    All LLM configuration (model, API key, temperature, etc.) is read from
    the .env file in project-utilities/llm-interface/.env

    Args:
        glosses: List of gloss strings
        num_sentences: Total number of sentences to generate (default: 100)
        batch_size: Number of sentences per batch (default: 20)

    Returns:
        List of dicts with 'glosses' and 'sentence' keys
    """
    # Get current model configuration
    model_info = get_current_model_info()
    print(f"\nLLM Configuration (from .env):")
    print(f"  Provider: {model_info['provider']}")
    print(f"  Model: {model_info['model']}")
    print(f"  Temperature: {model_info['temperature']}")
    print(f"  Max Tokens: {model_info['max_tokens']}")
    print(f"  API Key: {'Set' if model_info['api_key_set'] else 'NOT SET'}")

    # Initialize LLM provider from environment config
    try:
        llm = create_llm_provider(
            max_tokens=3000,  # Batch of 20 sentences needs ~1500-2000 tokens
            timeout=60  # 1 minute per batch is plenty
        )

        # Test with a simple prompt to verify it works
        test_response = llm.generate("Say 'test'", max_tokens=10)
        print(f"\n[SUCCESS] LLM is ready")
        print(f"  Test response: {test_response}")
    except Exception as e:
        print(f"\n[ERROR] LLM initialization failed: {e}")
        print("\nPlease check your .env configuration in project-utilities/llm-interface/")
        raise Exception(f"Could not initialize LLM: {e}")

    # Load prompt template from file
    prompt_file = Path(__file__).parent / PROMPT_TEMPLATE_FILE
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_file}")
        return []

    # Calculate number of batches
    num_batches = (num_sentences + batch_size - 1) // batch_size  # Ceiling division

    print(f"\nGenerating {num_sentences} sentences in {num_batches} batches of ~{batch_size} sentences each...")
    print(f"(Each batch takes 10-20 seconds)")

    all_sentences = []

    for batch_idx in range(num_batches):
        # Calculate how many sentences for this batch
        sentences_remaining = num_sentences - len(all_sentences)
        batch_num = min(batch_size, sentences_remaining)

        print(f"\n[Batch {batch_idx + 1}/{num_batches}] Generating {batch_num} sentences...")

        # Try batch with one retry on JSON parse errors
        max_retries = 1
        for retry in range(max_retries + 1):
            try:
                batch_sentences = generate_sentences_batch(glosses, batch_num, llm, prompt_template)
                all_sentences.extend(batch_sentences)
                print(f"  [OK] Generated {len(batch_sentences)} valid sentences (Total: {len(all_sentences)}/{num_sentences})")
                break  # Success, move to next batch

            except json.JSONDecodeError as e:
                if retry < max_retries:
                    print(f"  [RETRY] JSON parsing error, retrying... (attempt {retry + 1}/{max_retries + 1})")
                else:
                    print(f"  [ERROR] JSON parsing error in batch {batch_idx + 1}: {e}")
                    print(f"  Skipping this batch after {max_retries + 1} attempts...")

            except Exception as e:
                print(f"  [ERROR] Error in batch {batch_idx + 1}: {e}")
                print(f"  Skipping this batch and continuing...")
                break

    print(f"\n{'='*70}")
    print(f"Batch generation complete!")
    print(f"Successfully generated {len(all_sentences)} sentences")
    print(f"{'='*70}")

    return all_sentences




def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic ASL sentences using configured LLM'
    )
    parser.add_argument(
        '--num-glosses',
        type=int,
        required=True,
        help='Number of glosses to use (will select closest dataset <= this number)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=f'Output directory for generated dataset (default: {OUTPUT_REF_SENTENCE_DATASET_DIR} relative to project root)'
    )
    parser.add_argument(
        '--num-sentences',
        type=int,
        default=100,
        help='Total number of sentences to generate (default: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='Number of sentences per batch (default: 20)'
    )

    args = parser.parse_args()

    # Get config and find dataset
    config = get_config()
    dataset_splits = config.data_root / "dataset_splits"

    print(f"Searching for dataset with <= {args.num_glosses} glosses...")
    class_count, dataset_path = find_closest_dataset(args.num_glosses, dataset_splits)

    if dataset_path is None:
        print(f"Error: No dataset found with <= {args.num_glosses} glosses")
        sys.exit(1)

    print(f"Selected dataset: {class_count} classes")
    print(f"Dataset path: {dataset_path}")

    # Extract glosses
    print("\nExtracting glosses...")
    glosses = extract_glosses(dataset_path)
    print(f"Found {len(glosses)} glosses: {', '.join(glosses[:10])}{'...' if len(glosses) > 10 else ''}")

    # Generate sentences
    print("\nGenerating sentences with configured LLM...")
    sentences_data = generate_sentences(
        glosses,
        num_sentences=args.num_sentences,
        batch_size=args.batch_size
    )

    if not sentences_data:
        print("Error: Failed to generate sentences")
        sys.exit(1)

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use project root from config and append default output dir
        output_dir = config.project_root / OUTPUT_REF_SENTENCE_DATASET_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save output with class count in filename
    output_filename = OUTPUT_FILENAME_TEMPLATE.format(class_count=class_count)
    output_file = output_dir / output_filename
    with open(output_file, 'w') as f:
        json.dump(sentences_data, f, indent=2)

    print(f"\n{'='*70}")
    print("Dataset Generation Complete!")
    print(f"{'='*70}")
    print(f"Output file: {output_file}")
    print(f"Total sentences: {len(sentences_data)}")
    print(f"Glosses used: {len(glosses)}")

    # Print statistics
    word_counts = defaultdict(int)
    for item in sentences_data:
        word_counts[len(item['glosses'])] += 1

    print("\nSentence length distribution:")
    for length in sorted(word_counts.keys()):
        print(f"  {length} words: {word_counts[length]} sentences")

    print(f"\nSample output (first 3 sentences):")
    for i, item in enumerate(sentences_data[:3]):
        print(f"\n{i+1}. Sentence: {item['sentence']}")
        print(f"   Glosses: {item['glosses']}")


if __name__ == "__main__":
    main()
