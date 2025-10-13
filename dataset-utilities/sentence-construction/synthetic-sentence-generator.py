#!/usr/bin/env python3
"""
Synthetic Sentence Generator for ASL Testing

Generates synthetic sentences using glosses from WLASL dataset and Gemini API.
Creates a dataset with sentences using glosses from the dataset.

Usage:
    export GEMINI_API_KEY=your_api_key_here
    python synthetic-sentence-generator.py --num-glosses 20
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
import google.generativeai as genai

# Add parent directories to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_config

# ============================================================================
# CONFIGURATION
# ============================================================================
PROMPT_TEMPLATE_FILE = "syn_sentence_gen_prompt.txt"
OUTPUT_FILENAME_TEMPLATE = "synthetic_gloss_to_sentence_llm_dataset_{class_count}_glosses.json"
OUTPUT_REF_SENTENCE_DATASET_DIR = "datasets/synthetic_sentences"  # Relative to project root


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


def generate_sentences(glosses, api_key, num_sentences=300):
    """
    Generate sentences using Gemini API.

    Args:
        glosses: List of gloss strings
        api_key: Gemini API key
        num_sentences: Total number of sentences to generate

    Returns:
        List of dicts with 'glosses' and 'sentence' keys
    """
    genai.configure(api_key=api_key)

    # List and try available models
    print("\nChecking available models...")
    available_models = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
                print(f"  Found: {m.name}")
    except Exception as e:
        print(f"Warning: Could not list models: {e}")

    # Try models in order of preference
    model_names = available_models if available_models else [
        'models/gemini-1.5-flash-latest',
        'models/gemini-1.5-flash',
        'models/gemini-1.5-pro',
        'models/gemini-pro'
    ]

    model = None
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            # Test with a simple prompt to verify it works
            test_response = model.generate_content("Say 'test'")
            print(f"Using model: {model_name}")
            break
        except Exception as e:
            print(f"  Model {model_name} failed: {str(e)[:100]}")
            continue

    if model is None:
        print("\nNo compatible model found. Please check:")
        print("1. Your API key is valid")
        print("2. You have access to Gemini API")
        print("3. Try getting a new key from https://makersuite.google.com/app/apikey")
        raise Exception("Could not find a compatible Gemini model")

    # Load prompt template from file
    prompt_file = Path(__file__).parent / PROMPT_TEMPLATE_FILE
    try:
        with open(prompt_file, 'r') as f:
            prompt_template = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_file}")
        return []

    # Fill in the template
    gloss_list = ", ".join(glosses)
    prompt = prompt_template.format(
        num_sentences=num_sentences,
        gloss_list=gloss_list
    )

    print("Calling Gemini API to generate sentences...")
    response = model.generate_content(prompt)

    # Parse response
    try:
        # Extract JSON from response (handle markdown code blocks if present)
        response_text = response.text.strip()
        if response_text.startswith("```"):
            # Remove markdown code block markers
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        sentences_data = json.loads(response_text)

        # Validate and normalize
        validated_sentences = []
        gloss_set = set(g.upper() for g in glosses)

        for item in sentences_data:
            # Normalize glosses to uppercase
            item_glosses = [g.upper() for g in item['glosses']]

            # Validate glosses are in our set
            if all(g in gloss_set for g in item_glosses):
                validated_sentences.append({
                    'glosses': item_glosses,
                    'sentence': item['sentence']
                })
            else:
                print(f"Warning: Skipping sentence with invalid glosses: {item_glosses}")

        print(f"Generated {len(validated_sentences)} valid sentences")
        return validated_sentences

    except json.JSONDecodeError as e:
        print(f"Error parsing Gemini response: {e}")
        print(f"Response: {response.text[:500]}...")
        return []




def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic ASL sentences using Gemini API'
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

    args = parser.parse_args()

    # Get API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Set it with: export GEMINI_API_KEY=your_api_key_here")
        sys.exit(1)

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
    print("\nGenerating sentences with Gemini API...")
    sentences_data = generate_sentences(glosses, api_key)

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
