#!/usr/bin/env python3
"""
Compute missing plausibility scores for demo samples.

Finds samples with plausibility=0 and computes them using the PlausibilityScorer.
Also recalculates CTQI v3 composite score.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to Python path BEFORE loading .env
script_path = Path(__file__).resolve()
# Script is at: asl-v1/applications/show-and-tell/scripts/compute_missing_plausibility.py
# project-utilities is at: asl-v1/project-utilities
project_root = script_path.parents[3]  # asl-v1
sys.path.insert(0, str(project_root / "project-utilities"))

# Load .env file for API keys BEFORE any imports that need them
env_path = project_root / "project-utilities" / "llm_interface" / ".env"
if env_path.exists():
    print(f"Loading environment from: {env_path}")
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value
                if 'API' in key or 'KEY' in key:
                    print(f"  Set {key}=****{value[-4:] if len(value) > 4 else '****'}")
else:
    print(f"Warning: .env file not found at {env_path}")

# Now import the modules that need the API keys
from evaluation_metrics import (
    PlausibilityScorer,
    GEMINI_AVAILABLE,
    calculate_composite_score_v3,
)

DEMO_SAMPLES_DIR = Path(__file__).resolve().parents[1] / "demo-data" / "samples"


def compute_ctqi_v3(ga: float, cf1: float, p: float) -> float:
    """
    Compute CTQI v3 score.
    Formula: (GA/100) × (CF1/100) × (0.5 + 0.5 × P/100 × GA/100) × 100
    """
    ga_frac = ga / 100.0
    cf1_frac = cf1 / 100.0
    p_frac = p / 100.0

    # CTQI v3 formula
    composite = ga_frac * cf1_frac * (0.5 + 0.5 * p_frac * ga_frac) * 100.0
    return composite


def get_effective_ga(eval_data: dict) -> float:
    """Extract effective gloss accuracy from evaluation data."""
    # Try new field names first
    if 'effective_gloss_accuracy' in eval_data:
        return eval_data['effective_gloss_accuracy']
    if 'model_gloss_accuracy' in eval_data:
        return eval_data['model_gloss_accuracy']
    # Fall back to old field name
    if 'gloss_accuracy' in eval_data:
        return eval_data['gloss_accuracy']
    return 100.0  # Default if not found


def get_coverage_f1(eval_data: dict) -> float:
    """Extract coverage F1 from evaluation data."""
    if 'coverage_f1' in eval_data:
        return eval_data['coverage_f1']
    return 100.0  # Default if not found


def main():
    print("=" * 60)
    print("Computing Missing Plausibility Scores")
    print("=" * 60)

    if not GEMINI_AVAILABLE:
        print("ERROR: Gemini API not available. Cannot compute plausibility scores.")
        print("Make sure google-generativeai is installed: pip install google-generativeai")
        sys.exit(1)

    # Initialize scorer
    print("\nInitializing PlausibilityScorer...")
    scorer = PlausibilityScorer(verbose=True)

    # Find all sample metadata files
    samples_to_update = []

    for sample_dir in sorted(DEMO_SAMPLES_DIR.iterdir()):
        if not sample_dir.is_dir():
            continue

        metadata_path = sample_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        sample_id = metadata.get('id', sample_dir.name)
        precomputed = metadata.get('precomputed', {})
        evaluation = precomputed.get('evaluation', {})

        raw_eval = evaluation.get('raw', {})
        llm_eval = evaluation.get('llm', {})

        raw_p = raw_eval.get('plausibility', 0)
        llm_p = llm_eval.get('plausibility', 0)

        raw_sentence = precomputed.get('raw_sentence', '')
        llm_sentence = precomputed.get('llm_sentence', '')

        needs_raw = raw_p == 0 and raw_sentence
        needs_llm = llm_p == 0 and llm_sentence

        if needs_raw or needs_llm:
            samples_to_update.append({
                'path': metadata_path,
                'id': sample_id,
                'metadata': metadata,
                'raw_sentence': raw_sentence,
                'llm_sentence': llm_sentence,
                'needs_raw': needs_raw,
                'needs_llm': needs_llm,
                'current_raw_p': raw_p,
                'current_llm_p': llm_p,
            })

    if not samples_to_update:
        print("\nNo samples need plausibility score updates.")
        return

    print(f"\nFound {len(samples_to_update)} samples needing updates:")
    for sample in samples_to_update:
        needs = []
        if sample['needs_raw']:
            needs.append('raw')
        if sample['needs_llm']:
            needs.append('llm')
        print(f"  - {sample['id']}: needs {', '.join(needs)}")

    print("\n" + "-" * 60)

    for i, sample in enumerate(samples_to_update, 1):
        print(f"\n[{i}/{len(samples_to_update)}] Processing: {sample['id']}")

        metadata = sample['metadata']
        precomputed = metadata['precomputed']
        evaluation = precomputed.get('evaluation', {})
        raw_eval = evaluation.get('raw', {})
        llm_eval = evaluation.get('llm', {})

        # Compute raw plausibility if needed
        if sample['needs_raw'] and sample['raw_sentence']:
            print(f"  Raw sentence: '{sample['raw_sentence']}'")
            raw_p = scorer.calculate(sample['raw_sentence'], debug=True)
            if raw_p is not None:
                raw_eval['plausibility'] = raw_p
                print(f"  -> Raw plausibility: {raw_p:.1f}")
            else:
                print(f"  -> Failed to compute raw plausibility")
                raw_p = 0.0
        else:
            raw_p = raw_eval.get('plausibility', 0.0)

        # Compute LLM plausibility if needed
        if sample['needs_llm'] and sample['llm_sentence']:
            print(f"  LLM sentence: '{sample['llm_sentence']}'")
            llm_p = scorer.calculate(sample['llm_sentence'], debug=True)
            if llm_p is not None:
                llm_eval['plausibility'] = llm_p
                print(f"  -> LLM plausibility: {llm_p:.1f}")
            else:
                print(f"  -> Failed to compute LLM plausibility")
                llm_p = 0.0
        else:
            llm_p = llm_eval.get('plausibility', 0.0)

        # Recalculate CTQI v3 composite scores
        raw_ga = get_effective_ga(raw_eval)
        raw_cf1 = get_coverage_f1(raw_eval)
        raw_composite = compute_ctqi_v3(raw_ga, raw_cf1, raw_p)
        raw_eval['composite'] = raw_composite
        print(f"  -> Raw CTQI v3: {raw_composite:.1f} (GA={raw_ga:.1f}, CF1={raw_cf1:.1f}, P={raw_p:.1f})")

        llm_ga = get_effective_ga(llm_eval)
        llm_cf1 = get_coverage_f1(llm_eval)
        llm_composite = compute_ctqi_v3(llm_ga, llm_cf1, llm_p)
        llm_eval['composite'] = llm_composite
        print(f"  -> LLM CTQI v3: {llm_composite:.1f} (GA={llm_ga:.1f}, CF1={llm_cf1:.1f}, P={llm_p:.1f})")

        # Update metadata
        evaluation['raw'] = raw_eval
        evaluation['llm'] = llm_eval
        precomputed['evaluation'] = evaluation
        metadata['precomputed'] = precomputed

        # Save updated metadata
        with open(sample['path'], 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"  -> Saved to {sample['path'].name}")

    print("\n" + "=" * 60)
    print(f"Updated {len(samples_to_update)} samples successfully.")
    print("=" * 60)


if __name__ == '__main__':
    main()
