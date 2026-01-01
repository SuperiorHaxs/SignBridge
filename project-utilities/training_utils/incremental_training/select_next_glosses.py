#!/usr/bin/env python3
"""
select_next_glosses.py

Select the next batch of glosses to add to the model based on:
1. Embedding similarity/distinctiveness from existing classes
2. Sample availability (glosses with more samples are preferred)
3. Accuracy report from previous training (keep high-accuracy classes)

Usage:
    python select_next_glosses.py --embeddings gloss_embeddings.json --accuracy accuracy_report.json
    python select_next_glosses.py -e embeddings.json -a accuracy.json --num-to-add 5

Output:
    - next_glosses.json with recommended glosses to add
    - Prints ranked list of candidate glosses
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from collections import defaultdict

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def compute_distinctiveness_score(candidate_embedding, existing_embeddings):
    """
    Compute how distinct a candidate is from existing classes.

    Returns:
        Tuple of (distinctiveness_score, most_similar_class, max_similarity)

    Higher distinctiveness = more different from existing classes = better
    """
    if not existing_embeddings:
        return 1.0, None, 0.0

    similarities = []
    for gloss, emb in existing_embeddings.items():
        sim = cosine_similarity(candidate_embedding, emb)
        similarities.append((gloss, sim))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    most_similar = similarities[0][0]
    max_sim = similarities[0][1]

    # Distinctiveness is inverse of max similarity
    # We want candidates that are NOT too similar to existing classes
    distinctiveness = 1.0 - max_sim

    return distinctiveness, most_similar, max_sim


def load_accuracy_report(accuracy_path: Path):
    """Load accuracy report and extract keep/drop classes."""
    if not accuracy_path.exists():
        return None

    with open(accuracy_path, 'r') as f:
        report = json.load(f)

    return report


def load_embeddings(embeddings_path: Path):
    """Load embeddings file."""
    with open(embeddings_path, 'r') as f:
        data = json.load(f)
    return data


def rank_candidates(embeddings_data, accuracy_report=None,
                   min_samples=10, existing_classes=None):
    """
    Rank candidate glosses for addition to the model.

    Args:
        embeddings_data: Dict with embeddings for all glosses
        accuracy_report: Optional accuracy report to identify classes to keep
        min_samples: Minimum samples required for a candidate
        existing_classes: Set of existing class names (if not using accuracy report)

    Returns:
        List of (gloss, score_dict) tuples, sorted by recommendation score
    """
    embeddings = embeddings_data['embeddings']

    # Determine which classes are "existing" (in current model)
    if accuracy_report:
        keep_classes = set(g.upper() for g in accuracy_report['recommendations']['keep_classes'])
        drop_classes = set(g.upper() for g in accuracy_report['recommendations']['drop_classes'])
        all_existing = keep_classes | drop_classes
    elif existing_classes:
        keep_classes = existing_classes
        drop_classes = set()
        all_existing = existing_classes
    else:
        # Use 'is_existing' flag from embeddings
        keep_classes = set(g for g, data in embeddings.items() if data.get('is_existing', False))
        drop_classes = set()
        all_existing = keep_classes

    print(f"Existing classes to keep: {len(keep_classes)}")
    print(f"Existing classes to drop: {len(drop_classes)}")

    # Get embeddings for classes to keep
    keep_embeddings = {}
    for gloss in keep_classes:
        if gloss in embeddings:
            keep_embeddings[gloss] = embeddings[gloss]['embedding']

    print(f"Have embeddings for {len(keep_embeddings)} classes to keep")

    # Find candidate glosses (not in existing model)
    candidates = []
    for gloss, data in embeddings.items():
        if gloss not in all_existing:
            if data['num_samples'] >= min_samples:
                candidates.append((gloss, data))
            else:
                pass  # Skip glosses with too few samples

    print(f"Found {len(candidates)} candidate glosses with >= {min_samples} samples")

    # Score each candidate
    scored_candidates = []
    for gloss, data in candidates:
        embedding = data['embedding']
        num_samples = data['num_samples']

        # Compute distinctiveness from existing classes
        distinctiveness, most_similar, max_similarity = compute_distinctiveness_score(
            embedding, keep_embeddings
        )

        # Normalize sample count (log scale, capped)
        sample_score = min(np.log10(num_samples + 1) / np.log10(100), 1.0)

        # Combined score: balance distinctiveness and sample availability
        # Weight distinctiveness more heavily (0.7) vs samples (0.3)
        combined_score = 0.7 * distinctiveness + 0.3 * sample_score

        scored_candidates.append({
            'gloss': gloss,
            'distinctiveness': round(distinctiveness, 4),
            'most_similar_to': most_similar,
            'max_similarity': round(max_similarity, 4),
            'num_samples': num_samples,
            'sample_score': round(sample_score, 4),
            'combined_score': round(combined_score, 4)
        })

    # Sort by combined score (descending)
    scored_candidates.sort(key=lambda x: x['combined_score'], reverse=True)

    return scored_candidates, keep_classes, drop_classes


def generate_next_gloss_list(scored_candidates, keep_classes, num_to_add: int,
                             output_path: Path):
    """
    Generate the list of glosses for next training iteration.

    Args:
        scored_candidates: Ranked list of candidate glosses
        keep_classes: Set of existing classes to keep
        num_to_add: Number of new glosses to add
        output_path: Path to save the output
    """
    # Select top candidates
    selected = scored_candidates[:num_to_add]

    # Combine with existing classes to keep
    all_glosses = sorted(keep_classes) + [c['gloss'] for c in selected]

    result = {
        'timestamp': datetime.now().isoformat(),
        'total_classes': len(all_glosses),
        'existing_kept': len(keep_classes),
        'new_added': len(selected),
        'glosses': sorted(all_glosses),
        'new_glosses': [c['gloss'] for c in selected],
        'new_gloss_details': selected,
        'all_candidates_ranked': scored_candidates
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


def print_summary(result, scored_candidates):
    """Print human-readable summary."""
    print("\n" + "="*70)
    print("NEXT GLOSS SELECTION SUMMARY")
    print("="*70)

    print(f"\nTotal classes for next training: {result['total_classes']}")
    print(f"  Existing classes kept: {result['existing_kept']}")
    print(f"  New classes to add: {result['new_added']}")

    print(f"\n{'='*70}")
    print("NEW GLOSSES TO ADD (Ranked by distinctiveness)")
    print("="*70)
    for item in result['new_gloss_details']:
        print(f"  {item['gloss']:<20} score={item['combined_score']:.3f} "
              f"distinct={item['distinctiveness']:.3f} "
              f"samples={item['num_samples']} "
              f"similar_to={item['most_similar_to']}")

    print(f"\n{'='*70}")
    print("TOP 20 REMAINING CANDIDATES")
    print("="*70)
    remaining = [c for c in scored_candidates if c['gloss'] not in
                 [x['gloss'] for x in result['new_gloss_details']]][:20]
    for item in remaining:
        print(f"  {item['gloss']:<20} score={item['combined_score']:.3f} "
              f"distinct={item['distinctiveness']:.3f} "
              f"samples={item['num_samples']}")

    print(f"\n{'='*70}")
    print("GLOSS LIST FOR TRAINING")
    print("="*70)
    glosses = result['glosses']
    # Print in columns
    cols = 5
    for i in range(0, len(glosses), cols):
        row = glosses[i:i+cols]
        print("  " + "  ".join(f"{g:<15}" for g in row))


def main():
    parser = argparse.ArgumentParser(
        description="Select next glosses to add based on embeddings and accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--embeddings", "-e", type=Path, required=True,
                        help="Path to gloss embeddings JSON file")
    parser.add_argument("--accuracy", "-a", type=Path, default=None,
                        help="Path to accuracy report JSON file (optional)")
    parser.add_argument("--num-to-add", "-n", type=int, default=5,
                        help="Number of new glosses to add (default: 5)")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="Minimum samples required per gloss (default: 10)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output path for next gloss list")

    args = parser.parse_args()

    if args.output is None:
        args.output = Path(__file__).parent / "next_glosses.json"

    print(f"Loading embeddings from: {args.embeddings}")

    # Load embeddings
    embeddings_data = load_embeddings(args.embeddings)
    print(f"Loaded {embeddings_data['num_embeddings']} embeddings")

    # Load accuracy report if provided
    accuracy_report = None
    if args.accuracy and args.accuracy.exists():
        print(f"Loading accuracy report from: {args.accuracy}")
        accuracy_report = load_accuracy_report(args.accuracy)
        if accuracy_report:
            print(f"  Keep classes: {len(accuracy_report['recommendations']['keep_classes'])}")
            print(f"  Drop classes: {len(accuracy_report['recommendations']['drop_classes'])}")

    # Rank candidates
    print("\nRanking candidate glosses...")
    scored_candidates, keep_classes, drop_classes = rank_candidates(
        embeddings_data, accuracy_report, args.min_samples
    )

    if not scored_candidates:
        print("No candidate glosses found!")
        return 1

    # Generate next gloss list
    result = generate_next_gloss_list(
        scored_candidates, keep_classes, args.num_to_add, args.output
    )

    # Print summary
    print_summary(result, scored_candidates)

    print(f"\nSaved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
