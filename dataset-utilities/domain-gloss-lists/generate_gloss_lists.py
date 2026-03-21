#!/usr/bin/env python3
"""
Generate Domain-Specific Gloss Lists

Main entry point that:
1. Parses SignBridge app to discover domains
2. For each domain: filters WLASL 2000 to relevant candidates
3. Computes pose-based distance matrices
4. Uses farthest-point sampling to select maximally diverse subsets
5. Outputs model-compatible JSON files

Output format is directly usable by train_asl.py and inference API:
- "classes": list of glosses (for class_mapping)
- "class_to_index": gloss -> index mapping (for training)
- "class_index_mapping": index -> gloss mapping (for inference/production)

Usage:
    python generate_gloss_lists.py
    python generate_gloss_lists.py --common-count 30 --domain-count 40
    python generate_gloss_lists.py --output-dir datasets/domain-specific
    python generate_gloss_lists.py --no-pose-diversity  # skip pose analysis, use all candidates
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import date
from typing import Dict, List

# Add project paths
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(project_root))

from domain_config import (
    PROJECT_ROOT,
    SIGNBRIDGE_TEMPLATE,
    WLASL_CLASS_INDEX,
    PICKLE_DIR,
    OUTPUT_DIR,
    DEFAULT_COMMON_COUNT,
    DEFAULT_DOMAIN_COUNT,
    FORCED_COMMON,
    parse_signbridge_domains,
    load_wlasl_glosses,
    get_validated_candidates,
    get_validated_common_candidates,
)
from pose_distance import load_gloss_centroids, compute_distance_matrix
from select_diverse_subset import farthest_point_sampling


def build_gloss_list_json(
    glosses: List[str],
    domain: str,
    scenario: str,
    metadata: dict,
) -> dict:
    """
    Build a model-compatible gloss list JSON structure.

    Contains both training format (class_to_index) and
    inference format (class_index_mapping).
    """
    sorted_glosses = sorted(glosses)

    class_to_index = {g: i for i, g in enumerate(sorted_glosses)}
    class_index_mapping = {str(i): g.lower() for i, g in enumerate(sorted_glosses)}

    return {
        "domain": domain,
        "scenario": scenario,
        "num_classes": len(sorted_glosses),
        "classes": sorted_glosses,
        "class_to_index": class_to_index,
        "class_index_mapping": class_index_mapping,
        "metadata": metadata,
    }


def generate_domain_list(
    domain_info: dict,
    wlasl_glosses: set,
    target_count: int,
    centroids: Dict[str, 'np.ndarray'],
    use_pose_diversity: bool = True,
    exclude_glosses: set = None,
) -> dict:
    """Generate a single domain's gloss list.

    Args:
        exclude_glosses: Glosses to exclude (e.g. common list words).
            These are removed from the candidate pool before selection.
    """
    scenario = domain_info["scenario"]
    list_name = domain_info["list_name"]
    print(f"\n{'='*60}")
    print(f"Domain: {scenario}")
    print(f"{'='*60}")

    # Get validated candidates, then remove excluded glosses
    candidates = get_validated_candidates(scenario, wlasl_glosses)
    if exclude_glosses:
        before = len(candidates)
        candidates = [g for g in candidates if g not in exclude_glosses]
        removed = before - len(candidates)
        if removed:
            print(f"  Excluded {removed} glosses (reserved for common list)")
    print(f"  Valid candidates: {len(candidates)}")

    if len(candidates) <= target_count:
        print(f"  Using all {len(candidates)} candidates (fewer than target {target_count})")
        selected = candidates
        metrics = {
            "selection_method": "all_candidates",
            "candidate_pool_size": len(candidates),
            "selected_count": len(selected),
        }
    elif not use_pose_diversity:
        # Just take first N alphabetically
        selected = sorted(candidates)[:target_count]
        metrics = {
            "selection_method": "alphabetical_truncation",
            "candidate_pool_size": len(candidates),
            "selected_count": len(selected),
        }
    else:
        # Compute distance matrix for this domain's candidates
        from pose_distance import compute_distance_matrix
        dist_matrix, valid_glosses = compute_distance_matrix(candidates, centroids)

        if len(valid_glosses) <= target_count:
            print(f"  Only {len(valid_glosses)} glosses have pose data, using all")
            selected = valid_glosses
            metrics = {
                "selection_method": "all_with_pose_data",
                "candidate_pool_size": len(candidates),
                "selected_count": len(selected),
            }
        else:
            print(f"  Running farthest-point sampling: {len(valid_glosses)} -> {target_count}")
            selected, metrics = farthest_point_sampling(
                dist_matrix, valid_glosses, target_count
            )
            metrics["selection_method"] = "farthest_point_sampling"
            print(f"  Avg pairwise distance: {metrics['avg_pairwise_distance']:.4f}")
            print(f"  Min pairwise distance: {metrics['min_pairwise_distance']:.4f}")
            print(f"  Closest pair: {metrics.get('closest_pair', 'N/A')}")

    metrics["generated_at"] = str(date.today())
    metrics["source"] = "WLASL2000"

    print(f"  Selected {len(selected)} glosses: {sorted(selected)[:10]}...")

    return build_gloss_list_json(
        selected,
        domain=domain_info["domain_key"],
        scenario=scenario,
        metadata=metrics,
    )


def generate_common_list(
    wlasl_glosses: set,
    target_count: int,
    centroids: Dict[str, 'np.ndarray'],
    use_pose_diversity: bool = True,
) -> dict:
    """Generate the common everyday words list.

    Forced glosses (FORCED_COMMON) are always included. The remaining
    slots are filled by diversity sampling from the rest of the pool.
    """
    print(f"\n{'='*60}")
    print(f"Domain: Common Words")
    print(f"{'='*60}")

    candidates = get_validated_common_candidates(wlasl_glosses)
    print(f"  Valid candidates: {len(candidates)}")

    # Force-include FORCED_COMMON glosses that exist in WLASL 2000
    forced = [g for g in FORCED_COMMON if g in wlasl_glosses]
    forced_set = set(forced)
    remaining_candidates = [g for g in candidates if g not in forced_set]
    remaining_slots = max(0, target_count - len(forced))

    print(f"  Forced includes: {len(forced)} ({forced})")
    print(f"  Remaining slots to fill: {remaining_slots}")

    if remaining_slots == 0 or len(remaining_candidates) == 0:
        selected = forced
        metrics = {
            "selection_method": "forced_only",
            "forced_count": len(forced),
            "candidate_pool_size": len(candidates),
            "selected_count": len(forced),
        }
    elif len(remaining_candidates) <= remaining_slots:
        selected = forced + remaining_candidates
        metrics = {
            "selection_method": "forced_plus_all_remaining",
            "forced_count": len(forced),
            "candidate_pool_size": len(candidates),
            "selected_count": len(selected),
        }
    elif not use_pose_diversity:
        selected = forced + sorted(remaining_candidates)[:remaining_slots]
        metrics = {
            "selection_method": "forced_plus_alphabetical",
            "forced_count": len(forced),
            "candidate_pool_size": len(candidates),
            "selected_count": len(selected),
        }
    else:
        from pose_distance import compute_distance_matrix
        dist_matrix, valid_glosses = compute_distance_matrix(remaining_candidates, centroids)

        if len(valid_glosses) <= remaining_slots:
            selected = forced + valid_glosses
            metrics = {
                "selection_method": "forced_plus_all_with_pose_data",
                "forced_count": len(forced),
                "candidate_pool_size": len(candidates),
                "selected_count": len(selected),
            }
        else:
            print(f"  Running farthest-point sampling: {len(valid_glosses)} -> {remaining_slots}")
            sampled, metrics = farthest_point_sampling(
                dist_matrix, valid_glosses, remaining_slots
            )
            selected = forced + sampled
            metrics["selection_method"] = "forced_plus_farthest_point_sampling"
            metrics["forced_count"] = len(forced)
            print(f"  Avg pairwise distance: {metrics['avg_pairwise_distance']:.4f}")
            print(f"  Min pairwise distance: {metrics['min_pairwise_distance']:.4f}")
            print(f"  Closest pair: {metrics.get('closest_pair', 'N/A')}")

    metrics["generated_at"] = str(date.today())
    metrics["source"] = "WLASL2000"
    metrics["forced_glosses"] = forced

    print(f"  Selected {len(selected)} glosses: {sorted(selected)[:10]}...")

    return build_gloss_list_json(
        selected,
        domain="common",
        scenario="Common Words",
        metadata=metrics,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate domain-specific gloss lists from WLASL 2000",
    )
    parser.add_argument(
        "--common-count", type=int, default=DEFAULT_COMMON_COUNT,
        help=f"Number of common words to select (default: {DEFAULT_COMMON_COUNT})",
    )
    parser.add_argument(
        "--domain-count", type=int, default=DEFAULT_DOMAIN_COUNT,
        help=f"Number of words per domain (default: {DEFAULT_DOMAIN_COUNT})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-pose-diversity", action="store_true",
        help="Skip pose distance computation; use all valid candidates (faster)",
    )
    parser.add_argument(
        "--signbridge-template", type=Path, default=SIGNBRIDGE_TEMPLATE,
        help="Path to SignBridge index.html template",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load WLASL 2000 gloss set
    print("Loading WLASL 2000 gloss set...")
    wlasl_glosses = load_wlasl_glosses()
    print(f"  {len(wlasl_glosses)} glosses loaded")

    # 2. Parse SignBridge domains
    print(f"\nParsing SignBridge domains from {args.signbridge_template}...")
    domains = parse_signbridge_domains(args.signbridge_template)
    print(f"  Found {len(domains)} domains:")
    for d in domains:
        print(f"    - {d['scenario']} (key: {d['domain_key']}, file: {d['list_name']}.json)")

    # 3. Collect all candidate glosses across all domains + common
    all_candidates = set()
    all_candidates.update(get_validated_common_candidates(wlasl_glosses))
    for d in domains:
        all_candidates.update(get_validated_candidates(d["scenario"], wlasl_glosses))

    # 4. Load pose centroids (once, for all candidates)
    centroids = {}
    if not args.no_pose_diversity:
        print(f"\nLoading pose centroids for {len(all_candidates)} candidate glosses...")
        video_to_gloss_path = PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "video_to_gloss_mapping.json"
        if not video_to_gloss_path.exists():
            print(f"  WARNING: {video_to_gloss_path} not found, falling back to no-pose-diversity mode")
            args.no_pose_diversity = True
        elif not PICKLE_DIR.exists():
            print(f"  WARNING: {PICKLE_DIR} not found, falling back to no-pose-diversity mode")
            args.no_pose_diversity = True
        else:
            centroids = load_gloss_centroids(
                list(all_candidates), PICKLE_DIR, video_to_gloss_path
            )
            print(f"  Loaded centroids for {len(centroids)} glosses")

    # 5. Generate common words list
    common_result = generate_common_list(
        wlasl_glosses, args.common_count, centroids,
        use_pose_diversity=not args.no_pose_diversity,
    )
    common_path = output_dir / "common_words.json"
    with open(common_path, "w") as f:
        json.dump(common_result, f, indent=2)
    print(f"  -> Saved: {common_path}")

    # 6. Generate domain-specific lists
    #    Exclude common-list glosses from domain lists to avoid overlap
    common_glosses = set(common_result["classes"])
    all_results = {"common_words": common_result}

    for domain_info in domains:
        result = generate_domain_list(
            domain_info, wlasl_glosses, args.domain_count, centroids,
            use_pose_diversity=not args.no_pose_diversity,
            exclude_glosses=common_glosses,
        )
        filename = f"{domain_info['list_name']}.json"
        filepath = output_dir / filename
        with open(filepath, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  -> Saved: {filepath}")
        all_results[domain_info["list_name"]] = result

    # 7. Save generation report
    report = {
        "generated_at": str(date.today()),
        "signbridge_template": str(args.signbridge_template),
        "wlasl_gloss_count": len(wlasl_glosses),
        "common_count_setting": args.common_count,
        "domain_count_setting": args.domain_count,
        "pose_diversity_enabled": not args.no_pose_diversity,
        "centroids_loaded": len(centroids),
        "domains": {},
    }
    for name, result in all_results.items():
        report["domains"][name] = {
            "scenario": result["scenario"],
            "num_classes": result["num_classes"],
            "glosses": result["classes"],
            "metadata": result["metadata"],
        }
    report_path = output_dir / "generation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  -> Report: {report_path}")

    # Summary
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Files generated: {len(all_results) + 1} (incl. report)")
    for name, result in all_results.items():
        print(f"  {name}: {result['num_classes']} glosses")

    return 0


if __name__ == "__main__":
    sys.exit(main())
