#!/usr/bin/env python3
"""
Find which signer has the most videos on disk for a given model's gloss list,
and list video IDs for any specific signer.

Usage:
  # Rank signers by coverage for a model:
  python find_signer_videos.py --model <model_dir>

  # Show top N signers (default 10):
  python find_signer_videos.py --model <model_dir> --top 15

  # List video IDs for a specific signer:
  python find_signer_videos.py --model <model_dir> --signer 11

  # Both at once:
  python find_signer_videos.py --model <model_dir> --signer 11 --top 5

  # Override default paths:
  python find_signer_videos.py --model <model_dir> --metadata /path/to/WLASL_v0.3.json --videos /path/to/videos
"""

import argparse
import json
import os
from collections import defaultdict


def load_model_glosses(model_dir):
    """Load gloss list from a model's class_index_mapping.json."""
    mapping_path = os.path.join(model_dir, "class_index_mapping.json")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"No class_index_mapping.json in {model_dir}")

    with open(mapping_path) as f:
        mapping = json.load(f)

    glosses = {v.lower() for v in mapping.values()}
    print(f"Model: {os.path.basename(model_dir)}")
    print(f"Total glosses: {len(glosses)}")

    # Check for masked classes
    mask_path = os.path.join(model_dir, "masked_classes.json")
    if os.path.exists(mask_path):
        with open(mask_path) as f:
            mask = json.load(f)
        masked = set(n.lower() for n in mask.get("masked_class_names", []))
        print(f"Masked glosses: {len(masked)}")
        print(f"Effective glosses: {len(glosses) - len(masked)}")
    else:
        masked = set()

    return glosses, masked


def build_signer_index(metadata_path, videos_dir, glosses):
    """Build signer -> {gloss: video_id} index for videos that exist on disk."""
    with open(metadata_path) as f:
        data = json.load(f)

    signer_data = defaultdict(dict)
    for entry in data:
        gloss = entry.get("gloss", "").lower()
        if gloss not in glosses:
            continue
        for inst in entry.get("instances", []):
            vid = inst.get("video_id")
            sid = inst.get("signer_id")
            path = os.path.join(videos_dir, gloss, f"{vid}.mp4")
            if os.path.exists(path) and gloss not in signer_data[sid]:
                signer_data[sid][gloss] = vid

    return signer_data


def rank_signers(signer_data, glosses, top_n=10):
    """Rank signers by number of glosses covered."""
    results = []
    for sid, found in signer_data.items():
        results.append((sid, len(found), sorted(found.keys())))

    results.sort(key=lambda x: (-x[1], x[0]))

    total = len(glosses)
    print(f"\n{'='*60}")
    print(f"Top {min(top_n, len(results))} signers by coverage (out of {len(results)} total)")
    print(f"{'='*60}")

    for sid, count, signs in results[:top_n]:
        missing = sorted(glosses - set(signs))
        pct = count / total * 100
        print(f"\nSigner {sid:3d}: {count}/{total} signs ({pct:.0f}%)")
        if len(missing) <= 15:
            print(f"  Missing ({total - count}): {missing}")
        else:
            print(f"  Missing ({total - count}): {missing[:10]} + {total - count - 10} more...")

    return results


def show_signer_videos(signer_data, glosses, signer_id):
    """Show all video IDs for a specific signer."""
    if signer_id not in signer_data:
        print(f"\nSigner {signer_id}: no videos found on disk for these glosses.")
        return

    found = signer_data[signer_id]
    missing = sorted(glosses - set(found.keys()))
    total = len(glosses)

    print(f"\n{'='*60}")
    print(f"Signer {signer_id}: {len(found)}/{total} signs on disk")
    print(f"{'='*60}")
    print()

    for g in sorted(found.keys()):
        print(f"  {g:20s} -> {found[g]}")

    if missing:
        print(f"\n  Missing ({len(missing)}): {missing}")

        # Find fill-in candidates from other signers
        print(f"\n  Fill-in options from other signers:")
        for m in missing:
            candidates = []
            for sid, sg in signer_data.items():
                if sid != signer_id and m in sg:
                    candidates.append((sid, sg[m]))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                best = candidates[0]
                others = f" (+{len(candidates)-1} more)" if len(candidates) > 1 else ""
                print(f"    {m:20s} -> {best[1]} (signer {best[0]}){others}")
            else:
                print(f"    {m:20s} -> NO VIDEO ON DISK")


def main():
    parser = argparse.ArgumentParser(
        description="Find signers with max video coverage for a model's gloss list"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to model directory (must contain class_index_mapping.json)"
    )
    parser.add_argument(
        "--signer", type=int, default=None,
        help="Show video IDs for this specific signer"
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Number of top signers to show in ranking (default: 10)"
    )
    parser.add_argument(
        "--metadata", default="D:/Projects/WLASL/datasets/wlasl-kaggle/WLASL_v0.3.json",
        help="Path to WLASL_v0.3.json metadata file"
    )
    parser.add_argument(
        "--videos", default="D:/Projects/WLASL/datasets/wlasl-kaggle/videos",
        help="Path to videos directory"
    )

    args = parser.parse_args()

    glosses, masked = load_model_glosses(args.model)

    print(f"\nScanning videos in: {args.videos}")
    signer_data = build_signer_index(args.metadata, args.videos, glosses)
    print(f"Found {len(signer_data)} signers with at least 1 video on disk")

    if args.signer is None:
        # Rank mode — show top signers
        rank_signers(signer_data, glosses, args.top)
    else:
        # Show specific signer
        show_signer_videos(signer_data, glosses, args.signer)


if __name__ == "__main__":
    main()
