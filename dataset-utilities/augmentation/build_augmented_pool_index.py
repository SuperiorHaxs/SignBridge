#!/usr/bin/env python3
"""
Build index file for augmented pickle pool.

Creates a JSON index mapping gloss names to lists of pickle filenames,
enabling fast lookup during training without filesystem scans.

Usage:
    python build_augmented_pool_index.py
    python build_augmented_pool_index.py --pickle-dir /path/to/pickles
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config import get_config


def build_index(pickle_dir: Path, output_path: Path = None) -> dict:
    """
    Build an index of all pickle files organized by gloss.

    Args:
        pickle_dir: Directory containing gloss subdirectories with .pkl files
        output_path: Where to save the index (default: pickle_dir/pickle_index.json)

    Returns:
        dict mapping gloss names to lists of pickle filenames
    """
    if output_path is None:
        output_path = pickle_dir / "pickle_index.json"

    print(f"Building index for: {pickle_dir}")
    print(f"Output: {output_path}")

    index = defaultdict(list)
    total_files = 0

    # Iterate through gloss directories
    if not pickle_dir.exists():
        print(f"ERROR: Directory does not exist: {pickle_dir}")
        return {}

    gloss_dirs = sorted([d for d in pickle_dir.iterdir() if d.is_dir()])
    print(f"Found {len(gloss_dirs)} gloss directories")

    for gloss_dir in gloss_dirs:
        gloss_name = gloss_dir.name

        # Find all .pkl files in this gloss directory
        pkl_files = sorted(gloss_dir.glob("*.pkl"))

        if pkl_files:
            # Store just filenames (not full paths) for portability
            index[gloss_name] = [f.name for f in pkl_files]
            total_files += len(pkl_files)

            if len(pkl_files) > 0:
                print(f"  {gloss_name}: {len(pkl_files)} files")

    # Save index
    print(f"\nTotal: {total_files} pickle files across {len(index)} glosses")

    with open(output_path, 'w') as f:
        json.dump(dict(index), f, indent=2, sort_keys=True)

    print(f"Index saved to: {output_path}")

    return dict(index)


def main():
    parser = argparse.ArgumentParser(
        description="Build index for augmented pickle pool"
    )
    parser.add_argument(
        "--pickle-dir",
        type=Path,
        help="Directory containing pickle files (default: from config)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output index file path (default: pickle_dir/pickle_index.json)"
    )

    args = parser.parse_args()

    # Get pickle directory from config if not provided
    if args.pickle_dir:
        pickle_dir = args.pickle_dir
    else:
        config = get_config()
        pickle_dir = config.augmented_pool_pickle

    output_path = args.output

    index = build_index(pickle_dir, output_path)

    if index:
        print(f"\nSUCCESS: Index built with {len(index)} glosses")
        return 0
    else:
        print("\nERROR: Failed to build index")
        return 1


if __name__ == "__main__":
    sys.exit(main())
