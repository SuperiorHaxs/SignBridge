#!/usr/bin/env python3
"""
Create a targeted augmentation manifest.

Takes the full 50x train manifest and produces a new manifest where:
- Targeted classes (confusion pairs): keep all 50 samples (original + augmented)
- All other classes: keep only original files (no augmentation)

Usage:
    python create_targeted_manifest.py
    python create_targeted_manifest.py --targeted WHAT,YEAR,ORANGE,AFRICA,WALK,PLAY,WRONG,DRINK,CHAIR,TIME,CHANGE,WATER
    python create_targeted_manifest.py --targeted WHAT,YEAR --source path/to/train_manifest.json
"""

import argparse
import json
import copy
from pathlib import Path

# Default confusion pair targets (both sides of each pair)
DEFAULT_TARGETED = [
    # Fixable confusion pairs from original-dataset analysis:
    'WHAT', 'YEAR',       # WHAT -> YEAR (43.8%)
    'WHAT', 'CHANGE',     # WHAT -> CHANGE (31.2%)
    'ORANGE', 'AFRICA',   # ORANGE -> AFRICA (43.8%)
    'ORANGE', 'WATER',    # ORANGE -> WATER (25.0%)
    'WALK', 'PLAY',       # WALK -> PLAY (56.2%)
    'WRONG', 'ORANGE',    # WRONG -> ORANGE (50.0%)
    'WRONG', 'DRINK',     # WRONG -> DRINK (37.5%)
    'CHAIR', 'TIME',      # CHAIR -> TIME (25.0%)
    'YEAR', 'HELP',       # YEAR -> HELP (37.5%)
    # Structural pairs (may not help, but included for completeness):
    'COMPUTER', 'SON',    # COMPUTER -> SON (68.8%)
    'NEED', 'BOWLING',    # NEED -> BOWLING (56.2%)
]

DEFAULT_SOURCE = Path(__file__).resolve().parent.parent.parent / \
    'datasets' / 'augmented_pool' / 'splits' / '43_classes' / 'train_manifest.json'


def create_targeted_manifest(source_path, targeted_classes, output_path=None):
    """Create manifest with augmentation only for targeted classes."""
    with open(source_path, 'r') as f:
        manifest = json.load(f)

    targeted_set = set(g.lower() for g in targeted_classes)
    new_manifest = copy.deepcopy(manifest)

    total_original = 0
    total_new = 0
    stats = {}

    for gloss, families in new_manifest['classes'].items():
        old_count = sum(len(fam['files']) for fam in families)

        if gloss.lower() in targeted_set:
            # Keep all files (original + augmented)
            new_count = old_count
            stats[gloss.upper()] = f'{new_count} files (TARGETED - full augmentation)'
        else:
            # Keep only original files (no _aug_)
            for fam in families:
                fam['files'] = [f for f in fam['files'] if '_aug_' not in f]
            new_count = sum(len(fam['files']) for fam in families)
            stats[gloss.upper()] = f'{new_count} files (originals only)'

        total_original += old_count
        total_new += new_count

    # Update totals
    new_manifest['total_samples'] = total_new
    new_manifest['description'] = (
        f'Targeted augmentation manifest. '
        f'{len(targeted_set)} targeted classes at 50x, rest at originals only.'
    )
    new_manifest['targeted_classes'] = sorted(targeted_set)

    # Determine output path
    if output_path is None:
        output_path = source_path.replace('train_manifest.json', 'train_manifest_targeted.json')

    with open(output_path, 'w') as f:
        json.dump(new_manifest, f, indent=2)

    # Print summary
    print(f'Source: {source_path}')
    print(f'Output: {output_path}')
    print(f'Samples: {total_original} -> {total_new} ({total_new - total_original:+d})')
    print(f'Targeted classes ({len(targeted_set)}): {sorted(targeted_set)}')
    print()
    for gloss in sorted(stats.keys()):
        print(f'  {gloss:<18} {stats[gloss]}')

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Create targeted augmentation manifest')
    parser.add_argument('--source', type=str, default=str(DEFAULT_SOURCE),
                        help='Path to source train manifest')
    parser.add_argument('--targeted', type=str, default=None,
                        help='Comma-separated list of targeted classes (default: confusion pairs)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: train_manifest_targeted.json next to source)')
    args = parser.parse_args()

    targeted = [g.strip().upper() for g in args.targeted.split(',')] if args.targeted else DEFAULT_TARGETED
    # Deduplicate
    targeted = list(set(targeted))

    create_targeted_manifest(args.source, targeted, args.output)


if __name__ == '__main__':
    main()
