#!/usr/bin/env python3
"""
Gloss Categorizer - Categorize ASL glosses into semantic groups.

This module provides semantic categorization for ASL glosses from the WLASL dataset.
It maintains a master categorization file for all glosses and supports dynamic
filtering for any model size (50, 100, 125, 2000+ classes).

CLI Usage:
    # Generate master categorization from WLASL metadata
    python gloss_categorizer.py --generate --input WLASL_v0.3.json

    # Show categories for a specific model's glosses
    python gloss_categorizer.py --filter --model-mapping path/to/class_index_mapping.json

    # List all categories with counts
    python gloss_categorizer.py --list-categories

    # Get category for specific glosses
    python gloss_categorizer.py --lookup cook dance book

Library Usage:
    from categorization import GlossCategorizer

    categorizer = GlossCategorizer.load_master()
    filtered = categorizer.filter_by_glosses(my_model_glosses)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


# Seed categories from the 100-class model
# These are used as the initial categorization when generating the master file
SEED_CATEGORIES = {
    "Actions": ["cook", "dance", "drink", "eat", "finish", "forget", "give", "go", "graduate",
                "help", "kiss", "meet", "play", "study", "wait", "walk", "want", "work"],
    "People": ["cousin", "doctor", "family", "man", "mother", "secretary", "son", "wife", "woman"],
    "Objects": ["bed", "book", "chair", "computer", "hat", "jacket", "letter", "paper",
                "shirt", "table"],
    "Food": ["apple", "candy", "corn", "fish", "pizza", "water"],
    "Colors": ["black", "blue", "brown", "orange", "pink", "purple", "white", "yellow"],
    "Time": ["before", "birthday", "last", "later", "now", "thursday", "time", "year"],
    "Questions": ["what", "who"],
    "Descriptors": ["all", "cool", "dark", "deaf", "fine", "full", "hearing", "hot", "many",
                    "same", "short", "tall", "thin"],
    "Places": ["africa", "city", "school"],
    "Animals": ["bird", "cow", "dog"],
    "Activities": ["basketball", "bowling"],
    "Other": ["accident", "but", "can", "change", "cheat", "clothes", "color", "decide",
              "enjoy", "language", "like", "need", "no", "right", "tell", "thanksgiving",
              "wrong", "yes"]
}


class GlossCategorizer:
    """Categorize glosses into semantic groups."""

    def __init__(self, categories: Dict[str, List[str]], gloss_to_category: Dict[str, str] = None):
        """
        Initialize with category data.

        Args:
            categories: dict mapping category_name -> list of glosses
            gloss_to_category: optional reverse mapping (built if not provided)
        """
        self.categories = categories
        self.gloss_to_category = gloss_to_category or self._build_reverse_mapping()

    def _build_reverse_mapping(self) -> Dict[str, str]:
        """Build gloss -> category reverse mapping."""
        mapping = {}
        for category, glosses in self.categories.items():
            for gloss in glosses:
                mapping[gloss.lower()] = category
        return mapping

    @classmethod
    def load_master(cls) -> "GlossCategorizer":
        """Load the master WLASL categorization file."""
        data_dir = Path(__file__).parent / "data"
        master_file = data_dir / "wlasl_all_categories.json"

        if not master_file.exists():
            raise FileNotFoundError(
                f"Master categorization file not found: {master_file}\n"
                f"Generate it with: python gloss_categorizer.py --generate --input WLASL_v0.3.json"
            )

        return cls.from_json(str(master_file))

    @classmethod
    def from_json(cls, json_path: str) -> "GlossCategorizer":
        """Load categories from a JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        categories = data.get("categories", data)
        gloss_to_category = data.get("gloss_to_category", None)

        return cls(categories, gloss_to_category)

    @classmethod
    def from_seed(cls) -> "GlossCategorizer":
        """Create categorizer from seed categories (100-class model)."""
        return cls(SEED_CATEGORIES.copy())

    def get_category(self, gloss: str) -> str:
        """Get the category for a single gloss. Returns 'Other' if unknown."""
        return self.gloss_to_category.get(gloss.lower(), "Other")

    def get_all_categories(self) -> Dict[str, List[str]]:
        """Get the full category mapping."""
        return self.categories.copy()

    def filter_by_glosses(self, available_glosses: List[str]) -> Dict[str, List[str]]:
        """
        Filter categories to only include glosses from the given list.
        Returns: {category: [glosses]} with only non-empty categories.
        """
        # Normalize to lowercase for matching
        available_set = {g.lower() for g in available_glosses}

        filtered = defaultdict(list)
        for category, glosses in self.categories.items():
            for gloss in glosses:
                if gloss.lower() in available_set:
                    filtered[category].append(gloss)

        # Remove empty categories and convert to regular dict
        return {cat: glosses for cat, glosses in filtered.items() if glosses}

    def filter_by_model(self, class_mapping_path: str) -> Dict[str, List[str]]:
        """
        Filter categories based on a model's class_index_mapping.json.
        Convenience method that loads the mapping and calls filter_by_glosses.
        """
        with open(class_mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)

        # class_index_mapping.json format: {"0": "gloss1", "1": "gloss2", ...}
        glosses = list(mapping.values())
        return self.filter_by_glosses(glosses)

    def get_stats(self) -> Dict:
        """Get statistics: total glosses, glosses per category, etc."""
        stats = {
            "total_categories": len(self.categories),
            "total_glosses": sum(len(g) for g in self.categories.values()),
            "categories": {}
        }

        for category, glosses in sorted(self.categories.items()):
            stats["categories"][category] = len(glosses)

        return stats

    def add_glosses(self, glosses: List[str], category: str = "Other"):
        """Add glosses to a category (used during generation)."""
        if category not in self.categories:
            self.categories[category] = []

        for gloss in glosses:
            gloss_lower = gloss.lower()
            if gloss_lower not in self.gloss_to_category:
                self.categories[category].append(gloss_lower)
                self.gloss_to_category[gloss_lower] = category

    def to_json(self, output_path: str, name: str = None, description: str = None):
        """Save categories to a JSON file."""
        data = {
            "name": name or "WLASL Gloss Categories",
            "description": description or "Semantic categories for WLASL glosses",
            "version": "1.0",
            "total_glosses": sum(len(g) for g in self.categories.values()),
            "total_categories": len(self.categories),
            "categories": {cat: sorted(glosses) for cat, glosses in sorted(self.categories.items())},
            "gloss_to_category": dict(sorted(self.gloss_to_category.items()))
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"Saved categorization to: {output_path}")


def generate_master_categorization(wlasl_metadata_path: str, output_path: str):
    """
    Generate master categorization file from WLASL metadata.

    Reads all glosses from WLASL_v0.3.json and categorizes them:
    - Known glosses (from seed) keep their categories
    - Unknown glosses go to "Other" for manual review
    """
    print(f"Loading WLASL metadata from: {wlasl_metadata_path}")

    with open(wlasl_metadata_path, 'r', encoding='utf-8') as f:
        wlasl_data = json.load(f)

    # Extract all unique glosses
    all_glosses = set()
    for entry in wlasl_data:
        gloss = entry.get('gloss', '').lower()
        if gloss:
            all_glosses.add(gloss)

    print(f"Found {len(all_glosses)} unique glosses in WLASL metadata")

    # Start with seed categories
    categorizer = GlossCategorizer.from_seed()
    categorized_glosses = set(categorizer.gloss_to_category.keys())

    # Find uncategorized glosses
    uncategorized = all_glosses - categorized_glosses
    print(f"Already categorized: {len(categorized_glosses)} glosses")
    print(f"Uncategorized: {len(uncategorized)} glosses")

    # Add uncategorized glosses to "Other"
    categorizer.add_glosses(list(uncategorized), "Other")

    # Save to file
    categorizer.to_json(
        output_path,
        name="WLASL Complete Gloss Categories",
        description="Semantic categories for all glosses in WLASL dataset"
    )

    # Print stats
    stats = categorizer.get_stats()
    print(f"\nGenerated categorization:")
    print(f"  Total categories: {stats['total_categories']}")
    print(f"  Total glosses: {stats['total_glosses']}")
    print(f"\nGlosses per category:")
    for cat, count in stats['categories'].items():
        print(f"  {cat}: {count}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Gloss Categorizer - Categorize ASL glosses into semantic groups",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate master categorization from WLASL metadata
    python gloss_categorizer.py --generate --input path/to/WLASL_v0.3.json

    # Show categories for a specific model
    python gloss_categorizer.py --filter --model-mapping path/to/class_index_mapping.json

    # List all categories
    python gloss_categorizer.py --list-categories

    # Look up categories for specific glosses
    python gloss_categorizer.py --lookup cook dance book chair
        """
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", action="store_true",
                      help="Generate master categorization from WLASL metadata")
    mode.add_argument("--filter", action="store_true",
                      help="Filter categories to specific model's glosses")
    mode.add_argument("--list-categories", action="store_true",
                      help="List all categories and gloss counts")
    mode.add_argument("--lookup", nargs="+", metavar="GLOSS",
                      help="Look up categories for specific glosses")

    # Input/output options
    parser.add_argument("--input", "-i",
                        help="Input file (WLASL metadata JSON for --generate)")
    parser.add_argument("--output", "-o",
                        help="Output JSON file (default: data/wlasl_all_categories.json)")
    parser.add_argument("--model-mapping", "-m",
                        help="Path to model's class_index_mapping.json (for --filter)")

    args = parser.parse_args()

    # Handle each mode
    if args.generate:
        if not args.input:
            parser.error("--generate requires --input WLASL_v0.3.json")

        output = args.output or str(Path(__file__).parent / "data" / "wlasl_all_categories.json")
        generate_master_categorization(args.input, output)

    elif args.filter:
        if not args.model_mapping:
            parser.error("--filter requires --model-mapping path/to/class_index_mapping.json")

        try:
            categorizer = GlossCategorizer.load_master()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        filtered = categorizer.filter_by_model(args.model_mapping)

        # Load model info
        with open(args.model_mapping, 'r') as f:
            mapping = json.load(f)
        num_classes = len(mapping)

        print(f"Categories for {num_classes}-class model:")
        print(f"Model mapping: {args.model_mapping}")
        print("=" * 60)

        total_glosses = 0
        for category, glosses in sorted(filtered.items()):
            print(f"\n{category} ({len(glosses)} glosses):")
            print(f"  {', '.join(sorted(glosses))}")
            total_glosses += len(glosses)

        print(f"\n{'=' * 60}")
        print(f"Total: {len(filtered)} categories, {total_glosses} glosses")

    elif args.list_categories:
        try:
            categorizer = GlossCategorizer.load_master()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        stats = categorizer.get_stats()
        print("WLASL Gloss Categories")
        print("=" * 60)
        print(f"Total categories: {stats['total_categories']}")
        print(f"Total glosses: {stats['total_glosses']}")
        print("\nGlosses per category:")
        for cat, count in sorted(stats['categories'].items(), key=lambda x: -x[1]):
            print(f"  {cat:20s}: {count:5d} glosses")

    elif args.lookup:
        try:
            categorizer = GlossCategorizer.load_master()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        print("Gloss Categories:")
        print("-" * 40)
        for gloss in args.lookup:
            category = categorizer.get_category(gloss)
            print(f"  {gloss:20s} -> {category}")


if __name__ == "__main__":
    main()
