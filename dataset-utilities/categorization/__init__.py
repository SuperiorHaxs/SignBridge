"""
Gloss Categorization Module

Provides semantic categorization for ASL glosses.
Supports dynamic filtering for any model size (50, 100, 125+ classes).

Usage:
    from categorization import GlossCategorizer

    # Load master categorization (all WLASL glosses)
    categorizer = GlossCategorizer.load_master()

    # Filter to specific model's glosses
    my_glosses = ["cook", "dance", "book", ...]
    filtered_categories = categorizer.filter_by_glosses(my_glosses)

    # Or filter by model's class mapping file
    filtered_categories = categorizer.filter_by_model("path/to/class_index_mapping.json")
"""

from .gloss_categorizer import GlossCategorizer

__all__ = ["GlossCategorizer"]
