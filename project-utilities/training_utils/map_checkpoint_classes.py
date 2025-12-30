#!/usr/bin/env python3
"""
map_checkpoint_classes.py
Maps checkpoint class IDs to actual English gloss names
"""

import os
import sys
import torch
import pickle
import json
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_training_vocabulary():
    """Rebuild the training vocabulary from the same files used during training"""

    # Load checkpoint to get training files
    checkpoint_path = "./checkpoints/checkpoint_latest.pt"
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return None

    print("LOADING: Loading checkpoint to get training file list...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    train_files = checkpoint.get('train_files', [])

    if not train_files:
        print("ERROR: No training files found in checkpoint")
        return None

    print(f"SUCCESS: Found {len(train_files)} training files")

    # Extract glosses from training files (same as training process)
    print("EXTRACTING: Extracting glosses from training files...")
    labels = []

    for i, file_path in enumerate(train_files):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(train_files)} files...")

        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'gloss' in data:
                    labels.append(data['gloss'].upper())
                elif hasattr(data, '__len__') and len(data) >= 2:
                    labels.append(str(data[1]).upper())
        except Exception as e:
            continue

    print(f"SUCCESS: Extracted {len(labels)} labels")

    # Create vocabulary mapping (same as training)
    unique_glosses = sorted(set(labels))
    gloss_to_id = {gloss: i for i, gloss in enumerate(unique_glosses)}
    id_to_gloss = {i: gloss for i, gloss in enumerate(unique_glosses)}

    print(f"SUCCESS: Created vocabulary with {len(unique_glosses)} unique glosses")
    print(f"SAMPLE: First 10 glosses: {unique_glosses[:10]}")

    return id_to_gloss, gloss_to_id

def map_class_ids(class_ids):
    """Map class IDs to English gloss names"""

    print("MAPPING: Loading training vocabulary...")
    result = load_training_vocabulary()
    if result is None:
        return None

    id_to_gloss, gloss_to_id = result

    print(f"\nMAPPING: Mapping class IDs to English glosses...")
    print("="*50)

    mappings = {}
    for class_id in class_ids:
        if class_id in id_to_gloss:
            gloss = id_to_gloss[class_id]
            mappings[class_id] = gloss
            print(f"  CLASS_{class_id} -> '{gloss}'")
        else:
            mappings[class_id] = f"UNKNOWN_CLASS_{class_id}"
            print(f"  CLASS_{class_id} -> UNKNOWN (vocab size: {len(id_to_gloss)})")

    return mappings

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Map checkpoint class IDs to English glosses")
    parser.add_argument("class_ids", nargs="+", type=int, help="Class IDs to map (e.g., 145 348)")

    args = parser.parse_args()

    print("Checkpoint Class ID to English Gloss Mapper")
    print("="*50)
    print(f"INPUT: Class IDs to map: {args.class_ids}")
    print()

    mappings = map_class_ids(args.class_ids)

    if mappings:
        print(f"\nSUCCESS: Mapping Results:")
        print("="*30)
        for class_id, gloss in mappings.items():
            print(f"  CLASS_{class_id} = '{gloss}'")
    else:
        print("ERROR: Failed to create mappings")

if __name__ == "__main__":
    main()