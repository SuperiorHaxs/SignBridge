#!/usr/bin/env python3
"""
compute_gloss_embeddings.py

Extract embeddings from a trained model for both existing and candidate glosses.
Uses the model's learned representations (not handcrafted features) for selection.

Usage:
    python compute_gloss_embeddings.py --model-dir <path_to_model> --num-classes 100
    python compute_gloss_embeddings.py --model-dir ./models/wlasl_100_class_model -n 100 --output embeddings.json

Output:
    - embeddings.json with per-gloss mean embeddings
"""

import os
os.environ['PYTORCH_DISABLE_ONNX_METADATA'] = '1'
os.environ['TORCH_DYNAMO_DISABLE'] = '1'

import sys
import json
import argparse
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import configuration
from config import get_config

# Add openhands model path
config = get_config()
openhands_src = config.openhands_dir / "src"
sys.path.insert(0, str(openhands_src))

from openhands_modernized import (
    OpenHandsModel, OpenHandsConfig, WLASLOpenHandsDataset
)


def extract_cls_embedding(model, pose_data, attention_mask, finger_features=None):
    """
    Extract CLS token embedding from the model (before classifier).

    The OpenHandsModel architecture:
    1. PoseFlattener: (N, 3, T, 83) -> (N, T, 249)
    2. Concat finger features: (N, T, 279)
    3. Transformer: (N, T, 279) -> (N, T+1, hidden_size)
    4. CLS token extraction: hidden_states[:, 0, :] -> (N, hidden_size)
    5. Classifier: (N, hidden_size) -> (N, vocab_size)

    We extract at step 4 (before classifier).

    Args:
        model: OpenHandsModel
        pose_data: (batch, seq_len, 83, 3) tensor
        attention_mask: (batch, seq_len) tensor
        finger_features: optional (batch, seq_len, 30) tensor

    Returns:
        (batch, hidden_size) CLS token embeddings
    """
    with torch.no_grad():
        # Reshape for pose flattener: (N, T, 83, 3) -> (N, 3, T, 83)
        batch_size, seq_len = pose_data.shape[:2]
        pose_reshaped = pose_data.permute(0, 3, 1, 2)  # (N, 3, T, 83)

        # Flatten pose features
        pose_features = model.pose_flattener(pose_reshaped)  # (N, T, 249)

        # Add finger features if provided
        if finger_features is not None and model.config.use_finger_features:
            pose_features = torch.cat([pose_features, finger_features], dim=-1)  # (N, T, 279)

        # Get transformer hidden states
        hidden_states = model.transformer(pose_features, attention_mask)  # (N, T+1, hidden_size)

        # Extract CLS token (first position)
        cls_embedding = hidden_states[:, 0, :]  # (N, hidden_size)

        return cls_embedding


def load_model_from_dir(model_dir: Path):
    """Load trained model from directory."""
    model_dir = Path(model_dir)

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    model_config = OpenHandsConfig(
        num_pose_keypoints=config_dict.get('num_pose_keypoints', 83),
        pose_channels=config_dict.get('pose_channels', 3),
        pose_features=config_dict.get('pose_features', 279),
        use_finger_features=config_dict.get('use_finger_features', True),
        finger_features=config_dict.get('finger_features', 30),
        hidden_size=config_dict.get('hidden_size', 256),
        num_hidden_layers=config_dict.get('num_hidden_layers', 6),
        num_attention_heads=config_dict.get('num_attention_heads', 16),
        intermediate_size=config_dict.get('intermediate_size', 1024),
        max_position_embeddings=config_dict.get('max_position_embeddings', 257),
        dropout_prob=config_dict.get('dropout_prob', 0.2),
        vocab_size=config_dict.get('vocab_size', 100),
        use_cls_token=config_dict.get('use_cls_token', True)
    )

    mapping_path = model_dir / "class_index_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Class mapping not found: {mapping_path}")

    with open(mapping_path, 'r') as f:
        id_to_gloss = json.load(f)

    model = OpenHandsModel(model_config)

    weights_path = model_dir / "pytorch_model.bin"
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model from {model_dir}")
    print(f"  Classes: {model_config.vocab_size}")
    print(f"  Hidden size: {model_config.hidden_size}")

    return model, id_to_gloss, model_config


def get_all_available_glosses(pickle_pool: Path):
    """Get list of all available glosses in the pickle pool."""
    glosses = []
    for gloss_dir in pickle_pool.iterdir():
        if gloss_dir.is_dir():
            # Check if there are any pickle files
            pickle_files = list(gloss_dir.glob("*.pkl"))
            if pickle_files:
                glosses.append(gloss_dir.name.upper())
    return sorted(glosses)


def load_gloss_samples(gloss: str, pickle_pool: Path, max_samples: int = 50):
    """Load pose samples for a specific gloss."""
    gloss_dir = pickle_pool / gloss.lower()
    if not gloss_dir.exists():
        return []

    samples = []
    pickle_files = list(gloss_dir.glob("*.pkl"))[:max_samples]

    for pkl_file in pickle_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            samples.append({
                'file_path': str(pkl_file),
                'data': data
            })
        except Exception as e:
            print(f"  Warning: Failed to load {pkl_file}: {e}")

    return samples


def compute_gloss_embedding(model, samples, gloss, max_seq_length, device, use_finger_features=True):
    """Compute mean embedding for a gloss from its samples."""
    if not samples:
        return None, 0

    # Create temporary dataset
    file_paths = [s['file_path'] for s in samples]
    labels = [gloss] * len(samples)

    # Use a dummy gloss_to_id that maps this gloss to 0
    temp_gloss_to_id = {gloss: 0}

    try:
        dataset = WLASLOpenHandsDataset(
            file_paths, labels, temp_gloss_to_id,
            max_seq_length, augment=False, use_finger_features=use_finger_features
        )
    except Exception as e:
        print(f"  Warning: Failed to create dataset for {gloss}: {e}")
        return None, 0

    if len(dataset) == 0:
        return None, 0

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    all_embeddings = []
    model.eval()

    for batch in loader:
        pose_sequences = batch['pose_sequence'].to(device)
        attention_masks = batch['attention_mask'].to(device)
        finger_features = batch.get('finger_features')
        if finger_features is not None:
            finger_features = finger_features.to(device)

        # Extract CLS embeddings directly
        cls_embeddings = extract_cls_embedding(model, pose_sequences, attention_masks, finger_features)
        all_embeddings.append(cls_embeddings.cpu())

    if not all_embeddings:
        return None, 0

    # Concatenate and compute mean
    all_embeddings = torch.cat(all_embeddings, dim=0)
    mean_embedding = all_embeddings.mean(dim=0)
    std_embedding = all_embeddings.std().item()

    return {
        'embedding': mean_embedding.numpy().tolist(),
        'std': std_embedding,
        'n_samples': len(all_embeddings)
    }, len(all_embeddings)


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings for glosses using trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--model-dir", "-m", type=Path, required=True,
                        help="Path to model directory")
    parser.add_argument("--num-classes", "-n", type=int, required=True,
                        help="Number of classes model was trained on")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output path for embeddings (default: gloss_embeddings.json)")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Max samples per gloss for embedding computation (default: 50)")
    parser.add_argument("--max-seq-length", type=int, default=256,
                        help="Maximum sequence length (default: 256)")
    parser.add_argument("--include-existing", action="store_true",
                        help="Also compute embeddings for existing model classes")

    args = parser.parse_args()

    if args.output is None:
        args.output = Path(__file__).parent / "gloss_embeddings.json"

    print(f"Computing embeddings using model: {args.model_dir}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model, id_to_gloss, model_config = load_model_from_dir(args.model_dir)
    model = model.to(device)

    # Get existing glosses from model
    existing_glosses = set(g.upper() for g in id_to_gloss.values())
    print(f"Model has {len(existing_glosses)} existing classes")

    # Get config for pickle pool path
    proj_config = get_config()

    if args.num_classes not in proj_config.dataset_splits:
        raise ValueError(f"No dataset config for {args.num_classes} classes")

    splits = proj_config.dataset_splits[args.num_classes]
    pickle_pool = Path(splits.get('pickle_pool', ''))

    if not pickle_pool.exists():
        raise FileNotFoundError(f"Pickle pool not found: {pickle_pool}")

    # Get all available glosses
    all_glosses = get_all_available_glosses(pickle_pool)
    print(f"Found {len(all_glosses)} total glosses in pickle pool")

    # Determine which glosses to compute embeddings for
    if args.include_existing:
        glosses_to_embed = all_glosses
    else:
        # Only candidate glosses (not in current model)
        glosses_to_embed = [g for g in all_glosses if g not in existing_glosses]

    print(f"Computing embeddings for {len(glosses_to_embed)} glosses")

    # Compute embeddings for candidate glosses
    embeddings_dict = {}

    for i, gloss in enumerate(glosses_to_embed):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Processing {i + 1}/{len(glosses_to_embed)}: {gloss}")

        samples = load_gloss_samples(gloss, pickle_pool, args.max_samples)
        if not samples:
            print(f"  Warning: No samples found for {gloss}")
            continue

        result, n_samples = compute_gloss_embedding(
            model, samples, gloss,
            args.max_seq_length, device,
            use_finger_features=model_config.use_finger_features
        )

        if result is not None:
            embeddings_dict[gloss] = {
                'embedding': result['embedding'],
                'std': result['std'],
                'num_samples': n_samples,
                'is_existing': gloss in existing_glosses
            }

    # Also compute embeddings for existing classes if requested
    if args.include_existing:
        print(f"\nComputing embeddings for {len(existing_glosses)} existing classes...")
        for i, gloss in enumerate(sorted(existing_glosses)):
            if gloss in embeddings_dict:
                continue  # Already computed

            if (i + 1) % 20 == 0 or i == 0:
                print(f"  Processing existing {i + 1}/{len(existing_glosses)}: {gloss}")

            samples = load_gloss_samples(gloss, pickle_pool, args.max_samples)
            if not samples:
                continue

            result, n_samples = compute_gloss_embedding(
                model, samples, gloss,
                args.max_seq_length, device,
                use_finger_features=model_config.use_finger_features
            )

            if result is not None:
                embeddings_dict[gloss] = {
                    'embedding': result['embedding'],
                    'std': result['std'],
                    'num_samples': n_samples,
                    'is_existing': True
                }

    # Save embeddings
    output_data = {
        'model_dir': str(args.model_dir),
        'num_model_classes': len(existing_glosses),
        'num_embeddings': len(embeddings_dict),
        'embedding_dim': model_config.hidden_size,
        'embeddings': embeddings_dict
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(embeddings_dict)} embeddings to {args.output}")
    print(f"  Existing classes: {sum(1 for v in embeddings_dict.values() if v['is_existing'])}")
    print(f"  Candidate classes: {sum(1 for v in embeddings_dict.values() if not v['is_existing'])}")

    return 0


if __name__ == "__main__":
    exit(main())
