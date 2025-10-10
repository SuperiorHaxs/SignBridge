#!/usr/bin/env python3
"""
openhands_inference.py

Inference functions for OpenHands model.
Provides load_model_from_checkpoint and predict_pose_file functions.
"""

import torch
import json
from pathlib import Path
from openhands_modernized import OpenHandsConfig, OpenHandsModel, WLASLPoseProcessor


def load_model_from_checkpoint(checkpoint_path: str, vocab_size: int = 20):
    """
    Load OpenHands model from checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory containing pytorch_model.bin
        vocab_size: Number of classes (default 20)

    Returns:
        tuple: (model, id_to_gloss_mapping)
    """
    checkpoint_path = Path(checkpoint_path)

    # Load model weights
    model_file = checkpoint_path / "pytorch_model.bin"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    # Load vocabulary mapping
    vocab_file = checkpoint_path / "class_index_mapping.json"
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")

    with open(vocab_file, 'r') as f:
        id_to_gloss = json.load(f)

    # Create model config
    config = OpenHandsConfig(vocab_size=vocab_size)
    model = OpenHandsModel(config)

    # Load weights
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    print(f"SUCCESS: Loaded model from {checkpoint_path}")
    print(f"VOCAB: {vocab_size} classes")

    return model, id_to_gloss


def predict_pose_file(pickle_path: str, model=None, tokenizer=None, checkpoint_path: str = None):
    """
    Predict gloss from a pickle pose file.

    Args:
        pickle_path: Path to pickle file containing pose data
        model: Pre-loaded model (optional)
        tokenizer: Vocabulary mapping (id_to_gloss dict)
        checkpoint_path: Path to checkpoint if model not provided

    Returns:
        dict: Prediction results with 'gloss', 'confidence', 'top_k_predictions'
    """
    # Load model if not provided
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Either model or checkpoint_path must be provided")
        model, tokenizer = load_model_from_checkpoint(checkpoint_path)

    # Load and process pose data
    processor = WLASLPoseProcessor()
    pose_sequence = processor.load_pickle_pose(pickle_path)
    pose_sequence = processor.preprocess_pose_sequence(pose_sequence, augment=False)
    pose_sequence, attention_mask = processor.pad_or_truncate_sequence(pose_sequence, max_length=256)

    # Convert to tensors and add batch dimension
    pose_tensor = torch.tensor(pose_sequence, dtype=torch.float32).unsqueeze(0)  # (1, T, 75, 2)
    mask_tensor = torch.tensor(attention_mask, dtype=torch.float32).unsqueeze(0)  # (1, T)

    # Predict
    with torch.no_grad():
        logits = model(pose_tensor, mask_tensor)  # (1, vocab_size)
        probs = torch.softmax(logits, dim=-1)

        # Get top prediction
        confidence, pred_id = torch.max(probs, dim=-1)
        pred_id = pred_id.item()
        confidence = confidence.item()

        # Get top-k predictions
        top_k = 5
        top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)

        top_k_predictions = []
        for prob, idx in zip(top_probs[0], top_ids[0]):
            gloss = tokenizer.get(str(idx), f"UNKNOWN_{idx}")
            top_k_predictions.append({
                'gloss': gloss,
                'confidence': prob.item()
            })

    # Get predicted gloss
    predicted_gloss = tokenizer.get(str(pred_id), f"UNKNOWN_{pred_id}")

    return {
        'gloss': predicted_gloss,
        'confidence': confidence,
        'top_k_predictions': top_k_predictions
    }
