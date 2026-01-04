#!/usr/bin/env python3
"""
openhands_inference.py

Inference functions for OpenHands model.
Provides load_model_from_checkpoint and predict_pose_file functions.
"""

import torch
import numpy as np
import json
from pathlib import Path
from openhands_modernized import OpenHandsConfig, OpenHandsModel, WLASLPoseProcessor


def load_model_from_checkpoint(checkpoint_path: str, vocab_size: int = None):
    """
    Load OpenHands model from checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory containing pytorch_model.bin
        vocab_size: Number of classes (optional, will be read from config.json if available)

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

    # Try to load config from checkpoint directory
    config_file = checkpoint_path / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        # Create config from saved values
        config = OpenHandsConfig(
            num_pose_keypoints=config_dict.get('num_pose_keypoints', 75),
            pose_channels=config_dict.get('pose_channels', 2),
            pose_coord_features=config_dict.get('pose_coord_features', 150),
            use_finger_features=config_dict.get('use_finger_features', False),
            finger_features=config_dict.get('finger_features', 0),
            pose_features=config_dict.get('pose_features', 150),
            hidden_size=config_dict.get('hidden_size', 64),
            num_hidden_layers=config_dict.get('num_hidden_layers', 3),
            num_attention_heads=config_dict.get('num_attention_heads', 8),
            intermediate_size=config_dict.get('intermediate_size', 256),
            max_position_embeddings=config_dict.get('max_position_embeddings', 257),
            dropout_prob=config_dict.get('dropout_prob', 0.1),
            layer_norm_eps=config_dict.get('layer_norm_eps', 1e-12),
            vocab_size=config_dict.get('vocab_size', vocab_size or 20),
            use_cls_token=config_dict.get('use_cls_token', True)
        )
        print(f"Loaded config from {config_file}")
        print(f"  hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}, vocab={config.vocab_size}")
        print(f"  pose_features={config.pose_features}, use_finger_features={config.use_finger_features}")
    else:
        # Fallback to default config with provided vocab_size
        config = OpenHandsConfig(vocab_size=vocab_size or 20)
        print(f"Using default config with vocab_size={config.vocab_size}")

    model = OpenHandsModel(config)

    # Load weights
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    print(f"SUCCESS: Loaded model from {checkpoint_path}")
    print(f"VOCAB: {config.vocab_size} classes")

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
    import numpy as _np
    print(f'[DEBUG] Raw pose shape: {pose_sequence.shape}, shoulder dist: {_np.linalg.norm(pose_sequence[0,11,:2] - pose_sequence[0,12,:2]):.4f}')
    pose_sequence = processor.preprocess_pose_sequence(pose_sequence, augment=False)
    print(f'[DEBUG] After norm: shoulder dist: {_np.linalg.norm(pose_sequence[0,11,:2] - pose_sequence[0,12,:2]):.4f}')

    # Extract finger features before padding if model uses them
    finger_features_tensor = None
    if hasattr(model, 'config') and model.config.use_finger_features:
        import numpy as np
        finger_features = processor.extract_finger_features(pose_sequence)
        # Pad finger features to same length as pose sequence
        max_length = 256
        if len(finger_features) > max_length:
            finger_features = finger_features[:max_length]
        else:
            padding = np.zeros((max_length - len(finger_features), 30), dtype=np.float32)
            finger_features = np.concatenate([finger_features, padding], axis=0)
        finger_features_tensor = torch.tensor(finger_features, dtype=torch.float32).unsqueeze(0)  # (1, T, 30)

    pose_sequence, attention_mask = processor.pad_or_truncate_sequence(pose_sequence, max_length=256)

    # Convert to tensors and add batch dimension
    pose_tensor = torch.tensor(pose_sequence, dtype=torch.float32).unsqueeze(0)  # (1, T, 83, 3)
    mask_tensor = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)  # (1, T) - must be long like training!

    # Predict
    with torch.no_grad():
        logits = model(pose_tensor, mask_tensor, finger_features=finger_features_tensor)  # (1, vocab_size)
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
            idx_int = idx.item()  # Convert tensor to int
            gloss = tokenizer.get(str(idx_int), f"UNKNOWN_{idx_int}")
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
