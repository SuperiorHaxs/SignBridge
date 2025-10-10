#!/usr/bin/env python3
"""
openhands_modernized.py

OpenHands Architecture Implementation - ENHANCED
Upgraded to use full pose + hands representation for better handshape recognition.

Key Architecture Components:
1. Pose Flattener Encoder: 75-point (Pose+Hands) MediaPipe → 150 features
   - Full body pose: 33 points (vs 11 in original)
   - Full hand skeletons: 42 points (vs 10 in original)
   - Face excluded: 0 points (noisy, removed)
2. Enhanced Transformer: 6 layers, 256 dim, 16 heads (vs 3/64/8)
3. Long training: 1500 epochs with cosine annealing
4. Minimal augmentation: shear + rotation only

Expected Improvement: +10-15% accuracy from better hand detail
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import math


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class OpenHandsConfig:
    """Configuration for OpenHands model architecture - ENHANCED."""

    # Pose representation - UPGRADED to full pose + hands
    num_pose_keypoints: int = 75  # Full MediaPipe Pose (33) + Hands (42)
    pose_channels: int = 2        # x, y coordinates (confidence available but not used)
    pose_features: int = 150      # 75 * 2

    # Transformer architecture - UPGRADED for better capacity
    hidden_size: int = 256        # Increased from 64 (4x larger)
    num_hidden_layers: int = 6    # Increased from 3 (2x deeper)
    num_attention_heads: int = 16 # Increased from 8 (2x more heads)
    intermediate_size: int = 1024 # 4x hidden_size (scales with hidden_size)
    max_position_embeddings: int = 257  # 256 + 1 for CLS token

    # Training parameters
    dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12

    # Classification
    vocab_size: int = 50          # Number of sign classes
    use_cls_token: bool = True    # BERT-style classification


# =============================================================================
# MediaPipe Pose+Hands Extractor (75 points)
# =============================================================================

class MediaPipeSubset:
    """Extract 75-point pose+hands subset from full MediaPipe pose data."""

    # MediaPipe Holistic layout (576 total points):
    # Indices 0-32:   Pose (33 points)
    # Indices 33-500: Face (468 points) - EXCLUDED (noisy)
    # Indices 501-521: Left Hand (21 points)
    # Indices 522-542: Right Hand (21 points)
    # Indices 543-575: Pose World 3D (33 points) - EXCLUDED for now

    POSE_HAND_INDICES = list(range(0, 33)) + list(range(501, 543))
    # Total: 33 + 42 = 75 points

    @staticmethod
    def extract_27_points(full_pose_data: np.ndarray) -> np.ndarray:
        """
        DEPRECATED: Old 27-point extraction. Kept for backwards compatibility.
        Use extract_pose_hands_75() instead.
        """
        # Redirect to new 75-point extraction
        return MediaPipeSubset.extract_pose_hands_75(full_pose_data)

    @staticmethod
    def extract_pose_hands_75(full_pose_data: np.ndarray) -> np.ndarray:
        """
        Extract 75-point pose+hands from full MediaPipe pose data.

        Extracts:
        - Pose: indices 0-32 (33 points) - full body skeleton
        - Left Hand: indices 501-521 (21 points) - complete hand
        - Right Hand: indices 522-542 (21 points) - complete hand
        - Face: EXCLUDED (indices 33-500) - too noisy

        Args:
            full_pose_data: (frames, 576, 2) or (frames, 576, 3)

        Returns:
            subset_data: (frames, 75, 2) - x, y coordinates only
        """
        frames, total_points, coords = full_pose_data.shape

        # Extract only x, y coordinates (drop confidence if present)
        if coords == 3:
            full_pose_data = full_pose_data[:, :, :2]

        # Verify we have the expected number of keypoints
        if total_points < 543:
            raise ValueError(f"Expected at least 543 keypoints, got {total_points}")

        # Extract pose + hands indices
        pose_indices = list(range(0, 33))           # 33 pose points
        left_hand_indices = list(range(501, 522))   # 21 left hand points
        right_hand_indices = list(range(522, 543))  # 21 right hand points

        all_indices = pose_indices + left_hand_indices + right_hand_indices

        subset_data = full_pose_data[:, all_indices, :]  # (frames, 75, 2)

        return subset_data.astype(np.float32)


# =============================================================================
# Data Preprocessing and Augmentation
# =============================================================================

class PoseTransforms:
    """Pose data preprocessing and augmentation transforms."""

    @staticmethod
    def center_and_scale_normalize(pose_data: np.ndarray) -> np.ndarray:
        """
        Center and scale normalize pose keypoints.

        Args:
            pose_data: (frames, 75, 2) - x, y coordinates

        Returns:
            normalized_data: (frames, 75, 2)
        """
        frames, keypoints, channels = pose_data.shape
        normalized = pose_data.copy()

        for frame_idx in range(frames):
            frame = normalized[frame_idx]  # (75, 2)

            # Extract x, y coordinates
            xy_coords = frame[:, :]  # (75, 2)

            # Find valid keypoints (non-zero)
            valid_mask = (xy_coords != 0).any(axis=1)

            if valid_mask.sum() > 0:
                valid_coords = xy_coords[valid_mask]

                # Center around mean
                center = valid_coords.mean(axis=0)
                xy_coords = xy_coords - center

                # Scale by standard deviation
                std = valid_coords.std()
                if std > 0:
                    xy_coords = xy_coords / std

                # Update normalized frame
                normalized[frame_idx, :, :2] = xy_coords

        return normalized

    @staticmethod
    def apply_shear(pose_data: np.ndarray, shear_std: float = 0.1) -> np.ndarray:
        """Apply random shear transformation."""
        if np.random.random() > 0.5:  # 50% chance
            shear_x = np.random.normal(0, shear_std)
            shear_y = np.random.normal(0, shear_std)

            # Apply shear to x, y coordinates
            pose_data[:, :, 0] += shear_y * pose_data[:, :, 1]  # x += shear_y * y
            pose_data[:, :, 1] += shear_x * pose_data[:, :, 0]  # y += shear_x * x

        return pose_data

    @staticmethod
    def apply_rotation(pose_data: np.ndarray, rotation_std: float = 0.1) -> np.ndarray:
        """Apply random rotation transformation."""
        if np.random.random() > 0.5:  # 50% chance
            angle = np.random.normal(0, rotation_std)
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            # Rotation matrix
            x_coords = pose_data[:, :, 0].copy()
            y_coords = pose_data[:, :, 1].copy()

            pose_data[:, :, 0] = cos_a * x_coords - sin_a * y_coords
            pose_data[:, :, 1] = sin_a * x_coords + cos_a * y_coords

        return pose_data


# =============================================================================
# Pose Flattener Encoder
# =============================================================================

class PoseFlattener(nn.Module):
    """
    Flattens pose keypoints across the channel dimension.
    Transforms (N, 2, T, 75) -> (N, T, 150)

    UPDATED: Now handles 75 keypoints (pose+hands) with 2 channels (x, y)
    """

    def __init__(self, in_channels: int = 2, num_keypoints: int = 75):
        super().__init__()
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.output_features = in_channels * num_keypoints

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, time_steps, keypoints) = (N, 2, T, 75)

        Returns:
            flattened: (batch_size, time_steps, features) = (N, T, 150)
        """
        N, C, T, V = x.shape

        # Permute to (N, T, C, V) then flatten (C, V)
        x = x.permute(0, 2, 1, 3)  # (N, T, C, V)
        x = x.contiguous().view(N, T, C * V)  # (N, T, 150)

        return x


# =============================================================================
# Compact BERT-style Transformer
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, config: OpenHandsConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward."""

    def __init__(self, config: OpenHandsConfig):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.attention_output = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.dropout_prob)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.dropout_prob)
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_layer_norm(hidden_states + attention_output)

        # Feed-forward
        intermediate_output = F.gelu(self.intermediate(attention_output))
        layer_output = self.output(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        layer_output = self.output_layer_norm(attention_output + layer_output)

        return layer_output


class CompactTransformer(nn.Module):
    """Compact BERT-style transformer encoder."""

    def __init__(self, config: OpenHandsConfig):
        super().__init__()
        self.config = config

        # Input projection from pose features to hidden size
        self.input_projection = nn.Linear(config.pose_features, config.hidden_size)

        # Position embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # CLS token embedding
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, pose_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pose_features: (batch_size, seq_len, pose_features) = (N, T, 81)
            attention_mask: (batch_size, seq_len)

        Returns:
            hidden_states: (batch_size, seq_len + 1, hidden_size) if CLS token used
        """
        batch_size, seq_len = pose_features.shape[:2]

        # Project pose features to hidden size
        hidden_states = self.input_projection(pose_features)  # (N, T, hidden_size)

        # Add CLS token
        if self.config.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (N, 1, hidden_size)
            hidden_states = torch.cat([cls_tokens, hidden_states], dim=1)  # (N, T+1, hidden_size)
            seq_len += 1

            # Extend attention mask for CLS token
            if attention_mask is not None:
                cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
                attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        # Add position embeddings
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = hidden_states + position_embeddings

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Convert attention mask to proper format
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        # Apply transformer blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states, extended_attention_mask)

        return hidden_states


# =============================================================================
# OpenHands Model
# =============================================================================

class OpenHandsModel(nn.Module):
    """
    Complete OpenHands model - ENHANCED with pose+hands.

    Architecture:
    1. PoseFlattener: (N, 2, T, 75) -> (N, T, 150)
    2. CompactTransformer: (N, T, 150) -> (N, T+1, 256)
    3. Classification: CLS token -> vocab_size logits

    UPDATED: Now uses 75 keypoints (33 pose + 42 hands) with 2 channels (x, y)
    """

    def __init__(self, config: OpenHandsConfig):
        super().__init__()
        self.config = config

        # Pose encoder
        self.pose_flattener = PoseFlattener(
            in_channels=config.pose_channels,
            num_keypoints=config.num_pose_keypoints
        )

        # Transformer encoder
        self.transformer = CompactTransformer(config)

        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, pose_data: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pose_data: (batch_size, seq_len, 75, 2) - frames x keypoints x (x,y)
            attention_mask: (batch_size, seq_len) - mask for padding

        Returns:
            logits: (batch_size, vocab_size) - classification scores
        """
        # Reshape for pose flattener: (N, T, 75, 2) -> (N, 2, T, 75)
        batch_size, seq_len = pose_data.shape[:2]
        pose_data = pose_data.permute(0, 3, 1, 2)  # (N, 2, T, 75)

        # Flatten pose features
        pose_features = self.pose_flattener(pose_data)  # (N, T, 150)

        # Apply transformer
        hidden_states = self.transformer(pose_features, attention_mask)  # (N, T+1, 256)

        # Extract CLS token representation for classification
        if self.config.use_cls_token:
            cls_representation = hidden_states[:, 0, :]  # (N, 256)
        else:
            # Use mean pooling if no CLS token
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                hidden_states = hidden_states[:, 1:, :]  # Remove CLS position
                pooled = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                pooled = hidden_states.mean(dim=1)
            cls_representation = pooled

        # Classification
        cls_representation = self.dropout(cls_representation)
        logits = self.classifier(cls_representation)  # (N, vocab_size)

        return logits


# =============================================================================
# Data Processing Pipeline
# =============================================================================

class WLASLPoseProcessor:
    """WLASL pose data processor for OpenHands architecture."""

    def __init__(self):
        self.mediapipe_subset = MediaPipeSubset()
        self.transforms = PoseTransforms()

    def load_pickle_pose(self, pickle_path: str) -> np.ndarray:
        """Load pose data from WLASL pickle file and convert to 27-point format."""
        try:
            with open(pickle_path, 'rb') as f:
                pose_data = pickle.load(f)

            # Extract keypoints from various possible formats
            if isinstance(pose_data, dict):
                keypoints = None
                for key in ['keypoints', 'pose', 'landmarks', 'data', 'features']:
                    if key in pose_data:
                        keypoints = pose_data[key]
                        break

                if keypoints is None:
                    # Try to find the largest array
                    arrays = [v for v in pose_data.values() if isinstance(v, (np.ndarray, list))]
                    if arrays:
                        keypoints = max(arrays, key=lambda x: np.array(x).size)
                    else:
                        raise ValueError("No pose data found in dictionary")
            else:
                keypoints = pose_data

            # Convert to numpy array
            keypoints = np.array(keypoints, dtype=np.float32)

            # Handle MaskedArray
            if hasattr(keypoints, 'data'):
                keypoints = keypoints.data

            # Clean data
            keypoints = np.nan_to_num(keypoints, nan=0.0, posinf=0.0, neginf=0.0)

            # Ensure 3D shape: (frames, keypoints, coords)
            if len(keypoints.shape) == 2:
                keypoints = keypoints.reshape(1, -1, keypoints.shape[-1])
            elif len(keypoints.shape) == 4:
                keypoints = keypoints[:, 0, :, :]  # Remove person dimension

            # Check if already 75-point format (from augmented_75pt_20_class dataset)
            frames, num_keypoints, coords = keypoints.shape
            if num_keypoints == 75:
                # Already in 75-point format, just ensure 2D coords
                if coords == 3:
                    pose_75 = keypoints[:, :, :2]
                else:
                    pose_75 = keypoints
            else:
                # Extract 75-point pose+hands from 576-point format
                pose_75 = self.mediapipe_subset.extract_27_points(keypoints)  # Redirects to extract_pose_hands_75

            return pose_75

        except Exception as e:
            print(f"WARNING: Error loading {pickle_path}: {e}")
            # Return dummy data (75 points, 2 channels)
            return np.zeros((30, 75, 2), dtype=np.float32)

    def preprocess_pose_sequence(self, pose_sequence: np.ndarray, augment: bool = False) -> np.ndarray:
        """Preprocess pose sequence with normalization and optional augmentation."""
        # Center and scale normalize
        pose_sequence = self.transforms.center_and_scale_normalize(pose_sequence)

        # Apply augmentation during training
        if augment:
            pose_sequence = self.transforms.apply_shear(pose_sequence, shear_std=0.1)
            pose_sequence = self.transforms.apply_rotation(pose_sequence, rotation_std=0.1)

        return pose_sequence

    def pad_or_truncate_sequence(self, pose_sequence: np.ndarray, max_length: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """Pad or truncate pose sequence to fixed length."""
        seq_len = len(pose_sequence)

        if seq_len > max_length:
            # Truncate
            pose_sequence = pose_sequence[:max_length]
            attention_mask = np.ones(max_length, dtype=np.float32)
        else:
            # Pad
            padding_length = max_length - seq_len
            padding = np.zeros((padding_length, 75, 2), dtype=np.float32)  # Updated: 75 keypoints, 2 channels
            pose_sequence = np.vstack([pose_sequence, padding])
            attention_mask = np.concatenate([
                np.ones(seq_len, dtype=np.float32),
                np.zeros(padding_length, dtype=np.float32)
            ])

        return pose_sequence, attention_mask


# =============================================================================
# Dataset Class
# =============================================================================

class WLASLOpenHandsDataset(Dataset):
    """WLASL Dataset for OpenHands architecture."""

    def __init__(self, pose_files: List[str], labels: List[str], gloss_to_id: Dict[str, int],
                 max_seq_length: int = 256, augment: bool = False):
        self.pose_files = pose_files
        self.labels = labels
        self.gloss_to_id = gloss_to_id
        self.max_seq_length = max_seq_length
        self.augment = augment
        self.processor = WLASLPoseProcessor()

    def __len__(self):
        return len(self.pose_files)

    def __getitem__(self, idx):
        # Load and preprocess pose data
        pose_sequence = self.processor.load_pickle_pose(self.pose_files[idx])
        pose_sequence = self.processor.preprocess_pose_sequence(pose_sequence, augment=self.augment)
        pose_sequence, attention_mask = self.processor.pad_or_truncate_sequence(pose_sequence, self.max_seq_length)

        # Get label
        label_text = self.labels[idx]
        if label_text not in self.gloss_to_id:
            raise KeyError(f"Label '{label_text}' not found in vocabulary!")
        label_id = self.gloss_to_id[label_text]

        return {
            'pose_sequence': torch.tensor(pose_sequence, dtype=torch.float32),  # (T, 75, 2)
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float32),  # (T,)
            'label': torch.tensor(label_id, dtype=torch.long)
        }


# =============================================================================
# Dataset Loader
# =============================================================================

class WLASLDatasetLoader:
    """Load WLASL pickle dataset for OpenHands training."""

    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.pose_files = []
        self.labels = []
        self.gloss_to_id = {}
        self.id_to_gloss = {}

    def scan_dataset(self, max_files: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """Scan dataset directory for pickle files."""
        print(f"LOADING: Scanning WLASL dataset in {self.dataset_root}")

        all_pose_files = []
        all_labels = []

        for file_path in self.dataset_root.rglob("*.pkl"):
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

                # Extract gloss label
                if isinstance(data, dict) and 'gloss' in data:
                    gloss_label = data['gloss'].upper()
                    all_pose_files.append(str(file_path))
                    all_labels.append(gloss_label)

                    if max_files and len(all_pose_files) >= max_files:
                        break
            except:
                continue

        print(f"FOUND: {len(all_pose_files)} pose files")
        return all_pose_files, all_labels

    def create_vocabulary(self, labels: List[str], vocab_size: int) -> Dict[str, int]:
        """Create vocabulary from most frequent labels."""
        from collections import Counter

        label_counts = Counter(labels)
        most_frequent = label_counts.most_common(vocab_size)

        vocab = {gloss: idx for idx, (gloss, _) in enumerate(most_frequent)}

        print(f"VOCAB: Created vocabulary with {len(vocab)} classes")
        print(f"VOCAB: Most frequent classes: {list(vocab.keys())[:10]}")

        return vocab

    def load_dataset(self, vocab_size: int = 50, max_files_per_class: Optional[int] = None,
                    max_seq_length: int = 256, train_split: float = 0.8) -> Tuple[Dataset, Dataset, List[str]]:
        """Load and split dataset into train/validation."""

        # Scan dataset
        all_files, all_labels = self.scan_dataset()

        # Create vocabulary
        gloss_to_id = self.create_vocabulary(all_labels, vocab_size)
        id_to_gloss = {idx: gloss for gloss, idx in gloss_to_id.items()}

        # Filter files to vocabulary
        filtered_files = []
        filtered_labels = []

        for file_path, label in zip(all_files, all_labels):
            if label in gloss_to_id:
                filtered_files.append(file_path)
                filtered_labels.append(label)

        print(f"FILTERED: {len(filtered_files)} files match vocabulary")

        # Limit files per class if specified
        if max_files_per_class:
            from collections import defaultdict
            class_files = defaultdict(list)
            class_labels = defaultdict(list)

            for file_path, label in zip(filtered_files, filtered_labels):
                if len(class_files[label]) < max_files_per_class:
                    class_files[label].append(file_path)
                    class_labels[label].append(label)

            filtered_files = []
            filtered_labels = []
            for label in class_files:
                filtered_files.extend(class_files[label])
                filtered_labels.extend(class_labels[label])

            print(f"LIMITED: {len(filtered_files)} files after limiting to {max_files_per_class} per class")

        # Split into train/validation
        from sklearn.model_selection import train_test_split

        train_files, val_files, train_labels, val_labels = train_test_split(
            filtered_files, filtered_labels,
            test_size=1-train_split,
            stratify=filtered_labels,
            random_state=42
        )

        # Create datasets
        train_dataset = WLASLOpenHandsDataset(
            train_files, train_labels, gloss_to_id, max_seq_length, augment=True
        )
        val_dataset = WLASLOpenHandsDataset(
            val_files, val_labels, gloss_to_id, max_seq_length, augment=False
        )

        unique_glosses = list(gloss_to_id.keys())

        print(f"DATASET: Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        print(f"CLASSES: {len(unique_glosses)} unique glosses")

        return train_dataset, val_dataset, unique_glosses


if __name__ == "__main__":
    # Test the implementation - UPDATED for 75-point model
    print("Testing ENHANCED OpenHands Model (75-point pose+hands)")
    print("=" * 60)

    config = OpenHandsConfig(vocab_size=50)
    model = OpenHandsModel(config)

    # Test with dummy data - UPDATED dimensions
    batch_size = 4
    seq_len = 100
    dummy_pose = torch.randn(batch_size, seq_len, 75, 2)  # 75 keypoints, 2 channels
    dummy_mask = torch.ones(batch_size, seq_len)

    print(f"Input shape: {dummy_pose.shape}")  # (4, 100, 75, 2)
    print(f"Model config:")
    print(f"  Keypoints: {config.num_pose_keypoints}")
    print(f"  Channels: {config.pose_channels}")
    print(f"  Features: {config.pose_features}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print()

    logits = model(dummy_pose, dummy_mask)
    print(f"Output shape: {logits.shape}")  # Should be (4, 50)
    print()
    print("SUCCESS: Enhanced OpenHands architecture (75-point) working!")
    print("Expected improvements: +10-15% accuracy from full hand detail")

# =============================================================================
# Inference Functions
# =============================================================================

def load_model_from_checkpoint(checkpoint_path: str, vocab_size: int = 20):
    """Load OpenHands model from checkpoint directory."""
    checkpoint_path = Path(checkpoint_path)
    model_file = checkpoint_path / "pytorch_model.bin"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    vocab_file = checkpoint_path / "class_index_mapping.json"
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
    with open(vocab_file, 'r') as f:
        id_to_gloss = json.load(f)

    # Load config from file if it exists, otherwise use defaults
    config_file = checkpoint_path / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        config = OpenHandsConfig(**config_dict)
        print(f"Loaded config from {config_file}")
    else:
        config = OpenHandsConfig(vocab_size=vocab_size)
        print(f"No config.json found, using defaults")
    model = OpenHandsModel(config)
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    print(f"SUCCESS: Loaded model from {checkpoint_path}")
    return model, id_to_gloss

def predict_pose_file(pickle_path: str, model=None, tokenizer=None, checkpoint_path: str = None):
    """Predict gloss from a pickle pose file."""
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Either model or checkpoint_path must be provided")
        model, tokenizer = load_model_from_checkpoint(checkpoint_path)
    processor = WLASLPoseProcessor()
    pose_sequence = processor.load_pickle_pose(pickle_path)
    pose_sequence = processor.preprocess_pose_sequence(pose_sequence, augment=False)
    pose_sequence, attention_mask = processor.pad_or_truncate_sequence(pose_sequence, max_length=256)
    pose_tensor = torch.tensor(pose_sequence, dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.tensor(attention_mask, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(pose_tensor, mask_tensor)
        probs = torch.softmax(logits, dim=-1)
        confidence, pred_id = torch.max(probs, dim=-1)
        pred_id = pred_id.item()
        confidence = confidence.item()
        top_k = 5
        top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
        top_k_predictions = []
        for prob, idx in zip(top_probs[0], top_ids[0]):
            idx_int = idx.item()  # Convert tensor to int
            gloss = tokenizer.get(str(idx_int), f"UNKNOWN_{idx_int}")
            top_k_predictions.append({'gloss': gloss, 'confidence': prob.item()})
    predicted_gloss = tokenizer.get(str(pred_id), f"UNKNOWN_{pred_id}")
    return {'gloss': predicted_gloss, 'confidence': confidence, 'top_k_predictions': top_k_predictions}
