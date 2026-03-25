#!/usr/bin/env python3
"""
openhands_modernized.py

OpenHands Architecture Implementation - ENHANCED
Upgraded to use full pose + hands + face + z-coordinate + finger features.

Key Architecture Components:
1. Pose Flattener Encoder: 83-point MediaPipe xyz → 249 features
   - Full body pose: 33 points
   - Full hand skeletons: 42 points (21 per hand)
   - Minimal face: 8 points (for disambiguation)
   - Z-coordinate: depth information
2. Finger Feature Encoder: 30 derived features
   - Extension ratios: 5 per hand (finger straightness)
   - Spread angles: 4 per hand (between adjacent fingers)
   - Fingertip distances: 5 per hand (normalized)
   - Openness score: 1 per hand
3. Combined Features: 249 + 30 = 279 features per frame
4. Enhanced Transformer: 6 layers, 256 dim, 16 heads
5. Long training: 1500 epochs with cosine annealing

Expected Improvement: Better handshape recognition from explicit finger features
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

# Add landmarks-extraction to path for finger features
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent.parent
_landmarks_dir = _project_root / "dataset-utilities" / "landmarks-extraction"
if str(_landmarks_dir) not in sys.path:
    sys.path.insert(0, str(_landmarks_dir))

# Import finger feature extraction
try:
    from finger_features import (
        extract_finger_features_from_sequence,
        FINGER_FEATURE_COUNT,
    )
    FINGER_FEATURES_AVAILABLE = True
except ImportError:
    print("WARNING: finger_features module not found, finger features disabled")
    FINGER_FEATURES_AVAILABLE = False
    FINGER_FEATURE_COUNT = 0


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class OpenHandsConfig:
    """Configuration for OpenHands model architecture - ENHANCED."""

    # Pose representation - UPGRADED to full pose + hands + minimal face + z-coordinate
    num_pose_keypoints: int = 83  # Full MediaPipe Pose (33) + Hands (42) + Minimal Face (8)
    pose_channels: int = 3        # x, y, z coordinates (depth for better 3D understanding)
    pose_coord_features: int = 249  # 83 * 3 (raw coordinate features)

    # Finger features - derived hand shape features (position-invariant)
    use_finger_features: bool = True  # Whether to extract and use finger features
    finger_features: int = 30         # 15 per hand (extension, spread, distances, openness)

    # Motion features - hand velocity and activity detection
    use_motion_features: bool = True   # Whether to extract velocity + hand-detected features
    motion_features: int = 8           # 6 velocity + 2 hand-active flags per frame

    # Spatial features - trajectory, palm, location, handshape, face region, contact, repetition, interaction, rotation
    use_spatial_features: bool = True  # Whether to extract spatial features
    spatial_features: int = 40         # 26 original + 4 face region + 2 contact + 2 repetition + 4 interaction + 2 rotation

    # Total features per frame
    pose_features: int = 327      # 249 coords + 30 finger + 8 motion + 40 spatial (when all enabled)

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
    """Extract pose+hands subset from full MediaPipe pose data."""

    # MediaPipe Holistic layout (576 total points):
    # Indices 0-32:   Pose (33 points)
    # Indices 33-500: Face (468 points) - EXCLUDED (noisy)
    # Indices 501-521: Left Hand (21 points)
    # Indices 522-542: Right Hand (21 points)
    # Indices 543-575: Pose World 3D (33 points) - EXCLUDED for now

    # Full pose (33) + hands (42) = 75 points
    POSE_HAND_INDICES = list(range(0, 33)) + list(range(501, 543))

    # Upper-body only pose (12) + hands (42) = 54 points
    # Excludes: face landmarks (0-10), hips (23-24), legs (25-32)
    # Includes: shoulders (11-12), elbows (13-14), wrists (15-16), hand connections (17-22)
    UPPER_BODY_POSE_INDICES = list(range(11, 23))  # 12 upper body landmarks
    UPPER_BODY_HAND_INDICES = list(range(501, 543))  # 42 hand landmarks
    # Total: 12 + 42 = 54 points

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

    @staticmethod
    def extract_upper_body_54(full_pose_data: np.ndarray) -> np.ndarray:
        """
        Extract 54-point upper-body + hands from full MediaPipe pose data.

        Better for ASL: excludes face and lower body landmarks that vary
        based on camera framing (full body vs upper body videos).

        Extracts:
        - Upper body: indices 11-22 (12 points) - shoulders to hand connections
        - Left Hand: indices 501-521 (21 points)
        - Right Hand: indices 522-542 (21 points)
        - EXCLUDED: Face (0-10), Hips (23-24), Legs (25-32)

        Args:
            full_pose_data: (frames, 576, 2) or (frames, 576, 3)

        Returns:
            subset_data: (frames, 54, 2) - x, y coordinates only
        """
        frames, total_points, coords = full_pose_data.shape

        # Extract only x, y coordinates (drop confidence if present)
        if coords == 3:
            full_pose_data = full_pose_data[:, :, :2]

        if total_points < 543:
            raise ValueError(f"Expected at least 543 keypoints, got {total_points}")

        # Upper body pose (excludes face 0-10, hips 23-24, legs 25-32)
        upper_body_indices = list(range(11, 23))     # 12 upper body points
        left_hand_indices = list(range(501, 522))    # 21 left hand points
        right_hand_indices = list(range(522, 543))   # 21 right hand points

        all_indices = upper_body_indices + left_hand_indices + right_hand_indices

        subset_data = full_pose_data[:, all_indices, :]  # (frames, 54, 2)

        return subset_data.astype(np.float32)


# =============================================================================
# Data Preprocessing and Augmentation
# =============================================================================

class PoseTransforms:
    """Pose data preprocessing and augmentation transforms."""

    # Shoulder indices in 75-point format (indices 11-12 from original pose)
    # In 75-point: pose is 0-32, so shoulders are at indices 11 and 12
    LEFT_SHOULDER_IDX = 11
    RIGHT_SHOULDER_IDX = 12

    @staticmethod
    def center_and_scale_normalize(pose_data: np.ndarray, use_shoulder_center: bool = True) -> np.ndarray:
        """
        Center and scale normalize pose keypoints.

        Args:
            pose_data: (frames, N, 2) or (frames, N, 3) - x, y or x, y, z coordinates
            use_shoulder_center: If True, center on shoulders (robust to video framing).
                                 If False, center on mean of all valid keypoints.

        Returns:
            normalized_data: same shape as input (z-coordinate preserved if present)
        """
        frames, keypoints, channels = pose_data.shape
        normalized = pose_data.copy()

        for frame_idx in range(frames):
            frame = normalized[frame_idx]  # (N, channels)

            # Extract only x, y coordinates for normalization (preserve z if present)
            xy_coords = frame[:, :2].copy()  # (N, 2) - only x, y

            # Find valid keypoints (non-zero)
            valid_mask = (xy_coords != 0).any(axis=1)

            if valid_mask.sum() > 0:
                if use_shoulder_center and keypoints >= 33:
                    # Use shoulder midpoint as center (robust to video framing)
                    # Works for both 75-point and 54-point formats
                    left_shoulder = xy_coords[PoseTransforms.LEFT_SHOULDER_IDX]
                    right_shoulder = xy_coords[PoseTransforms.RIGHT_SHOULDER_IDX]

                    # Only use shoulders if both are valid
                    if (left_shoulder != 0).any() and (right_shoulder != 0).any():
                        center = (left_shoulder + right_shoulder) / 2
                    else:
                        # Fallback to mean if shoulders not detected
                        center = xy_coords[valid_mask].mean(axis=0)
                elif use_shoulder_center and keypoints == 54:
                    # For 54-point format, shoulders are at indices 0 and 1
                    # (since we start from index 11 in original, which becomes 0)
                    left_shoulder = xy_coords[0]  # Was index 11
                    right_shoulder = xy_coords[1]  # Was index 12

                    if (left_shoulder != 0).any() and (right_shoulder != 0).any():
                        center = (left_shoulder + right_shoulder) / 2
                    else:
                        center = xy_coords[valid_mask].mean(axis=0)
                else:
                    # Fallback: center on mean of all valid keypoints
                    center = xy_coords[valid_mask].mean(axis=0)

                xy_coords = xy_coords - center

                # Scale by shoulder width (more stable than std of all points)
                if use_shoulder_center and keypoints >= 13:
                    if keypoints >= 33:
                        shoulder_dist = np.linalg.norm(
                            xy_coords[PoseTransforms.LEFT_SHOULDER_IDX] -
                            xy_coords[PoseTransforms.RIGHT_SHOULDER_IDX]
                        )
                    else:  # 54-point format
                        shoulder_dist = np.linalg.norm(xy_coords[0] - xy_coords[1])

                    if shoulder_dist > 0.01:  # Avoid division by tiny values
                        xy_coords = xy_coords / shoulder_dist
                    else:
                        # Fallback to std if shoulders too close
                        std = xy_coords[valid_mask].std()
                        if std > 0:
                            xy_coords = xy_coords / std
                else:
                    # Fallback: scale by standard deviation
                    valid_coords = xy_coords[valid_mask]
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
    Complete OpenHands model - ENHANCED with pose+hands+face, z-coordinate, and finger features.

    Architecture:
    1. PoseFlattener: (N, 3, T, 83) -> (N, T, 249)
    2. Finger Features: (N, T, 30) - derived hand shape features
    3. Combined: (N, T, 279) = 249 + 30
    4. CompactTransformer: (N, T, 279) -> (N, T+1, 256)
    5. Classification: CLS token -> vocab_size logits

    UPDATED: Now uses 83 keypoints + 30 finger features = 279 features per frame
    """

    def __init__(self, config: OpenHandsConfig):
        super().__init__()
        self.config = config

        # Pose encoder (flattens 83 keypoints × 3 coords = 249 features)
        self.pose_flattener = PoseFlattener(
            in_channels=config.pose_channels,
            num_keypoints=config.num_pose_keypoints
        )

        # Transformer encoder (input is pose_features, which includes finger features if enabled)
        self.transformer = CompactTransformer(config)

        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(
        self,
        pose_data: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        finger_features: Optional[torch.Tensor] = None,
        motion_features: Optional[torch.Tensor] = None,
        spatial_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pose_data: (batch_size, seq_len, 83, 3) - frames x keypoints x (x,y,z)
            attention_mask: (batch_size, seq_len) - mask for padding
            finger_features: (batch_size, seq_len, 30) - optional finger features
            motion_features: (batch_size, seq_len, 8) - optional velocity + hand-active features
            spatial_features: (batch_size, seq_len, 26) - optional trajectory/palm/location/handshape features

        Returns:
            logits: (batch_size, vocab_size) - classification scores
        """
        # Reshape for pose flattener: (N, T, 83, 3) -> (N, 3, T, 83)
        batch_size, seq_len = pose_data.shape[:2]
        pose_data = pose_data.permute(0, 3, 1, 2)  # (N, 3, T, 83)

        # Flatten pose features
        pose_features = self.pose_flattener(pose_data)  # (N, T, 249)

        # Concatenate finger features if provided
        if finger_features is not None and self.config.use_finger_features:
            pose_features = torch.cat([pose_features, finger_features], dim=-1)  # (N, T, 279)

        # Concatenate motion features if model expects them
        if self.config.use_motion_features:
            if motion_features is not None:
                pose_features = torch.cat([pose_features, motion_features], dim=-1)  # (N, T, 287)
            else:
                # Pad with zeros for backward compatibility with old checkpoints/data
                zeros = torch.zeros(batch_size, seq_len, self.config.motion_features, device=pose_features.device)
                pose_features = torch.cat([pose_features, zeros], dim=-1)

        # Concatenate spatial features if model expects them
        if getattr(self.config, 'use_spatial_features', False):
            if spatial_features is not None:
                pose_features = torch.cat([pose_features, spatial_features], dim=-1)  # (N, T, 313)
            else:
                zeros = torch.zeros(batch_size, seq_len, self.config.spatial_features, device=pose_features.device)
                pose_features = torch.cat([pose_features, zeros], dim=-1)

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
        """Load pose data from WLASL pickle file in 83-point xyz format."""
        try:
            # Buffered reading for better performance on external drives
            with open(pickle_path, 'rb', buffering=1024*1024) as f:  # 1MB buffer
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

            # Check keypoint format and convert to 83-point xyz as needed
            frames, num_keypoints, coords = keypoints.shape

            if num_keypoints == 83:
                # Already in 83-point format
                if coords == 3:
                    # Perfect - 83pt xyz as expected
                    pose_83 = keypoints
                else:
                    # 83pt but only xy - pad z with zeros
                    pose_83 = np.zeros((frames, 83, 3), dtype=np.float32)
                    pose_83[:, :, :coords] = keypoints
            elif num_keypoints == 75:
                # 75-point format - pad with zeros for face landmarks
                if coords == 3:
                    pose_83 = np.zeros((frames, 83, 3), dtype=np.float32)
                    pose_83[:, :75, :] = keypoints
                else:
                    pose_83 = np.zeros((frames, 83, 3), dtype=np.float32)
                    pose_83[:, :75, :coords] = keypoints
            else:
                # Extract 83-point from 576-point format
                # Indices: 0-32 (pose), 501-521 (left hand), 522-542 (right hand), face minimal
                pose_indices = list(range(0, 33))
                left_hand_indices = list(range(501, 522))
                right_hand_indices = list(range(522, 543))
                # Minimal face: nose tip, mouth corners, chin, eyebrows, eye corners
                face_indices = [33 + i for i in [1, 61, 291, 152, 107, 336, 33, 263]]

                all_indices = pose_indices + left_hand_indices + right_hand_indices + face_indices

                if coords >= 3:
                    pose_83 = keypoints[:, all_indices, :3]
                else:
                    pose_83 = np.zeros((frames, 83, 3), dtype=np.float32)
                    pose_83[:, :, :coords] = keypoints[:, all_indices, :]

            return pose_83

        except Exception as e:
            import traceback
            print(f"WARNING: Error loading {pickle_path}: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise  # Let caller handle it instead of returning dummy data

    def preprocess_pose_sequence(self, pose_sequence: np.ndarray, augment: bool = False) -> np.ndarray:
        """Preprocess pose sequence with normalization and optional augmentation."""
        # Center and scale normalize
        pose_sequence = self.transforms.center_and_scale_normalize(pose_sequence)

        # Apply augmentation during training
        if augment:
            pose_sequence = self.transforms.apply_shear(pose_sequence, shear_std=0.1)
            pose_sequence = self.transforms.apply_rotation(pose_sequence, rotation_std=0.1)

        return pose_sequence

    def extract_finger_features(self, pose_sequence: np.ndarray) -> np.ndarray:
        """
        Extract finger features from pose sequence.

        Args:
            pose_sequence: (frames, 83, 3) or (frames, 83, 2) pose landmarks

        Returns:
            (frames, 30) finger features array
        """
        if not FINGER_FEATURES_AVAILABLE:
            # Return zeros if finger features not available
            return np.zeros((len(pose_sequence), 30), dtype=np.float32)

        # Hand positions in 83-point format: left hand at 33-53, right hand at 54-74
        # (same as 75pt since face landmarks are at the end)
        finger_features = extract_finger_features_from_sequence(
            pose_sequence[:, :, :2],  # Use only x, y for finger features
            left_hand_start=33,
            right_hand_start=54,
        )

        return finger_features.astype(np.float32)

    def extract_motion_features(self, pose_sequence: np.ndarray) -> np.ndarray:
        """
        Extract velocity and hand-activity features from pose sequence.

        Args:
            pose_sequence: (frames, 83, 3) pose landmarks

        Returns:
            (frames, 8) motion features array:
            - [0] left hand centroid velocity (magnitude)
            - [1] left hand centroid velocity max (rolling)
            - [2] right hand centroid velocity (magnitude)
            - [3] right hand centroid velocity max (rolling)
            - [4] left/right velocity ratio
            - [5] right/left velocity ratio
            - [6] left hand active flag (0 or 1)
            - [7] right hand active flag (0 or 1)
        """
        frames = len(pose_sequence)
        features = np.zeros((frames, 8), dtype=np.float32)

        if frames < 2:
            return features

        # Extract hand regions (xy only for velocity)
        left_hand = pose_sequence[:, 33:54, :2]   # (frames, 21, 2)
        right_hand = pose_sequence[:, 54:75, :2]   # (frames, 21, 2)

        # Centroids per frame
        left_centroid = np.mean(left_hand, axis=1)    # (frames, 2)
        right_centroid = np.mean(right_hand, axis=1)   # (frames, 2)

        # Frame-to-frame displacement magnitude
        left_vel = np.linalg.norm(np.diff(left_centroid, axis=0), axis=1)   # (frames-1,)
        right_vel = np.linalg.norm(np.diff(right_centroid, axis=0), axis=1)  # (frames-1,)

        # Feature 0,2: instantaneous velocity (shift by 1 so frame 0 = 0)
        features[1:, 0] = left_vel
        features[1:, 2] = right_vel

        # Feature 1,3: rolling max velocity (window=5)
        window = min(5, frames - 1)
        for i in range(1, frames):
            start = max(1, i - window + 1)
            features[i, 1] = np.max(features[start:i + 1, 0])
            features[i, 3] = np.max(features[start:i + 1, 2])

        # Feature 4,5: velocity ratios (with epsilon to avoid div/0)
        eps = 1e-6
        features[:, 4] = features[:, 0] / (features[:, 2] + eps)
        features[:, 5] = features[:, 2] / (features[:, 0] + eps)

        # Feature 6,7: hand active flags
        # Hand is "active" if displacement > threshold AND keypoints are non-zero
        active_thresh = 0.005
        left_present = np.any(np.abs(left_hand) > 1e-4, axis=(1, 2))
        right_present = np.any(np.abs(right_hand) > 1e-4, axis=(1, 2))

        features[1:, 6] = (left_vel > active_thresh).astype(np.float32) * left_present[1:].astype(np.float32)
        features[1:, 7] = (right_vel > active_thresh).astype(np.float32) * right_present[1:].astype(np.float32)

        return features

    def extract_spatial_features(self, pose_sequence: np.ndarray) -> np.ndarray:
        """
        Extract spatial features: trajectory, palm orientation, hand location, handshape dynamics,
        face region specificity, hand-body contact, repetition count, hand-hand interaction, wrist rotation.

        Args:
            pose_sequence: (frames, 83, 3) pose landmarks

        Returns:
            (frames, 40) spatial features array:
            Features 0-9: Movement Trajectory (per frame, both hands)
              [0] right hand dx (normalized movement direction x)
              [1] right hand dy (normalized movement direction y)
              [2] left hand dx
              [3] left hand dy
              [4] right hand circularity (rolling cross-product accumulation)
              [5] left hand circularity
              [6] right hand linearity (rolling net/path ratio)
              [7] left hand linearity
              [8] right hand cumulative path (normalized)
              [9] left hand cumulative path (normalized)
            Features 10-15: Palm Orientation (per frame, both hands)
              [10] right palm normal x
              [11] right palm normal y
              [12] right palm normal z
              [13] left palm normal x
              [14] left palm normal y
              [15] left palm normal z
            Features 16-21: Hand Location Relative to Body (per frame)
              [16] right hand y relative to nose
              [17] right hand x relative to nose
              [18] left hand y relative to nose
              [19] left hand x relative to nose
              [20] distance between hands
              [21] vertical zone indicator (0=face, 0.5=chest, 1=waist)
            Features 22-25: Handshape Dynamics (per frame)
              [22] right hand openness delta (change from previous frame)
              [23] left hand openness delta
              [24] right finger spread velocity
              [25] left finger spread velocity
        """
        frames = len(pose_sequence)
        features = np.zeros((frames, 40), dtype=np.float32)

        if frames < 3:
            return features

        # Key landmarks
        left_hand = pose_sequence[:, 33:54, :2]    # (frames, 21, 2)
        right_hand = pose_sequence[:, 54:75, :2]   # (frames, 21, 2)
        left_hand_3d = pose_sequence[:, 33:54, :]  # (frames, 21, 3) - for palm orientation
        right_hand_3d = pose_sequence[:, 54:75, :] # (frames, 21, 3)
        nose = pose_sequence[:, 0, :2]             # (frames, 2)
        left_shoulder = pose_sequence[:, 11, :2]
        right_shoulder = pose_sequence[:, 12, :2]
        chest = (left_shoulder + right_shoulder) / 2

        # Hand centroids
        lh_centroid = np.mean(left_hand, axis=1)   # (frames, 2)
        rh_centroid = np.mean(right_hand, axis=1)   # (frames, 2)

        # ── Trajectory features (0-9) ──────────────────────────────────

        # Displacements
        rh_disp = np.diff(rh_centroid, axis=0)  # (frames-1, 2)
        lh_disp = np.diff(lh_centroid, axis=0)

        rh_dist = np.linalg.norm(rh_disp, axis=1, keepdims=True)  # (frames-1, 1)
        lh_dist = np.linalg.norm(lh_disp, axis=1, keepdims=True)

        # [0-3] Normalized movement direction
        rh_dir = rh_disp / np.maximum(rh_dist, 1e-6)
        lh_dir = lh_disp / np.maximum(lh_dist, 1e-6)
        features[1:, 0] = rh_dir[:, 0]  # right dx
        features[1:, 1] = rh_dir[:, 1]  # right dy
        features[1:, 2] = lh_dir[:, 0]  # left dx
        features[1:, 3] = lh_dir[:, 1]  # left dy

        # [4-5] Circularity (rolling cross-product, window=7)
        window = min(7, frames - 1)
        for i in range(2, frames):
            start = max(1, i - window + 1)
            # Cross product of consecutive displacement vectors
            rh_seg = rh_disp[start-1:i]
            lh_seg = lh_disp[start-1:i]
            if len(rh_seg) >= 2:
                rh_cross = np.sum(rh_seg[:-1, 0] * rh_seg[1:, 1] - rh_seg[:-1, 1] * rh_seg[1:, 0])
                lh_cross = np.sum(lh_seg[:-1, 0] * lh_seg[1:, 1] - lh_seg[:-1, 1] * lh_seg[1:, 0])
                rh_path = np.sum(rh_dist[start-1:i, 0])
                lh_path = np.sum(lh_dist[start-1:i, 0])
                features[i, 4] = rh_cross / max(rh_path**2, 1e-6) * len(rh_seg)
                features[i, 5] = lh_cross / max(lh_path**2, 1e-6) * len(lh_seg)

        # [6-7] Linearity (rolling net/path, window=7)
        for i in range(1, frames):
            start = max(0, i - window)
            rh_net = np.linalg.norm(rh_centroid[i] - rh_centroid[start])
            lh_net = np.linalg.norm(lh_centroid[i] - lh_centroid[start])
            rh_path = np.sum(rh_dist[start:i, 0]) if i > start else 1e-6
            lh_path = np.sum(lh_dist[start:i, 0]) if i > start else 1e-6
            features[i, 6] = rh_net / max(rh_path, 1e-6)
            features[i, 7] = lh_net / max(lh_path, 1e-6)

        # [8-9] Cumulative path (normalized by shoulder width for scale invariance)
        shoulder_width = np.mean(np.linalg.norm(right_shoulder - left_shoulder, axis=1))
        scale = max(shoulder_width, 1e-4)
        rh_cum = np.cumsum(rh_dist[:, 0])
        lh_cum = np.cumsum(lh_dist[:, 0])
        features[1:, 8] = rh_cum / scale
        features[1:, 9] = lh_cum / scale

        # ── Palm Orientation features (10-15) ──────────────────────────

        # Palm normal from wrist(0), index_mcp(5), pinky_mcp(17)
        for hand_3d, offset in [(right_hand_3d, 10), (left_hand_3d, 13)]:
            wrist = hand_3d[:, 0, :]
            index_mcp = hand_3d[:, 5, :]
            pinky_mcp = hand_3d[:, 17, :]
            v1 = index_mcp - wrist
            v2 = pinky_mcp - wrist
            normal = np.cross(v1, v2)  # (frames, 3)
            norm_mag = np.linalg.norm(normal, axis=1, keepdims=True)
            normal = normal / np.maximum(norm_mag, 1e-6)
            features[:, offset] = normal[:, 0]
            features[:, offset + 1] = normal[:, 1]
            features[:, offset + 2] = normal[:, 2]

        # ── Hand Location features (16-21) ─────────────────────────────

        # [16-19] Position relative to nose
        features[:, 16] = rh_centroid[:, 1] - nose[:, 1]  # rh y rel nose
        features[:, 17] = rh_centroid[:, 0] - nose[:, 0]  # rh x rel nose
        features[:, 18] = lh_centroid[:, 1] - nose[:, 1]  # lh y rel nose
        features[:, 19] = lh_centroid[:, 0] - nose[:, 0]  # lh x rel nose

        # [20] Distance between hands (normalized by shoulder width)
        features[:, 20] = np.linalg.norm(rh_centroid - lh_centroid, axis=1) / scale

        # [21] Vertical zone (right hand): 0=face, 0.5=chest, 1.0=waist
        shoulder_to_nose = np.abs(chest[:, 1] - nose[:, 1])
        rh_y_offset = rh_centroid[:, 1] - nose[:, 1]
        features[:, 21] = np.clip(rh_y_offset / np.maximum(shoulder_to_nose, 1e-4), 0, 2) / 2

        # ── Handshape Dynamics features (22-25) ────────────────────────

        # Compute per-frame hand openness (avg fingertip-to-wrist distance)
        tip_indices = [4, 8, 12, 16, 20]
        for hand, offset in [(right_hand, 22), (left_hand, 23)]:
            wrist = hand[:, 0, :]  # (frames, 2)
            hand_size = np.linalg.norm(hand[:, 9, :] - wrist, axis=1)  # middle MCP to wrist
            tip_dists = np.mean([np.linalg.norm(hand[:, t, :] - wrist, axis=1) for t in tip_indices], axis=0)
            openness = tip_dists / np.maximum(hand_size, 1e-6)
            # Delta openness
            features[1:, offset] = np.diff(openness)

        # Finger spread velocity (avg spread angle change between frames)
        for hand, offset in [(right_hand, 24), (left_hand, 25)]:
            wrist = hand[:, 0, :]
            spreads = np.zeros(frames)
            for i in range(len(tip_indices) - 1):
                v1 = hand[:, tip_indices[i], :] - wrist
                v2 = hand[:, tip_indices[i + 1], :] - wrist
                cos_angle = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6)
                spreads += np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            spreads /= (len(tip_indices) - 1)
            features[1:, offset] = np.diff(spreads)

        # ── Face Region Specificity features (26-29) ──────────────────

        # Face landmarks in 83-point format: indices 75-82
        # 75=nose tip, 76=left mouth, 77=right mouth, 78=chin,
        # 79=left eyebrow, 80=right eyebrow, 81=left eye, 82=right eye
        mouth_center = (pose_sequence[:, 76, :2] + pose_sequence[:, 77, :2]) / 2
        chin = pose_sequence[:, 78, :2]
        forehead = (pose_sequence[:, 79, :2] + pose_sequence[:, 80, :2]) / 2

        # [26-27] Right hand distance to mouth and chin (normalized)
        features[:, 26] = np.linalg.norm(rh_centroid - mouth_center, axis=1) / scale
        features[:, 27] = np.linalg.norm(rh_centroid - chin, axis=1) / scale

        # [28-29] Left hand distance to mouth and chin (normalized)
        features[:, 28] = np.linalg.norm(lh_centroid - mouth_center, axis=1) / scale
        features[:, 29] = np.linalg.norm(lh_centroid - chin, axis=1) / scale

        # ── Hand-Body Contact Detection features (30-31) ──────────────

        # Contact = hand centroid within threshold distance of body landmark
        contact_thresh = 0.15 * scale  # ~15% of shoulder width

        # [30] Right hand contact score (min distance to nearest body landmark / threshold)
        rh_to_nose = np.linalg.norm(rh_centroid - nose, axis=1)
        rh_to_mouth = np.linalg.norm(rh_centroid - mouth_center, axis=1)
        rh_to_chin = np.linalg.norm(rh_centroid - chin, axis=1)
        rh_to_chest = np.linalg.norm(rh_centroid - chest, axis=1)
        rh_min_dist = np.minimum(np.minimum(rh_to_nose, rh_to_mouth), np.minimum(rh_to_chin, rh_to_chest))
        features[:, 30] = np.clip(1.0 - rh_min_dist / max(contact_thresh, 1e-6), 0, 1)

        # [31] Left hand contact score
        lh_to_nose = np.linalg.norm(lh_centroid - nose, axis=1)
        lh_to_mouth = np.linalg.norm(lh_centroid - mouth_center, axis=1)
        lh_to_chin = np.linalg.norm(lh_centroid - chin, axis=1)
        lh_to_chest = np.linalg.norm(lh_centroid - chest, axis=1)
        lh_min_dist = np.minimum(np.minimum(lh_to_nose, lh_to_mouth), np.minimum(lh_to_chin, lh_to_chest))
        features[:, 31] = np.clip(1.0 - lh_min_dist / max(contact_thresh, 1e-6), 0, 1)

        # ── Movement Repetition Count features (32-33) ────────────────

        # Detect direction reversals in y-axis (vertical bounces = repetitions)
        for hand_centroid, offset in [(rh_centroid, 32), (lh_centroid, 33)]:
            y_vel = np.diff(hand_centroid[:, 1])
            # Count sign changes in y-velocity (each reversal = half a repetition)
            sign_changes = np.diff(np.sign(y_vel))
            reversals = np.abs(sign_changes) > 1  # sign changed (not just zero-crossing)
            # Rolling count of reversals (window=frames), normalized
            cumulative = np.cumsum(np.concatenate([[0, 0], reversals.astype(np.float32)]))
            # Encode as repetition density (reversals per 10 frames)
            window = min(10, frames - 2)
            for t in range(2, frames):
                start = max(0, t - window)
                features[t, offset] = (cumulative[t] - cumulative[start]) / max(window, 1) * 10

        # ── Hand-to-Hand Interaction features (34-37) ─────────────────

        # [34] Hands distance (already have [20] but this is un-normalized for contact)
        hands_dist = np.linalg.norm(rh_centroid - lh_centroid, axis=1)

        # [34] Hand-hand contact score (1 when touching, 0 when far)
        hand_contact_thresh = 0.1 * scale
        features[:, 34] = np.clip(1.0 - hands_dist / max(hand_contact_thresh, 1e-6), 0, 1)

        # [35] Hand symmetry: are hands mirroring each other's movement?
        if len(rh_centroid) > 1:
            rh_vel_vec = np.diff(rh_centroid, axis=0)  # (frames-1, 2)
            lh_vel_vec = np.diff(lh_centroid, axis=0)
            # Mirror = opposite x, same y
            mirror_x = -lh_vel_vec[:, 0]  # flip left hand x
            rh_x = rh_vel_vec[:, 0]
            # Correlation of mirrored movements
            rh_norm = np.linalg.norm(rh_vel_vec, axis=1)
            lh_norm = np.linalg.norm(lh_vel_vec, axis=1)
            active = (rh_norm > 1e-4) & (lh_norm > 1e-4)
            if np.sum(active) > 2:
                cos_mirror = np.sum(rh_vel_vec[active] * np.column_stack([mirror_x[active], lh_vel_vec[active, 1]]), axis=1) / (rh_norm[active] * lh_norm[active] + 1e-6)
                features[1:, 35] = 0  # default
                features[1:, 35][active] = cos_mirror

        # [36] Hands crossing: whether hands have crossed over (left hand is right of right hand)
        features[:, 36] = (lh_centroid[:, 0] > rh_centroid[:, 0]).astype(np.float32)

        # [37] Relative vertical position of hands (positive = right hand above left)
        features[:, 37] = (lh_centroid[:, 1] - rh_centroid[:, 1]) / scale

        # ── Wrist Rotation features (38-39) ───────────────────────────

        # Track rotation of hand orientation over time using index-pinky vector angle
        for hand_3d, offset in [(right_hand_3d, 38), (left_hand_3d, 39)]:
            index_tip = hand_3d[:, 5, :2]   # index MCP
            pinky_tip = hand_3d[:, 17, :2]   # pinky MCP
            hand_vec = pinky_tip - index_tip
            hand_angle = np.arctan2(hand_vec[:, 1], hand_vec[:, 0])
            # Angular velocity (rotation speed per frame)
            features[1:, offset] = np.diff(hand_angle)
            # Wrap to [-pi, pi]
            features[:, offset] = np.where(features[:, offset] > np.pi, features[:, offset] - 2*np.pi, features[:, offset])
            features[:, offset] = np.where(features[:, offset] < -np.pi, features[:, offset] + 2*np.pi, features[:, offset])

        return features

    def pad_or_truncate_sequence(self, pose_sequence: np.ndarray, max_length: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """Pad or truncate pose sequence to fixed length."""
        seq_len = len(pose_sequence)
        # Get dimensions from actual data
        num_keypoints = pose_sequence.shape[1] if len(pose_sequence.shape) > 1 else 83
        num_coords = pose_sequence.shape[2] if len(pose_sequence.shape) > 2 else 3

        if seq_len > max_length:
            # Truncate
            pose_sequence = pose_sequence[:max_length]
            attention_mask = np.ones(max_length, dtype=np.float32)
        else:
            # Pad
            padding_length = max_length - seq_len
            padding = np.zeros((padding_length, num_keypoints, num_coords), dtype=np.float32)
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
    """WLASL Dataset for OpenHands architecture with finger features."""

    def __init__(self, pose_files: List[str], labels: List[str], gloss_to_id: Dict[str, int],
                 max_seq_length: int = 256, augment: bool = False, use_finger_features: bool = True,
                 use_motion_features: bool = True, use_spatial_features: bool = True,
                 cache_in_memory: bool = True, cache_dir: str = None):
        self.pose_files = pose_files
        self.labels = labels
        self.gloss_to_id = gloss_to_id
        self.max_seq_length = max_seq_length
        self.augment = augment
        self.use_finger_features = use_finger_features and FINGER_FEATURES_AVAILABLE
        self.use_motion_features = use_motion_features
        self.use_spatial_features = use_spatial_features
        self.processor = WLASLPoseProcessor()
        self.cache_in_memory = cache_in_memory
        self.cache_dir = cache_dir
        self._cache = {}  # idx -> raw pose numpy array (filled on first access)
        self._result_cache = {}  # idx -> full result dict (only for non-augmented datasets)

        # Pre-compute and cache all results for non-augmented datasets
        if cache_in_memory and not augment:
            self._precompute_cache()

    def _get_file_cache_key(self, filepath):
        """Get a cache key for a single file based on its path and processing settings."""
        import hashlib
        # Use filename + settings as key (filename is unique across the pool)
        basename = os.path.basename(filepath)
        key = hashlib.md5(
            f"{basename}|{self.max_seq_length}|{self.use_finger_features}|{self.use_motion_features}".encode()
        ).hexdigest()[:12]
        return f"{basename}_{key}"

    def _precompute_cache(self):
        """Pre-compute all results at init time so every epoch is instant.
        Uses per-file disk caching so results are reusable across different class subsets."""
        import time as _time
        import os

        disk_cache = {}  # file_cache_key -> result (loaded from disk)
        cache_hits = 0
        cache_misses = 0

        # Load existing per-file cache from disk
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_index_path = os.path.join(self.cache_dir, "cache_index.pt")
            if os.path.exists(cache_index_path):
                try:
                    print(f"  PRE-CACHE: Loading per-file disk cache...", end="", flush=True)
                    t0 = _time.perf_counter()
                    disk_cache = torch.load(cache_index_path, weights_only=False)
                    elapsed = _time.perf_counter() - t0
                    print(f" loaded {len(disk_cache)} cached files in {elapsed:.1f}s")
                except Exception as e:
                    print(f" failed ({e}), will recompute all")
                    disk_cache = {}

        # Build result cache, using disk cache where available
        # Save incrementally every 5000 new computations to survive crashes
        SAVE_INTERVAL = 5000
        t0 = _time.perf_counter()
        new_since_save = 0
        for idx in range(len(self.pose_files)):
            file_key = self._get_file_cache_key(self.pose_files[idx])
            if file_key in disk_cache:
                self._result_cache[idx] = disk_cache[file_key]
                cache_hits += 1
            else:
                try:
                    self._result_cache[idx] = self._compute_item(idx)
                except Exception as e:
                    # Skip corrupt/unloadable files - mark for removal
                    print(f"WARNING: Skipping corrupt file {self.pose_files[idx]}: {e}")
                    self._result_cache[idx] = None
                    cache_misses += 1
                    continue
                if self.cache_dir:
                    disk_cache[file_key] = self._result_cache[idx]
                    new_since_save += 1
                cache_misses += 1

            # Progress update + incremental save every SAVE_INTERVAL new computations
            if self.cache_dir and new_since_save >= SAVE_INTERVAL:
                elapsed = _time.perf_counter() - t0
                print(f"  PRE-CACHE: {idx+1}/{len(self.pose_files)} "
                      f"({cache_hits} cached, {cache_misses} computed, {elapsed:.0f}s) — saving checkpoint...",
                      end="", flush=True)
                try:
                    torch.save(disk_cache, cache_index_path)
                    print(f" saved {len(disk_cache)} files", flush=True)
                except Exception as e:
                    print(f" save failed: {e}", flush=True)
                new_since_save = 0
            elif (idx + 1) % 5000 == 0:
                elapsed = _time.perf_counter() - t0
                print(f"  PRE-CACHE: {idx+1}/{len(self.pose_files)} "
                      f"({cache_hits} cached, {cache_misses} computed, {elapsed:.0f}s)", flush=True)

        # Remove corrupt/skipped samples
        skipped = [idx for idx, val in self._result_cache.items() if val is None]
        if skipped:
            print(f"  PRE-CACHE: Removing {len(skipped)} corrupt samples from dataset")
            # Rebuild without corrupt entries
            valid_indices = [i for i in range(len(self.pose_files)) if self._result_cache.get(i) is not None]
            self.pose_files = [self.pose_files[i] for i in valid_indices]
            self.labels = [self.labels[i] for i in valid_indices]
            new_cache = {}
            for new_idx, old_idx in enumerate(valid_indices):
                new_cache[new_idx] = self._result_cache[old_idx]
            self._result_cache = new_cache
            self._cache = {}

        elapsed = _time.perf_counter() - t0
        print(f"  PRE-CACHE: {len(self.pose_files)} samples ready in {elapsed:.1f}s "
              f"({cache_hits} from cache, {cache_misses} computed)")

        # Final save
        if self.cache_dir and new_since_save > 0:
            try:
                t0 = _time.perf_counter()
                torch.save(disk_cache, cache_index_path)
                save_elapsed = _time.perf_counter() - t0
                size_mb = os.path.getsize(cache_index_path) / (1024 * 1024)
                print(f"  PRE-CACHE: Saved {len(disk_cache)} files to disk ({size_mb:.0f} MB) in {save_elapsed:.1f}s")
            except Exception as e:
                print(f"  PRE-CACHE: Warning - failed to save disk cache: {e}")

    def __len__(self):
        return len(self.pose_files)

    def _compute_item(self, idx):
        """Compute a single sample result (load, preprocess, finger features, tensorize)."""
        # Load pose data (from cache if available, otherwise from disk)
        if self.cache_in_memory and idx in self._cache:
            pose_sequence = self._cache[idx].copy()
        else:
            pose_sequence = self.processor.load_pickle_pose(self.pose_files[idx])
            if self.cache_in_memory:
                self._cache[idx] = pose_sequence.copy()

        pose_sequence = self.processor.preprocess_pose_sequence(pose_sequence, augment=self.augment)

        # Extract finger features BEFORE padding (on actual data only)
        if self.use_finger_features:
            finger_features = self.processor.extract_finger_features(pose_sequence)
        else:
            finger_features = None

        # Extract motion features BEFORE padding (on actual data only)
        use_motion = self.use_motion_features
        if use_motion:
            motion_features = self.processor.extract_motion_features(pose_sequence)
        else:
            motion_features = None

        # Extract spatial features BEFORE padding (on actual data only)
        use_spatial = self.use_spatial_features
        if use_spatial:
            spatial_features = self.processor.extract_spatial_features(pose_sequence)
        else:
            spatial_features = None

        # Pad/truncate pose sequence
        pose_sequence, attention_mask = self.processor.pad_or_truncate_sequence(pose_sequence, self.max_seq_length)

        # Pad finger features to match
        if self.use_finger_features:
            seq_len = len(finger_features)
            if seq_len > self.max_seq_length:
                finger_features = finger_features[:self.max_seq_length]
            elif seq_len < self.max_seq_length:
                padding = np.zeros((self.max_seq_length - seq_len, 30), dtype=np.float32)
                finger_features = np.vstack([finger_features, padding])

        # Pad motion features to match
        if use_motion and motion_features is not None:
            seq_len = len(motion_features)
            if seq_len > self.max_seq_length:
                motion_features = motion_features[:self.max_seq_length]
            elif seq_len < self.max_seq_length:
                padding = np.zeros((self.max_seq_length - seq_len, 8), dtype=np.float32)
                motion_features = np.vstack([motion_features, padding])

        # Pad spatial features to match
        if use_spatial and spatial_features is not None:
            n_spatial = spatial_features.shape[1]
            seq_len = len(spatial_features)
            if seq_len > self.max_seq_length:
                spatial_features = spatial_features[:self.max_seq_length]
            elif seq_len < self.max_seq_length:
                padding = np.zeros((self.max_seq_length - seq_len, n_spatial), dtype=np.float32)
                spatial_features = np.vstack([spatial_features, padding])

        # Get label
        label_text = self.labels[idx]
        if label_text not in self.gloss_to_id:
            raise KeyError(f"Label '{label_text}' not found in vocabulary!")
        label_id = self.gloss_to_id[label_text]

        result = {
            'pose_sequence': torch.tensor(pose_sequence, dtype=torch.float32),  # (T, 83, 3)
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float32),  # (T,)
            'label': torch.tensor(label_id, dtype=torch.long)
        }

        if self.use_finger_features:
            result['finger_features'] = torch.tensor(finger_features, dtype=torch.float32)  # (T, 30)

        if use_motion and motion_features is not None:
            result['motion_features'] = torch.tensor(motion_features, dtype=torch.float32)  # (T, 8)

        if use_spatial and spatial_features is not None:
            result['spatial_features'] = torch.tensor(spatial_features, dtype=torch.float32)  # (T, 26)

        return result

    def __getitem__(self, idx):
        # For pre-cached datasets, return directly
        if idx in self._result_cache:
            return self._result_cache[idx]

        # Compute on the fly (augmented datasets or cache disabled)
        result = self._compute_item(idx)

        # Cache for non-augmented datasets
        if self.cache_in_memory and not self.augment:
            self._result_cache[idx] = result

        return result


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
    # Test the implementation - UPDATED for 83-point xyz + finger features
    print("Testing ENHANCED OpenHands Model (83pt xyz + finger features)")
    print("=" * 60)

    config = OpenHandsConfig(vocab_size=50)
    model = OpenHandsModel(config)

    # Test with dummy data - UPDATED dimensions for 83pt xyz + finger features
    batch_size = 4
    seq_len = 100
    dummy_pose = torch.randn(batch_size, seq_len, 83, 3)  # 83 keypoints, 3 channels (xyz)
    dummy_mask = torch.ones(batch_size, seq_len)
    dummy_finger = torch.randn(batch_size, seq_len, 30)  # 30 finger features

    print(f"Pose shape: {dummy_pose.shape}")  # (4, 100, 83, 3)
    print(f"Finger features shape: {dummy_finger.shape}")  # (4, 100, 30)
    print(f"Model config:")
    print(f"  Keypoints: {config.num_pose_keypoints}")
    print(f"  Channels: {config.pose_channels}")
    print(f"  Coord features: {config.pose_coord_features}")
    print(f"  Finger features: {config.finger_features}")
    print(f"  Total features: {config.pose_features}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print()

    # Test WITH finger features
    logits = model(dummy_pose, dummy_mask, dummy_finger)
    print(f"Output shape (with finger features): {logits.shape}")  # Should be (4, 50)

    # Test WITHOUT finger features (backward compatible)
    config_no_finger = OpenHandsConfig(vocab_size=50, use_finger_features=False, pose_features=249)
    model_no_finger = OpenHandsModel(config_no_finger)
    logits_no_finger = model_no_finger(dummy_pose, dummy_mask)
    print(f"Output shape (no finger features): {logits_no_finger.shape}")  # Should be (4, 50)

    print()
    print("SUCCESS: Enhanced OpenHands architecture working!")
    print("Features: 83 keypoints × 3 coords + 30 finger features = 279 features per frame")

# =============================================================================
# Inference Functions
# =============================================================================

def load_model_from_checkpoint(checkpoint_path: str, vocab_size: int = 20):
    """Load OpenHands model from checkpoint directory.

    Returns:
        tuple: (model, id_to_gloss, masked_class_ids)
               masked_class_ids is a list of class indices to mask, or empty list if no masking
    """
    checkpoint_path = Path(checkpoint_path)
    model_file = checkpoint_path / "pytorch_model.bin"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    vocab_file = checkpoint_path / "class_index_mapping.json"
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
    with open(vocab_file, 'r') as f:
        id_to_gloss = json.load(f)

    # Load masked classes config (optional)
    masked_class_ids = []
    masked_classes_file = checkpoint_path / "masked_classes.json"
    if masked_classes_file.exists():
        with open(masked_classes_file, 'r') as f:
            masked_config = json.load(f)
        masked_class_ids = masked_config.get('masked_class_ids', [])
        masked_names = masked_config.get('masked_class_names', [])
        print(f"Loaded class masking: {len(masked_class_ids)} classes masked")
        print(f"  Masked: {', '.join(masked_names)}")

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
    if masked_class_ids:
        print(f"EFFECTIVE CLASSES: {config.vocab_size - len(masked_class_ids)} (after masking)")
    return model, id_to_gloss, masked_class_ids

def predict_pose_file(pickle_path: str, model=None, tokenizer=None, checkpoint_path: str = None,
                       use_finger_features: bool = True, masked_class_ids=None):
    """Predict gloss from a pickle pose file.

    Args:
        pickle_path: Path to pickle file containing pose data
        model: Pre-loaded model (optional)
        tokenizer: Vocabulary mapping (id_to_gloss dict)
        checkpoint_path: Path to checkpoint if model not provided
        use_finger_features: Whether to use finger features
        masked_class_ids: List of class indices to mask (set to -inf before softmax)
    """
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Either model or checkpoint_path must be provided")
        model, tokenizer, masked_class_ids = load_model_from_checkpoint(checkpoint_path)

    processor = WLASLPoseProcessor()
    pose_sequence = processor.load_pickle_pose(pickle_path)
    pose_sequence = processor.preprocess_pose_sequence(pose_sequence, augment=False)

    # Extract finger features if enabled
    finger_tensor = None
    if use_finger_features and FINGER_FEATURES_AVAILABLE and model.config.use_finger_features:
        finger_features = processor.extract_finger_features(pose_sequence)
        # Pad finger features
        seq_len = len(finger_features)
        max_length = 256
        if seq_len > max_length:
            finger_features = finger_features[:max_length]
        elif seq_len < max_length:
            padding = np.zeros((max_length - seq_len, 30), dtype=np.float32)
            finger_features = np.vstack([finger_features, padding])
        finger_tensor = torch.tensor(finger_features, dtype=torch.float32).unsqueeze(0)

    pose_sequence, attention_mask = processor.pad_or_truncate_sequence(pose_sequence, max_length=256)
    pose_tensor = torch.tensor(pose_sequence, dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.tensor(attention_mask, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(pose_tensor, mask_tensor, finger_tensor)

        # Apply class masking if configured
        if masked_class_ids:
            for class_id in masked_class_ids:
                logits[:, class_id] = float('-inf')

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
