#!/usr/bin/env python3
"""
pose_transformer_architecture.py
Modern Transformer-based architecture for ASL pose sequence classification
Designed to replace CNN+LSTM for 50+ class breakthrough performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class PoseTransformerConfig:
    """Configuration for Pose Transformer model."""
    pose_dim: int = 1152  # 576 keypoints * 2 coords
    vocab_size: int = 50
    max_seq_length: int = 150

    # Transformer architecture
    d_model: int = 512  # Model dimension
    nhead: int = 8      # Number of attention heads
    num_encoder_layers: int = 6  # Transformer encoder layers
    dim_feedforward: int = 2048  # FFN dimension
    dropout: float = 0.1

    # Pose processing
    pose_embed_dim: int = 256  # Pose embedding dimension
    temporal_pool: str = "attention"  # "mean", "max", "attention"


class PoseEmbedding(nn.Module):
    """Embed pose sequences into model dimension."""

    def __init__(self, config: PoseTransformerConfig):
        super().__init__()
        self.config = config

        # Project pose features to embedding dimension
        self.pose_projection = nn.Sequential(
            nn.Linear(config.pose_dim, config.pose_embed_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.pose_embed_dim, config.d_model)
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, pose_sequences: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose_sequences: (batch_size, seq_len, pose_dim)
        Returns:
            embedded_poses: (batch_size, seq_len, d_model)
        """
        # Project poses to model dimension
        embedded = self.pose_projection(pose_sequences)
        embedded = self.layer_norm(embedded)

        return embedded


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)  # (batch_size, seq_len, d_model)
        return x


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence aggregation."""

    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len) - True for valid positions
        Returns:
            pooled: (batch_size, d_model)
        """
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)

        # Convert attention mask for MultiheadAttention (False for positions to mask)
        if attention_mask is not None:
            attention_mask_bool = attention_mask.bool()
            key_padding_mask = ~attention_mask_bool  # Invert mask
        else:
            key_padding_mask = None

        attended, _ = self.attention(
            query, x, x,
            key_padding_mask=key_padding_mask
        )

        return attended.squeeze(1)  # (batch_size, d_model)


class PoseTransformer(nn.Module):
    """
    Modern Transformer architecture for pose sequence classification.

    Architecture:
    1. Pose Embedding: Project poses to model dimension
    2. Positional Encoding: Add temporal information
    3. Transformer Encoder: Multi-head self-attention layers
    4. Temporal Pooling: Aggregate sequence for classification
    5. Classification Head: Final prediction
    """

    def __init__(self, config: PoseTransformerConfig):
        super().__init__()
        self.config = config

        # Pose embedding
        self.pose_embedding = PoseEmbedding(config)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )

        # Temporal pooling
        if config.temporal_pool == "attention":
            self.pooling = AttentionPooling(config.d_model)
        elif config.temporal_pool == "mean":
            self.pooling = None  # Will use manual mean pooling
        elif config.temporal_pool == "max":
            self.pooling = None  # Will use manual max pooling
        else:
            raise ValueError(f"Unknown temporal_pool: {config.temporal_pool}")

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.vocab_size)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, pose_sequences: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        """
        Args:
            pose_sequences: (batch_size, seq_len, pose_dim)
            attention_mask: (batch_size, seq_len) - True for valid positions
            labels: (batch_size,) - class labels for loss calculation
        Returns:
            dict with 'loss' and 'logits' if labels provided, else just logits tensor
        """
        # 1. Embed poses
        embedded = self.pose_embedding(pose_sequences)  # (batch_size, seq_len, d_model)

        # 2. Add positional encoding
        embedded = self.pos_encoding(embedded)

        # 3. Apply dropout
        embedded = F.dropout(embedded, p=self.config.dropout, training=self.training)

        # 4. Transformer encoding
        # Create source key padding mask for transformer
        if attention_mask is not None:
            # Convert to boolean and invert for transformer (False = mask, True = keep)
            attention_mask_bool = attention_mask.bool()
            src_key_padding_mask = ~attention_mask_bool
        else:
            src_key_padding_mask = None

        encoded = self.transformer_encoder(
            embedded,
            src_key_padding_mask=src_key_padding_mask
        )  # (batch_size, seq_len, d_model)

        # 5. Temporal pooling
        if self.config.temporal_pool == "attention":
            pooled = self.pooling(encoded, attention_mask)
        elif self.config.temporal_pool == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).float()
                pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                pooled = encoded.mean(dim=1)
        elif self.config.temporal_pool == "max":
            if attention_mask is not None:
                # Masked max pooling
                mask_expanded = attention_mask.unsqueeze(-1).bool()
                encoded_masked = encoded.masked_fill(~mask_expanded, float('-inf'))
                pooled = encoded_masked.max(dim=1)[0]
            else:
                pooled = encoded.max(dim=1)[0]

        # 6. Classification
        logits = self.classifier(pooled)  # (batch_size, vocab_size)

        # Calculate loss if labels provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {'loss': loss, 'logits': logits}
        else:
            return logits


def create_pose_transformer_config(num_classes: int) -> PoseTransformerConfig:
    """Create optimized transformer config based on number of classes."""

    if num_classes <= 20:
        # Smaller model for 20 classes
        return PoseTransformerConfig(
            vocab_size=num_classes,
            d_model=256,
            nhead=8,
            num_encoder_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            pose_embed_dim=128
        )
    elif num_classes <= 50:
        # Medium model for 50 classes - optimized for breakthrough
        return PoseTransformerConfig(
            vocab_size=num_classes,
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            dim_feedforward=2048,
            dropout=0.15,
            pose_embed_dim=256,
            temporal_pool="attention"
        )
    else:
        # Large model for 100+ classes
        return PoseTransformerConfig(
            vocab_size=num_classes,
            d_model=768,
            nhead=12,
            num_encoder_layers=8,
            dim_feedforward=3072,
            dropout=0.2,
            pose_embed_dim=384
        )


if __name__ == "__main__":
    # Test the transformer architecture
    config = create_pose_transformer_config(50)
    model = PoseTransformer(config)

    # Test forward pass
    batch_size, seq_len = 4, 100
    pose_sequences = torch.randn(batch_size, seq_len, config.pose_dim)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    logits = model(pose_sequences, attention_mask)
    print(f"Input shape: {pose_sequences.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ Transformer architecture test successful!")