#!/usr/bin/env python3
"""
Test data shapes for transformer training.
"""

import sys
import os
import torch
import numpy as np
sys.path.append('.')

from openhands_modernized import WLASLDatasetLoader

def test_data_shapes():
    """Test that data shapes are correct for transformer training."""

    print("Testing data shapes for transformer training...")

    # Load a small dataset
    dataset_root = r'C:\Users\padwe\OneDrive\WLASL-proj\wlasl-kaggle\wlasl_poses_complete\conservative_split_50_class'
    loader = WLASLDatasetLoader(dataset_root)
    train_dataset, val_dataset, unique_glosses = loader.load_dataset(
        max_files_per_class=1,
        max_seq_length=150
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Number of classes: {len(unique_glosses)}")

    if len(train_dataset) > 0:
        sample = train_dataset[0]
        pose_seq = sample['pose_sequence']
        attn_mask = sample['attention_mask']
        label = sample['label']

        print(f"\nSample data shapes:")
        print(f"  Pose sequence: {pose_seq.shape} (expected: (150, 1152))")
        print(f"  Attention mask: {attn_mask.shape} (expected: (150,))")
        print(f"  Label: {label} (type: {type(label)})")

        # Check data range
        print(f"\nData ranges:")
        print(f"  Pose data: {pose_seq.min():.3f} to {pose_seq.max():.3f}")
        print(f"  Non-zero poses: {(pose_seq != 0).sum()} / {pose_seq.numel()}")

        # Test batch creation
        from torch.utils.data import DataLoader
        dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False)
        batch = next(iter(dataloader))

        print(f"\nBatch shapes:")
        print(f"  Batch pose sequences: {batch['pose_sequence'].shape} (expected: (2, 150, 1152))")
        print(f"  Batch attention masks: {batch['attention_mask'].shape} (expected: (2, 150))")
        print(f"  Batch labels: {batch['label'].shape} (expected: (2,))")

        # Test with transformer model
        try:
            from pose_transformer_architecture import PoseTransformer, PoseTransformerConfig

            config = PoseTransformerConfig(
                pose_dim=1152,
                vocab_size=len(unique_glosses),
                max_seq_length=150
            )

            model = PoseTransformer(config)
            model.eval()

            with torch.no_grad():
                output = model(batch['pose_sequence'], batch['attention_mask'])
                print(f"\nTransformer output shape: {output['logits'].shape} (expected: (2, {len(unique_glosses)}))")
                print("✓ Transformer forward pass successful!")

        except Exception as e:
            print(f"\n✗ Transformer test failed: {e}")

        return True
    else:
        print("✗ No data samples found!")
        return False

if __name__ == "__main__":
    test_data_shapes()