# OpenHands-Modernized Architecture Improvements

This document outlines the key architectural improvements made in the OpenHands-Modernized implementation compared to the original OpenHands model.

## Architecture Comparison

| Component               | Original OpenHands           | OpenHands-Modernized             |
|-------------------------|------------------------------|----------------------------------|
| Pose Keypoints          | 27 (upper body only)         | 75 (body + hands)                |
| Feature Dimension       | 54 (27 × 2)                  | 150 (75 × 2)                     |
| Architecture            | Fixed (64h, 3L, 8heads)      | Configurable (Small/Large)       |
| Parameters (Small)      | ~179K                        | ~179K                            |
| Parameters (Large)      | N/A                          | ~2.7M                            |
| Activation              | ReLU                         | GELU                             |

## Key Improvements

### 1. Enhanced Pose Representation (27 → 75 keypoints)

**Original**: Used 27 keypoints representing upper body pose only
- Limited hand detail (only 10 hand points)
- Insufficient for capturing fine handshapes critical for ASL

**Modernized**: Uses full 75-point MediaPipe representation
- Full body pose: 33 points (complete skeleton)
- Complete hand skeletons: 42 points (21 per hand)
- Face excluded: Removed as noisy and not critical for ASL

### 2. Configurable Model Size

**Original**: Single fixed configuration
- Hidden size: 64
- Layers: 3
- Attention heads: 8
- Total parameters: ~179K

**Modernized**: Two preset configurations

**Small Model** (CPU-optimized):
- Hidden size: 64
- Layers: 3
- Attention heads: 8
- Total parameters: ~179K
- Use case: Fast training, small datasets (≤20 classes)

**Large Model** (Performance-optimized):
- Hidden size: 256
- Layers: 6
- Attention heads: 16
- Total parameters: ~2.7M
- Use case: High accuracy, larger datasets (50+ classes)

### 3. Modern Activation Function (ReLU → GELU)

**Original**: Used ReLU activation
- Standard but can cause "dead neurons"
- Harsh gradient cutoff at zero

**Modernized**: Uses GELU (Gaussian Error Linear Unit)
- Smoother gradients for better optimization
- Industry standard for transformer models (BERT, GPT)
- Better convergence in deep networks

## Implementation Details

### Model Architecture Files
- `src/openhands_modernized.py`: Core model implementation
- `models/training-scripts/train_asl.py`: Training pipeline

### Key Classes
- `OpenHandsConfig`: Configuration dataclass for model architecture
- `OpenHandsModel`: Main model class with transformer encoder
- `MediaPipeSubset`: 75-point pose extraction from MediaPipe data
- `CompactTransformer`: BERT-style transformer with GELU activation

## References

- Original OpenHands: [AI4Bharat OpenHands](https://github.com/AI4Bharat/OpenHands)
- MediaPipe Holistic: Full 576-point pose tracking
- GELU Paper: [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415)
