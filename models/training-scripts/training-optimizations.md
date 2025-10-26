# Training Pipeline Improvements

This document outlines the training pipeline improvements implemented in `train_asl.py` to enhance model performance and convergence.

## Summary of Changes

| Component          | Previous Configuration        | Improved Configuration          | Impact                    |
|--------------------|-------------------------------|---------------------------------|---------------------------|
| Model Size (Large) | 128 hidden, 6 layers         | 256 hidden, 6 layers            | 4x model capacity         |
| Optimizer          | SGD with momentum            | AdamW                           | Better transformer training|
| Learning Rate      | Fixed 1e-4                   | CosineAnnealing (1e-4 → 1e-6)   | Improved convergence      |
| Weight Decay       | 0.008                        | 0.001                           | Less aggressive regularization |
| Batch Size         | 16                           | 32                              | More stable gradients     |

## Detailed Improvements

### 1. Model Capacity Increase

**Change**: Updated "large" model preset from 128 to 256 hidden dimensions
- **Location**: `train_asl.py`, line 77
- **Previous**: `default_hidden = 128`
- **Current**: `default_hidden = 256`

**Rationale**:
- The openhands-modernized architecture was designed with 256 hidden dimensions
- 128 hidden size is only 50% of the intended capacity
- Model parameters scale with hidden_size², so 256 hidden provides ~4x more capacity
- Critical for learning complex patterns in 50+ class tasks

**Technical Details**:
```python
# Small model (unchanged)
hidden_size=64, layers=3, heads=8  # ~179K parameters

# Large model (improved)
hidden_size=256, layers=6, heads=16  # ~2.7M parameters
intermediate_size=1024  # Automatically scaled (4x hidden_size)
```

### 2. Optimizer Change: SGD → AdamW

**Change**: Switched from SGD with momentum to AdamW optimizer
- **Location**: `train_asl.py`, line 524
- **Previous**: `torch.optim.SGD(lr=1e-4, momentum=0.9, weight_decay=0.008)`
- **Current**: `torch.optim.AdamW(lr=1e-4, betas=(0.9, 0.999), weight_decay=0.001)`

**Rationale**:
- **Adaptive learning rates**: Each parameter gets its own learning rate based on gradient history
- **Better for transformers**: Industry standard for BERT-style architectures
- **Momentum tracking**: First moment (β₁=0.9) and second moment (β₂=0.999) for stable updates
- **Decoupled weight decay**: Proper L2 regularization that doesn't interfere with adaptive rates

**Why SGD was suboptimal**:
- Fixed learning rate for all parameters regardless of gradient magnitude
- Requires more careful tuning of learning rate schedule
- Slower convergence on transformer architectures

### 3. Learning Rate Scheduler: Fixed → Cosine Annealing

**Change**: Added cosine annealing learning rate schedule
- **Location**: `train_asl.py`, lines 527-533
- **Previous**: `scheduler = None` (fixed learning rate)
- **Current**: `CosineAnnealingLR(T_max=num_epochs, eta_min=1e-6)`

**Rationale**:
The learning rate follows a cosine curve from initial value to minimum:

```
LR(epoch) = eta_min + (lr_initial - eta_min) × (1 + cos(π × epoch / T_max)) / 2
```

**Benefits**:
- **Early training**: High LR (1e-4) for rapid initial learning
- **Mid training**: Gradual decrease allows finer adjustments
- **Late training**: Low LR (1e-6) enables precise convergence
- **Smooth transitions**: Cosine curve prevents abrupt changes

**Why fixed LR failed**:
With 1500 epochs at constant 1e-4:
- Steps remain too large in later epochs (1000-1500)
- Model oscillates around optimal weights without settling
- Cannot achieve fine-grained convergence
- Loses potential 5-10% accuracy from poor final convergence

### 4. Weight Decay Reduction

**Change**: Reduced weight decay for augmented dataset training
- **Location**: `train_asl.py`, line 491
- **Previous**: `weight_decay = 0.008` (high regularization)
- **Current**: `weight_decay = 0.001` (balanced regularization)

**Rationale**:
Weight decay adds penalty: `Loss = CrossEntropy + λ × Σ(weights²)`

**Why reduction helps**:
- **Larger model**: 256-hidden model has 4x more capacity than 128-hidden
- **Large dataset**: Augmented dataset (~15x more samples) reduces overfitting risk
- **AdamW compatibility**: AdamW handles regularization better than SGD
- **Flexibility needed**: 50-class task requires learning complex patterns

**Previous issue**:
- 0.008 weight decay was tuned for small model (128 hidden) on smaller dataset
- Too aggressive for larger model, preventing it from using full capacity
- Artificially limited learning of nuanced class distinctions

### 5. Batch Size Increase

**Change**: Increased batch size for more stable training
- **Location**: `train_asl.py`, line 489
- **Previous**: `batch_size = 16`
- **Current**: `batch_size = 32`

**Rationale**:
Larger batches provide:
- **Stable gradients**: Averaging over more samples reduces noise
- **Better generalization**: Less overfitting to individual sample noise
- **AdamW synergy**: Adaptive optimizers benefit from stable gradient estimates
- **Efficient computation**: Better GPU/CPU utilization (though slightly slower per epoch)

**Technical trade-off**:
- Per-epoch time increases (~2x)
- But convergence improves (fewer epochs needed)
- Net result: Better final accuracy for similar total training time

## Configuration Summary

### For Augmented Dataset (50 classes)

```python
# Model
model_size = "large"              # 256 hidden, 6 layers, 16 heads
total_parameters = ~2.7M

# Training
optimizer = AdamW
lr = 1e-4                         # Initial learning rate
lr_schedule = CosineAnnealing     # Decays to 1e-6
weight_decay = 0.001
batch_size = 32
max_epochs = 1500
early_stopping = 30               # Optional

# Data
sequence_length = 128
augmentation = False              # Using pre-generated augmented pool
```

### For Original Dataset (20 classes)

```python
# Model (can use smaller)
model_size = "small"              # 64 hidden, 3 layers, 8 heads

# Training (same improvements apply)
optimizer = AdamW
lr = 1e-4
lr_schedule = CosineAnnealing
weight_decay = 0.001
batch_size = 16                   # Smaller due to limited data
```

## Implementation Notes

### Checkpoint Compatibility

When changing model size, checkpoints become incompatible. Use `--force-fresh` flag:

```bash
python train_asl.py --classes 50 --model-size large --force-fresh
```

### Scheduler Placement

The scheduler steps after validation (line 612):
```python
# Validation
val_results = evaluate_model_improved(model, val_loader, device)

# Step scheduler AFTER validation
scheduler.step()
```

This ensures learning rate updates occur at consistent points in training.

### Architecture Agnostic

All improvements are architecture-agnostic and will benefit any model trained with this pipeline:
- AdamW works with any PyTorch model
- CosineAnnealingLR works with any optimizer
- Hyperparameters apply regardless of architecture

## References

- **AdamW**: [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- **Cosine Annealing**: [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
- **Transformer Training**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
