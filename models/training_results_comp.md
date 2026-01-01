# Sign Language Recognition - Training Results Comparison

## Literature Benchmarks

### Video-Based Results (WLASL 2019-2024)
| Model | WLASL100 | WLASL300 | WLASL1000 | WLASL2000 | Approach |
|-------|----------|----------|-----------|-----------|----------|
| I3D Baseline | 65.89% | 56.14% | 47.33% | 32.48% | 3D CNN on video |
| I3D + Transformer | 74.47% | 57.86% | 45.13% | 34.66% | Hybrid video+transformer |
| Multi-stream CNN | 81.38% | 73.43% | 63.61% | 47.26% | SOTA 2021 |
| Sign2Pose | 80.9% | 64.21% | 49.46% | 38.65% | Pose-guided video |

### Pose-Based Results (OpenHands 2021)
| Dataset | Accuracy | Classes | Approach |
|---------|----------|---------|----------|
| WLASL2000 | 30.6% | 2000 | SL-GCN on pose sequences |
| INCLUDE (Indian) | 93.5% | 263 | SL-GCN on pose sequences |
| AUTSL (Turkish) | 95.02% | 226 | SL-GCN on pose sequences |
| GSL (Greek) | 95.4% | 310 | SL-GCN on pose sequences |
| CSL (Chinese) | 94.8% | 178 | SL-GCN on pose sequences |
| LSA64 (Argentinian) | 97.8% | 64 | SL-GCN on pose sequences |

## Our Experimental Results

### CNN-LSTM Experiments
| Experiment | Classes | Architecture | Keypoints | Train Samples | Val Acc | Test Acc | Top-3 Acc | Baseline | Improvement | Status |
|------------|---------|--------------|-----------|---------------|---------|----------|-----------|----------|-------------|--------|
| Exp 1 | 100 | Transformer | 27 | ~900 | 1.49% | - | - | 1.0% | 1.49x | Failed |
| Exp 2 | 100 | CNN+LSTM | 27 | ~900 | 1.50% | - | - | 1.0% | 1.5x | Failed |
| Exp 3 | 20 | CNN+LSTM | 27 | ~274 | 10.91% | - | - | 5.0% | 2.2x | Success |
| Exp 4a | 50 | CNN+LSTM | 27 | ~614 | 4.07% | - | - | 2.0% | 2.0x | Partial |
| Exp 4b | 50 | CNN+LSTM (100 epochs) | 27 | ~614 | 4.88% | - | - | 2.0% | 2.4x | Overfitting |
| Exp 4c | 50 | CNN+LSTM (Reduced) | 27 | ~614 | 1.63% | - | - | 2.0% | 0.8x | Failed |
| Exp 4d | 50 | CNN+LSTM + Class Balance | 27 | ~614 | 8.94% | - | - | 2.0% | 4.5x | Success |
| Enhanced 20-Class | 20 | CNN+LSTM | 27 | ~2460 | - | 27.61% | - | 5.0% | 5.5x | Success |
| 50-Class v1 | 50 | CNN+LSTM | 27 | ~5520 | - | 8.18% | - | 2.0% | 4.1x | Below Target |
| 50-Class v2 | 50 | CNN+LSTM | 27 | ~5520 | - | 5.89% | 13.72% | 2.0% | 2.9x | Regression |

### OpenHands Transformer Experiments
| Experiment | Classes | Architecture | Keypoints | Train Samples | Val Acc | Test Acc | Top-3 Acc | Baseline | Improvement | Status |
|------------|---------|--------------|-----------|---------------|---------|----------|-----------|----------|-------------|--------|
| OpenHands 27pt Baseline | 20 | OpenHands (3-layer) | 27 | ~1200 | 10.91% | - | - | 5.0% | 2.2x | Baseline |
| OpenHands 75pt Original | 20 | OpenHands (6-layer) | 75 | ~1200 | 20.00% | - | - | 5.0% | 4.0x | Major Improvement |
| OpenHands 75pt Augmented | 20 | OpenHands (6-layer) | 75 | ~775 | **42.47%** | - | 75.29% | 5.0% | **8.5x** | **BEST for 20** |
| OpenHands 50-Class | 50 | OpenHands (3-layer) | 27 | ~6754 | 9.38% | - | 18.27% | 2.0% | 4.7x | Success |
| OpenHands 50-Class Large (Aug) | 50 | OpenHands (6-layer) | 75 | ~8,892 | 43.64% | - | 58.18% | 2.0% | 21.8x | Overfitting |
| OpenHands 50-Class Small (Aug+D0.25) | 50 | OpenHands (3-layer) | 75 | ~9,234 | **47.27%** | - | 50.91% | 2.0% | **23.6x** | **BEST for 50** |
| OpenHands 100-Class (Aug) | 100 | OpenHands (3-layer) | 75 | ~18,468 | 46.81% | ~40% | 69.55% | 1.0% | 46.8x | Previous Best |
| OpenHands 100-Class Large (Aug+3D+Finger) | 100 | OpenHands (6-layer) | 83 | ~18,468 | **48.65%** | - | - | 1.0% | **48.7x** | **BEST for 100** |

### Architecture Specifications

#### CNN-LSTM Models
| Model | Parameters | Channels | LSTM Hidden | Dropout | Regularization |
|-------|-----------|----------|-------------|---------|----------------|
| Exp 1-2 | ~700K | 64→128→256 | 160 | Standard | Basic |
| Exp 3 | ~450K | 64→128 | 128 | Standard | Basic |
| Exp 4a-b | ~718K | 64→128→256 | 160 | Enhanced | Enhanced Aug |
| Exp 4c-d | ~447K | 64→128 | 128 | Enhanced | Class Balance + Aug |

#### OpenHands Transformer Models
| Model | Parameters | Hidden Size | Layers | Attention Heads | Intermediate Size | Dropout |
|-------|-----------|-------------|--------|-----------------|-------------------|---------|
| 27pt Baseline | ~175K | 64 | 3 | 8 | 256 | 0.1 |
| 75pt Original | ~8M | 128 | 6 | 16 | 512 | 0.1 |
| 75pt Augmented | ~8M | 128 | 6 | 16 | 512 | 0.1 |
| 50-Class | ~175K | 64 | 3 | 8 | 256 | 0.1 |
| 50-Class Large (Aug) | ~4.8M | 256 | 6 | 16 | 1024 | 0.25 |
| 50-Class Small (Aug+D0.25) | ~175K | 64 | 3 | 8 | 256 | 0.25 |
| 100-Class (Aug) | ~175K | 64 | 3 | 8 | 256 | 0.1 |
| 100-Class Large (Aug+3D+Finger) | ~19M | 256 | 6 | 16 | 1024 | 0.2 |
