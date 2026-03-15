# Training Results

## 43-Class Model - Original Dataset (No Augmentation)

**Configuration:**
- Architecture: OpenHands (small) - 64 hidden, 3 layers, 187K params
- Dataset: Original only (no augmentation)
- Dropout: 0.2
- Weight decay: 0.01
- Learning rate: 0.0001
- Early stopping: 15 epochs patience
- Date: 2026-03-10

**Final Results:**
- Best Validation Accuracy (Top-1): **69.23%**
- Final Epoch: 266/1500 (early stopped at 50 epochs no improvement)
- Train Loss: 0.7861
- Train Accuracy: 100.00% (fully overfit)
- Val Accuracy (Top-1): 64.62% (at final epoch)
- Val Accuracy (Top-3): 80.00% (at final epoch)
- LR at stop: 0.000099

**Confusion Pairs Identified (from checkpoint at epoch 198):**

| Class | Top-1 | Top-3 | Main Confusion |
|---|---|---|---|
| COMPUTER | 0.0% | 0.0% | SON (68.8%) |
| NEED | 0.0% | 0.0% | BOWLING (56.2%) |
| WHAT | 0.0% | 0.0% | YEAR (43.8%), CHANGE (31.2%) |
| ORANGE | 6.2% | 25.0% | AFRICA (43.8%) |
| WALK | 6.2% | 68.8% | PLAY (56.2%) |
| WRONG | 6.2% | 81.2% | ORANGE (50.0%), DRINK (37.5%) |
| CHAIR | 31.2% | 62.5% | TIME (25.0%) |
| YEAR | 43.8% | 87.5% | HELP (37.5%) |

**Key Observations:**
- 100% train accuracy with 69.23% val = severe overfitting due to small original dataset (~3-14 samples per class)
- 8 classes below 50% Top-1 accuracy
- COMPUTER->SON and NEED->BOWLING are structural confusion pairs (persist even with augmentation)
- Other 6 confusion pairs are resolved by data augmentation

---

## 43-Class Model - Augmented Dataset (Production)

**Configuration:**
- Architecture: OpenHands (small) - 64 hidden, 3 layers
- Dataset: Augmented (50 samples per class via manifest)
- Dropout: 0.3
- Weight decay: 0.001

**Results:**
- Top-1 Accuracy: **80.97%**
- Top-3 Accuracy: **91.62%**
- All classes at 100% except COMPUTER (0%), NEED (0%), LATER (0%)

**Improvement from augmentation: +11.74 pp (69.23% -> 80.97%)**

---

## 100-Class Model - Healthcare Domain Dataset (Kaggle GPU)

**Configuration:**
- Architecture: OpenHands (small) - 64 hidden, 3 layers, 191K params
- Dataset: Healthcare domain augmented (100 samples per class via manifest)
- Classes: 43 base + 57 healthcare domain (pain, hospital, nurse, medicine, emergency, etc.)
- Augmentations: 30 per video (9 techniques), 100 target samples per class
- Total training samples: 9,988 (100 classes × ~100 samples)
- Dropout: 0.3
- Weight decay: 0.001
- Label smoothing: 0.1
- Learning rate: 0.0001 with warmup (150 epochs) + cosine annealing
- Early stopping: 100 epochs patience
- Finger features: Enabled (279 features = 249 coords + 30 finger)
- Trained on: Kaggle GPU
- Date: 2026-03-12

**Final Results:**
- Best Validation Accuracy (Top-1): **63.16%**
- Best Validation Accuracy (Top-3): **81.28%**
- Best epoch: 467/1500 (manually stopped, plateauing)
- Train Loss: 0.9376
- Train Accuracy: 98.29%
- LR at stop: 0.000087

**Key Observations:**
- 63.16% Top-1 on 100 classes = 63.2x random baseline (1.0%)
- Train/val gap of ~36 points indicates overfitting — next experiment should increase regularization
- Pre-caching reduced epoch time significantly (660s initial cache, then fast epochs)
- Warmup phase (150 epochs) was important — accuracy accelerated after warmup completed

**Next Steps:**
- Try dropout 0.5, weight decay 0.005, label smoothing 0.2
- Consider increasing augmentations to 200 samples per class
- Compare with smart-selected 100-class model (non-domain)

---

## 100-Class Model - Higher Regularization Experiment (Kaggle GPU)

**Configuration:**
- Architecture: OpenHands (small) - 64 hidden, 3 layers, 191K params
- Dataset: Healthcare domain augmented (100 samples per class via manifest)
- Classes: Same 100 as previous run
- Augmentations: 30 per video (9 techniques), 100 target samples per class
- Dropout: 0.5 (increased from 0.3)
- Weight decay: 0.005 (increased from 0.001)
- Label smoothing: 0.2 (increased from 0.1)
- Learning rate: 0.0001 with warmup + cosine annealing
- Early stopping: 100 epochs patience
- Finger features: Enabled (279 features)
- Trained on: Kaggle GPU
- Date: 2026-03-12

**Final Results:**
- Best Validation Accuracy (Top-1): **61.45%**
- Best Validation Accuracy (Top-3): **81.75%**
- Best epoch: 615/1500 (still running)
- Train Accuracy: 95.90%
- Train Loss: 1.8188
- LR at stop: 0.000074

**Key Observations:**
- Higher regularization reduced train/val gap from 36pp to 32pp — less overfitting
- But Top-1 peaked lower (60.06% vs 63.16%) — regularization too aggressive with only 100 samples/class
- The data diversity bottleneck is the limiting factor, not regularization
- Top-3 accuracy similar (~80.6% vs 81.3%) — model learns same broad patterns

**Conclusion:**
- More regularization alone doesn't help — need more diverse training data
- Next experiment: v2 100-gloss list with 200 samples/class (60 aug/video)

---

## 100-Class Model - v2 Gloss List + 12 Augmentation Techniques (Kaggle GPU)

**Configuration:**
- Architecture: OpenHands (small) - 64 hidden, 3 layers, 191K params
- Dataset: v2 healthcare domain augmented (200 samples per class via manifest)
- Classes: 100 (v2 list — 4 swaps from v1: removed cold/hurt/please/sorry, added dizzy/doctor/patient/stress)
- Augmentations: 60 per video (12 techniques), 200 target samples per class
- New techniques: temporal_warp, signer_proportion, viewpoint_3d (added to existing 9)
- Total training samples: 19,991 (from 617 original source videos)
- Dropout: 0.3
- Weight decay: 0.001
- Label smoothing: 0.1
- Learning rate: 0.0001 with warmup + cosine annealing
- Early stopping: 100 epochs patience
- Finger features: Enabled (279 features)
- Trained on: Kaggle GPU
- Date: 2026-03-13

**Final Results:**
- Best Validation Accuracy (Top-1): **~61.04%**
- Best Validation Accuracy (Top-3): **82.32%**
- Best epoch: ~260/1500 (manually stopped)
- Train Accuracy: 98.14%
- Train Loss: 0.9662
- LR at stop: 0.000098

**Key Observations:**
- Doubling augmentation (100→200 samples/class) and adding 3 new techniques did NOT improve Top-1 (61% vs 63% v1, 61% higher-reg)
- Top-3 improved slightly (82.3% vs 81.3%/81.8%) — broader coverage but same precision ceiling
- Train/val gap ~37pp — overfitting persists despite 2x more data
- 617 source videos is the fundamental bottleneck — augmenting the same few videos more does not add real diversity
- Model converges faster (epoch 260 vs 467/615) with more data, but to same plateau

**Conclusion:**
- Augmentation quantity/variety alone cannot overcome the source video diversity limit
- Need more original source videos (real data) to break past ~63% Top-1
- Video crawler tool created to address this — fetch new ASL videos from YouTube

---

## 57-Class Model - Healthcare Subset (Kaggle GPU)

**Configuration:**
- Architecture: OpenHands (small) - 64 hidden, 3 layers, 188K params
- Dataset: 57-class healthcare subset of 100-class v2 (augmented, ~300 samples/class)
- Classes: 57 healthcare-relevant glosses from the 100-class v2 list
- Total training samples: 17,138
- Dropout: 0.3
- Weight decay: 0.001
- Label smoothing: 0.1
- Learning rate: 0.0001 with warmup + cosine annealing
- Early stopping: 100 epochs patience
- Finger features: Enabled (279 features)
- Validation set: 64 samples (~1 per class — too small, unreliable metric)
- Trained on: Kaggle GPU
- Date: 2026-03-13

**Final Results:**
- Best Validation Accuracy (Top-1): **62.50%**
- Final Val Accuracy (Top-1): 54.69%
- Final Val Accuracy (Top-3): 79.69%
- Final epoch: 181/1500 (early stopped after 100 epochs no improvement)
- Train Accuracy: 97.78%
- Train Loss: 0.8638
- LR at stop: 0.000100

**Key Observations:**
- 62.50% Top-1 on 57 classes = 35.7x random baseline (1.75%)
- Validation set too small (64 samples / 57 classes ≈ 1 per class) — accuracy metric is noisy/unreliable
- Similar plateau as 100-class models (~62%) despite fewer classes
- Manifests initially had wrong filenames (v2 `_balance` naming vs actual v1 `_aug_XX` naming) — rebuilt from actual files on disk
- Next run: rebuilt manifests with proper 80/20 train/val split (~59 val samples/class) for reliable metrics

---

## 57-Class Model - Proper Val Split (Kaggle GPU)

**Configuration:**
- Architecture: OpenHands (small) - 64 hidden, 3 layers, 188K params
- Dataset: 57-class healthcare subset, proper stratified family-level split
- Split method: `stratified_family_split.py` — families kept intact, no data leakage
- Train: 287 families, 17,776 samples (71.4%)
- Val: 57 families, 3,601 samples (14.5%) — ~63 per class, different source videos from train
- Test: 57 families, 3,504 samples (14.1%)
- Dropout: 0.3
- Weight decay: 0.001
- Label smoothing: 0.1
- Learning rate: 0.0001 with warmup + cosine annealing
- Early stopping: 15 epochs patience (stopped after 100 epochs no improvement)
- Finger features: Enabled (279 features)
- Trained on: Kaggle GPU
- Date: 2026-03-14

**Final Results:**
- Best Validation Accuracy (Top-1): **69.76%**
- Final Val Accuracy (Top-1): 66.81%
- Final Val Accuracy (Top-3): 82.09%
- Final epoch: 600/1500 (early stopped after 100 epochs no improvement)
- Train Accuracy: 99.50%
- Train Loss: 0.7671
- LR at stop: 0.000075

**Key Observations:**
- 69.76% Top-1 on 57 classes = 39.9x random baseline (1.75%)
- First result with proper val set (3,601 samples vs previous 64) — this is a reliable metric
- Train/val gap ~30pp — still overfitting but less than 100-class models (~37pp)
- Higher accuracy than 100-class models (69.76% vs ~63%) — fewer classes helps
- Top-3 at 82% — model has reasonable broad coverage
- Next step: run per-class accuracy analysis to identify and drop classes below 80% Top-1

---

## 77-Class Model - Combined Pruned Classes (Kaggle GPU)

**Configuration:**
- Architecture: OpenHands (small) - 64 hidden, 3 layers
- Dataset: 77-class combined (40 from 43-class model + 38 from 57-class healthcare, -1 overlap DOCTOR)
- Class selection: Only classes with top1>=80% AND top3>=90% from prior models
- Train: 30,275 samples (520 families)
- Val: 4,629 samples (79 families, ~60 per class)
- Test: 4,636 samples (79 families)
- Dropout: 0.3
- Weight decay: 0.001
- Label smoothing: 0.1
- Learning rate: 0.0001 with warmup + cosine annealing
- Early stopping: 100 epochs patience
- Finger features: Enabled (279 features)
- Trained on: Kaggle GPU
- Date: 2026-03-14

**Results (in progress — epoch 250):**
- Best Validation Accuracy (Top-1): **68.39%**
- Best Validation Accuracy (Top-3): **89.41%**
- Train Accuracy: 97.31%
- Train Loss: 1.0476
- LR: 0.000099

**Key Observations:**
- 68.39% Top-1 on 77 classes = 52.6x random baseline (1.3%)
- Top-3 at 89.4% — strong broad coverage
- Train/val gap ~29pp — similar to 57-class model
- Still training (epoch 250/1500, early stopping at 100 epochs no improvement)
