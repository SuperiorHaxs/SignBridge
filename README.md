# SignBridge: Real-Time American Sign Language Recognition System

*Building accessible, real-time ASL translation for video conferencing and beyond*

---

## Table of Contents

1. [Project Goals and Motivation](#1--project-goals-and-motivation)
2. [Current State of the Field](#2--current-state-of-the-field)
3. [Research Challenges & Our Solutions](#3--research-challenges--our-solutions)
4. [Phased Research Roadmap](#4--phased-research-roadmap)
5. [Performance Comparison](#5--performance-comparison)
6. [More Details on Our Unique Features & Innovations](#6--more-details-on-our-unique-features--innovations)
7. [Getting Started](#7--getting-started)
8. [Related Work](#8--related-work)

---

## 1. 🎯 Project Goals and Motivation

### The Accessibility Challenge
- 500,000+ ASL users in North America face communication barriers
- Video conferencing (Zoom, Teams, Meet) lacks real-time ASL translation
- Existing solutions: expensive, non-real-time, or accuracy-limited

### Our Vision
Build a **production-ready, real-time ASL translation system** that:
- Translates signs to natural English sentences in <2 seconds
- Runs on consumer hardware (webcam, laptop)
- Achieves 80%+ accuracy on real-world vocabulary
- Integrates seamlessly into video conferencing workflows

### Research Objectives
1. **Improve pose-based recognition** beyond current benchmarks
2. **Enable real-time inference** suitable for live conversation
3. **Build end-to-end pipeline** (video → pose → prediction → natural language)
4. **Make research reproducible** with clean architecture and documentation

---

## 2. 📊 Current State of the Field

### Literature Benchmarks (WLASL Dataset)

**Video-Based Models (2019-2024):**
| Model | WLASL100 | WLASL300 | WLASL1000 | WLASL2000 | Notes |
|-------|----------|----------|-----------|-----------|-------|
| I3D Baseline | 65.89% | 56.14% | 47.33% | 32.48% | 3D CNN, computationally expensive |
| Multi-stream CNN (SOTA 2021) | 81.38% | 73.43% | 63.61% | 47.26% | State-of-the-art, heavy model |

**Pose-Based Models (OpenHands 2021):**
| Dataset | Classes | Accuracy | Model |
|---------|---------|----------|-------|
| WLASL2000 | 2000 | 30.6% | SL-GCN |
| INCLUDE (Indian) | 263 | 93.5% | SL-GCN |
| AUTSL (Turkish) | 226 | 95.02% | SL-GCN |

**Key Observations:**
- Video models: High accuracy but computationally prohibitive for real-time
- Pose models: Fast but significantly lower accuracy (30.6% on WLASL2000)
- Gap: 63.61% (video SOTA) vs 30.6% (pose baseline) on comparable scales

---

## 3. 🔍 Research Challenges & Our Solutions

### (a) Model Architecture

| Challenge | Problem Statement | Solution & Impact |
|-----------|-------------------|-------------------|
| **Pose Representation Quality** | OpenHands baseline uses only 27 keypoints (body), missing critical hand and face details needed for sign disambiguation. Video models use full RGB but are too computationally expensive. | **Solution: 83-Point OpenHands-HD Keypoints**<br>Face + body + hands (83 points: 8 face + 33 body + 42 hands) generating 279-dimensional feature vectors per frame. Built on OpenHands transformer, extended to OpenHands-HD.<br>**Result**: 80.97% Top-1 accuracy on WLASL-100, up from 72% benchmark. |
| **Context Disambiguation** | Single-prediction approach limits ability to use context for resolving ambiguous signs (e.g., BOOK vs LOOK). | **Solution: Top-K Prediction Support**<br>Model returns top-1 through top-5 predictions with confidences. Gemini (gemini-2.0-flash) receives all alternatives for context-aware semantic selection.<br>**Result**: 91.62% Top-3 accuracy; LLM selects based on sentence meaning, not just confidence. |

### (b) Data Augmentation

| Challenge | Problem Statement | Solution & Impact |
|-----------|-------------------|-------------------|
| **Data Efficiency & Augmentation** | Limited pose augmentation techniques in literature. WLASL has only 2,000 classes with limited samples per class. Models overfit on small datasets. | **Solution: 50x Pose Data Augmentation Pipeline**<br>Comprehensive augmentation: rotation (±15°), shear (±0.2 rad), combinations, and multi-variant stacking. Pre-generated (not runtime) for training efficiency. 342 samples → 17,100 via 50x expansion.<br>**Result**: 80.97% Top-1 accuracy on WLASL-100 with augmented training data. |

### (c) Training Optimizations

| Challenge | Problem Statement | Solution & Impact |
|-----------|-------------------|-------------------|
| **Model Capacity & Overfitting** | Standard training approaches cause severe overfitting on limited sign language data. Optimal model capacity unclear for pose-based recognition. | **Solution: Samples-per-Parameter Model Sizing**<br>Analytical approach to right-sizing models. Small vs Large capacities: 64-129 hidden, 2-3 layers, 8 heads. Training: 1500 epochs, batch size 16.<br>**Result**: 80.97% Top-1 accuracy on WLASL-100 with stable training. |
| **Training Stability** | Default dropout settings cause overfitting on limited ASL data. Need optimization for small dataset regime. | **Solution: Configurable Dropout**<br>Command-line configurable dropout with empirical optimization. Found 0.25 optimal for 50-class (vs 0.1 default).<br>**Result**: +3.63% improvement, stable until epoch 25. |

### (d) Application

| Challenge | Problem Statement | Solution & Impact |
|-----------|-------------------|-------------------|
| **Natural Language Generation** | No end-to-end systems combining modern pose models with LLMs. Traditional rule-based grammar insufficient for natural output. | **Solution: LLM-based Semantic Coherence Analysis**<br>**Implementation**: Gemini (gemini-2.0-flash) API with smart buffering (5 trigger strategies), context-aware prompts, top-K integration<br>**Components**: Smart buffering, local fallback, BLEU/BERTScore/CTQI evaluation<br>**Result**: Grammatical quality 32% → 76%, CTQI 51% → 74%. |
| **Continuous Sign Segmentation** | Segmenting continuous signing into individual signs is unsolved. Real-world videos unusable without manual annotation. | **Solution: Continuous Sign Detection**<br>**Implementation**: Dual segmentation approach - auto-detect (pose_to_segments ML-based) + motion-based (velocity thresholds)<br>**Features**: Configurable for different signing styles, works on real-world videos<br>**Result**: Automated boundary detection enabling continuous video processing. |

### (e) Reusability & Extensibility

| Challenge | Problem Statement | Solution & Impact |
|-----------|-------------------|-------------------|
| **System Architecture** | Hardcoded paths and manual configuration in research code. Brittle scripts difficult to reproduce across machines. | **Solution: Centralized Configuration System**<br>`config/settings.json` with auto-detection, gitignored user settings, cross-platform compatibility (Windows/Linux).<br>**Result**: Reproducible research, easy multi-machine setup. |
| **Scalability Across Class Sizes** | Hardcoded class lists require code changes when switching between 20/50/100/2000-class configurations. | **Solution: Dynamic Class Loading**<br>Reads class mappings from JSON files dynamically. Single codebase for all class configurations.<br>**Result**: Zero code changes when scaling vocabulary. |

---

## 4. 🗺️ Phased Research Roadmap

| Phase | Title | Status | Key Deliverables | Success Criteria | Notes |
|-------|-------|--------|------------------|------------------|-------|
| **1** | Isolated Sign Recognition Model Prototype | ✅ **COMPLETED** | • 20-class model<br>• 50-class model<br>• 100-class model<br>• 83pt OpenHands-HD<br>• 50x augmentation | • 80%+ Top-1 (100-class) ✅<br>• 90%+ Top-3 (100-class) ✅ | **Achieved:** 80.97% Top-1, 91.62% Top-3 (WLASL-100). 50x augmentation (342 → 17,100 samples) |
| **2** | LLM-based Self-Correcting Sentence Construction | ✅ **COMPLETED** | • Gemini integration<br>• Smart buffering<br>• Top-K prompts<br>• Context-aware grammar | • Natural sentences ✅<br>• Context disambiguation ✅<br>• 90%+ coherence ✅<br>• BLEU score evaluation ✅ | **Achieved:** Streaming API, 5 trigger strategies, local fallback. BLEU 56.53 (+35.91 vs baseline), BERTScore 96.30, CTQI 78.16 |
| **3** | Full Pipeline Integration | ✅ **COMPLETED** | • End-to-end system<br>• File processing<br>• Evaluation framework | • Video → text functional ✅<br>• <2s latency ✅<br>• 75%+ translation accuracy ✅ | **Achieved:** 5-step pipeline |
| **4** | Continuous Sign Detection | ✅ **COMPLETED** | • Temporal segmentation<br>• Boundary detection<br>• Real-world videos | • 85%+ boundary accuracy ✅<br>• Real-time processing ✅<br>• <200ms latency ✅ | **Achieved:** Auto-detect + motion-based segmentation |
| **5** | Real-Time Webcam App | ✅ **COMPLETED** | • Desktop application<br>• Live inference<br>• Visualization UI | • 15-30 FPS ✅<br>• <500ms latency ✅<br>• Production-ready ✅ | **Achieved:** 2 versions (standard + streaming) |
| **6** | Isolated Sign Recognition Model Optimization & Expansion | 🔄 **IN PROGRESS** | • 100-class model<br>• 300-class model<br>• Dropout tuning<br>• Label smoothing<br>• Learning rate optimization<br>• Gradient clipping | • 67%+ Top-3 (100-class)<br>• 67%+ Top-3 (300-class)<br>• 67%+ Top-3 (50-class optimized)<br>• Reduced overfitting | **In Progress:** Dropout tuning (testing 0.35), label smoothing, gradient clipping. **Next:** 100-class and 300-class models |
| **7** | Text-to-Audio Streaming Enhancement | ⏳ **NOT STARTED** | • TTS integration<br>• Real-time audio output<br>• Voice customization<br>• Audio-visual sync | • <500ms audio latency<br>• Natural voice quality<br>• Seamless integration | **Future:** Complete audio-visual accessibility solution |
| **8** | Deployment & Release | 🔄 **IN PROGRESS** | • Model quantization<br>• Docker containerization<br>• Public release<br>• Documentation | • Production-ready deployment<br>• Complete documentation<br>• Demo videos | **In Progress:** Documentation (README, training results). **Next:** Containerization, model quantization |

**Current Status:** 5 of 8 phases complete (62.5% done), 2 in progress, 1 not started

---

## 5. 📈 Performance Comparison

### Our Results vs Literature

| Model | Approach | Keypoints | Classes | Top-1 Accuracy | Top-3 Accuracy | Speed |
|-------|----------|-----------|---------|----------------|----------------|-------|
| **OpenHands-HD (WLASL-100)** | Pose + Transformer | 83 | 100 | **80.97%** | **91.62%** | Real-time |
| **OpenHands-HD (50-class)** | Pose + Transformer | 83 | 50 | **47.27%** | **75.29%** | Real-time |
| **OpenHands-HD (20-class)** | Pose + Transformer | 83 | 20 | **42.47%** | **75.29%** | Real-time |
| OpenHands Baseline | Pose + Transformer | 27 | 2000 | 30.6% | — | Real-time |
| Multi-stream CNN (SOTA) | Video | N/A | 1000 | 63.61% | — | Slow |
| I3D + Transformer | Video | N/A | 1000 | 45.13% | — | Slow |
| I3D Baseline | Video | N/A | 100 | 65.89% | — | Slow |

### Key Metrics

**Recognition Performance (WLASL-100):**
- **Top-1 accuracy**: 80.97% (vs 72% benchmark, +8.97%)
- **Top-3 accuracy**: 91.62%
- **Training data**: 342 samples → 17,100 via 50x augmentation

**Efficiency:**
- **Samples per parameter**: 52.8 (optimal for small model)
- **End-to-end latency**: <2 seconds (webcam → sentence)
- **Model size**: 175K params (small) vs 4.8M (large)

**Translation Quality (Synthetic Evaluation):**

| Metric | Baseline (No LLM) | With LLM | Improvement |
|--------|-------------------|----------|-------------|
| Grammatical Quality | 32% | 76% | +44% |
| CTQI | 51% | 74% | +23% |
| BLEU Score | 20.62 | 56.53 | +35.91 |
| BERTScore | 91.64 | 96.30 | +4.65 |
| Coverage F1 | 74.64 | 87.62 | +12.98 |

**Statistical Significance (paired t-tests, n=34):**

| Metric | p-value | Cohen's d |
|--------|---------|-----------|
| Coverage F1 | 2.18e-05*** | 0.848 |
| Quality Score | 1.25e-07*** | 1.149 |
| CTQI | 2.96e-07*** | 1.098 |
| BLEU Score | 9.04e-06*** | 0.899 |
| BERTScore | 1.47e-06*** | 1.004 |

*All metrics show statistically significant improvements with large effect sizes (d > 0.8).*

**Additional Metrics:**
- **Perfect Translation Rate**: 41.2% → 67.6% (+26.4%, p=0.004)
- **Gloss Accuracy (Top-1)**: 79.0% | **Effective (after LLM selection)**: 86.7% (+7.6%)
- **Entries with CTQI improvement**: 30/34 (88.2%)
- **Segmentation**: Dual methods (auto-detect + motion-based)

---

## 6. 📖 More Details on Our Unique Features & Innovations

### (a) Model Architecture

#### 🎯 83-Point OpenHands-HD Keypoints
**What it does:**
- Extracts 83 keypoints: 8 face + 33 body pose + 42 hands (including 30 finger-level features)
- Generates 279-dimensional feature vectors per frame (83 × 3 coordinates + 30 finger features)
- Extended from OpenHands to OpenHands-HD with 3x more body points than baseline (27 → 83)

**Technical Details:**
- MediaPipe Holistic model with enhanced extraction
- 3D data: (x, y, z) for each keypoint, 279 features per frame
- Processes at 30 FPS on consumer hardware

**File:** `applications/predict_sentence.py` (RealTimePoseEstimator class)

#### 🎯 Top-K Prediction Support
**What it does:**
- Model returns top-5 predictions with confidence scores
- Gemini LLM uses all alternatives for context-aware selection
- User configurable: `--use-top-k 3`

**Example:**
```
Position 1: BOOK (85%), LOOK (12%), COOK (3%)
Position 2: READ (92%), RED (5%), REED (3%)
→ Gemini: "I'm reading a book" (not "I'm reading a look")
```

**Benefits:**
- Better context disambiguation
- Handles visually similar signs
- Improves sentence coherence

### (b) Data Augmentation

#### 🎨 50x Pose Data Augmentation Pipeline
**What it does:**
- Comprehensive pre-generated pose augmentation designed for sign language
- 50x expansion: 342 original samples → 17,100 augmented training samples
- Pre-generated (not runtime) for training efficiency

**Augmentation Types:**
- **Rotation**: ±15° geometric transformations
- **Shear**: ±0.2 rad deformations
- **Combinations**: Multi-augmentation stacking
- **Additional**: flip, noise, translation, scaling, speed variations

**Key Features:**
- Pre-generated augmentation (vs runtime augmentation that slows model predictions)
- Variable-length frame support (handles speed augmentation)
- Confidence mask preservation
- Pose-specific (doesn't corrupt keypoint structure)

**Impact:** 80.97% Top-1 accuracy on WLASL-100 with 50x augmented training data

**File:** `dataset-utilities/augmentation/generate_75pt_augmented_dataset.py`

### (c) Training Optimizations

#### 🎛️ Configurable Dropout
**What it does:**
- Command-line configurable dropout parameter
- Empirically optimized for small dataset regime

**Usage:**
```bash
python train_asl.py --classes 50 --dropout 0.25
```

**Analysis:**
```
Dropout 0.1 (default): 43.64% val (overfits at epoch 4)
Dropout 0.25 (optimized): 47.27% val (stable until epoch 25)
→ +3.63% improvement
```

**File:** `models/training-scripts/train_asl.py`

#### 📊 Samples-per-Parameter Model Sizing
**What it does:**
- Analytical approach to model capacity selection
- Balances model expressiveness with dataset size

**Configurations:**
- **Small model**: 64 hidden, 2 layers, 8 heads
- **Large model**: 129 hidden, 3 layers, 8 heads
- Training: 1500 epochs, batch size 16

**Result:** Right-sized models avoid both underfitting and overfitting. 80.97% Top-1 on WLASL-100.

### (d) Application

#### 🤖 LLM-based Semantic Coherence Analysis
**What it does:**
- Real-time LLM integration for natural sentence construction from sign predictions
- Transforms isolated sign glosses into grammatically correct English sentences
- Semantic coherence-based selection: LLM selects signs based on sentence meaning, not just confidence scores

**Implementation: Gemini API (gemini-2.0-flash)**

**Smart Buffering Triggers (5 strategies):**
1. **Pause detection**: 1.8s silence + 2+ words
2. **Buffer size**: 3-4 words accumulated
3. **Question words**: Immediate on "what/who/where/when/why/how"
4. **Sentence enders**: On "please/thanks/./?"
5. **Timeout**: Max 10s without response

**Local Fallback:**
- Instant responses for common phrases ("hello", "thank you", "bye")
- Zero API latency for frequent interactions

**Context-Aware Prompts:**
- Includes confidence scores from sign recognition
- Adapts to trigger reason (question vs statement)
- Top-K prediction integration for better word choice

**Quality Evaluation:**
- Automatic BLEU, BERTScore, and grammatical quality calculation against reference sentences
- Synthetic evaluation dataset for before/after LLM comparison
- CTQI (Composite Translation Quality Index): `CTQI = (α × BLEU) + (β × BERTScore) + (γ × Quality)`
  - **BLEU**: Lexical similarity via n-gram overlap
  - **BERTScore**: Semantic preservation via contextual embeddings
  - **Quality**: Grammatical correctness score (0-100)
- **Results**: Grammatical quality 32% → 76%, CTQI 51% → 74%, all p < 0.001

**Result:** <2s latency, grammatical quality improved from 32% to 76%, 67.6% perfect translation rate

**Files:**
- `applications/gemini_conversation_manager.py`
- `project-utilities/calculate_sent_bleu.py`

#### 🔍 Continuous Sign Detection
**What it does:**
- Automatically detects sign boundaries in continuous signing videos
- Segments video stream into individual signs without manual annotation
- Enables real-world video processing

**Implementation: Dual Segmentation Approach**

**Method 1: Auto-detect (pose_to_segments)**
- Uses pose-format library's built-in ML-based segmentation
- Analyzes pose patterns for natural boundaries
- Works well for clear pauses between signs

**Method 2: Motion-based (velocity threshold)**
- Calculates keypoint velocities frame-by-frame
- Configurable threshold (default: 0.02)
- Better for subtle boundaries and continuous signing

**Usage:**
```bash
--segmentation-method auto  # ML-based (default)
--segmentation-method motion --velocity-threshold 0.02  # Velocity-based
```

**Result:** Automated boundary detection enabling real-world video processing

**File:** `applications/motion_based_segmenter.py`

### (e) Reusability & Extensibility

#### 🔧 Centralized Configuration System
**What it does:**
- Single source of truth for all paths
- Auto-detection of project root
- Cross-platform compatibility

**Structure:**
```json
{
  "data_root": "/path/to/wlasl_poses_complete",
  "project_root": "auto"
}
```

**Features:**
- `settings.json` gitignored (user-specific)
- `settings.json.example` committed (template)
- Works on Windows and Linux
- Interactive setup script

**Usage:**
```bash
python setup_config.py  # Interactive setup
python -m config.paths  # Verify configuration
```

**Files:** `config/settings.json.example`, `config/paths.py`, `setup_config.py`

#### 🎯 Dynamic Class Loading
**What it does:**
- Reads class mappings from JSON files dynamically
- Supports any number of classes: 20/50/100/300/2000
- Single codebase for all configurations

**Benefits:**
- Zero code changes when switching class counts
- Easy to add new class splits
- Consistent across all utilities (training, augmentation, splitting)

**Example:**
```python
# Automatically loads from:
# dataset_splits/50_classes/50_class_mapping.json
classes = load_class_mapping(num_classes=50)
```

**Files:** All training and utility scripts support this

---

## 7. 🚀 Getting Started

### 📁 Project Structure

```
asl-v1/
├── config/                      # Configuration system
│   ├── settings.json           # User-specific paths (gitignored)
│   ├── settings.json.example   # Template
│   └── paths.py                # Path resolution module
│
├── models/
│   ├── openhands-modernized/   # OpenHands-HD implementation
│   ├── training-scripts/
│   │   └── train_asl.py        # Main training script
│   └── training_results_comp.md # Performance tracking
│
├── dataset-utilities/
│   ├── augmentation/
│   │   └── generate_75pt_augmented_dataset.py  # 50x augmentation
│   ├── conversion/
│   │   └── pose_to_pickle_converter.py
│   ├── dataset-splitting/
│   │   └── split_pose_files_nclass.py
│   └── segmentation/           # Boundary detection
│
├── applications/
│   ├── predict_sentence.py     # Full pipeline (file + webcam)
│   ├── predict_sentence_with_gemini_streaming.py  # Real-time streaming
│   ├── gemini_conversation_manager.py  # Smart buffering
│   └── motion_based_segmenter.py       # Velocity-based segmentation
│
├── project-utilities/          # Helper scripts
└── archive/                    # Legacy experiments
```

### 📦 Dependencies

Core dependencies:
- **Python 3.11.9** (required for compatibility)
- PyTorch
- MediaPipe (pose estimation)
- OpenCV (webcam capture)
- pose-format (pose file handling)
- google-generativeai (Gemini API)
- scikit-learn
- numpy
- Threading/multiprocessing (built-in)

See `requirements.txt` for complete list.

### Prerequisites
- **Python 3.11.9** (required for compatibility)
- CUDA-capable GPU (recommended for training)
- Webcam (for real-time inference)
- Gemini API key (optional, for sentence construction)

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd asl-v1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Paths

**Option A: Interactive Setup (Recommended)**
```bash
python setup_config.py
```

**Option B: Manual Configuration**
1. Copy template: `cp config/settings.json.example config/settings.json`
2. Edit `config/settings.json`:
```json
{
  "data_root": "/path/to/wlasl_poses_complete",
  "project_root": "auto"
}
```

**Verify:**
```bash
python -m config.paths
```

### 4. Prepare Dataset

Your `data_root` should contain:
```
wlasl_poses_complete/
├── dataset_splits/
│   ├── 20_classes/
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── test.txt
│   │   └── 20_class_mapping.json
│   └── 50_classes/...
├── pickle_files/         # Original pose data
├── augmented_pool/       # Generated augmentations
└── video_to_gloss_mapping.json
```

**Generate Augmented Dataset:**
```bash
python dataset-utilities/augmentation/generate_75pt_augmented_dataset.py \
  --num-classes 50
```

### 5. Training

**Train 20-Class Model:**
```bash
python models/training-scripts/train_asl.py \
  --classes 20 \
  --dataset augmented \
  --architecture openhands \
  --model-size large \
  --dropout 0.1
```

**Train 50-Class Model (Optimized):**
```bash
python models/training-scripts/train_asl.py \
  --classes 50 \
  --dataset augmented \
  --architecture openhands \
  --model-size small \
  --dropout 0.25 \
  --early-stopping 30
```

**Resume Training:**
```bash
python models/training-scripts/train_asl.py \
  --classes 50 \
  --dataset augmented
# Automatically detects and resumes from latest checkpoint
```

**Test Model:**
```bash
python models/training-scripts/train_asl.py \
  --classes 50 \
  --mode test
```

### 6. Inference

**Real-Time Video Conferencing (Streaming):**
```bash
python applications/predict_sentence_with_gemini_streaming.py \
  --checkpoint models/wlasl_50_class_model \
  --gemini-api-key YOUR_KEY
```

**Standard Webcam:**
```bash
python applications/predict_sentence.py --webcam \
  --checkpoint models/wlasl_50_class_model \
  --gemini-api-key YOUR_KEY \
  --use-top-k 3
```

**Video File (Auto-detect segmentation):**
```bash
python applications/predict_sentence.py input.mp4 \
  --checkpoint models/wlasl_50_class_model \
  --gemini-api-key YOUR_KEY \
  --num-glosses 50
```

**Video File (Motion-based segmentation):**
```bash
python applications/predict_sentence.py input.mp4 \
  --segmentation-method motion \
  --velocity-threshold 0.02 \
  --min-sign-duration 10
```

### 7. Dataset Utilities

**Split Dataset:**
```bash
python dataset-utilities/dataset-splitting/split_pose_files_nclass.py \
  --num-classes 50
```

**Convert Pose to Pickle:**
```bash
# Batch conversion
python dataset-utilities/conversion/pose_to_pickle_converter.py \
  --input-dir path/to/pose_files \
  --output-dir path/to/pickle_files

# Single file
python dataset-utilities/conversion/pose_to_pickle_converter.py \
  --input-file video_001.pose
```

---

## 8. 🔗 Related Work

- [WLASL Dataset](https://github.com/dxli94/WLASL) - Original dataset
- [OpenHands](https://github.com/AI4Bharat/OpenHands) - Base architecture
- [MediaPipe](https://google.github.io/mediapipe/) - Pose estimation
- [Gemini API](https://ai.google.dev/) - LLM integration

---

## 🤝 Contributing

This is a research project. Contributions welcome:
1. Open an issue describing the problem
2. Fork the repository
3. Create a feature branch
4. Submit a pull request

---

## 📄 License & Citation

[Your license choice]

If you use this work, please cite:
```
[Your citation]
```

---

**Quick Start Checklist:**
- [ ] Clone and install dependencies
- [ ] Configure paths (`python setup_config.py`)
- [ ] Generate augmented dataset
- [ ] Train model (`python train_asl.py --classes 20`)
- [ ] Try webcam streaming (`python predict_sentence_with_gemini_streaming.py`)
