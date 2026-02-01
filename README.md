# SignBridge: Real-Time American Sign Language Recognition System

*Building accessible, real-time ASL translation for video conferencing and beyond*

---

## Table of Contents

1. [Abstract](#1--abstract)
2. [Research Questions, Hypotheses & Engineering Goals](#2--research-questions-hypotheses--engineering-goals)
3. [Current State of the Field](#3--current-state-of-the-field)
4. [Experimental Design Overview](#4--experimental-design-overview)
5. [Phased Research Roadmap](#5--phased-research-roadmap)
6. [SignBridge Performance](#6--signbridge-performance)
7. [More Details on Our Unique Features & Innovations](#7--more-details-on-our-unique-features--innovations)
8. [Getting Started](#8--getting-started)
9. [Related Work](#9--related-work)

---

## 1. 📄 Abstract

Around 70 million Deaf individuals worldwide rely on sign languages, yet fewer than 20% of their digital content receives accurate captions. This is because current automated systems struggle with dynamic, continuous signing, leading to grammatically incorrect and unusable translations.

We present SignBridge, an American Sign Language (ASL) translation system that combines high-density pose estimation with an LLM for contextual selection from multiple sign candidates. SignBridge addresses three central challenges: individual word-level sign language accuracy in data-scarce scenarios, lack of grammatically correct sentence construction from sign predictions, and inadequate, multi-dimensional translation quality metrics. SignBridge's core innovation enables coherence-based selection rather than relying solely on vision model confidence scores for individual words.

Our approach enhances the OpenHands model to OpenHands-HD (expanding from 27 to 83 body points including detailed finger tracking), applies 50x pose data augmentation for training diversity, and uses transformer architecture to generate Top-K sign predictions. Google's Gemini LLM (gemini-2.0-flash) then performs contextual recovery by selecting semantically appropriate signs from the Top-K predictions.

Testing on the WLASL-100 dataset shows substantial improvements: word-level prediction accuracy increased from 72% benchmark to 80.97% for Top-1 predictions and 91.62% for Top-3 predictions. Sentence-level grammatical quality rose from 32% to 76%. To address the absence of comprehensive translation quality assessment frameworks, we introduce Composite Translation Quality Index (CTQI)—a new score that integrates lexical similarity, semantic preservation, grammatical structure, and completeness—improved from 51% to 74%.

Taken together, SignBridge offers a foundation for more reliable and practical real-time ASL translation systems, helping reduce communication barriers for the Deaf community in education, employment and digital communication.

---

## 2. 🎯 Research Questions, Hypotheses & Engineering Goals

### Research Questions

- **RQ1:** Can tracking more body points improve sign language recognition accuracy (benchmark 72%) when training data is limited?
- **RQ2:** Can AI language models improve translation accuracy by considering sentence meaning rather than evaluating each sign independently?
- **RQ3:** Can measuring multiple aspects of translation quality together provide better assessment than single measurements?

### Hypotheses

- **H1: Enhanced Body Tracking.** Tracking 3x more body points instead of standard 27, combined with artificially expanding the training dataset by 10-20 times, will improve word-level recognition accuracy compared to current methods.
- **H2: AI-Powered Word Selection.** Using an AI Large Language Model to select words based on sentence meaning will produce higher quality translations than choosing the highest-confidence prediction from traditional science frameworks and machine learning models — particularly when the correct sign appears among the top predictions but not as the first choice.
- **H3: Multi-Dimensional Quality Measurement.** A combined quality score measuring word accuracy, meaning preservation, and grammatical correctness will reveal improvements that no single measurement captures alone.

### Engineering Goals

1. Build a sign recognition system that tracks 83 body points including detailed finger positions
2. Integrate an AI language model to construct grammatically correct sentences from sign predictions
3. Develop Composite Translation Quality Index (CTQI), a combined quality scoring framework measuring multiple translation dimensions
4. Create a working prototype demonstrating practical ASL-to-English translation (SignBridge)

### Expected Outcomes

- Recognition accuracy reaching 80%+ among top-3 predictions
- Grammatical quality of translations significantly improved over word-by-word output
- Validation that combined quality measurement outperforms single metrics for usage in real world applications
- A functional system demonstrating the viability of AI-enhanced sign language translation

---

## 3. 📊 Current State of the Field

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

**Key Observations:**
- Video models: High accuracy but computationally prohibitive for real-time
- Pose models: Fast but significantly lower accuracy (30.6% on WLASL2000)
- Gap: 63.61% (video SOTA) vs 30.6% (pose baseline) on comparable scales

---

## 4. 🔬 Experimental Design Overview

All development and experimentation for this research project were conducted on personal computing equipment using publicly available and synthetically augmented datasets. This develops and evaluates a three-component ASL translation system:

### Component 1: OpenHands-HD for Pose-Based Sign Recognition

- **Original Dataset:** WLASL (Word-Level ASL), with the WLASL-100 subset (100 sign classes, 342 samples expanded 50x to 17,100 via augmentation, 30% held out for validation/test)
- **Synthetic Data Augmentation:** Pre-generated augmentation with rotation (±15°), shear (±0.2 rad), producing 50x expansion
- **OpenHands-HD Development:** Enhanced MediaPipe Holistic extraction of 83 keypoints (8 face + 33 body + 42 hands), yielding 279-dimensional feature vectors per frame
- **Model Training:** Transformer encoder, small vs. large capacity (64–129 hidden, 2–3 layers, 8 heads), 1500 epochs, batch size 16
- **Output:** Top-K predictions with confidence scores per video

### Component 2: LLM Integration Library for Semantic Sentence Construction

- **Plug-and-play LLM integration;** prototype uses Google Gemini (gemini-2.0-flash)
- **Input:** Top-3 predictions per sign position with confidence scores, rolling window predictions
- **Semantic Coherence Analysis:** LLM selects signs based on semantic coherence (not just confidence), multi-pass prompt engineering, prevents duplicate selections, adds grammatical elements
- **Output:** Grammatically correct English sentences

### Component 3: CTQI Framework for Comprehensive Evaluation

- **CTQI Formula:** `CTQI = (α × BLEU) + (β × BERTScore) + (γ × Quality)`
  - **BLEU** for lexical similarity
  - **BERTScore** for semantic preservation
  - **Quality** for grammatical correctness (0–100)
- **Evaluation:** 25-entry synthetic sentence dataset with ground truth, comparing before-LLM (confidence-based) vs. after-LLM (semantic coherence), with statistical analysis of all quality metrics

---

## 5. 🗺️ Phased Research Roadmap

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

## 6. 📈 SignBridge Performance

### Isolated Word-Level Sign Recognition Accuracy

OpenHands-HD achieves 80.97% Top-1 accuracy on WLASL-100, surpassing the original OpenHands pose-based baseline (71.57%) and approaching the video-based state-of-the-art (81.38%) while maintaining real-time inference speed.

| Model | Dataset | Approach | Keypoints | Top-1 Accuracy | Top-3 Accuracy | Speed |
|-------|---------|----------|-----------|----------------|----------------|-------|
| I3D Baseline | WLASL-100 | Video | N/A | 65.89% | — | Slow |
| Multi-stream CNN (SOTA) | WLASL-100 | Video | N/A | 81.38% | — | Slow |
| OpenHands Baseline | WLASL-100 | Pose + Transformer | 27 | 71.57% | — | Real-time |
| **OpenHands-HD (Ours)** | WLASL-100 | Pose + Transformer | 83 | **80.97%** | **91.62%** | Real-time |

### End-to-End Sentence-Level Translation

Statistical analysis of SignBridge results using paired t-tests (n=34 sentence pairs) demonstrates significant improvements across all evaluation metrics. For gloss-level selection accuracy, Coverage F1—which measures the overlap of content words between the generated and reference sentences—improved from 74.64 to 87.62 (t(33) = 4.944, p < 0.001, Cohen's d = 0.848), representing a large effect size. This improvement indicates that the LLM pipeline more accurately selects contextually appropriate glosses from the model's top-k predictions, resulting in translations that better capture the intended meaning.

For overall translation quality, the Quality Score (a reference-free grammaticality measure based on GPT-2 perplexity) improved substantially from 39.38 to 74.56 (t(33) = 6.700, p < 0.001, Cohen's d = 1.149), representing a large effect size. Additionally, the Perfect Translation Rate—a binary metric indicating whether all glosses in a sentence were correctly predicted—increased from 41.2% (14/34) to 67.6% (23/34), with p=0.004, confirming this improvement is statistically significant. The Composite Translation Quality Index (CTQI, introduced by SignBridge), which combines Gloss Accuracy (40%), Quality Score (40%), and Perfect Translation Rate (20%), improved from 55.56 to 78.16 (t(33) = 6.403, p < 0.001, Cohen's d = 1.098). Overall, 88.2% of test entries (30/34) showed improvement in CTQI, demonstrating consistent gains across the evaluation dataset.

| Metric | Baseline (No LLM) | With LLM | Improvement |
|--------|-------------------|----------|-------------|
| Coverage F1 | 74.64 | 87.62 | +12.98 |
| Quality Score | 39.38 | 74.56 | +35.18 |
| Perfect Translation Rate | 41.2% (14/34) | 67.6% (23/34) | +26.4% |
| CTQI (introduced by SignBridge) | 55.56 | 78.16 | +22.60 |
| Entries with CTQI improvement | — | 30/34 (88.2%) | — |

---

## 7. 📖 More Details on Our Unique Features & Innovations

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

## 8. 🚀 Getting Started

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

## 9. 🔗 Related Work

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
