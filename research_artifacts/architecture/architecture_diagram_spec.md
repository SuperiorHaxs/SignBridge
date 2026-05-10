# SignBridge System Architecture — Diagram Specification

Use this document to recreate the architecture in Visio or PowerPoint.
Each section describes a **box/group** you should draw, with arrows showing data flow.

---

## LAYOUT OVERVIEW (Left-to-Right Flow, 5 Columns)

```
┌─────────────┐   ┌──────────────────┐   ┌──────────────────────┐   ┌─────────────────────┐   ┌──────────────────┐
│  COLUMN 1   │   │    COLUMN 2      │   │      COLUMN 3        │   │     COLUMN 4        │   │    COLUMN 5      │
│  Data       │──>│  Preprocessing   │──>│  Model Training &    │──>│  LLM-Enhanced       │──>│  Application     │
│  Sources    │   │  & Augmentation  │   │  Inference Engine     │   │  Sentence Builder   │   │  Layer           │
└─────────────┘   └──────────────────┘   └──────────────────────┘   └─────────────────────┘   └──────────────────┘
                                                                                                       │
                                                                                                       v
                                                                                              ┌──────────────────┐
                                                                                              │    COLUMN 6      │
                                                                                              │  Evaluation &    │
                                                                                              │  Metrics         │
                                                                                              └──────────────────┘
```

---

## COLUMN 1: DATA SOURCES (Blue shading)

**Group Title:** "Data Sources"

### Box 1A: WLASL Dataset
- Label: **WLASL-100 Dataset**
- Sub-text: "2,038 videos · 100 ASL glosses"
- Icon suggestion: database/cylinder

### Box 1B: Video Corpus
- Label: **Raw Video Files**
- Sub-text: "MP4/AVI sign language clips"
- Icon suggestion: film strip

### Box 1C: Gloss Vocabulary
- Label: **Gloss List**
- Sub-text: "100-class vocabulary mapping"
- File ref: `gloss_list_100_class.json`

**Arrow:** All three boxes feed right into Column 2

---

## COLUMN 2: PREPROCESSING & AUGMENTATION (Green shading)

**Group Title:** "Preprocessing & Augmentation Pipeline"

### Box 2A: Keypoint Extraction
- Label: **MediaPipe Pose Estimation**
- Sub-text: "83 keypoints per frame"
- Detail bullets (inside or as callout):
  - 27 body landmarks
  - 21 left hand landmarks
  - 21 right hand landmarks
  - 8 face landmarks (mouth region)
  - 6 additional finger features

### Box 2B: Feature Engineering
- Label: **Feature Vector Construction**
- Sub-text: "279 dimensions per frame"
- Detail:
  - 249 coordinate features (x, y, z)
  - 30 finger angle/distance features
  - Position-invariant normalization

### Box 2C: Data Augmentation Engine
- Label: **Augmentation Pipeline**
- Sub-text: "6 augmentation strategies"
- Detail bullets:
  - Temporal scaling (speed ±20%)
  - Spatial scaling (size ±15%)
  - Rotation (±15°)
  - Translation shift
  - Gaussian noise injection
  - Mirror reflection
- File ref: `augmentation_config.py`

### Box 2D: Smart Sample Selection
- Label: **Smart Selection**
- Sub-text: "Balanced class distribution"
- Detail: Min 8 samples/class guaranteed

**Internal arrows:** 2A → 2B → 2C → 2D (vertical stack, flowing downward)
**Arrow out:** 2D → Column 3

---

## COLUMN 3: MODEL TRAINING & INFERENCE (Orange shading)

**Group Title:** "OpenHands-HD Recognition Engine"

### Box 3A: Model Architecture (largest box, central)
- Label: **OpenHands-HD Transformer**
- Sub-text: "BERT-style sign language classifier"
- Detail (draw as stacked internal layers):
  ```
  ┌────────────────────────────┐
  │   Input: 279-dim features  │
  │   (variable-length seq)    │
  ├────────────────────────────┤
  │   Linear Projection        │
  │   → 256 hidden dims        │
  ├────────────────────────────┤
  │   Positional Encoding      │
  │   (learned embeddings)     │
  ├────────────────────────────┤
  │   6× Transformer Encoder   │
  │   Layers (16 attention     │
  │   heads, GELU activation)  │
  ├────────────────────────────┤
  │   [CLS] Token Pooling      │
  ├────────────────────────────┤
  │   Classification Head      │
  │   → 100 ASL classes        │
  └────────────────────────────┘
  ```

### Box 3B: Training Pipeline
- Label: **Training Loop**
- Sub-text: "AdamW optimizer, cosine LR schedule"
- Detail:
  - Batch size: 32
  - Learning rate: 1e-4
  - Label smoothing: 0.1
  - Dropout: 0.3

### Box 3C: Inference Output
- Label: **Top-K Predictions**
- Sub-text: "Returns ranked candidates with confidence scores"
- Example callout:
  ```
  Sign input → ["hello" (0.72), "hi" (0.15), "hey" (0.08)]
  ```

**Internal arrows:** 3B feeds into 3A, 3A → 3C
**Arrow out:** 3C → Column 4 (sends Top-K predictions)

---

## COLUMN 4: LLM-ENHANCED SENTENCE BUILDER (Purple shading)

**Group Title:** "Gemini LLM Sentence Construction"

### Box 4A: Word Sequence Buffer
- Label: **Sign Sequence Accumulator**
- Sub-text: "Collects per-sign Top-K predictions"
- Visual: show a timeline with multiple sign slots:
  ```
  Sign 1         Sign 2         Sign 3
  [Top-K list]   [Top-K list]   [Top-K list]
  ```

### Box 4B: LLM Semantic Selection
- Label: **Google Gemini 2.0 Flash**
- Sub-text: "Semantic coherence word selection"
- Detail:
  - Receives all Top-K candidates per sign
  - Selects most contextually coherent word
  - Considers full sentence context
  - API: `google.generativeai`

### Box 4C: Grammar & Output
- Label: **Sentence Assembly**
- Sub-text: "Grammatically correct English output"
- Example callout:
  ```
  ASL signs: [HELLO] [HOW] [YOU]
  → "Hello, how are you?"
  ```

**Internal arrows:** 4A → 4B → 4C
**Arrow out:** 4C → Column 5

---

## COLUMN 5: APPLICATION LAYER (Teal shading)

**Group Title:** "Show-and-Tell Demo Application"

### Box 5A: Streamlit Web App
- Label: **Streamlit Interface**
- Sub-text: "Interactive demo UI"
- Detail:
  - Live webcam capture
  - Video file upload
  - Real-time sign recognition display

### Box 5B: Demo Data
- Label: **Pre-recorded Samples**
- Sub-text: "Curated demo videos with ground truth"
- File ref: `samples.json`

### Box 5C: Output Display
- Label: **Translation Output**
- Sub-text: "Real-time ASL → English display"
- Detail:
  - Per-sign predictions with confidence
  - Full sentence translation
  - Side-by-side video + text view

**Internal arrows:** 5A ↔ 5B, 5A → 5C

---

## COLUMN 6: EVALUATION & METRICS (Red/Pink shading — place below Columns 3-5)

**Group Title:** "Evaluation Framework"

### Box 6A: Word-Level Metrics
- Label: **Classification Metrics**
- Sub-text: "Per-sign accuracy"
- Key numbers:
  - Top-1 Accuracy: **80.97%**
  - Top-3 Accuracy: **91.62%**
  - Top-5 Accuracy: **95.17%**

### Box 6B: Sentence-Level Metrics
- Label: **CTQI v2 Score**
- Sub-text: "Composite Translation Quality Index"
- Key numbers:
  - Without LLM: 55.56
  - With LLM: **78.16**
  - Improvement: **+40.7%**

### Box 6C: Ablation Studies
- Label: **Ablation Analysis**
- Sub-text: "Component contribution analysis"
- Detail:
  - Keypoint density comparison (27 vs 83)
  - Augmentation impact study
  - LLM vs no-LLM comparison

### Box 6D: Statistical Validation
- Label: **Statistical Tests**
- Sub-text: "Rigorous significance testing"
- Detail:
  - Paired t-test: p < 0.001
  - Cohen's d > 0.8 (large effect)
  - Inter-rater agreement (Fleiss' κ)

**Arrows:** Dotted/dashed arrows from Columns 3, 4, 5 pointing down into Column 6 (evaluation draws from all stages)

---

## CROSS-CUTTING COMPONENTS (Draw as a horizontal bar below everything)

### Bar: Training Utilities
- Label: **Training & Dataset Utilities**
- Sub-items (as small boxes inside the bar):
  - Incremental Training Manager
  - Targeted Manifest Creator
  - Domain-Specific Training
  - Kaggle Integration

---

## SUGGESTED COLOR SCHEME

| Component               | Color (Hex) | Meaning              |
|-------------------------|-------------|----------------------|
| Data Sources            | #4472C4     | Blue — raw inputs    |
| Preprocessing           | #70AD47     | Green — preparation  |
| Model Engine            | #ED7D31     | Orange — core ML     |
| LLM Sentence Builder    | #7B2D8E     | Purple — AI/LLM      |
| Application Layer       | #00B0F0     | Teal — user-facing   |
| Evaluation Framework    | #FF6B6B     | Red — validation     |
| Utilities Bar           | #A5A5A5     | Gray — support       |

---

## KEY ARROWS & DATA FLOW LABELS

| From → To                          | Arrow Label                            |
|-------------------------------------|----------------------------------------|
| Data Sources → Preprocessing        | "Raw video files"                      |
| Keypoint Extraction → Features      | "83 keypoints × 3 coords"             |
| Features → Augmentation             | "279-dim feature vectors"              |
| Augmentation → Model                | "Augmented training samples"           |
| Model → Top-K Predictions           | "Softmax probabilities"                |
| Top-K → Gemini LLM                  | "Top-K candidates per sign"            |
| Gemini → Sentence Assembly          | "Selected words + context"             |
| Sentence Assembly → App             | "English sentence"                     |
| All stages → Evaluation             | "Metrics collection" (dashed)          |

---

## POSTER LAYOUT SUGGESTION

For a science fair poster (typically 48" × 36" or 36" × 24"):

```
┌──────────────────────────────────────────────────────────────────────┐
│                    SignBridge: System Architecture                    │
├──────────┬──────────┬──────────────┬─────────────┬──────────────────┤
│          │          │              │             │                  │
│  DATA    │  PRE-    │   MODEL      │   GEMINI    │   APPLICATION   │
│  SOURCES │ PROCESS  │   ENGINE     │   LLM       │   LAYER         │
│          │  &       │              │   SENTENCE   │                  │
│  [1A]    │  AUGMENT │   [3A model  │   BUILDER   │   [5A]          │
│  [1B]    │          │    diagram]  │             │   [5B]          │
│  [1C]    │  [2A]    │              │   [4A]      │   [5C]          │
│          │  [2B]    │   [3B]       │   [4B]      │                  │
│          │  [2C]    │   [3C]       │   [4C]      │                  │
│          │  [2D]    │              │             │                  │
├──────────┴──────────┴──────────────┴─────────────┴──────────────────┤
│                     EVALUATION FRAMEWORK                             │
│         [6A]          [6B]           [6C]          [6D]              │
├──────────────────────────────────────────────────────────────────────┤
│                     Training & Dataset Utilities                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## ICON SUGGESTIONS

- Database cylinder → WLASL Dataset
- Film strip → Video files
- Skeleton/stick figure → MediaPipe keypoints
- Neural network nodes → Transformer model
- Brain/sparkle → Gemini LLM
- Monitor/screen → Streamlit app
- Chart/graph → Evaluation metrics
- Gear/wrench → Utilities
