# ASL-v1: American Sign Language Recognition System

A comprehensive sign language recognition system using pose estimation and deep learning models for translating ASL videos to English text.

## 🎯 Features

- **Multiple Model Architectures**: OpenHands-Modernized, Transformer, CNN-LSTM
- **Flexible Class Support**: Train on 20, 50, or 100 sign classes
- **Real-time Translation**: Webcam support for live sign language translation
- **Data Augmentation**: Advanced augmentation pipeline for improved model performance
- **End-to-End Pipeline**: Video → Pose → Segmentation → Prediction → Sentence Construction
- **Centralized Configuration**: Easy path management for multi-machine setup

## 📁 Project Structure

```
asl-v1/
├── config/                      # Configuration system
│   ├── __init__.py
│   ├── paths.py                # Path configuration module
│   ├── settings.json           # User-specific paths (UPDATE THIS!)
│   └── MIGRATION_GUIDE.md      # Configuration guide
│
├── models/                      # Model architectures and training
│   ├── openhands-modernized/   # OpenHands model implementation
│   ├── Transformer/            # Transformer-based model
│   ├── CNN-LSTM/               # CNN-LSTM model
│   └── training-scripts/       # Training scripts
│       └── train_asl.py        # Main training script
│
├── dataset-utilities/          # Dataset processing tools
│   ├── augmentation/           # Data augmentation scripts
│   ├── conversion/             # Format conversion utilities
│   ├── dataset-splitting/      # Train/val/test splitting
│   ├── segmentation/           # Sign segmentation tools
│   └── visualization/          # Visualization utilities
│
├── applications/               # End-user applications
│   ├── predict_sentence.py    # Video to sentence translation
│   ├── predict_sentence_with_gemini_streaming.py
│   └── motion_based_segmenter.py
│
├── project-utilities/          # Helper utilities
│
├── archive/                    # Legacy experiments (not maintained)
│
└── setup_config.py             # Interactive configuration setup

```

## 🚀 Quick Start

### 1. Clone the Repository

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

Edit `config/settings.json`:
```json
{
  "data_root": "/path/to/your/dataset-root",
  "project_root": "auto"
}
```

**Verify Configuration:**
```bash
python -m config.paths
```

### 4. Prepare Your Dataset

Your dataset root should contain:
- `pickle_files/` - Pose data in pickle format
- `pose_files/` - Pose files
- `dataset_splits/` - Train/val/test splits
- `class_index_mapping_XX.json` - Class mapping files
- `video_to_gloss_mapping.json` - Video metadata

See [config/MIGRATION_GUIDE.md](config/MIGRATION_GUIDE.md) for details.

## 🏋️ Training

### Train on 20 Classes (Default)
```bash
python models/training-scripts/train_asl.py --classes 20 --dataset original
```

### Train on 50 Classes with Augmented Data
```bash
python models/training-scripts/train_asl.py --classes 50 --dataset augmented
```

### Advanced Training Options
```bash
python models/training-scripts/train_asl.py \
  --classes 20 \
  --dataset augmented \
  --architecture openhands \
  --model-size large \
  --early-stopping 15 \
  --force-fresh
```

**Options:**
- `--classes`: Number of classes (20, 50, 100)
- `--dataset`: Dataset type (original, augmented)
- `--architecture`: Model type (openhands, transformer)
- `--model-size`: Model size (small, large)
- `--early-stopping`: Early stopping patience
- `--force-fresh`: Start fresh training (ignore checkpoints)

### Test Trained Model
```bash
python models/training-scripts/train_asl.py --classes 20 --mode test
```

## 🎥 Inference

### Video to Sentence Translation
```bash
python applications/predict_sentence.py input_video.mp4 \
  --checkpoint ./models/wlasl_20_class_model \
  --gemini-api-key YOUR_API_KEY
```

### Real-time Webcam Translation
```bash
python applications/predict_sentence.py --webcam \
  --checkpoint ./models/wlasl_20_class_model \
  --gemini-api-key YOUR_API_KEY
```

**Options:**
- `--checkpoint`: Path to trained model checkpoint
- `--gemini-api-key`: Gemini API key for sentence construction (optional)
- `--segmentation-method`: Segmentation method (auto, motion)
- `--use-top-k`: Use top-k predictions (1-5)

## 🛠️ Dataset Utilities

### Split Dataset
```bash
# Split pose files into train/val/test
python dataset-utilities/dataset-splitting/split_pose_files_nclass.py --num-classes 20
```

### Generate Augmented Dataset
```bash
python dataset-utilities/augmentation/generate_augmented_dataset.py \
  --source /path/to/pickle_files \
  --target /path/to/output
```

### Convert Video to Pose
```bash
python dataset-utilities/conversion/video_to_pose_extraction.py
```

## 📊 Research Project Timeline

This project follows a phased research approach, progressing from isolated sign recognition to a complete real-time translation system.

| Phase | Title | Status | Planned Completion | Key Deliverables | Success Criteria |
|-------|-------|--------|-------------------|------------------|------------------|
| **Phase 1** | Isolated Sign Recognition<br>(OpenHands-Modernized) | 🔄 **IN PROGRESS** | TBD | • Trained 20-class model<br>• Trained 50-class model<br>• Architecture paper | • 80%+ Top-3 accuracy (20-class)<br>• 60%+ Top-3 accuracy (50-class)<br>• Published baseline results |
| **Phase 2** | LLM Sentence Constructor | ⏳ NOT STARTED | TBD | • LLM integration module<br>• Sign-to-sentence pipeline<br>• Grammar correction system | • Grammatically correct sentences<br>• Context-based disambiguation<br>• 90%+ sentence coherence |
| **Phase 3** | Full Pipeline Integration | ⏳ NOT STARTED | TBD | • End-to-end system<br>• Batch processing capability<br>• Evaluation framework | • Functional sign sequence → English text<br>• <2s latency per sign<br>• 75%+ translation accuracy |
| **Phase 4** | Continuous Sign Detection | ⏳ NOT STARTED | TBD | • Temporal segmentation model<br>• Boundary detection system<br>• Continuous recognition pipeline | • 85%+ boundary detection accuracy<br>• Real-time processing capability<br>• <200ms segmentation latency |
| **Phase 5** | Real-Time Webcam Application | ⏳ NOT STARTED | TBD | • Desktop application<br>• Real-time inference pipeline<br>• User interface | • 15-30 FPS processing<br>• <500ms end-to-end latency<br>• Deployable application |
| **Phase 6** | Optimization & Deployment | ⏳ NOT STARTED | TBD | • Model quantization<br>• Performance optimization<br>• Documentation & demos | • 2x speed improvement<br>• Production-ready code<br>• Public release |

## 📊 Model Performance

| Model | Classes | Dataset | Top-1 Acc | Top-3 Acc | Notes |
|-------|---------|---------|-----------|-----------|-------|
| OpenHands | 20 | Original | 27.6% | - | Baseline |
| OpenHands | 20 | Augmented | TBD | TBD | With augmentation |
| OpenHands | 50 | Augmented | TBD | TBD | 50-class model |

## 🔧 Configuration System

The project uses a centralized configuration system for easy path management:

- **All paths in one place**: `config/settings.json`
- **Auto-detection**: Project root auto-detected
- **Easy migration**: Just update `settings.json` on new machine
- **Cross-platform**: Works on Windows and Linux

See [config/MIGRATION_GUIDE.md](config/MIGRATION_GUIDE.md) for complete documentation.

## 📦 Dependencies

Core dependencies:
- Python 3.8+
- PyTorch
- MediaPipe
- OpenCV
- pose-format
- google-generativeai (for Gemini API)
- scikit-learn
- numpy

See `requirements.txt` for complete list.

## 🤝 Contributing

This is a personal research project. If you find issues or have suggestions:

1. Open an issue describing the problem
2. Fork the repository
3. Create a feature branch
4. Submit a pull request

## 📝 Notes

- **Archive folder**: Contains legacy experiments, not actively maintained
- **External datasets**: Store datasets on external drive, configure via `settings.json`
- **Model checkpoints**: Saved in `models/wlasl_XX_class_model/`
- **Training outputs**: Checkpoints support resuming training

## 🎓 Citation

If you use this work, please cite:
```
[Your citation information]
```

## 📄 License

[Your license choice - e.g., MIT, Apache 2.0]

## 🔗 Related Projects

- [WLASL Dataset](https://github.com/dxli94/WLASL)
- [MediaPipe](https://google.github.io/mediapipe/)
- [OpenHands](https://github.com/AI4Bharat/OpenHands)

## 📧 Contact

[Your contact information]

---

**Getting Started Checklist:**

- [ ] Clone repository
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Configure paths (`python setup_config.py`)
- [ ] Verify configuration (`python -m config.paths`)
- [ ] Prepare dataset in configured location
- [ ] Run training (`python models/training-scripts/train_asl.py`)
- [ ] Test model (`python models/training-scripts/train_asl.py --mode test`)
- [ ] Try webcam inference (`python applications/predict_sentence.py --webcam`)

For detailed setup instructions, see [config/MIGRATION_GUIDE.md](config/MIGRATION_GUIDE.md)
