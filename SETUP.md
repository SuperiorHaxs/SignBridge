# ASL-v1 Setup Guide

Complete guide for setting up ASL-v1 Sign Language Recognition system on a new machine.

## Quick Start

After cloning the repository, run the interactive setup script:

```bash
# Recommended: Use Python 3.11.9
py -3.11 setup.py

# Or with system Python (if 3.11.9 is default)
python setup.py
```

The setup script will guide you through all necessary configuration steps.

---

## System Requirements

### Minimum Requirements
- **Python**: 3.11.9 (REQUIRED - this project was created with this version)
  - Download: https://www.python.org/downloads/release/python-3119/
  - **Important**: Some dependencies (like mediapipe) may not work with Python 3.13+
- **Storage**: 50GB+ free space for datasets
- **RAM**: 8GB+ (16GB recommended)
- **OS**: Windows, Linux, or macOS

### Recommended for Training
- **RAM**: 16GB+
- **Storage**: 100GB+ (SSD recommended)
- **GPU**: NVIDIA GPU with CUDA support (10-50x faster than CPU)
- **CPU**: Multi-core processor

---

## Manual Setup (Alternative to setup.py)

If you prefer manual setup or the script doesn't work for your system:

### 1. Clone Repository

```bash
git clone https://github.com/SuperiorHaxs/asl-v1.git
cd asl-v1
```

### 2. Create Virtual Environment

**IMPORTANT: Use Python 3.11.9 for best compatibility**

```bash
# Create virtual environment with Python 3.11.9
py -3.11 -m venv venv

# Or if 3.11.9 is your default Python
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 3. Install Dependencies

**For CPU training:**
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

**For GPU training (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**For GPU training (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Note:** Install PyTorch first with the correct CUDA version, then install other dependencies from requirements.txt. The requirements.txt includes:
- mediapipe (pose estimation)
- pose-format (pose processing)
- opencv-python (video processing)
- numpy, pandas, scikit-learn (data processing)
- matplotlib, seaborn (visualization)
- And other utilities

### 4. Configure Data Paths

Edit `config/settings.json`:

```json
{
  "data_root": "/absolute/path/to/datasets/wlasl_poses_complete",
  "project_root": "auto"
}
```

Replace `/absolute/path/to/datasets/wlasl_poses_complete` with your actual path.

### 5. Setup Dataset Structure

**IMPORTANT:** Dataset files are NOT included in the repository (too large).

**Step-by-step instructions to set up datasets:**

1. **Create datasets directory in your cloned repo:**
   ```bash
   cd asl-v1
   mkdir datasets
   ```

2. **Copy your wlasl_poses_complete folder:**
   ```bash
   # From another system or external drive
   cp -r /path/to/wlasl_poses_complete datasets/

   # Or on Windows
   xcopy /E /I D:\wlasl_poses_complete datasets\wlasl_poses_complete
   ```

3. **Verify the directory structure:**
   ```
   asl-v1/
   └── datasets/
       └── wlasl_poses_complete/
           ├── dataset_splits/
           │   ├── 20_classes/
           │   │   └── original/pickle_from_pose_split_20_class/
           │   │       ├── train/ (20 class folders with .pkl files)
           │   │       ├── test/
           │   │       └── val/
           │   └── 50_classes/
           │       └── original/pickle_from_pose_split_50_class/
           │           ├── train/ (50 class folders with .pkl files)
           │           ├── test/
           │           └── val/
           └── (this will be created next) →

   asl-v1/
   └── datasets/
       └── augmented_pool/
           └── pickle/
               ├── accident/
               ├── apple/
               └── ... (896 class folders)
               └── pickle_index.json
   ```

**Where to get dataset files:**
- Another system where you've already set up ASL-v1
- Original WLASL dataset source after processing with pose extraction
- **Note:** The augmented_pool goes in `datasets/augmented_pool/`, NOT inside wlasl_poses_complete

### 6. Setup Augmented Pool

The augmented pool is **REQUIRED** for training (not optional).

**Option A: Copy pre-generated augmented pool**

If you have augmented data from another system:

```bash
# Copy from another system/drive
cp -r /source/augmented_pool datasets/
```

**Option B: Generate augmented data** (time-intensive)

```bash
# Navigate to augmentation utilities
cd dataset-utilities/augmentation

# Follow augmentation scripts to generate data
# (Refer to augmentation documentation)
```

### 7. Create Augmented Pool Index

If you copied augmented data without the index file:

```python
import json
from pathlib import Path
from collections import defaultdict

pool_root = Path("datasets/augmented_pool/pickle")
index = defaultdict(list)

print("Scanning augmented pool...")
for class_dir in pool_root.iterdir():
    if class_dir.is_dir():
        class_name = class_dir.name.upper()
        for pkl_file in class_dir.glob("*.pkl"):
            index[class_name].append(pkl_file.name)

print(f"Indexed {len(index)} classes")

with open(pool_root / "pickle_index.json", 'w') as f:
    json.dump(dict(index), f, indent=2)

print("Index created successfully")
```

### 8. Verify Setup

Run the verification script:

```bash
cd models/training-scripts

# Verify 20-class setup
python training_pre_processing_setup.py --classes 20

# Verify 50-class setup
python training_pre_processing_setup.py --classes 50
```

---

## Starting Training

### Quick Test (5-10 epochs)

```bash
cd models/training-scripts

# 20 classes
python train_asl.py --classes 20 --architecture openhands --model-size small --dataset augmented --early-stopping 10

# 50 classes
python train_asl.py --classes 50 --architecture openhands --model-size small --dataset augmented --early-stopping 10
```

### Full Training

```bash
# 20 classes (2-4 min/epoch on CPU)
python train_asl.py --classes 20 --architecture openhands --model-size small --dataset augmented

# 50 classes (5-10 min/epoch on CPU)
python train_asl.py --classes 50 --architecture openhands --model-size small --dataset augmented

# With larger model (better accuracy, slower)
python train_asl.py --classes 20 --architecture openhands --model-size large --dataset augmented
```

### Testing Trained Model

```bash
# Test on held-out test set
python train_asl.py --classes 20 --architecture openhands --model-size small --test
```

---

## Expected Results

### 20 Classes
- **Random baseline**: 5%
- **Target validation accuracy**: 20-30%
- **Training time (CPU)**: 2-4 minutes/epoch
- **Training time (GPU)**: 15-30 seconds/epoch

### 50 Classes
- **Random baseline**: 2%
- **Target validation accuracy**: 12-18%
- **Training time (CPU)**: 5-10 minutes/epoch
- **Training time (GPU)**: 30-60 seconds/epoch

---

## Troubleshooting

### "Configuration failed to load"
- Check that `config/settings.json` exists
- Verify `data_root` path is correct
- Ensure path uses forward slashes `/` or escaped backslashes `\\\\`

### "Dataset directory not found"
- Verify dataset files are copied to correct location
- Check directory structure matches requirements
- Run verification script to identify missing directories

### "Index file not found"
- Create augmented pool index (see step 7 above)
- Or copy `pickle_index.json` from another system

### "Out of memory" during training
- Reduce batch size: Edit line 489 in `train_asl.py`, change `batch_size = 16` to `batch_size = 8`
- Reduce sequence length: Edit line 48, change `MAX_SEQ_LENGTH = 128` to `MAX_SEQ_LENGTH = 64`
- Use smaller model: Change `--model-size large` to `--model-size small`

### Training is very slow
- **Use GPU if available** (10-50x faster than CPU)
- Verify datasets are on local drive, not external/network drive
- Close other applications to free up CPU/RAM
- Consider using smaller model for faster iteration

### "Windows Error 1450"
- This has been fixed in recent versions
- Update to latest code: `git pull`
- Ensure using `os.listdir()` instead of pathlib in dataset loading

---

## Advanced Options

### Resume Training

Training automatically resumes from last checkpoint:

```bash
# Just run the same command again
python train_asl.py --classes 20 --architecture openhands --model-size small --dataset augmented
```

### Force Fresh Training

Ignore existing checkpoint:

```bash
python train_asl.py --classes 20 --architecture openhands --model-size small --dataset augmented --force-fresh
```

### Custom Hyperparameters

```bash
# Custom model size
python train_asl.py --classes 20 --architecture openhands --hidden-size 96 --num-layers 4 --dataset augmented

# Early stopping
python train_asl.py --classes 20 --architecture openhands --model-size small --dataset augmented --early-stopping 50
```

---

## File Structure Reference

```
asl-v1/
├── setup.py                      # Interactive setup script
├── SETUP.md                      # This file
├── config/
│   ├── settings.json             # User configuration (UPDATE THIS)
│   ├── paths.py                  # Path management system
│   └── __init__.py
├── models/
│   ├── openhands-modernized/     # OpenHands model architecture
│   └── training-scripts/
│       ├── train_asl.py          # Main training script
│       └── training_pre_processing_setup.py  # Verification script
├── datasets/                     # Your datasets (NOT in git)
│   └── wlasl_poses_complete/
│       ├── dataset_splits/
│       └── ../augmented_pool/
└── venv/                         # Virtual environment (NOT in git)
```

---

## Getting Help

- **Issues**: https://github.com/SuperiorHaxs/asl-v1/issues
- **Documentation**: See README.md for project overview
- **Training guide**: See models/training_results_comp.md for results

---

## Next Steps After Setup

1. **Quick test**: Run 20-class training for 10 epochs to verify everything works
2. **Monitor progress**: Check validation accuracy improves over epochs
3. **Full training**: Run overnight for 100-500 epochs
4. **Evaluate**: Test on held-out test set
5. **Scale up**: Try 50 or 100 classes with larger models

Good luck with your ASL recognition training! 🚀
