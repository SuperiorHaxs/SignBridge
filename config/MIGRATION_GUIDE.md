# Configuration System Migration Guide

## Overview

The ASL-v1 project now uses a centralized configuration system to manage all file paths. This makes it easy to move the project between computers and keeps datasets on external drives.

## 🎯 Key Benefits

✅ **No hardcoded paths** - All paths in one place
✅ **Easy migration** - Just update one config file
✅ **External drive support** - Dataset can be anywhere
✅ **Cross-platform** - Works on Windows & Linux

## 📁 Configuration Files

### 1. `config/settings.json` (Main Configuration)
```json
{
  "data_root": "C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/wlasl_poses_complete",
  "project_root": "auto"
}
```

- **data_root**: Full path to your dataset root directory
- **project_root**: Set to "auto" for automatic detection

### 2. `config/paths.py` (Path Configuration Module)
- Loads settings from `settings.json`
- Provides all dataset, model, and utility paths
- Auto-validates configuration

### 3. `setup_config.py` (Interactive Setup Script)
- Run this to configure paths interactively
- Validates dataset structure
- Creates proper `settings.json`

## 🚀 Quick Start

### First Time Setup (Current Computer)
Already configured with your current paths!

### Setting Up on New Computer

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd asl-v1
   ```

2. **Run interactive setup**
   ```bash
   python setup_config.py
   ```

3. **Or manually edit `config/settings.json`**
   ```json
   {
     "data_root": "/path/to/your/dataset-root",
     "project_root": "auto"
   }
   ```

4. **Verify configuration**
   ```bash
   python -m config.paths
   ```

## 📂 What Dataset Root Should Contain

Your dataset root directory should have:
- `pickle_files/` - Pickle format pose data
- `pose_files/` - Pose format files
- `dataset_splits/` - Train/val/test splits
- `class_index_mapping_20.json` - Class mappings
- `class_index_mapping_50.json`
- `class_index_mapping_100.json`
- `video_to_gloss_mapping.json` - Video metadata

## 🔧 Using Configuration in Scripts

### Import and Use
```python
from config import get_config

# Get configuration
config = get_config()

# Use paths
dataset_path = config.dataset_root
pickle_dir = config.pickle_files_dir
class_mapping = config.get_class_mapping(20)
```

### Available Paths

**Dataset Paths:**
- `config.dataset_root` - Main dataset directory
- `config.pickle_files_dir` - Pickle files
- `config.pose_files_dir` - Pose files
- `config.videos_dir` - Video files
- `config.dataset_splits[20/50/100]` - Split configurations

**Model Paths:**
- `config.models_dir` - Models directory
- `config.checkpoints_dir` - Training checkpoints
- `config.get_model_dir(20/50/100)` - Model for N classes

**Metadata:**
- `config.wlasl_metadata` - WLASL metadata JSON
- `config.nslt_300_metadata` - NSLT-300 metadata
- `config.video_to_gloss_mapping` - Video mappings
- `config.get_class_mapping(N)` - Class mapping for N classes

## ✅ Files Updated to Use Config System

### Core Training Scripts
- ✅ `models/training-scripts/train_asl.py`

### Dataset Utilities
- ✅ `dataset-utilities/dataset-splitting/split_pose_files.py`
- ✅ `dataset-utilities/dataset-splitting/split_pose_files_nclass.py`

### Files Still Using Old Paths (Archive)
The `archive/` directory files were intentionally not updated since they're legacy code. Update them individually if needed using the same pattern.

## 🔄 Migration Pattern

### Before (Hardcoded):
```python
DATASET_ROOT = "C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/wlasl_poses_complete"
vocab_path = f"{DATASET_ROOT}/class_index_mapping_20.json"
```

### After (Config System):
```python
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config import get_config

config = get_config()
DATASET_ROOT = str(config.dataset_root)
vocab_path = config.get_class_mapping(20)
```

## 🐛 Troubleshooting

### "Path does not exist" Warning
```
WARNING: Configuration needs attention!
  - data_root does not exist: /path/to/dataset
```

**Solution**: Update `config/settings.json` with correct path or run `python setup_config.py`

### Import Error: "No module named 'config'"
**Solution**: Make sure you're running from project root or add:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

### Still See Example Paths
**Solution**: The config system will warn you. Update `config/settings.json` with your actual paths.

## 📋 Updating More Files

To update additional files (in `dataset-utilities/`, `project-utilities/`, etc.):

1. Add config import at the top
2. Replace hardcoded paths with config paths
3. Test the script

Example template:
```python
#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config import get_config

# Load config
config = get_config()

# Use config paths
DATASET_ROOT = str(config.dataset_root)
# ... rest of your script
```

## 🎉 Next Steps

1. ✅ Configuration system created
2. ✅ Core files updated
3. ⏭️ Test on current computer
4. ⏭️ Create .gitignore
5. ⏭️ Initialize git repository
6. ⏭️ Push to GitHub
7. ⏭️ Clone on new computer
8. ⏭️ Run `python setup_config.py` on new computer
9. ⏭️ Start training!

## 📝 Notes

- The configuration is committed to git with your current paths
- When cloning, just update `config/settings.json` once
- All scripts will automatically use the new paths
- No need to edit individual files
