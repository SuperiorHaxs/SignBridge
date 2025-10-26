# Dataset Setup Guide

This guide explains how to use the automated dataset setup system for creating train/val/test splits.

## Overview

The `training_pre_processing_setup.py` script now provides **automated dataset splitting** in addition to verification. It eliminates the need to manually run separate splitting and conversion scripts.

## Workflow

### Step 1: Create Dataset Splits

For **100 classes** (or any class count not yet configured):

```bash
python training_pre_processing_setup.py --classes 100 --create-splits
```

This will:
1. Determine which classes to include (builds on existing 20/50-class if available)
2. Split `.pose` files from `pose_files/` into train/val/test (70/15/15 ratio)
3. Convert `.pose` files to `.pkl` format for each split
4. Create directory structure:
   ```
   dataset_splits/100_classes/original/
   ├── pose_split_100_class/
   │   ├── train/ (class folders with .pose files)
   │   ├── val/   (class folders with .pose files)
   │   └── test/  (class folders with .pose files)
   └── pickle_split_100_class/
       ├── train/ (class folders with .pkl files)
       ├── val/   (class folders with .pkl files)
       └── test/  (class folders with .pkl files)
   ```

### Step 2: Update Configuration

After splits are created, update `config/settings.json` to add the new paths:

```json
{
  "dataset_splits": {
    "100": {
      "train_original": "dataset_splits/100_classes/original/pickle_split_100_class/train",
      "val": "dataset_splits/100_classes/original/pickle_split_100_class/val",
      "test": "dataset_splits/100_classes/original/pickle_split_100_class/test",
      "train_augmented": "augmented_pool/pickle/75pt_augmented_pool",
      "class_mapping": "dataset_splits/100_classes/class_mapping.json"
    }
  }
}
```

### Step 3: Verify Setup

```bash
python training_pre_processing_setup.py --classes 100 --verify-only
```

This checks:
- ✓ Directory structure exists
- ✓ All class folders present
- ✓ Sample counts per class
- ✓ Augmented pool coverage
- ✓ Class mapping file

### Step 4: Start Training

```bash
python train_asl.py --classes 100 --dataset augmented --model-size large
```

## Usage Modes

### Verify Existing Setup (Default)
```bash
python training_pre_processing_setup.py --classes 50
```

Runs all verification checks and provides helpful error messages if issues are found.

### Verify Only (No Auto-Create Suggestions)
```bash
python training_pre_processing_setup.py --classes 50 --verify-only
```

Only verifies, doesn't offer to create missing splits.

### Create New Splits
```bash
python training_pre_processing_setup.py --classes 100 --create-splits
```

Automatically creates train/val/test splits from `.pose` files.

## Class Selection Logic

The system uses **dynamic frequency-based incremental selection** - no hardcoded class lists!

### How It Works:

**For 20 classes (first time)**:
1. Count video frequency for all classes in `video_to_gloss_mapping.json`
2. Select top 20 most frequent classes
3. Save to `dataset_splits/20_classes/class_mapping.json`
4. Split and convert files

**For 50 classes**:
1. Check if `dataset_splits/50_classes/class_mapping.json` exists
   - If yes: Use it (stable, reproducible)
   - If no: Continue to step 2
2. Load existing 20 classes from `dataset_splits/20_classes/class_mapping.json`
3. Count frequencies of remaining classes (excluding the 20)
4. Select next 30 most frequent classes
5. Save combined 50 classes to `dataset_splits/50_classes/class_mapping.json`
6. Split and convert files

**For 100 classes**:
1. Check if `dataset_splits/100_classes/class_mapping.json` exists
   - If yes: Use it (stable, reproducible)
   - If no: Continue to step 2
2. Load existing 50 classes from `dataset_splits/50_classes/class_mapping.json`
3. Count frequencies of remaining classes (excluding the 50)
4. Select next 50 most frequent classes
5. Save combined 100 classes to `dataset_splits/100_classes/class_mapping.json`
6. Split and convert files

### Key Benefits:

✅ **No hardcoded lists** - All selection is data-driven from metadata
✅ **Stable** - Once `class_mapping.json` is created, classes are locked in
✅ **Incremental** - 50-class includes all 20-class; 100-class includes all 50-class
✅ **Reproducible** - Same metadata always produces same class selection
✅ **Flexible** - Can manually edit `class_mapping.json` if needed

### Class Mapping File Format:

```json
{
  "classes": ["accident", "apple", "bath", ...],
  "num_classes": 20,
  "creation_method": "frequency_based_incremental"
}
```

## Requirements

- `pose_format` library (for `.pose` to `.pkl` conversion)
- `tqdm` (for progress bars)
- All `.pose` files in `pose_files/` directory
- Valid `video_to_gloss_mapping.json`

## Troubleshooting

### "pose_format not available"
Install the pose_format library:
```bash
pip install pose-format
```

### "No .pose files found"
Ensure `.pose` files are in the configured `pose_files/` directory.

### "No configuration found for N classes"
Update `config/settings.json` to add configuration for the desired class count.

## Architecture

The system uses a **modular architecture** - no code duplication!

```
dataset-utilities/
├── dataset-splitting/
│   └── split_pose_files_nclass.py     ← Splitting logic (standalone or imported)
└── conversion/
    └── pose_to_pickle_converter.py    ← Conversion logic (standalone or imported)

models/training-scripts/
└── training_pre_processing_setup.py   ← Orchestrator (imports utilities)
```

### Utilities Can Be Used Standalone:

**Split pose files directly:**
```bash
python dataset-utilities/dataset-splitting/split_pose_files_nclass.py --num-classes 100
```

**Convert pose to pickle directly:**
```bash
python dataset-utilities/conversion/pose_to_pickle_converter.py --input-dir <pose_dir> --output-dir <pkl_dir>
```

### Or Use Integrated Workflow:

```bash
python training_pre_processing_setup.py --classes 100 --create-splits
```

The integrated workflow:
- Imports functions from both utilities
- Orchestrates the complete setup (split + convert + verify)
- Single command for everything

## Previous Manual Workflow (No Longer Needed)

Previously, you had to:
1. Run `split_pose_files_nclass.py` to split .pose files
2. Run `pose_to_pickle_converter.py` three times (train/val/test)
3. Manually update `config/settings.json`
4. Classes were hardcoded in Python

Now:
- ✅ Run `--create-splits` - does all of this automatically!
- ✅ Classes selected dynamically from metadata
- ✅ Classes saved to `class_mapping.json` for stability
