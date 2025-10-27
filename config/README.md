# Configuration Setup

## First Time Setup

1. Copy the example settings file:
   ```bash
   cp config/settings.json.example config/settings.json
   ```

2. Edit `config/settings.json` with your local paths:
   ```json
   {
     "data_root": "/your/actual/path/to/datasets/wlasl_poses_complete",
     "project_root": "auto"
   }
   ```

## Settings

- **data_root**: Absolute path to your WLASL dataset root directory
  - Should contain `dataset_splits/`, `pose_files/`, etc.
  - Example: `"C:/Users/yourname/Projects/WLASL-proj/asl-v1/datasets/wlasl_poses_complete"`

- **project_root**: Project root directory
  - Use `"auto"` to auto-detect (recommended)
  - Or provide absolute path if needed

## Note

- `settings.json` is gitignored (contains your local paths)
- `settings.json.example` is committed (template for others)
- Never commit your personal `settings.json` with local paths
