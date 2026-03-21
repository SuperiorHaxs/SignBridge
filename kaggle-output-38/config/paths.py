"""
Path Configuration Module for ASL-v1 Project

Centralized path management system that:
- Provides all dataset, model, and output paths
- Supports auto-detection of project root
- Validates paths on first use
- Easy to configure for different machines
"""

import os
import json
from pathlib import Path
import sys


class PathConfig:
    """Centralized path configuration for ASL-v1 project"""

    def __init__(self, config_file=None):
        """
        Initialize path configuration

        Args:
            config_file: Optional path to settings.json file
        """
        # Auto-detect project root (where config/ directory is)
        self.project_root = self._detect_project_root()

        # Load settings from JSON
        if config_file is None:
            config_file = self.project_root / "config" / "settings.json"

        self.config_file = Path(config_file)
        self.settings = self._load_settings()

        # Set base paths
        self._setup_base_paths()

        # Validate configuration
        self._validate_config()

        # Setup all derived paths
        self._setup_all_paths()

    def _detect_project_root(self):
        """Auto-detect project root directory"""
        # Start from this file's directory
        current = Path(__file__).resolve().parent.parent

        # Verify this is the project root (should contain config/ directory)
        if (current / "config").exists():
            return current

        # Fallback: try to find from current working directory
        cwd = Path.cwd()
        if (cwd / "config").exists():
            return cwd

        # Last resort: use parent of config directory
        return Path(__file__).resolve().parent.parent

    def _load_settings(self):
        """Load settings from JSON file"""
        if not self.config_file.exists():
            print(f"WARNING: Config file not found: {self.config_file}")
            print("Using default settings. Run 'python setup_config.py' to configure.")
            return {
                "data_root": "/path/to/dataset-root",
                "project_root": "auto"
            }

        try:
            with open(self.config_file, 'r') as f:
                settings = json.load(f)
            return settings
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in config file: {e}")
            print("Using default settings.")
            return {
                "data_root": "/path/to/dataset-root",
                "project_root": "auto"
            }

    def _setup_base_paths(self):
        """Setup base paths from settings"""
        # Project root
        if self.settings.get("project_root") == "auto":
            self.project_root = self._detect_project_root()
        else:
            self.project_root = Path(self.settings["project_root"])

        # Data root (external drive or dataset location)
        data_root_str = self.settings.get("data_root", "/path/to/dataset-root")
        self.data_root = Path(data_root_str)

        # Check for environment variable override
        if "DATA_ROOT" in os.environ:
            self.data_root = Path(os.environ["DATA_ROOT"])
            print(f"INFO: Using DATA_ROOT from environment: {self.data_root}")

    def _validate_config(self):
        """Validate configuration and warn about example paths"""
        issues = []

        # Check if data_root is still example path
        data_root_str = str(self.data_root)
        if "path/to" in data_root_str.lower() or "update" in data_root_str.lower():
            issues.append(f"  - data_root is still example path: {data_root_str}")

        # Check if data_root exists
        if not self.data_root.exists():
            issues.append(f"  - data_root does not exist: {self.data_root}")

        if issues:
            print("\n" + "="*70)
            print("WARNING: Configuration needs attention!")
            print("="*70)
            for issue in issues:
                print(issue)
            print("\nTo fix:")
            print(f"  1. Run: python setup_config.py")
            print(f"  2. Or manually edit: {self.config_file}")
            print("="*70 + "\n")

    def _setup_all_paths(self):
        """Setup all derived paths"""
        # ==================== DATASET PATHS ====================

        # Base dataset directory
        self.dataset_root = self.data_root

        # Augmented pool (central storage for all augmented files)
        self.augmented_pool_root = self.data_root.parent / "augmented_pool"
        self.augmented_pool_pickle = self.augmented_pool_root / "pickle"
        self.augmented_pool_pose = self.augmented_pool_root / "pose"
        self.augmented_pool_index = self.augmented_pool_pickle / "pickle_index.json"

        # Dataset splits for different class counts
        # Check if settings.json has dataset_splits defined (takes precedence)
        if 'dataset_splits' in self.settings:
            # Load from settings.json and convert relative paths to absolute
            self.dataset_splits = {}
            for num_classes_str, paths in self.settings['dataset_splits'].items():
                num_classes = int(num_classes_str)
                self.dataset_splits[num_classes] = {}
                for key, path_str in paths.items():
                    if path_str.startswith('../'):
                        # Relative path (e.g., ../augmented_pool/splits/100_classes/train_manifest.json)
                        # Resolve relative to data_root's parent (datasets/)
                        relative_path = path_str.replace('../', '')
                        self.dataset_splits[num_classes][key] = self.dataset_root.parent / relative_path
                    elif key == 'train_augmented':
                        # Legacy: special handling for old augmented pool path
                        self.dataset_splits[num_classes][key] = self.augmented_pool_pickle
                    else:
                        # Regular paths (relative to data_root)
                        self.dataset_splits[num_classes][key] = self.dataset_root / path_str
        else:
            # Fallback to hardcoded defaults if not in settings.json
            self.dataset_splits = {
                20: {
                    'train_original': self.dataset_root / "dataset_splits/20_classes/original/pickle_from_pose_split_20_class/train",
                    'train_augmented': self.augmented_pool_pickle,  # Use central augmented pool
                    'test': self.dataset_root / "dataset_splits/20_classes/original/pickle_from_pose_split_20_class/test",
                    'val': self.dataset_root / "dataset_splits/20_classes/original/pickle_from_pose_split_20_class/val",
                    'class_mapping': self.dataset_root / "dataset_splits/20_classes/20_class_mapping.json",
                },
                50: {
                    'train_original': self.dataset_root / "dataset_splits/50_classes/original/pickle_from_pose_split_50_class/train",
                    'train_augmented': self.augmented_pool_pickle,  # Use central augmented pool
                    'test': self.dataset_root / "dataset_splits/50_classes/original/pickle_from_pose_split_50_class/test",
                    'val': self.dataset_root / "dataset_splits/50_classes/original/pickle_from_pose_split_50_class/val",
                    'class_mapping': self.dataset_root / "dataset_splits/50_classes/50_class_mapping.json",
                },
                100: {
                    'train_original': self.dataset_root / "dataset_splits/100_classes/original/pickle_split_100_class/train",
                    'train_augmented': self.augmented_pool_pickle,  # Use central augmented pool
                    'test': self.dataset_root / "dataset_splits/100_classes/original/pickle_split_100_class/test",
                    'val': self.dataset_root / "dataset_splits/100_classes/original/pickle_split_100_class/val",
                    'class_mapping': self.dataset_root / "dataset_splits/100_classes/class_mapping.json",
                }
            }

        # Pickle files directory
        self.pickle_files_dir = self.dataset_root / "pickle_files"

        # Pose files directory
        self.pose_files_dir = self.dataset_root / "pose_files"

        # Video files directory
        self.videos_dir = self.dataset_root.parent / "videos"

        # Metadata files
        self.metadata_dir = self.dataset_root.parent
        self.wlasl_metadata = self.metadata_dir / "WLASL_v0.3.json"
        self.nslt_300_metadata = self.metadata_dir / "nslt_300.json"
        self.nslt_2000_metadata = self.metadata_dir / "nslt_2000.json"
        self.video_to_gloss_mapping = self.dataset_root / "video_to_gloss_mapping.json"

        # Class mapping files
        self.class_mapping_20 = self.dataset_root / "class_index_mapping_20.json"
        self.class_mapping_50 = self.dataset_root / "class_index_mapping_50.json"
        self.class_mapping_100 = self.dataset_root / "class_index_mapping_100.json"

        # Vocab file
        self.vocab_file = self.dataset_root / "vocab.json"

        # ==================== MODEL PATHS ====================

        # Models directory (in project)
        self.models_dir = self.project_root / "models"

        # Model-specific directories
        self.openhands_dir = self.models_dir / "openhands-modernized"
        self.transformer_dir = self.models_dir / "Transformer"
        self.cnn_lstm_dir = self.models_dir / "CNN-LSTM"

        # Training outputs
        self.training_outputs = self.models_dir / "training-outputs"
        self.checkpoints_dir = self.models_dir / "checkpoints"

        # Saved models for different class counts
        self.saved_models = {
            20: self.models_dir / "wlasl_20_class_model",
            50: self.models_dir / "wlasl_50_class_model",
            100: self.models_dir / "wlasl_100_class_model"
        }

        # ==================== APPLICATION PATHS ====================

        self.applications_dir = self.project_root / "applications"

        # ==================== UTILITY PATHS ====================

        # Dataset utilities
        self.dataset_utils_dir = self.project_root / "dataset-utilities"
        self.augmentation_dir = self.dataset_utils_dir / "augmentation"
        self.conversion_dir = self.dataset_utils_dir / "conversion"
        self.segmentation_dir = self.dataset_utils_dir / "segmentation"
        self.visualization_dir = self.dataset_utils_dir / "visualization"

        # Project utilities
        self.project_utils_dir = self.project_root / "project-utilities"

        # Temporary and output directories
        self.temp_dir = self.project_root / "temp"
        self.output_dir = self.project_root / "output"
        self.experiments_dir = self.project_root / "experiments"

    def get_class_mapping(self, num_classes):
        """Get class mapping file path for specified number of classes"""
        mapping = {
            20: self.class_mapping_20,
            50: self.class_mapping_50,
            100: self.class_mapping_100
        }
        return mapping.get(num_classes)

    def get_dataset_paths(self, num_classes):
        """Get dataset paths for specified number of classes"""
        return self.dataset_splits.get(num_classes, {})

    def get_model_dir(self, num_classes):
        """Get model directory for specified number of classes"""
        return self.saved_models.get(num_classes)

    def ensure_dir_exists(self, path):
        """Ensure directory exists, create if it doesn't"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __repr__(self):
        """String representation of config"""
        return f"""PathConfig(
    project_root={self.project_root}
    data_root={self.data_root}
    dataset_root={self.dataset_root}
)"""


# Global config instance (singleton pattern)
_global_config = None


def get_config(config_file=None, force_reload=False):
    """
    Get global configuration instance (singleton)

    Args:
        config_file: Optional path to settings.json
        force_reload: Force reload configuration

    Returns:
        PathConfig instance
    """
    global _global_config

    if _global_config is None or force_reload:
        _global_config = PathConfig(config_file)

    return _global_config


def print_config():
    """Print current configuration for debugging"""
    config = get_config()

    print("\n" + "="*70)
    print("ASL-v1 PATH CONFIGURATION")
    print("="*70)

    print(f"\nBASE PATHS:")
    print(f"  Project Root:  {config.project_root}")
    print(f"  Data Root:     {config.data_root}")
    print(f"  Dataset Root:  {config.dataset_root}")

    print(f"\nDATASET PATHS:")
    print(f"  Pickle Files:  {config.pickle_files_dir}")
    print(f"  Pose Files:    {config.pose_files_dir}")
    print(f"  Videos:        {config.videos_dir}")

    print(f"\nMODEL PATHS:")
    print(f"  Models Dir:    {config.models_dir}")
    print(f"  Checkpoints:   {config.checkpoints_dir}")

    print(f"\nMETADATA:")
    print(f"  WLASL:         {config.wlasl_metadata}")
    print(f"  NSLT 300:      {config.nslt_300_metadata}")

    print(f"\nCLASS MAPPINGS:")
    print(f"  20 classes:    {config.class_mapping_20}")
    print(f"  50 classes:    {config.class_mapping_50}")
    print(f"  100 classes:   {config.class_mapping_100}")

    print("="*70 + "\n")


if __name__ == "__main__":
    # Test configuration
    print_config()
