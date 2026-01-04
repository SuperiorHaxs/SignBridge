#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_synthetic_dataset.py
Batch evaluation of synthetic ASL sentence dataset

This script:
1. Loads synthetic gloss-to-sentence dataset
2. For each entry:
   - Creates concatenated pose file from glosses using sentence_to_pickle.py
   - Calculates baseline BLEU (glosses → simple sentence)
   - Runs prediction using predict_sentence.py with metadata-based segmentation
   - Calculates model BLEU (model prediction → reference sentence)
3. Generates comparison table showing improvements

Usage:
    python evaluate_synthetic_dataset.py [options]

    --dataset PATH            Path to synthetic dataset JSON (default: ../../datasets/synthetic_sentences/synthetic_gloss_to_sentence_llm_dataset_50_glosses.json)
    --checkpoint PATH         Path to model checkpoint directory
    --no-llm                  Disable LLM sentence construction (use simple gloss joining)
    --output-dir PATH         Output directory for results (default: ./evaluation_results)
    --limit N                 Limit evaluation to first N entries (default: all)
    --num-glosses N           Number of glosses in model (default: 50)
    --skip-existing           Skip entries that already have results
"""

import os
import sys
import json
import subprocess
import argparse
import tempfile
import shutil
import string
import pickle
from pathlib import Path
from datetime import datetime
import traceback
import numpy as np
import torch

# Suppress HuggingFace Hub warnings and progress bars
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Add parent directories to path for imports
# synthetic_evaluation -> evaluation_metrics -> project-utilities
project_utilities_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_utilities_dir)

# Import modular evaluation metrics from the parent evaluation_metrics package
from evaluation_metrics import (
    calculate_gloss_accuracy,
    calculate_perfect_translation_rate,
    calculate_bleu_score,
    calculate_bert_score,
    calculate_quality_score,
    calculate_composite_score,
    calculate_coverage,
    QualityScorer,
    METRIC_WEIGHTS,
    BLEU_AVAILABLE,
    BERTSCORE_AVAILABLE,
    QUALITY_SCORING_AVAILABLE,
)

# Print availability status
print(f"Metrics availability: BLEU={BLEU_AVAILABLE}, BERTScore={BERTSCORE_AVAILABLE}, Quality={QUALITY_SCORING_AVAILABLE}")

# Import two-stage pipeline and LLM interface for two-phase mode
TWO_STAGE_AVAILABLE = False
try:
    from llm_interface.two_stage_pipeline import TwoStagePipeline
    from llm_interface import create_llm_provider
    TWO_STAGE_AVAILABLE = True
    print("Two-stage pipeline: AVAILABLE")
except ImportError as e:
    print(f"Two-stage pipeline: NOT AVAILABLE ({e})")

# ============================================================================


class SyntheticDatasetEvaluator:
    """Evaluate synthetic ASL sentence dataset"""

    def __init__(
        self,
        dataset_path,
        checkpoint_path=None,
        use_llm=True,
        output_dir="./evaluation_results",
        num_glosses=50,
        skip_existing=False,
        keep_poses=False,
        no_confidence_scores=False,
        split="all",
        quiet=False,
        use_two_stage=False,
        prompt_file=None,
        use_manifest=False,
        no_mask=False
    ):
        self.dataset_path = dataset_path
        self.checkpoint_path = checkpoint_path
        self.use_llm = use_llm
        self.output_dir = output_dir
        self.skip_existing = skip_existing

        # Auto-detect num_glosses from checkpoint config if provided
        if checkpoint_path:
            checkpoint = Path(checkpoint_path)
            if checkpoint.suffix in ['.pt', '.bin', '.pth']:
                config_path = checkpoint.parent / "config.json"
            else:
                config_path = checkpoint / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    model_config = json.load(f)
                detected_glosses = model_config.get('vocab_size')
                if detected_glosses:
                    print(f"Auto-detected num_glosses={detected_glosses} from checkpoint config")
                    num_glosses = detected_glosses

        self.num_glosses = num_glosses
        self.keep_poses = keep_poses
        self.no_confidence_scores = no_confidence_scores
        self.split = split
        self.quiet = quiet
        self.use_two_stage = use_two_stage and TWO_STAGE_AVAILABLE
        self.prompt_file = prompt_file
        self.use_manifest = use_manifest
        self.no_mask = no_mask

        # Manifest-based mode setup
        self.manifest = None
        self.pickle_pool = None
        self.model = None
        self.id_to_gloss = None

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Load manifest and model if using manifest mode
        if self.use_manifest:
            self._load_manifest_and_model()

        # Initialize LLM provider and pipelines
        self.llm_provider = None
        self.two_stage_pipeline = None
        self.one_stage_prompt_template = None

        if self.use_two_stage:
            try:
                self.llm_provider = create_llm_provider()
                self.two_stage_pipeline = TwoStagePipeline(self.llm_provider)
                print("Two-stage pipeline: INITIALIZED")
            except Exception as e:
                print(f"Warning: Failed to initialize two-stage pipeline: {e}")
                self.use_two_stage = False
        elif self.use_llm and self.use_manifest:
            # One-stage LLM for manifest mode
            try:
                self.llm_provider = create_llm_provider()
                # Load one-stage prompt template
                prompt_path = self.prompt_file or Path(__file__).parent.parent.parent / "llm_interface" / "prompts" / "llm_prompt_topk.txt"
                if Path(prompt_path).exists():
                    self.one_stage_prompt_template = Path(prompt_path).read_text(encoding='utf-8')
                    print(f"One-stage LLM: INITIALIZED (prompt: {Path(prompt_path).name})")
                else:
                    print(f"Warning: One-stage prompt not found at {prompt_path}")
                    self.use_llm = False
            except Exception as e:
                print(f"Warning: Failed to initialize one-stage LLM: {e}")
                self.use_llm = False

        # Load dataset
        self.dataset = self._load_dataset()

        # Results tracking
        self.results = []

        # Initialize quality scorer using the modular library
        self.quality_scorer = None
        if QUALITY_SCORING_AVAILABLE:
            self.quality_scorer = QualityScorer(verbose=not self.quiet)

    def _load_dataset(self):
        """Load synthetic dataset JSON"""
        print(f"Loading dataset: {self.dataset_path}")

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        with open(self.dataset_path, 'r') as f:
            dataset = json.load(f)

        print(f"SUCCESS: Loaded {len(dataset)} entries")
        return dataset

    def _load_manifest_and_model(self):
        """Load val_manifest.json and initialize model for manifest-based evaluation"""
        print("\n[MANIFEST MODE] Loading manifest and model...")

        # Add project root to path for imports
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        sys.path.insert(0, str(project_root))

        try:
            from config import get_config
            config = get_config()
        except ImportError as e:
            raise RuntimeError(f"Cannot use manifest mode - config module not available: {e}")

        # Get paths from config for this num_glosses
        if self.num_glosses not in config.dataset_splits:
            raise ValueError(f"No dataset config for {self.num_glosses} classes. "
                           f"Available: {list(config.dataset_splits.keys())}")

        splits = config.dataset_splits[self.num_glosses]

        # Determine which manifest to use based on split parameter
        if self.split == "test":
            manifest_key = 'test_manifest'
        else:
            manifest_key = 'val_manifest'  # Default to val for "val" or "all"

        if manifest_key not in splits:
            raise ValueError(f"No {manifest_key} defined for {self.num_glosses} classes in config")

        manifest_path = splits[manifest_key]
        pickle_pool_path = splits['pickle_pool']

        print(f"  Loading manifest ({self.split}): {manifest_path}")
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        self.pickle_pool = Path(pickle_pool_path)
        print(f"  Pickle pool: {self.pickle_pool}")
        print(f"  Manifest has {len(self.manifest['classes'])} classes, {self.manifest['total_samples']} samples")

        # Load model
        self._load_model_for_manifest()

        print("[MANIFEST MODE] Ready")

    def _load_model_for_manifest(self):
        """Load the model for direct pickle prediction"""
        # Add openhands path
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        sys.path.insert(0, str(project_root))

        from config import get_config
        config = get_config()

        openhands_src = config.openhands_dir / "src"
        sys.path.insert(0, str(openhands_src))

        from openhands_modernized import OpenHandsModel, OpenHandsConfig, WLASLPoseProcessor

        # Load from checkpoint or default model
        if self.checkpoint_path:
            checkpoint = Path(self.checkpoint_path)
            # If checkpoint looks like a file path (ends with .pt or .bin), use its parent directory
            if checkpoint.suffix in ['.pt', '.bin', '.pth']:
                model_dir = checkpoint.parent
            else:
                model_dir = checkpoint
        else:
            model_dir = config.openhands_dir / "production-models" / f"wlasl_{self.num_glosses}_class_50s_model"

        print(f"  Loading model from: {model_dir}")

        # Load config
        config_path = model_dir / "config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        model_config = OpenHandsConfig(
            num_pose_keypoints=config_dict.get('num_pose_keypoints', 83),
            pose_channels=config_dict.get('pose_channels', 3),
            pose_features=config_dict.get('pose_features', 279),
            use_finger_features=config_dict.get('use_finger_features', True),
            finger_features=config_dict.get('finger_features', 30),
            hidden_size=config_dict.get('hidden_size', 256),
            num_hidden_layers=config_dict.get('num_hidden_layers', 6),
            num_attention_heads=config_dict.get('num_attention_heads', 16),
            intermediate_size=config_dict.get('intermediate_size', 1024),
            max_position_embeddings=config_dict.get('max_position_embeddings', 257),
            dropout_prob=config_dict.get('dropout_prob', 0.2),
            vocab_size=config_dict.get('vocab_size', self.num_glosses),
            use_cls_token=config_dict.get('use_cls_token', True)
        )

        # Load class mapping
        mapping_path = model_dir / "class_index_mapping.json"
        with open(mapping_path, 'r') as f:
            self.id_to_gloss = json.load(f)

        # Load masked classes config (optional, unless --no-mask is set)
        self.masked_class_ids = []
        if self.no_mask:
            print("  Class masking: DISABLED (--no-mask flag)")
        else:
            masked_classes_file = model_dir / "masked_classes.json"
            if masked_classes_file.exists():
                with open(masked_classes_file, 'r') as f:
                    masked_config = json.load(f)
                self.masked_class_ids = masked_config.get('masked_class_ids', [])
                masked_names = masked_config.get('masked_class_names', [])
                print(f"  Loaded class masking: {len(self.masked_class_ids)} classes masked")
                print(f"    Masked: {', '.join(masked_names)}")

        # Create processor for feature extraction
        self.processor = WLASLPoseProcessor()
        self.model_config = model_config
        self.max_seq_length = 256

        # Create and load model
        self.model = OpenHandsModel(model_config)
        # Use the checkpoint file if provided, otherwise look for standard files
        if self.checkpoint_path and Path(self.checkpoint_path).is_file():
            weights_path = Path(self.checkpoint_path)
        elif (model_dir / "best_model.pt").exists():
            weights_path = model_dir / "best_model.pt"
        else:
            weights_path = model_dir / "pytorch_model.bin"
        state_dict = torch.load(weights_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"  Model loaded: {model_config.vocab_size} classes, {model_config.hidden_size} hidden")

    def _get_pickle_file_for_gloss(self, gloss):
        """
        Get the pickle file path for a gloss from the val_manifest.
        Returns the first (original, non-augmented) pickle file.
        """
        gloss_lower = gloss.lower()
        if gloss_lower not in self.manifest['classes']:
            return None

        # Get the first family (video) and first file (original, not augmented)
        families = self.manifest['classes'][gloss_lower]
        if not families:
            return None

        first_family = families[0]
        files = first_family.get('files', [])
        if not files:
            return None

        # First file should be the original (e.g., "37884.pkl", not "37884_aug_00.pkl")
        original_file = files[0]
        return self.pickle_pool / gloss_lower / original_file

    def _predict_from_pickle(self, pickle_path):
        """
        Run prediction on a single pickle file.
        Returns top-k predictions as dict with confidence scores.

        Uses the same processing as WLASLOpenHandsDataset for consistency.
        """
        # Use processor to load and preprocess (same as dataset does)
        pose_sequence = self.processor.load_pickle_pose(str(pickle_path))
        pose_sequence = self.processor.preprocess_pose_sequence(pose_sequence, augment=False)

        # Extract finger features BEFORE padding (on actual data only)
        finger_features = None
        if self.model_config.use_finger_features:
            finger_features = self.processor.extract_finger_features(pose_sequence)

        # Pad/truncate pose sequence using processor method
        pose_sequence, attention_mask = self.processor.pad_or_truncate_sequence(
            pose_sequence, self.max_seq_length
        )

        # Pad finger features to match
        if finger_features is not None:
            seq_len = len(finger_features)
            if seq_len > self.max_seq_length:
                finger_features = finger_features[:self.max_seq_length]
            elif seq_len < self.max_seq_length:
                padding = np.zeros((self.max_seq_length - seq_len, 30), dtype=np.float32)
                finger_features = np.vstack([finger_features, padding])

        # Convert to tensors with batch dimension
        pose_tensor = torch.from_numpy(pose_sequence).float().unsqueeze(0)  # (1, T, 83, 3)
        attention_tensor = torch.from_numpy(attention_mask).long().unsqueeze(0)  # (1, T)

        finger_tensor = None
        if finger_features is not None:
            finger_tensor = torch.from_numpy(finger_features).float().unsqueeze(0)  # (1, T, 30)

        # Run model (must pass attention_mask as 2nd positional arg, like accuracy analysis does)
        with torch.no_grad():
            outputs = self.model(pose_tensor, attention_tensor, finger_tensor)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs

            # Apply class masking if configured
            if hasattr(self, 'masked_class_ids') and self.masked_class_ids:
                for class_id in self.masked_class_ids:
                    logits[:, class_id] = float('-inf')

            probs = torch.softmax(logits, dim=-1)

        # Get top-3 predictions
        top_k = 3
        top_probs, top_indices = torch.topk(probs[0], top_k)

        result = {
            'top_prediction': self.id_to_gloss[str(top_indices[0].item())].lower(),
            'confidence': top_probs[0].item(),
            'top_k': []
        }

        for i in range(top_k):
            idx = top_indices[i].item()
            result['top_k'].append({
                'gloss': self.id_to_gloss[str(idx)].lower(),
                'confidence': top_probs[i].item()
            })

        return result

    def _predict_glosses_from_manifest(self, glosses):
        """
        Predict each gloss using pickle files from manifest.
        Returns list of prediction dicts with top-k info.
        """
        predictions = []

        for gloss in glosses:
            pickle_path = self._get_pickle_file_for_gloss(gloss)

            if pickle_path is None or not pickle_path.exists():
                print(f"  WARNING: No pickle file found for gloss '{gloss}'")
                predictions.append({
                    'top_prediction': 'UNKNOWN',
                    'confidence': 0.0,
                    'top_k': []
                })
                continue

            try:
                pred = self._predict_from_pickle(pickle_path)
                predictions.append(pred)

                if not self.quiet:
                    top1 = pred['top_prediction']
                    conf = pred['confidence']
                    match = "[OK]" if top1.upper() == gloss.upper() else "[X]"
                    print(f"  {gloss}: {top1} ({conf*100:.1f}%) {match}")

            except Exception as e:
                print(f"  ERROR predicting {gloss}: {e}")
                predictions.append({
                    'top_prediction': 'ERROR',
                    'confidence': 0.0,
                    'top_k': []
                })

        return predictions

    def _run_one_stage_llm(self, predicted_glosses):
        """
        Run one-stage LLM to generate sentence from predicted glosses.

        Args:
            predicted_glosses: List of prediction dicts with top_k info

        Returns:
            tuple: (predicted_sentence, llm_response_json)
        """
        if not self.llm_provider or not self.one_stage_prompt_template:
            # Fallback to simple joining
            top1_glosses = [p['top_prediction'] for p in predicted_glosses]
            sentence = ' '.join(top1_glosses) + '.'
            return sentence, sentence

        # Format gloss details for the prompt
        gloss_lines = []
        for i, pred in enumerate(predicted_glosses, 1):
            top_k = pred.get('top_k', [])
            if top_k:
                options = ', '.join([f"'{g['gloss']}' ({g['confidence']*100:.1f}%)" for g in top_k])
                gloss_lines.append(f"Position {i}: {options}")
            else:
                gloss_lines.append(f"Position {i}: '{pred['top_prediction']}' ({pred['confidence']*100:.1f}%)")

        gloss_details = '\n'.join(gloss_lines)

        # Build prompt
        prompt = self.one_stage_prompt_template.replace('{gloss_details}', gloss_details)

        try:
            # Call LLM
            response = self.llm_provider.generate(prompt)

            if not self.quiet:
                print(f"  [ONE-STAGE LLM] Response received")

            # Parse JSON response
            import re
            json_match = re.search(r'\{[\s\S]*"selections"[\s\S]*"sentence"[\s\S]*\}', response)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    sentence = data.get('sentence', '')
                    selections = data.get('selections', [])
                    if not self.quiet:
                        print(f"  [ONE-STAGE LLM] Selections: {selections}")
                        print(f"  [ONE-STAGE LLM] Sentence: '{sentence}'")
                    return sentence, response
                except json.JSONDecodeError:
                    pass

            # If JSON parsing fails, use raw response
            return response.strip(), response

        except Exception as e:
            print(f"  [ONE-STAGE LLM] Error: {e}")
            # Fallback to simple joining
            top1_glosses = [p['top_prediction'] for p in predicted_glosses]
            sentence = ' '.join(top1_glosses) + '.'
            return sentence, sentence

    def _create_concatenated_pose(self, glosses, entry_id):
        """
        Create concatenated pose file from glosses using sentence_to_pickle.py
        Returns: (pose_file_path, metadata_file_path) or (None, None) on error
        """
        if not self.quiet:
            print(f"\n  Creating concatenated pose from glosses: {', '.join(glosses)}")

        # Build command for sentence_to_pickle.py (in pose_utils directory)
        project_utilities_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sentence_to_pickle_script = os.path.join(
            project_utilities_dir,
            "pose_utils",
            "sentence_to_pickle.py"
        )

        # Join glosses into a sentence string
        sentence_str = ' '.join(glosses)

        cmd = [
            sys.executable,
            sentence_to_pickle_script,
            "--sentence", sentence_str,
            "--pose-mode",  # Use pose concatenation mode
            "--num-glosses", str(self.num_glosses),
            "--output-dir", self.output_dir
        ]

        # Add split parameter if not using default
        if self.split != "all":
            cmd.extend(["--split", self.split])

        if not self.quiet:
            print(f"  COMMAND: {' '.join(cmd[:5])}...")  # Abbreviated

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=60
            )

            # Parse output to find generated files
            # Look for "FINAL OUTPUT:" and "SEGMENT METADATA:" lines
            output_lines = result.stdout.strip().split('\n')

            pose_file = None
            metadata_file = None

            # Look for file paths in output
            for line in output_lines:
                if 'FINAL OUTPUT:' in line:
                    # Extract path from "FINAL OUTPUT: path/to/file.pose"
                    parts = line.split('FINAL OUTPUT:', 1)
                    if len(parts) == 2:
                        pose_file = parts[1].strip()
                elif 'SEGMENT METADATA:' in line:
                    # Extract path from "SEGMENT METADATA: path/to/file_segments.json"
                    parts = line.split('SEGMENT METADATA:', 1)
                    if len(parts) == 2:
                        metadata_file = parts[1].strip()

            # Verify files exist
            if pose_file and metadata_file and os.path.exists(pose_file) and os.path.exists(metadata_file):
                if not self.quiet:
                    print(f"  SUCCESS: Pose file created: {pose_file}")
                    print(f"  SUCCESS: Metadata created: {metadata_file}")
                return pose_file, metadata_file
            else:
                if not self.quiet:
                    print(f"  ERROR: Expected files not found")
                    print(f"  Pose file: {pose_file}")
                    print(f"  Metadata file: {metadata_file}")
                    print(f"  LAST 10 LINES OF OUTPUT:")
                    for line in output_lines[-10:]:
                        print(f"    {line}")
                return None, None

        except subprocess.TimeoutExpired:
            print(f"  ERROR: sentence_to_pickle.py timed out (60s)")
            return None, None
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: sentence_to_pickle.py failed with exit code {e.returncode}")
            print(f"  STDERR: {e.stderr}")
            return None, None
        except Exception as e:
            print(f"  ERROR: Unexpected error: {e}")
            traceback.print_exc()
            return None, None

    def _extract_top1_glosses(self, predicted_glosses):
        """
        Extract top-1 predictions from predicted glosses.
        Handles both string predictions and dict predictions with top-k info.

        Args:
            predicted_glosses: List of predictions (can be strings or dicts)

        Returns:
            List of top-1 gloss strings
        """
        top1_glosses = []

        for gloss in predicted_glosses:
            if isinstance(gloss, str):
                # Simple string prediction
                top1_glosses.append(gloss)
            elif isinstance(gloss, dict):
                # Dict with top-k predictions - extract top-1
                # Try different possible keys for top-1 prediction
                top1 = gloss.get('gloss', gloss.get('top_prediction', gloss.get('prediction', 'UNKNOWN')))
                top1_glosses.append(top1)
            else:
                # Fallback
                top1_glosses.append(str(gloss))

        return top1_glosses

    def _analyze_llm_choices(self, predicted_glosses, llm_sentence):
        """
        Analyze which predictions from top-3 LLM actually used

        Returns: dict with 'used_alternatives', 'alternative_positions', 'total_glosses'
        """
        if not predicted_glosses or not llm_sentence:
            return {
                'used_alternatives': 0,
                'alternative_positions': [],
                'total_glosses': 0
            }

        try:
            # Normalize LLM sentence
            translator = str.maketrans('', '', string.punctuation)
            llm_normalized = llm_sentence.translate(translator).strip().lower()
            llm_words = llm_normalized.split()

            used_alternatives = 0
            alternative_positions = []

            for idx, gloss_pred in enumerate(predicted_glosses):
                if not isinstance(gloss_pred, dict):
                    # No top-k info, skip
                    continue

                # Get top-1 and alternatives
                top1 = gloss_pred.get('gloss', gloss_pred.get('top_prediction', '')).lower()
                alternatives = []
                alternative_details = []  # Store confidence info

                # Try to get top-2 and top-3 from top_k array
                if 'top_k' in gloss_pred and len(gloss_pred['top_k']) > 1:
                    # Extract alternatives (skip first one which is top-1)
                    for item in gloss_pred['top_k'][1:3]:  # Get top-2 and top-3
                        alternatives.append(item['gloss'].lower())
                        alternative_details.append({
                            'gloss': item['gloss'],
                            'confidence': item['confidence']
                        })
                elif 'alternatives' in gloss_pred:
                    alternatives = [alt.lower() for alt in gloss_pred['alternatives'][:2]]
                    alternative_details = [{'gloss': alt, 'confidence': None} for alt in gloss_pred['alternatives'][:2]]
                elif 'top_2' in gloss_pred:
                    alternatives.append(gloss_pred['top_2'].lower())
                    alternative_details.append({'gloss': gloss_pred['top_2'], 'confidence': None})
                    if 'top_3' in gloss_pred:
                        alternatives.append(gloss_pred['top_3'].lower())
                        alternative_details.append({'gloss': gloss_pred['top_3'], 'confidence': None})

                if not alternatives:
                    continue

                # Check if top-1 appears in LLM sentence
                top1_used = any(top1 in word or word in top1 for word in llm_words)

                # Check if any alternative appears in LLM sentence
                alt_used = False
                used_alt = None
                used_alt_detail = None
                for alt_idx, alt in enumerate(alternatives):
                    if any(alt in word or word in alt for word in llm_words):
                        alt_used = True
                        used_alt = alt
                        used_alt_detail = alternative_details[alt_idx]
                        break

                # If alternative used but not top-1, LLM chose alternative
                if alt_used and not top1_used:
                    used_alternatives += 1
                    # Get top-1 confidence from top_k if available
                    top1_confidence = None
                    if 'top_k' in gloss_pred and len(gloss_pred['top_k']) > 0:
                        top1_confidence = gloss_pred['top_k'][0]['confidence']

                    alternative_positions.append({
                        'position': idx,
                        'top1': top1,
                        'top1_confidence': top1_confidence,
                        'used': used_alt,
                        'used_confidence': used_alt_detail['confidence'] if used_alt_detail else None,
                        'rank': alt_idx + 2  # +2 because we skip top-1 (rank 2 = top-2, rank 3 = top-3)
                    })

            return {
                'used_alternatives': used_alternatives,
                'alternative_positions': alternative_positions,
                'total_glosses': len(predicted_glosses)
            }

        except Exception as e:
            print(f"  ERROR: Alternative analysis failed: {e}")
            return {
                'used_alternatives': 0,
                'alternative_positions': [],
                'total_glosses': len(predicted_glosses)
            }

    def _extract_sentence_from_response(self, llm_response):
        """
        Extract just the sentence from LLM response (handles both JSON and plain text).

        Args:
            llm_response: LLM output (JSON or plain sentence)

        Returns:
            Clean sentence string
        """
        if not llm_response:
            return ""

        try:
            import json
            import re

            # Debug: Print first 500 chars of raw response
            if not self.quiet:
                print(f"  DEBUG: Raw LLM response (first 500 chars): {llm_response[:500]}")

            # Strip markdown code fences if present
            cleaned_response = llm_response.strip()
            if cleaned_response.startswith('```'):
                # Remove opening fence (```json or ```)
                cleaned_response = re.sub(r'^```(?:json)?\s*', '', cleaned_response)
                # Remove closing fence
                cleaned_response = re.sub(r'\s*```\s*$', '', cleaned_response)
                cleaned_response = cleaned_response.strip()
                if not self.quiet:
                    print(f"  DEBUG: Stripped markdown fences. Cleaned response: {cleaned_response[:300]}")

            # Try to parse as JSON
            json_match = re.search(r'\{[\s\S]*"selections"[\s\S]*"sentence"[\s\S]*\}', cleaned_response)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    sentence = data.get('sentence', '')
                    if not self.quiet:
                        print(f"  DEBUG: Extracted sentence from JSON: '{sentence}'")
                    return sentence if sentence else llm_response
                except json.JSONDecodeError as e:
                    if not self.quiet:
                        print(f"  DEBUG: JSON parsing failed: {e}")
                    pass

            # Not JSON or parsing failed, return cleaned response
            if not self.quiet:
                print(f"  DEBUG: No JSON found, returning cleaned response")
            return cleaned_response

        except Exception as e:
            if not self.quiet:
                print(f"  DEBUG: Exception in _extract_sentence_from_response: {e}")
            return llm_response

    def _extract_effective_glosses(self, predicted_glosses, llm_sentence):
        """
        Extract which glosses from top-k the LLM actually used in the generated sentence.
        First tries to parse JSON format with explicit selections.
        Falls back to reverse-engineering if JSON parsing fails.

        Args:
            predicted_glosses: List of predictions with top_k info (dicts)
            llm_sentence: The LLM-generated sentence (or JSON with selections)

        Returns:
            List of effective gloss strings (what LLM chose from top-k)
        """
        if not predicted_glosses or not llm_sentence:
            return []

        try:
            # First, try to parse as JSON
            import json
            import re

            # Strip markdown code fences if present
            cleaned_response = llm_sentence.strip()
            if cleaned_response.startswith('```'):
                # Remove opening fence (```json or ```)
                cleaned_response = re.sub(r'^```(?:json)?\s*', '', cleaned_response)
                # Remove closing fence
                cleaned_response = re.sub(r'\s*```\s*$', '', cleaned_response)
                cleaned_response = cleaned_response.strip()

            # Try to extract JSON from response
            sentence_for_fallback = None  # Track sentence from JSON for fallback
            json_match = re.search(r'\{[\s\S]*"selections"[\s\S]*"sentence"[\s\S]*\}', cleaned_response)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    selections = data.get('selections', [])
                    sentence = data.get('sentence', '')
                    sentence_for_fallback = sentence  # Save for fallback

                    # Validate: selections count matches position count
                    if len(selections) == len(predicted_glosses):
                        # Additional validation: check if sentence uses all selections
                        sentence_lower = sentence.lower()
                        all_used = all(sel.lower() in sentence_lower for sel in selections)

                        if all_used:
                            print(f"  [JSON PARSE SUCCESS] Extracted {len(selections)} selections from JSON")
                            return [str(sel).upper() for sel in selections]
                        else:
                            print(f"  [JSON VALIDATION WARNING] Not all selections used in sentence, falling back to reverse engineering")
                    else:
                        print(f"  [JSON VALIDATION WARNING] Selection count mismatch ({len(selections)} vs {len(predicted_glosses)}), falling back")
                except json.JSONDecodeError as e:
                    print(f"  [JSON PARSE WARNING] Invalid JSON format: {e}, falling back to reverse engineering")

            # Fallback: Reverse engineer from sentence (original logic)
            # Use sentence from JSON if available, otherwise use cleaned response
            sentence_to_parse = sentence_for_fallback if sentence_for_fallback else cleaned_response

            # Normalize LLM sentence
            translator = str.maketrans('', '', string.punctuation)
            llm_normalized = sentence_to_parse.translate(translator).strip().lower()
            llm_words = llm_normalized.split()

            effective_glosses = []

            for idx, gloss_pred in enumerate(predicted_glosses):
                if not isinstance(gloss_pred, dict):
                    # Simple string prediction, use as-is
                    effective_glosses.append(str(gloss_pred))
                    continue

                # Get top-k predictions
                top_k = gloss_pred.get('top_k', [])
                if not top_k:
                    # No top-k info, use top_prediction
                    effective_glosses.append(gloss_pred.get('top_prediction', 'UNKNOWN'))
                    continue

                # Try to find which gloss from top-k appears in the LLM sentence
                matched_gloss = None
                for candidate in top_k:
                    gloss_text = candidate['gloss'].lower()
                    # Check if this gloss (or its stem) appears in LLM sentence
                    # Use bidirectional substring matching and prefix matching
                    for word in llm_words:
                        if (gloss_text in word or word in gloss_text or
                            (len(gloss_text) >= 4 and word.startswith(gloss_text[:4]))):
                            matched_gloss = candidate['gloss']
                            break
                    if matched_gloss:
                        break

                # Use matched gloss if found, otherwise use top-1
                if matched_gloss:
                    effective_glosses.append(matched_gloss)
                else:
                    # No match found in LLM sentence, default to top-1
                    effective_glosses.append(top_k[0]['gloss'])

            return effective_glosses

        except Exception as e:
            print(f"  ERROR: Effective gloss extraction failed: {e}")
            # Fallback to top-1 predictions
            return self._extract_top1_glosses(predicted_glosses)

    # NOTE: Metric calculation methods moved to evaluation_metrics.metrics module
    # Use imported functions: calculate_gloss_accuracy, calculate_quality_score, calculate_coverage, calculate_composite_score

    def _calculate_baseline_metrics(self, predicted_top1_glosses, original_glosses, reference_sentence):
        """
        Calculate baseline metrics (predicted top-1 glosses → simple sentence vs reference)
        Returns: (bleu_score, bertscore_f1, quality_score, coverage_dict)
        This represents the baseline performance without LLM sentence construction.

        Args:
            predicted_top1_glosses: Top-1 predicted glosses from model
            original_glosses: Original input glosses from dataset (for coverage calculation)
            reference_sentence: Reference sentence for BLEU/BERTScore
        """
        # Create simple baseline sentence by joining predicted top-1 glosses
        baseline_sentence = ' '.join(predicted_top1_glosses) + '.'

        if not self.quiet:
            print(f"  BASELINE (predicted top-1): '{baseline_sentence}'")
            print(f"  REFERENCE: '{reference_sentence}'")

        # Calculate BLEU using modular function
        bleu_score = calculate_bleu_score(baseline_sentence, reference_sentence)
        if bleu_score is not None and not self.quiet:
            print(f"  BASELINE BLEU: {bleu_score:.2f}")

        # Calculate BERTScore using modular function
        bertscore_f1 = calculate_bert_score(baseline_sentence, reference_sentence)
        if bertscore_f1 is not None and not self.quiet:
            print(f"  BASELINE BERTScore: {bertscore_f1:.2f}")

        # Calculate Quality Score using modular function
        quality_score = calculate_quality_score(baseline_sentence, scorer=self.quality_scorer)
        if quality_score is not None and not self.quiet:
            print(f"  BASELINE Quality: {quality_score:.2f}")

        # Calculate Coverage using modular function
        coverage = calculate_coverage(reference_sentence, baseline_sentence)
        if coverage['recall'] is not None and not self.quiet:
            print(f"  BASELINE Coverage: Recall={coverage['recall']:.1f}%, Precision={coverage['precision']:.1f}%, F1={coverage['f1']:.1f}%")

        return bleu_score, bertscore_f1, quality_score, coverage

    def _run_prediction(self, pose_file, metadata_file):
        """
        Run predict_sentence.py with metadata-based segmentation
        Returns: predicted_sentence or None on error
        """
        if not self.quiet:
            print(f"\n  Running prediction with pose + metadata...")

        # Build command for predict_sentence.py (go up to asl-v1, then to applications)
        # Path: synthetic_evaluation -> evaluation_metrics -> project-utilities -> asl-v1 -> applications
        predict_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "applications",
            "predict_sentence.py"
        )

        cmd = [
            sys.executable,
            predict_script,
            "--pose-file", pose_file,
            "--segment-metadata", metadata_file,
            "--num-glosses", str(self.num_glosses),
            "--use-top-k", "3"  # Enable top-3 predictions for LLM
        ]

        # Add checkpoint if provided
        if self.checkpoint_path:
            cmd.extend(["--checkpoint", self.checkpoint_path])

        # For two-stage mode, always disable LLM in predict_sentence.py
        # We'll use our own TwoStagePipeline instead
        if self.use_two_stage:
            cmd.append("--no-llm")
            if not self.quiet:
                print(f"  [TWO-STAGE MODE] Getting model predictions only...")
        # Add --no-llm flag if LLM is disabled
        elif not self.use_llm:
            cmd.append("--no-llm")

        # Add --no-confidence-scores flag if enabled
        if self.no_confidence_scores:
            cmd.append("--no-confidence-scores")

        # Add custom prompt file if provided
        if self.prompt_file:
            cmd.extend(["--prompt-file", self.prompt_file])

        # Debug: Show if LLM is enabled
        if not self.quiet:
            print(f"  DEBUG: LLM enabled: {self.use_llm}")
            print(f"  COMMAND: {' '.join(cmd[:8])}... (full command has {len(cmd)} args)")  # Show more args

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=120
            )

            # Print stderr if not empty (for debugging)
            if result.stderr and result.stderr.strip():
                print(f"  DEBUG: STDERR output from predict_sentence.py:")
                print(f"  {result.stderr}")

            # Parse output to extract final sentence (handle multiline JSON responses)
            import json
            import re

            stdout_full = result.stdout
            output_lines = stdout_full.strip().split('\n')

            # Debug: Check what we got back
            if not self.quiet:
                has_glosses_json = any("GLOSSES_JSON:" in line for line in output_lines)
                has_final_sentence = any("FINAL SENTENCE:" in line for line in output_lines)
                print(f"  DEBUG: Subprocess output has FINAL SENTENCE: {has_final_sentence}, GLOSSES_JSON: {has_glosses_json}")

            predicted_sentence = None
            predicted_glosses = []

            # Extract FINAL SENTENCE (may be multiline JSON)
            sentence_match = re.search(r"FINAL SENTENCE:\s*'(.*?)'(?:\n|$)", stdout_full, re.DOTALL)
            if sentence_match:
                raw_output = sentence_match.group(1)

                # Try to parse as JSON first
                json_match = re.search(r'\{[\s\S]*"selections"[\s\S]*"sentence"[\s\S]*\}', raw_output)
                if json_match:
                    try:
                        data = json.loads(json_match.group(0))
                        # Store full JSON for later parsing
                        predicted_sentence = raw_output
                        if not self.quiet:
                            print(f"  [JSON DETECTED] LLM returned JSON format")
                    except json.JSONDecodeError:
                        # Not valid JSON, treat as plain sentence
                        predicted_sentence = raw_output
                else:
                    # Plain sentence format
                    predicted_sentence = raw_output

            # Print DEBUG_INFO lines and extract GLOSSES_JSON
            for line in output_lines:
                # Print DEBUG_INFO lines from subprocess
                if "DEBUG_INFO:" in line and not self.quiet:
                    print(f"  {line}")

                if "GLOSSES_JSON:" in line:
                    # Extract glosses with top-k info from JSON
                    parts = line.split("GLOSSES_JSON:", 1)
                    if len(parts) == 2:
                        try:
                            import json
                            predicted_glosses = json.loads(parts[1].strip())
                            # Debug: Check the structure
                            if predicted_glosses and not self.quiet:
                                first_gloss = predicted_glosses[0]
                                if isinstance(first_gloss, str):
                                    print(f"  DEBUG: Glosses are STRINGS (no top-k info): {first_gloss}")
                                elif isinstance(first_gloss, dict):
                                    print(f"  DEBUG: Glosses are DICTS (has top-k): keys={list(first_gloss.keys())}")
                                else:
                                    print(f"  DEBUG: Glosses are unknown type: {type(first_gloss)}")
                        except json.JSONDecodeError as json_err:
                            if not self.quiet:
                                print(f"  WARNING: Failed to parse GLOSSES_JSON: {json_err}")
                            predicted_glosses = []

            if predicted_sentence or (self.use_two_stage and predicted_glosses):
                # Two-stage mode: use TwoStagePipeline to process predictions
                if self.use_two_stage and predicted_glosses and self.two_stage_pipeline:
                    if not self.quiet:
                        print(f"  [TWO-STAGE] Running Stage 1 (Selection) + Stage 2 (Sentence)...")

                    # Convert predicted_glosses to format expected by TwoStagePipeline
                    # Expected format: [{'gloss': str, 'confidence': float, 'top_k': [...]}]
                    formatted_glosses = []
                    for g in predicted_glosses:
                        if isinstance(g, dict):
                            # Already has top_k info
                            formatted = {
                                'gloss': g.get('top_prediction', g.get('gloss', 'UNKNOWN')),
                                'confidence': g.get('confidence', 0.0),
                                'top_k': g.get('top_k', [])
                            }
                            formatted_glosses.append(formatted)
                        else:
                            # Simple string, no top-k
                            formatted_glosses.append({
                                'gloss': str(g),
                                'confidence': 1.0,
                                'top_k': [{'gloss': str(g), 'confidence': 1.0}]
                            })

                    try:
                        # Run two-stage pipeline
                        result = self.two_stage_pipeline.run_full_pipeline(formatted_glosses)

                        # Extract results
                        stage1 = result.get('stage1', {})
                        stage2 = result.get('stage2', {})
                        final_sentence = result.get('final_sentence', '')
                        selected_glosses = result.get('selected_glosses', [])

                        if not self.quiet:
                            print(f"  [STAGE 1] Selections: {selected_glosses}")
                            print(f"  [STAGE 2] Sentence: '{final_sentence}'")

                        # Build a JSON-like response for compatibility with existing parsing
                        import json
                        two_stage_response = json.dumps({
                            'selections': selected_glosses,
                            'sentence': final_sentence,
                            'stage1_raw': stage1.get('raw_response', ''),
                            'stage2_raw': stage2.get('raw_response', '')
                        })

                        print(f"PREDICTED SENTENCE: '{final_sentence}'")
                        print(f"PREDICTED GLOSSES (selected): {', '.join(selected_glosses)}")
                        return two_stage_response, predicted_glosses

                    except Exception as e:
                        print(f"  [TWO-STAGE ERROR] {e}")
                        traceback.print_exc()
                        # Fall back to no-LLM output
                        pass

                # Regular output (one-phase or fallback)
                print(f"PREDICTED SENTENCE: '{predicted_sentence}'")
                if predicted_glosses:
                    # Extract display names from glosses (handle both dicts and strings)
                    display_glosses = []
                    for g in predicted_glosses:
                        if isinstance(g, dict):
                            display_glosses.append(g.get('gloss', g.get('top_prediction', 'UNKNOWN')))
                        else:
                            display_glosses.append(str(g))
                    print(f"PREDICTED GLOSSES: {', '.join(display_glosses)}")
                return predicted_sentence, predicted_glosses
            else:
                print(f"  ERROR: Could not extract predicted sentence from output")
                print(f"  LAST 10 LINES:")
                for line in output_lines[-10:]:
                    print(f"    {line}")
                return None, []

        except subprocess.TimeoutExpired:
            print(f"  ERROR: predict_sentence.py timed out (120s)")
            return None, []
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: predict_sentence.py failed with exit code {e.returncode}")
            print(f"  STDERR: {e.stderr[-500:]}")  # Last 500 chars
            return None, []
        except Exception as e:
            print(f"  ERROR: Unexpected error: {e}")
            traceback.print_exc()
            return None, []

    def _calculate_model_metrics(self, predicted_glosses, predicted_sentence, reference_sentence):
        """
        Calculate model metrics (LLM sentence with top-3 predictions vs reference)
        Returns: (bleu_score, bertscore_f1, quality_score)
        Directly compares predicted sentence (constructed by LLM from top-3) to reference sentence.
        This represents the performance with LLM sentence construction.
        """
        if not predicted_sentence:
            return None, None, None

        if not self.quiet:
            print(f"  MODEL PREDICTION (LLM with top-3): '{predicted_sentence}'")
            print(f"  REFERENCE: '{reference_sentence}'")

        # Calculate BLEU using modular function
        bleu_score = calculate_bleu_score(predicted_sentence, reference_sentence)
        if bleu_score is not None and not self.quiet:
            print(f"  MODEL BLEU: {bleu_score:.2f}")

        # Calculate BERTScore using modular function
        bertscore_f1 = calculate_bert_score(predicted_sentence, reference_sentence)
        if bertscore_f1 is not None and not self.quiet:
            print(f"  MODEL BERTScore: {bertscore_f1:.2f}")

        # Calculate Quality Score using modular function
        quality_score = calculate_quality_score(predicted_sentence, scorer=self.quality_scorer)
        if quality_score is not None and not self.quiet:
            print(f"  MODEL Quality: {quality_score:.2f}")

        return bleu_score, bertscore_f1, quality_score

    def evaluate_entry(self, entry, entry_id):
        """
        Evaluate a single dataset entry
        Returns: result dict
        """
        glosses = entry['glosses']
        reference_sentence = entry['sentence']

        if not self.quiet:
            print("\n" + "="*80)
            print(f"EVALUATING ENTRY {entry_id + 1}")
            print("="*80)
            print(f"GLOSSES: {', '.join(glosses)}")
            print(f"REFERENCE: '{reference_sentence}'")

        result = {
            'entry_id': entry_id,
            'glosses': glosses,
            'reference_sentence': reference_sentence,
            'baseline_bleu': None,
            'baseline_bertscore': None,
            'baseline_quality': None,
            'baseline_composite': None,
            'model_bleu': None,
            'model_bertscore': None,
            'model_quality': None,
            'model_composite': None,
            'predicted_sentence': None,
            'predicted_glosses': [],
            'improvement_bleu': None,
            'improvement_bertscore': None,
            'improvement_quality': None,
            'improvement_composite': None,
            # Gloss coverage metrics (comparative)
            'baseline_coverage_recall': None,
            'baseline_coverage_precision': None,
            'baseline_coverage_f1': None,
            'model_coverage_recall': None,
            'model_coverage_precision': None,
            'model_coverage_f1': None,
            'improvement_coverage_recall': None,
            'improvement_coverage_precision': None,
            'improvement_coverage_f1': None,
            'missing_words': [],
            'hallucinated_words': [],
            # LLM alternative usage tracking
            'llm_used_alternatives': 0,
            'llm_alternative_details': [],
            # Gloss prediction accuracy (top-1 only)
            'gloss_accuracy': None,
            'gloss_correct': 0,
            'gloss_total': 0,
            'gloss_mismatches': [],
            # Effective gloss accuracy (what LLM chose from top-k)
            'effective_gloss_accuracy': None,
            'effective_gloss_correct': 0,
            'effective_gloss_total': 0,
            'effective_gloss_mismatches': [],
            'effective_glosses': [],
            # Perfect Translation Rate (PTR) - binary metric for CTQI
            'baseline_ptr': None,
            'model_ptr': None,
            'status': 'pending'
        }

        try:
            # Initialize pose_file and metadata_file for cleanup handling
            pose_file = None
            metadata_file = None

            # Branch based on evaluation mode
            if self.use_manifest:
                # MANIFEST MODE: Predict each gloss individually from pickle files
                if not self.quiet:
                    print(f"\n  [MANIFEST MODE] Predicting from pickle files...")

                predicted_glosses = self._predict_glosses_from_manifest(glosses)
                result['predicted_glosses'] = predicted_glosses

                # Use two-stage pipeline or simple joining for sentence
                if self.use_two_stage and self.two_stage_pipeline:
                    if not self.quiet:
                        print(f"  [TWO-STAGE] Running Stage 1 (Selection) + Stage 2 (Sentence)...")
                    two_stage_result = self.two_stage_pipeline.run_full_pipeline(predicted_glosses)
                    predicted_sentence = two_stage_result.get('final_sentence', '')
                    predicted_response = json.dumps(two_stage_result)
                    if not self.quiet:
                        selections = two_stage_result.get('selected_glosses', [])
                        print(f"  [STAGE 1] Selections: {selections}")
                        print(f"  [STAGE 2] Sentence: '{predicted_sentence}'")
                elif self.use_llm and self.llm_provider:
                    # One-stage LLM for manifest mode
                    if not self.quiet:
                        print(f"  [ONE-STAGE LLM] Running sentence generation...")
                    predicted_sentence, predicted_response = self._run_one_stage_llm(predicted_glosses)
                else:
                    # Simple gloss joining as fallback (--no-llm)
                    top1_glosses = [p['top_prediction'] for p in predicted_glosses]
                    predicted_sentence = ' '.join(top1_glosses) + '.'
                    predicted_response = predicted_sentence

                result['predicted_sentence'] = predicted_sentence
                result['predicted_response_raw'] = predicted_response

            else:
                # POSE MODE: Create concatenated pose + metadata (original behavior)
                # Step 1: Create concatenated pose + metadata
                pose_file, metadata_file = self._create_concatenated_pose(glosses, entry_id)

                if not pose_file or not metadata_file:
                    result['status'] = 'failed_pose_creation'
                    return result

                # Step 2: Run prediction (with top-3 for LLM)
                predicted_response, predicted_glosses = self._run_prediction(pose_file, metadata_file)

                # Extract clean sentence from response (handles JSON or plain text)
                predicted_sentence = self._extract_sentence_from_response(predicted_response)

                # Store both raw response and clean sentence
                result['predicted_sentence'] = predicted_sentence
                result['predicted_response_raw'] = predicted_response  # For selection extraction
                result['predicted_glosses'] = predicted_glosses

            if not predicted_sentence:
                result['status'] = 'failed_prediction'
                return result

            # Step 2b: Analyze LLM alternative usage
            alternative_analysis = self._analyze_llm_choices(predicted_glosses, predicted_sentence)
            result['llm_used_alternatives'] = alternative_analysis['used_alternatives']
            result['llm_alternative_details'] = alternative_analysis['alternative_positions']

            # Step 3: Extract top-1 predictions for baseline calculation
            top1_glosses = self._extract_top1_glosses(predicted_glosses)

            # Step 3b: Calculate gloss prediction accuracy (top-1)
            gloss_accuracy_result = calculate_gloss_accuracy(top1_glosses, glosses)
            result['gloss_accuracy'] = gloss_accuracy_result['accuracy']
            result['gloss_correct'] = gloss_accuracy_result['correct']
            result['gloss_total'] = gloss_accuracy_result['total']
            result['gloss_mismatches'] = gloss_accuracy_result['mismatches']

            if not self.quiet:
                print(f"\n  MODEL GLOSS ACCURACY (top-1): {result['gloss_correct']}/{result['gloss_total']} ({result['gloss_accuracy']:.1f}%)")
                print(f"  TOP-1 GLOSSES: {', '.join(top1_glosses)}")
                print(f"  ORIGINAL GLOSSES: {', '.join(glosses)}")
                if result['gloss_mismatches']:
                    print(f"  TOP-1 MISMATCHES:")
                    for mm in result['gloss_mismatches']:
                        print(f"    Position {mm['position']}: '{mm['predicted']}' vs '{mm['original']}'")

            # Step 3c: Extract effective glosses (what LLM chose from top-k) and calculate accuracy
            # Use raw response for JSON parsing, fallback to sentence for reverse engineering
            effective_glosses = self._extract_effective_glosses(predicted_glosses, predicted_response)
            result['effective_glosses'] = effective_glosses

            effective_accuracy_result = calculate_gloss_accuracy(effective_glosses, glosses)
            result['effective_gloss_accuracy'] = effective_accuracy_result['accuracy']
            result['effective_gloss_correct'] = effective_accuracy_result['correct']
            result['effective_gloss_total'] = effective_accuracy_result['total']
            result['effective_gloss_mismatches'] = effective_accuracy_result['mismatches']

            if not self.quiet:
                print(f"\n  EFFECTIVE GLOSS ACCURACY (LLM-selected from top-k): {result['effective_gloss_correct']}/{result['effective_gloss_total']} ({result['effective_gloss_accuracy']:.1f}%)")
                print(f"  EFFECTIVE GLOSSES: {', '.join(effective_glosses)}")
                if result['effective_gloss_mismatches']:
                    print(f"  EFFECTIVE MISMATCHES:")
                    for mm in result['effective_gloss_mismatches']:
                        pos = mm['position']
                        print(f"    Position {pos}: LLM chose '{mm['predicted']}' but should be '{mm['original']}'")

                        # Show top-k predictions at this position
                        if pos < len(predicted_glosses) and isinstance(predicted_glosses[pos], dict):
                            top_k = predicted_glosses[pos].get('top_k', [])
                            if top_k:
                                print(f"      Available top-k predictions:")
                                for i, candidate in enumerate(top_k[:3], 1):  # Show top-3
                                    gloss_name = candidate['gloss']
                                    conf = candidate['confidence']
                                    # Check if this is the correct answer
                                    is_correct = gloss_name.upper() == mm['original'].upper()
                                    marker = " * CORRECT!" if is_correct else ""
                                    print(f"        {i}. {gloss_name} ({conf*100:.1f}%){marker}")

                # Show improvement from top-1 to effective
                if result['gloss_accuracy'] is not None and result['effective_gloss_accuracy'] is not None:
                    improvement = result['effective_gloss_accuracy'] - result['gloss_accuracy']
                    print(f"  ACCURACY IMPROVEMENT (effective - top-1): {improvement:+.1f}%")

            # Step 4: Calculate baseline metrics (from top-1 predicted glosses)
            baseline_bleu, baseline_bertscore, baseline_quality, baseline_coverage = self._calculate_baseline_metrics(
                top1_glosses, glosses, reference_sentence
            )
            result['baseline_bleu'] = baseline_bleu
            result['baseline_bertscore'] = baseline_bertscore
            result['baseline_quality'] = baseline_quality
            result['baseline_coverage_recall'] = baseline_coverage['recall']
            result['baseline_coverage_precision'] = baseline_coverage['precision']
            result['baseline_coverage_f1'] = baseline_coverage['f1']

            # Calculate baseline PTR (Perfect Translation Rate)
            result['baseline_ptr'] = calculate_perfect_translation_rate(top1_glosses, glosses)

            # Calculate baseline CTQI composite score (gloss_accuracy + quality + PTR)
            result['baseline_composite'] = calculate_composite_score(
                gloss_accuracy=result['gloss_accuracy'],
                quality=baseline_quality,
                perfect_translation_rate=result['baseline_ptr']
            )

            # Step 5: Calculate model metrics (from LLM sentence with top-3)
            model_bleu, model_bertscore, model_quality = self._calculate_model_metrics(predicted_glosses, predicted_sentence, reference_sentence)
            result['model_bleu'] = model_bleu
            result['model_bertscore'] = model_bertscore
            result['model_quality'] = model_quality

            # Step 5b: Calculate model coverage (vs reference sentence)
            model_coverage = calculate_coverage(reference_sentence, predicted_sentence)
            result['model_coverage_recall'] = model_coverage['recall']
            result['model_coverage_precision'] = model_coverage['precision']
            result['model_coverage_f1'] = model_coverage['f1']
            result['missing_words'] = model_coverage['missing_words']
            result['hallucinated_words'] = model_coverage['hallucinated_words']

            # Calculate model PTR (Perfect Translation Rate) using effective glosses
            result['model_ptr'] = calculate_perfect_translation_rate(effective_glosses, glosses)

            # Calculate model CTQI composite score (effective_gloss_accuracy + quality + PTR)
            result['model_composite'] = calculate_composite_score(
                gloss_accuracy=result['effective_gloss_accuracy'],
                quality=model_quality,
                perfect_translation_rate=result['model_ptr']
            )

            # Print model coverage metrics
            if not self.quiet and model_coverage['recall'] is not None:
                print(f"\n  MODEL Coverage: Recall={model_coverage['recall']:.1f}%, Precision={model_coverage['precision']:.1f}%, F1={model_coverage['f1']:.1f}%")
                if model_coverage['missing_words']:
                    print(f"    Missing: {', '.join(model_coverage['missing_words'])}")
                if model_coverage['hallucinated_words']:
                    print(f"    Hallucinated: {', '.join(model_coverage['hallucinated_words'])}")

            # Print LLM alternative usage
            if not self.quiet:
                if result['llm_used_alternatives'] > 0:
                    print(f"\n  GEMINI ALTERNATIVE USAGE: {result['llm_used_alternatives']}/{alternative_analysis['total_glosses']} positions used non-top-1 predictions")
                    for detail in result['llm_alternative_details']:
                        rank_str = f"top-{detail.get('rank', '?')}"
                        top1_conf = detail.get('top1_confidence')
                        used_conf = detail.get('used_confidence')
                        if top1_conf is not None and used_conf is not None:
                            print(f"    Position {detail['position']}: Used '{detail['used']}' ({rank_str}, {used_conf*100:.1f}%) instead of '{detail['top1']}' (top-1, {top1_conf*100:.1f}%)")
                        else:
                            print(f"    Position {detail['position']}: Used '{detail['used']}' ({rank_str}) instead of '{detail['top1']}' (top-1)")
                else:
                    print(f"\n  GEMINI ALTERNATIVE USAGE: 0/{alternative_analysis['total_glosses']} positions (all top-1)")

            # Step 6: Calculate improvements
            if baseline_bleu is not None and model_bleu is not None:
                result['improvement_bleu'] = model_bleu - baseline_bleu
                if not self.quiet:
                    print(f"\n  BLEU IMPROVEMENT: {result['improvement_bleu']:+.2f} ({baseline_bleu:.2f} -> {model_bleu:.2f})")

            if baseline_bertscore is not None and model_bertscore is not None:
                result['improvement_bertscore'] = model_bertscore - baseline_bertscore
                if not self.quiet:
                    print(f"  BERTScore IMPROVEMENT: {result['improvement_bertscore']:+.2f} ({baseline_bertscore:.2f} -> {model_bertscore:.2f})")

            if baseline_quality is not None and model_quality is not None:
                result['improvement_quality'] = model_quality - baseline_quality
                if not self.quiet:
                    print(f"  QUALITY IMPROVEMENT: {result['improvement_quality']:+.2f} ({baseline_quality:.2f} -> {model_quality:.2f})")

            if result['baseline_composite'] is not None and result['model_composite'] is not None:
                result['improvement_composite'] = result['model_composite'] - result['baseline_composite']
                if not self.quiet:
                    print(f"  COMPOSITE IMPROVEMENT: {result['improvement_composite']:+.2f} ({result['baseline_composite']:.2f} -> {result['model_composite']:.2f})")

            # Calculate coverage improvements
            if result['baseline_coverage_recall'] is not None and result['model_coverage_recall'] is not None:
                result['improvement_coverage_recall'] = result['model_coverage_recall'] - result['baseline_coverage_recall']
                result['improvement_coverage_precision'] = result['model_coverage_precision'] - result['baseline_coverage_precision']
                result['improvement_coverage_f1'] = result['model_coverage_f1'] - result['baseline_coverage_f1']
                if not self.quiet:
                    print(f"  COVERAGE IMPROVEMENT:")
                    print(f"    Recall: {result['improvement_coverage_recall']:+.1f}% ({result['baseline_coverage_recall']:.1f}% -> {result['model_coverage_recall']:.1f}%)")
                    print(f"    Precision: {result['improvement_coverage_precision']:+.1f}% ({result['baseline_coverage_precision']:.1f}% -> {result['model_coverage_precision']:.1f}%)")
                    print(f"    F1: {result['improvement_coverage_f1']:+.1f}% ({result['baseline_coverage_f1']:.1f}% -> {result['model_coverage_f1']:.1f}%)")

            # In quiet mode, print reference and predicted sentence
            if self.quiet:
                print(f"[{entry_id + 1}] Reference: {reference_sentence}")
                print(f"    Predicted: {result.get('predicted_sentence', 'N/A')}")

            result['status'] = 'success'

        except Exception as e:
            if not self.quiet:
                print(f"\nERROR: Entry evaluation failed: {e}")
                traceback.print_exc()
            result['status'] = 'failed_exception'
            result['error'] = str(e)

        finally:
            # Cleanup pose files unless --keep-poses is set
            if not self.keep_poses and pose_file and metadata_file:
                try:
                    if os.path.exists(pose_file):
                        os.remove(pose_file)
                        if not self.quiet:
                            print(f"\n  CLEANUP: Deleted {os.path.basename(pose_file)}")
                    if os.path.exists(metadata_file):
                        os.remove(metadata_file)
                        if not self.quiet:
                            print(f"  CLEANUP: Deleted {os.path.basename(metadata_file)}")
                except Exception as cleanup_error:
                    if not self.quiet:
                        print(f"  WARNING: Failed to cleanup files: {cleanup_error}")

        return result

    def evaluate_dataset(self, limit=None):
        """
        Evaluate entire dataset (or limited subset)
        """
        if not self.quiet:
            print("\n" + "="*80)
            print("SYNTHETIC DATASET EVALUATION")
            print("="*80)
            print(f"Dataset: {self.dataset_path}")
            print(f"Total entries: {len(self.dataset)}")
            if limit:
                print(f"Limit: First {limit} entries")
            print(f"Output directory: {self.output_dir}")
            print(f"Checkpoint: {self.checkpoint_path or 'None (using default model)'}")
            print(f"LLM: {'Enabled' if self.use_llm else 'Disabled (simple gloss joining)'}")
            print(f"Keep poses: {'Yes (poses will be saved)' if self.keep_poses else 'No (poses will be deleted after evaluation)'}")
            print("\nEvaluation Metrics:")
            quality_available = QUALITY_SCORING_AVAILABLE and self.quality_scorer is not None and self.quality_scorer.is_available
            print(f"  BLEU Score: {'Enabled' if BLEU_AVAILABLE else 'DISABLED'}")
            print(f"  BERTScore: {'Enabled' if BERTSCORE_AVAILABLE else 'DISABLED'}")
            print(f"  Quality Score (GPT-2): {'Enabled' if quality_available else 'DISABLED'}")
            print(f"  Composite Score: {'Enabled' if (BLEU_AVAILABLE or BERTSCORE_AVAILABLE or quality_available) else 'DISABLED'}")
            print("="*80)
        else:
            quality_available = QUALITY_SCORING_AVAILABLE and self.quality_scorer is not None and self.quality_scorer.is_available

        # Process entries
        entries_to_process = self.dataset[:limit] if limit else self.dataset

        for i, entry in enumerate(entries_to_process):
            result = self.evaluate_entry(entry, i)
            self.results.append(result)

            # Save intermediate results after each entry
            self._save_results()

        # Generate final report
        self._generate_report()

    def _save_results(self):
        """Save results to JSON file"""
        results_file = os.path.join(self.output_dir, "evaluation_results.json")

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        if not self.quiet:
            print(f"\nSaved intermediate results: {results_file}")

    def _generate_report(self):
        """Generate evaluation report"""
        if not self.quiet:
            print("\n" + "="*80)
            print("EVALUATION REPORT")
            print("="*80)

        # Calculate statistics
        total = len(self.results)
        successful = sum(1 for r in self.results if r['status'] == 'success')
        failed = total - successful

        # Calculate BLEU statistics
        baseline_bleu_scores = [r['baseline_bleu'] for r in self.results if r['baseline_bleu'] is not None]
        model_bleu_scores = [r['model_bleu'] for r in self.results if r['model_bleu'] is not None]
        bleu_improvements = [r['improvement_bleu'] for r in self.results if r['improvement_bleu'] is not None]

        # Calculate BERTScore statistics
        baseline_bertscore_scores = [r['baseline_bertscore'] for r in self.results if r['baseline_bertscore'] is not None]
        model_bertscore_scores = [r['model_bertscore'] for r in self.results if r['model_bertscore'] is not None]
        bertscore_improvements = [r['improvement_bertscore'] for r in self.results if r['improvement_bertscore'] is not None]

        # Calculate Quality Score statistics
        baseline_quality_scores = [r['baseline_quality'] for r in self.results if r['baseline_quality'] is not None]
        model_quality_scores = [r['model_quality'] for r in self.results if r['model_quality'] is not None]
        quality_improvements = [r['improvement_quality'] for r in self.results if r['improvement_quality'] is not None]

        # Calculate Composite Score (CTQI) statistics
        baseline_composite_scores = [r['baseline_composite'] for r in self.results if r['baseline_composite'] is not None]
        model_composite_scores = [r['model_composite'] for r in self.results if r['model_composite'] is not None]
        composite_improvements = [r['improvement_composite'] for r in self.results if r['improvement_composite'] is not None]

        # Calculate Perfect Translation Rate (PTR) statistics
        baseline_ptr_scores = [r['baseline_ptr'] for r in self.results if r['baseline_ptr'] is not None]
        model_ptr_scores = [r['model_ptr'] for r in self.results if r['model_ptr'] is not None]
        # Count perfect translations (PTR = 100)
        baseline_perfect_count = sum(1 for p in baseline_ptr_scores if p == 100)
        model_perfect_count = sum(1 for p in model_ptr_scores if p == 100)

        # Calculate Coverage statistics (comparative metric)
        baseline_coverage_recalls = [r['baseline_coverage_recall'] for r in self.results if r['baseline_coverage_recall'] is not None]
        baseline_coverage_precisions = [r['baseline_coverage_precision'] for r in self.results if r['baseline_coverage_precision'] is not None]
        baseline_coverage_f1s = [r['baseline_coverage_f1'] for r in self.results if r['baseline_coverage_f1'] is not None]

        model_coverage_recalls = [r['model_coverage_recall'] for r in self.results if r['model_coverage_recall'] is not None]
        model_coverage_precisions = [r['model_coverage_precision'] for r in self.results if r['model_coverage_precision'] is not None]
        model_coverage_f1s = [r['model_coverage_f1'] for r in self.results if r['model_coverage_f1'] is not None]

        coverage_recall_improvements = [r['improvement_coverage_recall'] for r in self.results if r['improvement_coverage_recall'] is not None]
        coverage_precision_improvements = [r['improvement_coverage_precision'] for r in self.results if r['improvement_coverage_precision'] is not None]
        coverage_f1_improvements = [r['improvement_coverage_f1'] for r in self.results if r['improvement_coverage_f1'] is not None]

        model_perfect_recall_count = sum(1 for r in self.results if r['model_coverage_recall'] == 100.0)
        entries_with_missing = sum(1 for r in self.results if r['missing_words'])
        entries_with_hallucinations = sum(1 for r in self.results if r['hallucinated_words'])
        total_missing = sum(len(r['missing_words']) for r in self.results)
        total_hallucinated = sum(len(r['hallucinated_words']) for r in self.results)

        # Gloss Prediction Accuracy Statistics (Top-1)
        gloss_accuracies = [r['gloss_accuracy'] for r in self.results if r['gloss_accuracy'] is not None]
        total_correct = sum(r['gloss_correct'] for r in self.results if r['gloss_correct'] is not None)
        total_glosses = sum(r['gloss_total'] for r in self.results if r['gloss_total'] is not None)
        perfect_predictions = sum(1 for acc in gloss_accuracies if acc == 100.0) if gloss_accuracies else 0

        # Effective Gloss Accuracy Statistics (LLM-selected from top-k)
        effective_gloss_accuracies = [r['effective_gloss_accuracy'] for r in self.results if r['effective_gloss_accuracy'] is not None]
        effective_total_correct = sum(r['effective_gloss_correct'] for r in self.results if r['effective_gloss_correct'] is not None)
        effective_total_glosses = sum(r['effective_gloss_total'] for r in self.results if r['effective_gloss_total'] is not None)
        effective_perfect_predictions = sum(1 for acc in effective_gloss_accuracies if acc == 100.0) if effective_gloss_accuracies else 0

        # In quiet mode, skip all verbose console output
        if not self.quiet:
            # Verbose console output
            print(f"\nOVERALL STATISTICS:")
            print(f"  Total entries: {total}")
            print(f"  Successful: {successful} ({successful/total*100:.1f}%)")
            print(f"  Failed: {failed} ({failed/total*100:.1f}%)")

            if gloss_accuracies:
                print(f"\nMODEL GLOSS ACCURACY - TOP-1 (Predicted vs Original Input Glosses):")
                print(f"  Overall: {total_correct}/{total_glosses} ({total_correct/total_glosses*100:.1f}%)")
                print(f"  Average per Entry: {sum(gloss_accuracies)/len(gloss_accuracies):.1f}%")
                print(f"  Min: {min(gloss_accuracies):.1f}%")
                print(f"  Max: {max(gloss_accuracies):.1f}%")
                print(f"  Perfect Predictions: {perfect_predictions}/{len(gloss_accuracies)} ({perfect_predictions/len(gloss_accuracies)*100:.1f}%)")

            if effective_gloss_accuracies:
                print(f"\nEFFECTIVE GLOSS ACCURACY - LLM-SELECTED FROM TOP-K:")
                print(f"  Overall: {effective_total_correct}/{effective_total_glosses} ({effective_total_correct/effective_total_glosses*100:.1f}%)")
                print(f"  Average per Entry: {sum(effective_gloss_accuracies)/len(effective_gloss_accuracies):.1f}%")
                print(f"  Min: {min(effective_gloss_accuracies):.1f}%")
                print(f"  Max: {max(effective_gloss_accuracies):.1f}%")
                print(f"  Perfect Predictions: {effective_perfect_predictions}/{len(effective_gloss_accuracies)} ({effective_perfect_predictions/len(effective_gloss_accuracies)*100:.1f}%)")

                # Show improvement from top-1 to effective
                if gloss_accuracies:
                    avg_improvement = sum(effective_gloss_accuracies)/len(effective_gloss_accuracies) - sum(gloss_accuracies)/len(gloss_accuracies)
                    overall_improvement = (effective_total_correct/effective_total_glosses*100) - (total_correct/total_glosses*100)
                    print(f"\nGLOSS ACCURACY IMPROVEMENT (Effective - Top-1):")
                    print(f"  Overall Improvement: {overall_improvement:+.1f}% ({total_correct}/{total_glosses} -> {effective_total_correct}/{effective_total_glosses})")
                    print(f"  Average Improvement: {avg_improvement:+.1f}%")
                    print(f"  Perfect Predictions Gained: {effective_perfect_predictions - perfect_predictions}")

                    # Analyze effective mismatches: was correct answer in top-k?
                    print(f"\nEFFECTIVE MISMATCH ANALYSIS:")
                correct_in_topk = 0
                correct_not_in_topk = 0
                for r in self.results:
                    if r['effective_gloss_mismatches']:
                        for mm in r['effective_gloss_mismatches']:
                            pos = mm['position']
                            if pos < len(r['predicted_glosses']) and isinstance(r['predicted_glosses'][pos], dict):
                                top_k = r['predicted_glosses'][pos].get('top_k', [])
                                # Check if correct answer is in top-k
                                is_in_topk = any(
                                    candidate['gloss'].upper() == mm['original'].upper()
                                    for candidate in top_k[:3]
                                )
                                if is_in_topk:
                                    correct_in_topk += 1
                                else:
                                    correct_not_in_topk += 1

                total_effective_mismatches = correct_in_topk + correct_not_in_topk
                if total_effective_mismatches > 0:
                    print(f"  Total Effective Mismatches: {total_effective_mismatches}")
                    print(f"  Correct answer IN top-3 (LLM selection error): {correct_in_topk} ({correct_in_topk/total_effective_mismatches*100:.1f}%)")
                    print(f"  Correct answer NOT in top-3 (model error): {correct_not_in_topk} ({correct_not_in_topk/total_effective_mismatches*100:.1f}%)")
                    if correct_in_topk > 0:
                        print(f"  -> Potential improvement with better LLM prompting: {correct_in_topk} glosses")

        # LLM Alternative Usage Statistics (needed for file writing)
        total_alternatives_used = sum(r['llm_used_alternatives'] for r in self.results if r['llm_used_alternatives'] is not None)
        entries_with_alternatives = sum(1 for r in self.results if r['llm_used_alternatives'] and r['llm_used_alternatives'] > 0)
        total_gloss_positions = sum(len(r['predicted_glosses']) for r in self.results if r['predicted_glosses'])

        # Generate detailed table (skip in quiet mode)
        if not self.quiet:
            self._generate_table()

        # Save report
        report_file = os.path.join(self.output_dir, "evaluation_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("GLOSS SELECTION DEBUG REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Dataset: {self.dataset_path}\n")
            f.write(f"Total entries: {total}\n")
            f.write(f"Successful: {successful} ({successful/total*100:.1f}%)\n")
            f.write(f"Failed: {failed} ({failed/total*100:.1f}%)\n\n")

            # 1. MODEL GLOSS ACCURACY (Top-1 predictions)
            if gloss_accuracies and total_glosses > 0:
                f.write("="*80 + "\n")
                f.write("1. MODEL GLOSS ACCURACY - TOP-1 (Before LLM Selection)\n")
                f.write("="*80 + "\n")
                f.write(f"Overall Accuracy: {total_correct}/{total_glosses} ({total_correct/total_glosses*100:.1f}%)\n")
                f.write(f"Perfect Predictions (all glosses correct): {perfect_predictions}/{len(gloss_accuracies)} entries ({perfect_predictions/len(gloss_accuracies)*100:.1f}%)\n")
                f.write(f"Average Per Entry: {sum(gloss_accuracies)/len(gloss_accuracies):.1f}%\n\n")

            # 2. EFFECTIVE GLOSS ACCURACY (LLM-selected from top-k)
            if effective_gloss_accuracies and effective_total_glosses > 0:
                f.write("="*80 + "\n")
                f.write("2. EFFECTIVE GLOSS ACCURACY (After LLM Selection from Top-K)\n")
                f.write("="*80 + "\n")
                f.write(f"Overall Accuracy: {effective_total_correct}/{effective_total_glosses} ({effective_total_correct/effective_total_glosses*100:.1f}%)\n")
                f.write(f"Perfect Predictions (all glosses correct): {effective_perfect_predictions}/{len(effective_gloss_accuracies)} entries ({effective_perfect_predictions/len(effective_gloss_accuracies)*100:.1f}%)\n")
                f.write(f"Average Per Entry: {sum(effective_gloss_accuracies)/len(effective_gloss_accuracies):.1f}%\n")
                if gloss_accuracies and total_glosses > 0:
                    overall_improvement = (effective_total_correct/effective_total_glosses*100) - (total_correct/total_glosses*100)
                    f.write(f"\nImprovement from LLM Selection: {overall_improvement:+.1f}% ({total_correct}/{total_glosses} -> {effective_total_correct}/{effective_total_glosses})\n")
                    f.write(f"Perfect Predictions Gained: {effective_perfect_predictions - perfect_predictions} entries\n\n")

            # 3. SELECTION SUCCESS/FAILURE BREAKDOWN
            f.write("="*80 + "\n")
            f.write("3. GLOSS SELECTION BREAKDOWN\n")
            f.write("="*80 + "\n")
            f.write(f"Total Gloss Positions: {effective_total_glosses}\n")
            if effective_total_glosses > 0:
                f.write(f"Correctly Selected from Top-K: {effective_total_correct} ({effective_total_correct/effective_total_glosses*100:.1f}%)\n")
                f.write(f"Incorrectly Selected from Top-K: {effective_total_glosses - effective_total_correct} ({(effective_total_glosses - effective_total_correct)/effective_total_glosses*100:.1f}%)\n\n")
            else:
                f.write(f"Correctly Selected from Top-K: 0 (N/A - no evaluations succeeded)\n")
                f.write(f"Incorrectly Selected from Top-K: 0 (N/A - no evaluations succeeded)\n\n")

            # 4. DETAILED MISMATCH ANALYSIS
            if effective_gloss_accuracies:
                # Analyze effective mismatches
                correct_in_topk = 0
                correct_not_in_topk = 0
                mismatch_entries = []

                for r in self.results:
                    if r['effective_gloss_mismatches']:
                        entry_mismatches = []
                        for mm in r['effective_gloss_mismatches']:
                            pos = mm['position']
                            is_in_topk = False
                            top_k_info = []

                            if pos < len(r['predicted_glosses']) and isinstance(r['predicted_glosses'][pos], dict):
                                top_k = r['predicted_glosses'][pos].get('top_k', [])
                                for i, candidate in enumerate(top_k[:3], 1):
                                    top_k_info.append({
                                        'rank': i,
                                        'gloss': candidate['gloss'],
                                        'confidence': candidate['confidence'],
                                        'is_correct': candidate['gloss'].upper() == mm['original'].upper()
                                    })
                                    if candidate['gloss'].upper() == mm['original'].upper():
                                        is_in_topk = True

                                if is_in_topk:
                                    correct_in_topk += 1
                                else:
                                    correct_not_in_topk += 1

                            entry_mismatches.append({
                                'position': pos,
                                'selected': mm['predicted'],
                                'correct': mm['original'],
                                'in_topk': is_in_topk,
                                'top_k_info': top_k_info
                            })

                        if entry_mismatches:
                            mismatch_entries.append({
                                'entry_id': r['entry_id'] + 1,
                                'glosses': r['glosses'],
                                'reference': r['reference_sentence'],
                                'predicted': r['predicted_sentence'],
                                'mismatches': entry_mismatches
                            })

                total_effective_mismatches = correct_in_topk + correct_not_in_topk
                if total_effective_mismatches > 0:
                    f.write("="*80 + "\n")
                    f.write("4. MISMATCH ANALYSIS\n")
                    f.write("="*80 + "\n")
                    f.write(f"Total Mismatches: {total_effective_mismatches}\n")
                    f.write(f"  - LLM Selection Errors (correct answer WAS in top-3): {correct_in_topk} ({correct_in_topk/total_effective_mismatches*100:.1f}%)\n")
                    f.write(f"  - Model Prediction Errors (correct answer NOT in top-3): {correct_not_in_topk} ({correct_not_in_topk/total_effective_mismatches*100:.1f}%)\n")
                    if correct_in_topk > 0:
                        f.write(f"\n** POTENTIAL IMPROVEMENT: {correct_in_topk} glosses could be fixed with better LLM prompting **\n")
                    f.write("\n")

                    # 5. DETAILED BREAKDOWN BY ENTRY
                    if mismatch_entries:
                        f.write("="*80 + "\n")
                        f.write("5. ENTRIES WITH SELECTION ERRORS (Detailed Breakdown)\n")
                        f.write("="*80 + "\n\n")

                        for entry_info in mismatch_entries:
                            f.write(f"Entry {entry_info['entry_id']}: {', '.join(entry_info['glosses'])}\n")
                            f.write(f"  Reference: {entry_info['reference']}\n")
                            f.write(f"  Predicted: {entry_info['predicted']}\n")

                            # Show selected glosses
                            entry_id = entry_info['entry_id'] - 1
                            if entry_id < len(self.results):
                                selected_glosses = self.results[entry_id].get('effective_glosses', [])
                                if selected_glosses:
                                    f.write(f"  Selected Glosses: {', '.join(selected_glosses)}\n")

                            f.write(f"  Mismatches ({len(entry_info['mismatches'])}):\n")

                            for mm in entry_info['mismatches']:
                                error_type = "LLM SELECTION ERROR" if mm['in_topk'] else "MODEL PREDICTION ERROR"
                                f.write(f"\n    Position {mm['position']}: [{error_type}]\n")
                                f.write(f"      LLM Selected: '{mm['selected']}' (WRONG)\n")
                                f.write(f"      Should Be: '{mm['correct']}' (CORRECT)\n")

                                if mm['top_k_info']:
                                    f.write(f"      Available in Top-3:\n")
                                    for tk in mm['top_k_info']:
                                        marker = " ← CORRECT!" if tk['is_correct'] else ""
                                        f.write(f"        {tk['rank']}. {tk['gloss']} ({tk['confidence']*100:.1f}%){marker}\n")
                                else:
                                    f.write(f"      (Top-k info not available)\n")

                            f.write("\n")
                f.write("\n")

            f.write("\n" + "="*80 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("="*80 + "\n\n")

            # BLEU Summary
            if baseline_bleu_scores:
                f.write(f"BASELINE BLEU: {sum(baseline_bleu_scores)/len(baseline_bleu_scores):.2f}\n")
            if model_bleu_scores:
                f.write(f"MODEL BLEU: {sum(model_bleu_scores)/len(model_bleu_scores):.2f}\n")
            if bleu_improvements:
                f.write(f"AVERAGE BLEU IMPROVEMENT: {sum(bleu_improvements)/len(bleu_improvements):+.2f}\n")
                positive_bleu = [i for i in bleu_improvements if i > 0]
                if positive_bleu:
                    f.write(f"ENTRIES WITH BLEU IMPROVEMENT: {len(positive_bleu)}/{len(bleu_improvements)} ({len(positive_bleu)/len(bleu_improvements)*100:.1f}%)\n")
                    f.write(f"AVERAGE BLEU IMPROVEMENT (of improved entries): {sum(positive_bleu)/len(positive_bleu):+.2f}\n")
            f.write("\n")

            # BERTScore Summary
            if baseline_bertscore_scores:
                f.write(f"BASELINE BERTScore: {sum(baseline_bertscore_scores)/len(baseline_bertscore_scores):.2f}\n")
            if model_bertscore_scores:
                f.write(f"MODEL BERTScore: {sum(model_bertscore_scores)/len(model_bertscore_scores):.2f}\n")
            if bertscore_improvements:
                f.write(f"AVERAGE BERTScore IMPROVEMENT: {sum(bertscore_improvements)/len(bertscore_improvements):+.2f}\n")
                positive_bertscore = [i for i in bertscore_improvements if i > 0]
                if positive_bertscore:
                    f.write(f"ENTRIES WITH BERTScore IMPROVEMENT: {len(positive_bertscore)}/{len(bertscore_improvements)} ({len(positive_bertscore)/len(bertscore_improvements)*100:.1f}%)\n")
                    f.write(f"AVERAGE BERTScore IMPROVEMENT (of improved entries): {sum(positive_bertscore)/len(positive_bertscore):+.2f}\n")
            f.write("\n")

            # Quality Score Summary
            if baseline_quality_scores:
                f.write(f"BASELINE QUALITY: {sum(baseline_quality_scores)/len(baseline_quality_scores):.2f}\n")
            if model_quality_scores:
                f.write(f"MODEL QUALITY: {sum(model_quality_scores)/len(model_quality_scores):.2f}\n")
            if quality_improvements:
                f.write(f"AVERAGE QUALITY IMPROVEMENT: {sum(quality_improvements)/len(quality_improvements):+.2f}\n")
                positive_quality = [i for i in quality_improvements if i > 0]
                if positive_quality:
                    f.write(f"ENTRIES WITH QUALITY IMPROVEMENT: {len(positive_quality)}/{len(quality_improvements)} ({len(positive_quality)/len(quality_improvements)*100:.1f}%)\n")
                    f.write(f"AVERAGE QUALITY IMPROVEMENT (of improved entries): {sum(positive_quality)/len(positive_quality):+.2f}\n")
            f.write("\n")

            # Coverage Summary (Comparative Metric - vs Reference Sentence)
            if baseline_coverage_recalls:
                f.write("BASELINE COVERAGE (vs Reference Sentence):\n")
                f.write(f"AVERAGE RECALL: {sum(baseline_coverage_recalls)/len(baseline_coverage_recalls):.1f}%\n")
                f.write(f"AVERAGE PRECISION: {sum(baseline_coverage_precisions)/len(baseline_coverage_precisions):.1f}%\n")
                f.write(f"AVERAGE F1: {sum(baseline_coverage_f1s)/len(baseline_coverage_f1s):.1f}%\n")
            if model_coverage_recalls:
                f.write("MODEL COVERAGE (vs Reference Sentence):\n")
                f.write(f"AVERAGE RECALL: {sum(model_coverage_recalls)/len(model_coverage_recalls):.1f}%\n")
                f.write(f"AVERAGE PRECISION: {sum(model_coverage_precisions)/len(model_coverage_precisions):.1f}%\n")
                f.write(f"AVERAGE F1: {sum(model_coverage_f1s)/len(model_coverage_f1s):.1f}%\n")
                f.write(f"PERFECT RECALL: {model_perfect_recall_count}/{len(model_coverage_recalls)} ({model_perfect_recall_count/len(model_coverage_recalls)*100:.1f}%)\n")
                f.write(f"ENTRIES WITH MISSING WORDS: {entries_with_missing}/{total} ({total_missing} total)\n")
                f.write(f"ENTRIES WITH HALLUCINATIONS: {entries_with_hallucinations}/{total} ({total_hallucinated} total)\n")
            if coverage_recall_improvements:
                f.write(f"AVERAGE RECALL IMPROVEMENT: {sum(coverage_recall_improvements)/len(coverage_recall_improvements):+.1f}%\n")
                f.write(f"AVERAGE PRECISION IMPROVEMENT: {sum(coverage_precision_improvements)/len(coverage_precision_improvements):+.1f}%\n")
                f.write(f"AVERAGE F1 IMPROVEMENT: {sum(coverage_f1_improvements)/len(coverage_f1_improvements):+.1f}%\n")
                positive_recall = [i for i in coverage_recall_improvements if i > 0]
                if positive_recall:
                    f.write(f"ENTRIES WITH RECALL IMPROVEMENT: {len(positive_recall)}/{len(coverage_recall_improvements)} ({len(positive_recall)/len(coverage_recall_improvements)*100:.1f}%)\n")
            f.write("\n")

            # Perfect Translation Rate (PTR) Summary
            if baseline_ptr_scores:
                f.write(f"BASELINE PERFECT TRANSLATIONS: {baseline_perfect_count}/{len(baseline_ptr_scores)} ({baseline_perfect_count/len(baseline_ptr_scores)*100:.1f}%)\n")
            if model_ptr_scores:
                f.write(f"MODEL PERFECT TRANSLATIONS: {model_perfect_count}/{len(model_ptr_scores)} ({model_perfect_count/len(model_ptr_scores)*100:.1f}%)\n")
            if baseline_ptr_scores and model_ptr_scores:
                ptr_improvement = model_perfect_count - baseline_perfect_count
                f.write(f"PERFECT TRANSLATION IMPROVEMENT: {ptr_improvement:+d} entries\n")
            f.write("\n")

            # Composite Score (CTQI) Summary
            f.write("CTQI WEIGHTS: Gloss Accuracy=40%, Quality=40%, PTR=20%\n")
            if baseline_composite_scores:
                f.write(f"BASELINE CTQI: {sum(baseline_composite_scores)/len(baseline_composite_scores):.2f}\n")
            if model_composite_scores:
                f.write(f"MODEL CTQI: {sum(model_composite_scores)/len(model_composite_scores):.2f}\n")
            if composite_improvements:
                f.write(f"AVERAGE CTQI IMPROVEMENT: {sum(composite_improvements)/len(composite_improvements):+.2f}\n")
                positive_composite = [i for i in composite_improvements if i > 0]
                if positive_composite:
                    f.write(f"ENTRIES WITH CTQI IMPROVEMENT: {len(positive_composite)}/{len(composite_improvements)} ({len(positive_composite)/len(composite_improvements)*100:.1f}%)\n")
                    f.write(f"AVERAGE CTQI IMPROVEMENT (of improved entries): {sum(positive_composite)/len(positive_composite):+.2f}\n")
            f.write("\n")

            # LLM Alternative Usage Statistics
            if total_gloss_positions > 0:
                f.write("GEMINI ALTERNATIVE USAGE (Top-2 or Top-3 instead of Top-1):\n")
                f.write(f"TOTAL GLOSS POSITIONS: {total_gloss_positions}\n")
                f.write(f"ALTERNATIVES USED: {total_alternatives_used} ({total_alternatives_used/total_gloss_positions*100:.1f}%)\n")
                f.write(f"ENTRIES WITH ALTERNATIVES: {entries_with_alternatives}/{len(self.results)} ({entries_with_alternatives/len(self.results)*100:.1f}%)\n")
                f.write("\n")

            # Add detailed results
            f.write("\nDETAILED RESULTS:\n")
            f.write("-"*80 + "\n")
            for r in self.results:
                f.write(f"\nEntry {r['entry_id'] + 1}:\n")
                f.write(f"  Glosses: {', '.join(r['glosses'])}\n")
                f.write(f"  Reference: {r['reference_sentence']}\n")
                if r['predicted_sentence']:
                    f.write(f"  Predicted: {r['predicted_sentence']}\n")

                # Gloss prediction accuracy (top-1)
                if r['gloss_accuracy'] is not None:
                    f.write(f"  Model Gloss Accuracy (top-1): {r['gloss_correct']}/{r['gloss_total']} ({r['gloss_accuracy']:.1f}%)\n")
                    if r['gloss_mismatches']:
                        f.write(f"  Top-1 Mismatches:\n")
                        for mm in r['gloss_mismatches']:
                            f.write(f"    Position {mm['position']}: '{mm['predicted']}' vs '{mm['original']}'\n")

                # Effective gloss accuracy (LLM-selected from top-k)
                if r['effective_gloss_accuracy'] is not None:
                    f.write(f"  Effective Gloss Accuracy (LLM-selected): {r['effective_gloss_correct']}/{r['effective_gloss_total']} ({r['effective_gloss_accuracy']:.1f}%)\n")
                    if r['effective_glosses']:
                        f.write(f"  Effective Glosses: {', '.join(r['effective_glosses'])}\n")
                    if r['effective_gloss_mismatches']:
                        f.write(f"  Effective Mismatches:\n")
                        for mm in r['effective_gloss_mismatches']:
                            pos = mm['position']
                            f.write(f"    Position {pos}: LLM chose '{mm['predicted']}' but should be '{mm['original']}'\n")

                            # Show top-k predictions at this position
                            if pos < len(r['predicted_glosses']) and isinstance(r['predicted_glosses'][pos], dict):
                                top_k = r['predicted_glosses'][pos].get('top_k', [])
                                if top_k:
                                    f.write(f"      Available top-k predictions:\n")
                                    for i, candidate in enumerate(top_k[:3], 1):  # Show top-3
                                        gloss_name = candidate['gloss']
                                        conf = candidate['confidence']
                                        # Check if this is the correct answer
                                        is_correct = gloss_name.upper() == mm['original'].upper()
                                        marker = " [CORRECT!]" if is_correct else ""
                                        f.write(f"        {i}. {gloss_name} ({conf*100:.1f}%){marker}\n")
                    if r['gloss_accuracy'] is not None:
                        improvement = r['effective_gloss_accuracy'] - r['gloss_accuracy']
                        if improvement != 0:
                            f.write(f"  Gloss Accuracy Improvement: {improvement:+.1f}%\n")

                # BLEU (grouped: baseline, model, improvement)
                if r['baseline_bleu'] is not None or r['model_bleu'] is not None:
                    if r['baseline_bleu'] is not None:
                        f.write(f"  Baseline BLEU: {r['baseline_bleu']:.2f}\n")
                    if r['model_bleu'] is not None:
                        f.write(f"  Model BLEU: {r['model_bleu']:.2f}\n")
                    if r['improvement_bleu'] is not None:
                        f.write(f"  BLEU Improvement: {r['improvement_bleu']:+.2f}\n")

                # BERTScore (grouped: baseline, model, improvement)
                if r['baseline_bertscore'] is not None or r['model_bertscore'] is not None:
                    if r['baseline_bertscore'] is not None:
                        f.write(f"  Baseline BERTScore: {r['baseline_bertscore']:.2f}\n")
                    if r['model_bertscore'] is not None:
                        f.write(f"  Model BERTScore: {r['model_bertscore']:.2f}\n")
                    if r['improvement_bertscore'] is not None:
                        f.write(f"  BERTScore Improvement: {r['improvement_bertscore']:+.2f}\n")

                # Quality (grouped: baseline, model, improvement)
                if r['baseline_quality'] is not None or r['model_quality'] is not None:
                    if r['baseline_quality'] is not None:
                        f.write(f"  Baseline Quality: {r['baseline_quality']:.2f}\n")
                    if r['model_quality'] is not None:
                        f.write(f"  Model Quality: {r['model_quality']:.2f}\n")
                    if r['improvement_quality'] is not None:
                        f.write(f"  Quality Improvement: {r['improvement_quality']:+.2f}\n")

                # Coverage (grouped: baseline, model, improvement)
                if r['baseline_coverage_recall'] is not None or r['model_coverage_recall'] is not None:
                    if r['baseline_coverage_recall'] is not None:
                        f.write(f"  Baseline Coverage Recall: {r['baseline_coverage_recall']:.1f}%\n")
                        f.write(f"  Baseline Coverage Precision: {r['baseline_coverage_precision']:.1f}%\n")
                        f.write(f"  Baseline Coverage F1: {r['baseline_coverage_f1']:.1f}%\n")
                    if r['model_coverage_recall'] is not None:
                        f.write(f"  Model Coverage Recall: {r['model_coverage_recall']:.1f}%\n")
                        f.write(f"  Model Coverage Precision: {r['model_coverage_precision']:.1f}%\n")
                        f.write(f"  Model Coverage F1: {r['model_coverage_f1']:.1f}%\n")
                    if r['improvement_coverage_recall'] is not None:
                        f.write(f"  Coverage Recall Improvement: {r['improvement_coverage_recall']:+.1f}%\n")
                        f.write(f"  Coverage Precision Improvement: {r['improvement_coverage_precision']:+.1f}%\n")
                        f.write(f"  Coverage F1 Improvement: {r['improvement_coverage_f1']:+.1f}%\n")
                    if r['missing_words']:
                        f.write(f"  Missing Words: {', '.join(r['missing_words'])}\n")
                    if r['hallucinated_words']:
                        f.write(f"  Hallucinated Words: {', '.join(r['hallucinated_words'])}\n")

                # PTR (Perfect Translation Rate)
                if r['baseline_ptr'] is not None or r['model_ptr'] is not None:
                    baseline_ptr_str = "Yes" if r['baseline_ptr'] == 100 else "No"
                    model_ptr_str = "Yes" if r['model_ptr'] == 100 else "No"
                    f.write(f"  Baseline Perfect Translation: {baseline_ptr_str}\n")
                    f.write(f"  Model Perfect Translation: {model_ptr_str}\n")

                # CTQI Composite (grouped: baseline, model, improvement)
                if r['baseline_composite'] is not None or r['model_composite'] is not None:
                    if r['baseline_composite'] is not None:
                        f.write(f"  Baseline CTQI: {r['baseline_composite']:.2f}\n")
                    if r['model_composite'] is not None:
                        f.write(f"  Model CTQI: {r['model_composite']:.2f}\n")
                    if r['improvement_composite'] is not None:
                        f.write(f"  CTQI Improvement: {r['improvement_composite']:+.2f}\n")

                # LLM alternative usage
                if r['llm_used_alternatives'] is not None:
                    total_positions = len(r['predicted_glosses']) if r['predicted_glosses'] else 0
                    f.write(f"  LLM Alternatives Used: {r['llm_used_alternatives']}/{total_positions}\n")
                    if r['llm_alternative_details']:
                        for detail in r['llm_alternative_details']:
                            rank_str = f"top-{detail.get('rank', '?')}"
                            top1_conf = detail.get('top1_confidence')
                            used_conf = detail.get('used_confidence')
                            if top1_conf is not None and used_conf is not None:
                                f.write(f"    Position {detail['position']}: Used '{detail['used']}' ({rank_str}, {used_conf*100:.1f}%) instead of '{detail['top1']}' (top-1, {top1_conf*100:.1f}%)\n")
                            else:
                                f.write(f"    Position {detail['position']}: Used '{detail['used']}' ({rank_str}) instead of '{detail['top1']}' (top-1)\n")

                f.write(f"  Status: {r['status']}\n")

        if not self.quiet:
            print(f"\nReport saved: {report_file}")

    def _generate_table(self):
        """Generate comparison table (markdown format)"""
        print("\n" + "="*80)
        print("COMPARISON TABLE")
        print("="*80)

        table_file = os.path.join(self.output_dir, "comparison_table.md")

        with open(table_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write("# Synthetic Dataset Evaluation - Comparison Table\n\n")
            f.write("| Entry | Glosses | Reference | Predicted | Base BLEU | Model BLEU | BLEU Improve | Base BERT | Model BERT | BERT Improve | Base Quality | Model Quality | Quality Improve | Base Composite | Model Composite | Composite Improve | Base Cov Recall | Model Cov Recall | Cov Recall Improve | Base Cov Precision | Model Cov Precision | Cov Precision Improve | Status |\n")
            f.write("|-------|---------|-----------|-----------|-----------|------------|--------------|-----------|------------|--------------|--------------|---------------|-----------------|----------------|-----------------|-------------------|-----------------|------------------|--------------------|--------------------|---------------------|-----------------------|--------|\n")

            # Write rows
            for r in self.results:
                entry_id = r['entry_id'] + 1
                glosses = ', '.join(r['glosses'])
                ref_sentence = r['reference_sentence']
                pred_sentence = r['predicted_sentence'] or 'N/A'

                # BLEU scores
                baseline_bleu = f"{r['baseline_bleu']:.2f}" if r['baseline_bleu'] is not None else 'N/A'
                model_bleu = f"{r['model_bleu']:.2f}" if r['model_bleu'] is not None else 'N/A'
                improvement_bleu = f"{r['improvement_bleu']:+.2f}" if r['improvement_bleu'] is not None else 'N/A'

                # BERTScore scores
                baseline_bertscore = f"{r['baseline_bertscore']:.2f}" if r['baseline_bertscore'] is not None else 'N/A'
                model_bertscore = f"{r['model_bertscore']:.2f}" if r['model_bertscore'] is not None else 'N/A'
                improvement_bertscore = f"{r['improvement_bertscore']:+.2f}" if r['improvement_bertscore'] is not None else 'N/A'

                # Quality scores
                baseline_quality = f"{r['baseline_quality']:.2f}" if r['baseline_quality'] is not None else 'N/A'
                model_quality = f"{r['model_quality']:.2f}" if r['model_quality'] is not None else 'N/A'
                improvement_quality = f"{r['improvement_quality']:+.2f}" if r['improvement_quality'] is not None else 'N/A'

                # Composite scores
                baseline_composite = f"{r['baseline_composite']:.2f}" if r['baseline_composite'] is not None else 'N/A'
                model_composite = f"{r['model_composite']:.2f}" if r['model_composite'] is not None else 'N/A'
                improvement_composite = f"{r['improvement_composite']:+.2f}" if r['improvement_composite'] is not None else 'N/A'

                # Coverage scores (comparative)
                baseline_cov_recall = f"{r['baseline_coverage_recall']:.1f}%" if r['baseline_coverage_recall'] is not None else 'N/A'
                model_cov_recall = f"{r['model_coverage_recall']:.1f}%" if r['model_coverage_recall'] is not None else 'N/A'
                improvement_cov_recall = f"{r['improvement_coverage_recall']:+.1f}%" if r['improvement_coverage_recall'] is not None else 'N/A'

                baseline_cov_precision = f"{r['baseline_coverage_precision']:.1f}%" if r['baseline_coverage_precision'] is not None else 'N/A'
                model_cov_precision = f"{r['model_coverage_precision']:.1f}%" if r['model_coverage_precision'] is not None else 'N/A'
                improvement_cov_precision = f"{r['improvement_coverage_precision']:+.1f}%" if r['improvement_coverage_precision'] is not None else 'N/A'

                status = r['status']

                f.write(f"| {entry_id} | {glosses} | {ref_sentence} | {pred_sentence} | {baseline_bleu} | {model_bleu} | {improvement_bleu} | {baseline_bertscore} | {model_bertscore} | {improvement_bertscore} | {baseline_quality} | {model_quality} | {improvement_quality} | {baseline_composite} | {model_composite} | {improvement_composite} | {baseline_cov_recall} | {model_cov_recall} | {improvement_cov_recall} | {baseline_cov_precision} | {model_cov_precision} | {improvement_cov_precision} | {status} |\n")

        print(f"Table saved: {table_file}")

        # Also print first 10 rows to console
        print("\nFirst 10 entries:")
        print("-"*80)
        for i, r in enumerate(self.results[:10]):
            print(f"{i+1}. {', '.join(r['glosses'])}")
            print(f"   Reference: {r['reference_sentence']}")
            print(f"   Predicted: {r['predicted_sentence'] or 'N/A'}")

            # Format BLEU scores
            baseline_bleu_str = f"{r['baseline_bleu']:.2f}" if r['baseline_bleu'] is not None else 'N/A'
            model_bleu_str = f"{r['model_bleu']:.2f}" if r['model_bleu'] is not None else 'N/A'
            improvement_bleu_str = f"{r['improvement_bleu']:+.2f}" if r['improvement_bleu'] is not None else 'N/A'

            print(f"   BLEU: {baseline_bleu_str} -> {model_bleu_str} ({improvement_bleu_str})")

            # Format BERTScore scores
            if r['baseline_bertscore'] is not None or r['model_bertscore'] is not None:
                baseline_bertscore_str = f"{r['baseline_bertscore']:.2f}" if r['baseline_bertscore'] is not None else 'N/A'
                model_bertscore_str = f"{r['model_bertscore']:.2f}" if r['model_bertscore'] is not None else 'N/A'
                improvement_bertscore_str = f"{r['improvement_bertscore']:+.2f}" if r['improvement_bertscore'] is not None else 'N/A'
                print(f"   BERTScore: {baseline_bertscore_str} -> {model_bertscore_str} ({improvement_bertscore_str})")

            # Format Quality scores
            if r['baseline_quality'] is not None or r['model_quality'] is not None:
                baseline_quality_str = f"{r['baseline_quality']:.2f}" if r['baseline_quality'] is not None else 'N/A'
                model_quality_str = f"{r['model_quality']:.2f}" if r['model_quality'] is not None else 'N/A'
                improvement_quality_str = f"{r['improvement_quality']:+.2f}" if r['improvement_quality'] is not None else 'N/A'
                print(f"   Quality: {baseline_quality_str} -> {model_quality_str} ({improvement_quality_str})")

            # Format PTR (Perfect Translation Rate)
            if r['baseline_ptr'] is not None or r['model_ptr'] is not None:
                baseline_ptr_str = "Yes" if r['baseline_ptr'] == 100 else "No"
                model_ptr_str = "Yes" if r['model_ptr'] == 100 else "No"
                print(f"   Perfect Translation: {baseline_ptr_str} -> {model_ptr_str}")

            # Format CTQI Composite scores
            if r['baseline_composite'] is not None or r['model_composite'] is not None:
                baseline_composite_str = f"{r['baseline_composite']:.2f}" if r['baseline_composite'] is not None else 'N/A'
                model_composite_str = f"{r['model_composite']:.2f}" if r['model_composite'] is not None else 'N/A'
                improvement_composite_str = f"{r['improvement_composite']:+.2f}" if r['improvement_composite'] is not None else 'N/A'
                print(f"   CTQI: {baseline_composite_str} -> {model_composite_str} ({improvement_composite_str})")

            print()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Batch evaluation of synthetic ASL sentence dataset"
    )
    parser.add_argument(
        "--dataset",
        default="../../datasets/synthetic_sentences/synthetic_gloss_to_sentence_llm_dataset_50_glosses.json",
        help="Path to synthetic dataset JSON"
    )
    parser.add_argument(
        "--checkpoint",
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM sentence construction (use simple gloss joining)"
    )
    parser.add_argument(
        "--output-dir",
        default="./evaluation_results",
        help="Output directory for results (default: ./evaluation_results)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit evaluation to first N entries (default: all)"
    )
    parser.add_argument(
        "--num-glosses",
        type=int,
        default=50,
        help="Number of glosses in model (default: 50)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip entries that already have results"
    )
    parser.add_argument(
        "--keep-poses",
        action="store_true",
        help="Keep pose and metadata files after evaluation (default: delete to save space)"
    )
    parser.add_argument(
        "--no-confidence-scores",
        action="store_true",
        help="Send only top-k words to LLM without confidence scores (LLM selects based on semantic fit only)"
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "val", "all"],
        default="all",
        help="Dataset split to use for pose files (default: all - uses all data)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal console output (only show glosses, reference, predicted sentence/glosses)"
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Use two-stage LLM pipeline (Stage 1: Selection, Stage 2: Sentence) instead of one-phase"
    )
    parser.add_argument(
        "--prompt-file",
        help="Custom prompt file path for one-stage LLM (overrides default prompt)"
    )
    parser.add_argument(
        "--use-manifest",
        action="store_true",
        help="Use val_manifest.json pickle files for prediction (matches accuracy analysis exactly)"
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Disable class masking (use all classes even if masked_classes.json exists)"
    )

    args = parser.parse_args()

    try:
        # Create evaluator
        evaluator = SyntheticDatasetEvaluator(
            dataset_path=args.dataset,
            checkpoint_path=args.checkpoint,
            use_llm=not args.no_llm,  # Enable LLM by default unless --no-llm is set
            output_dir=args.output_dir,
            num_glosses=args.num_glosses,
            skip_existing=args.skip_existing,
            keep_poses=args.keep_poses,
            no_confidence_scores=args.no_confidence_scores,
            split=args.split,
            quiet=args.quiet,
            use_two_stage=args.two_stage,
            prompt_file=args.prompt_file,
            use_manifest=args.use_manifest,
            no_mask=args.no_mask
        )

        # Run evaluation
        evaluator.evaluate_dataset(limit=args.limit)

        if not args.quiet:
            print("\n" + "="*80)
            print("EVALUATION COMPLETE")
            print("="*80)
            print(f"Results saved to: {args.output_dir}")

        return 0

    except KeyboardInterrupt:
        print("\n\nEVALUATION INTERRUPTED by user")
        return 1
    except Exception as e:
        print(f"\nERROR: Evaluation failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
