#!/usr/bin/env python3
"""
incremental_trainer.py

Orchestrates the incremental training pipeline:
1. Analyze per-class accuracy of existing model
2. Compute embeddings for existing and candidate glosses
3. Select next batch of glosses to add

This script generates a gloss list that you can pass to the existing
training scripts (training_pre_processing_setup.py and train_asl.py).

Usage:
    python incremental_trainer.py --model-dir <path> --num-classes 100
    python incremental_trainer.py -m ./models/wlasl_100_class_model -n 100 --threshold 70 --add 5

Output:
    - accuracy_report.json
    - gloss_embeddings.json
    - next_glosses.json (final gloss list for training)
"""

import os
os.environ['PYTORCH_DISABLE_ONNX_METADATA'] = '1'
os.environ['TORCH_DYNAMO_DISABLE'] = '1'

import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))


def run_step(script_name: str, args: list, description: str):
    """Run a step of the pipeline."""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print("="*70)

    script_path = script_dir / script_name
    cmd = [sys.executable, str(script_path)] + args

    print(f"Running: {' '.join(cmd)}")
    print("-"*70)

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\nERROR: {script_name} failed with return code {result.returncode}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate incremental training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example workflow:
  1. Train initial model with N classes
  2. Run this script to get next gloss list
  3. Use gloss list with training_pre_processing_setup.py
  4. Train new model with train_asl.py
  5. Repeat from step 2

The pipeline generates:
  - accuracy_report.json: Per-class accuracy analysis
  - gloss_embeddings.json: Model embeddings for similarity
  - next_glosses.json: Final gloss list for training
        """
    )

    parser.add_argument("--model-dir", "-m", type=Path, required=True,
                        help="Path to trained model directory")
    parser.add_argument("--num-classes", "-n", type=int, required=True,
                        help="Number of classes the model was trained on")
    parser.add_argument("--threshold", "-t", type=float, default=50.0,
                        help="Accuracy threshold for keeping classes (default: 50%%)")
    parser.add_argument("--add", "-a", type=int, default=5,
                        help="Number of new glosses to add (default: 5)")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="Minimum samples required per gloss (default: 10)")
    parser.add_argument("--max-embedding-samples", type=int, default=50,
                        help="Max samples per class for embeddings (default: 50)")
    parser.add_argument("--output-dir", "-o", type=Path, default=None,
                        help="Output directory (default: incremental_training/)")
    parser.add_argument("--skip-accuracy", action="store_true",
                        help="Skip accuracy analysis (use existing report)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding computation (use existing)")

    args = parser.parse_args()

    # Set output directory
    output_dir = args.output_dir or script_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_report = output_dir / "accuracy_report.json"
    embeddings_file = output_dir / "gloss_embeddings.json"
    next_glosses_file = output_dir / "next_glosses.json"

    print("="*70)
    print("INCREMENTAL TRAINING PIPELINE")
    print("="*70)
    print(f"Model directory: {args.model_dir}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Accuracy threshold: {args.threshold}%")
    print(f"Glosses to add: {args.add}")
    print(f"Min samples per gloss: {args.min_samples}")
    print(f"Output directory: {output_dir}")

    # Step 1: Analyze per-class accuracy
    if not args.skip_accuracy:
        success = run_step(
            "analyze_per_class_accuracy.py",
            [
                "--model-dir", str(args.model_dir),
                "--num-classes", str(args.num_classes),
                "--threshold", str(args.threshold),
                "--output", str(accuracy_report)
            ],
            "Analyzing per-class accuracy"
        )
        if not success:
            return 1
    else:
        print(f"\nSkipping accuracy analysis (using existing: {accuracy_report})")
        if not accuracy_report.exists():
            print(f"ERROR: Accuracy report not found: {accuracy_report}")
            return 1

    # Step 2: Compute embeddings
    if not args.skip_embeddings:
        success = run_step(
            "compute_gloss_embeddings.py",
            [
                "--model-dir", str(args.model_dir),
                "--num-classes", str(args.num_classes),
                "--max-samples", str(args.max_embedding_samples),
                "--output", str(embeddings_file),
                "--include-existing"  # Include existing classes for comparison
            ],
            "Computing gloss embeddings"
        )
        if not success:
            return 1
    else:
        print(f"\nSkipping embedding computation (using existing: {embeddings_file})")
        if not embeddings_file.exists():
            print(f"ERROR: Embeddings file not found: {embeddings_file}")
            return 1

    # Step 3: Select next glosses
    success = run_step(
        "select_next_glosses.py",
        [
            "--embeddings", str(embeddings_file),
            "--accuracy", str(accuracy_report),
            "--num-to-add", str(args.add),
            "--min-samples", str(args.min_samples),
            "--output", str(next_glosses_file)
        ],
        "Selecting next glosses"
    )
    if not success:
        return 1

    # Print final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)

    with open(next_glosses_file, 'r') as f:
        result = json.load(f)

    print(f"\nNext training configuration:")
    print(f"  Total classes: {result['total_classes']}")
    print(f"  Kept from current model: {result['existing_kept']}")
    print(f"  New classes added: {result['new_added']}")

    print(f"\nNew glosses to add:")
    for item in result['new_gloss_details']:
        print(f"  - {item['gloss']} (score: {item['combined_score']:.3f}, "
              f"samples: {item['num_samples']})")

    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print("="*70)
    print(f"""
1. Review the gloss list in: {next_glosses_file}

2. Create training data splits:
   python training_pre_processing_setup.py \\
       --gloss-list {next_glosses_file}

3. Train the new model:
   python train_asl.py --num-classes {result['total_classes']}

4. Run this pipeline again on the new model to continue expansion
""")

    # Also create a simple gloss list file for easy use
    simple_list_file = output_dir / "gloss_list.txt"
    with open(simple_list_file, 'w') as f:
        for gloss in result['glosses']:
            f.write(f"{gloss}\n")

    print(f"Simple gloss list saved to: {simple_list_file}")

    return 0


if __name__ == "__main__":
    exit(main())
