#!/usr/bin/env python3
"""
incremental_trainer.py

Iterative training pipeline to build up to a target number of classes
while maintaining high accuracy.

Workflow:
1. Analyze per-class accuracy of current model
2. Keep classes above accuracy threshold
3. Use smart selector to fill gap to target (e.g., 100 classes)
4. Output gloss list for next training iteration

Iterate until convergence (most classes are stable high-accuracy).

Usage:
    python incremental_trainer.py --model-dir <path> --num-classes 43 --target 100
    python incremental_trainer.py -m ./models/wlasl_43_class_model -n 43 -T 100 --threshold 70

Output:
    - accuracy_report.json: Per-class accuracy analysis
    - smart_selection_report.json: Candidate evaluation details
    - gloss_list_<N>_class.json: Final gloss list for training
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


def extract_keep_classes(accuracy_report_path: Path, threshold: float):
    """Extract classes above threshold from accuracy report."""
    with open(accuracy_report_path, 'r') as f:
        report = json.load(f)

    keep_classes = report['recommendations']['keep_classes']
    drop_classes = report['recommendations']['drop_classes']

    return keep_classes, drop_classes, report['summary']


def save_keep_classes_list(keep_classes: list, output_path: Path):
    """Save keep classes to a JSON file for smart selector."""
    # Convert to lowercase for consistency
    glosses = [g.lower() for g in keep_classes]
    with open(output_path, 'w') as f:
        json.dump(glosses, f)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Iterative training pipeline to reach target class count",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Iterative Workflow:
  1. Train initial model (e.g., 43 high-accuracy classes)
  2. Run this script to generate 100-class gloss list
  3. Train new 100-class model
  4. Run this script again on new model
  5. Repeat until accuracy stabilizes

Example iteration:
  Iteration 1: 43 keep + 57 new = 100 -> Train -> 60 classes pass threshold
  Iteration 2: 60 keep + 40 new = 100 -> Train -> 75 classes pass threshold
  Iteration 3: 75 keep + 25 new = 100 -> Train -> 85 classes pass threshold
  ...converges to stable high-accuracy 100-class model
        """
    )

    parser.add_argument("--model-dir", "-m", type=Path, required=True,
                        help="Path to trained model directory")
    parser.add_argument("--num-classes", "-n", type=int, required=True,
                        help="Number of classes the model was trained on")
    parser.add_argument("--target", "-T", type=int, default=100,
                        help="Target number of classes (default: 100)")
    parser.add_argument("--threshold", "-t", type=float, default=70.0,
                        help="Accuracy threshold for keeping classes (default: 70%%)")
    parser.add_argument("--min-samples", type=int, default=15,
                        help="Minimum samples required per candidate gloss (default: 15)")
    parser.add_argument("--max-candidates", type=int, default=200,
                        help="Maximum candidates to evaluate in smart selector (default: 200)")
    parser.add_argument("--output-dir", "-o", type=Path, default=None,
                        help="Output directory (default: incremental_training/)")
    parser.add_argument("--skip-accuracy", action="store_true",
                        help="Skip accuracy analysis (use existing report)")

    args = parser.parse_args()

    # Set output directory
    output_dir = args.output_dir or script_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_report = output_dir / "accuracy_report.json"
    keep_classes_file = output_dir / "keep_classes_temp.json"

    print("="*70)
    print("INCREMENTAL TRAINING PIPELINE")
    print("="*70)
    print(f"Model directory: {args.model_dir}")
    print(f"Current classes: {args.num_classes}")
    print(f"Target classes: {args.target}")
    print(f"Accuracy threshold: {args.threshold}%")
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

    # Step 2: Extract keep classes
    print(f"\n{'='*70}")
    print("STEP: Extracting high-accuracy classes")
    print("="*70)

    keep_classes, drop_classes, summary = extract_keep_classes(accuracy_report, args.threshold)

    print(f"\nAccuracy Summary:")
    print(f"  Overall accuracy: {summary['overall_accuracy']}%")
    print(f"  Classes above {args.threshold}% threshold: {len(keep_classes)}")
    print(f"  Classes below threshold: {len(drop_classes)}")

    # Calculate gap to target
    num_to_add = args.target - len(keep_classes)

    if num_to_add <= 0:
        print(f"\n*** Already at or above target! ***")
        print(f"  Keep classes: {len(keep_classes)}")
        print(f"  Target: {args.target}")
        print(f"\nNo new classes needed. Consider raising the threshold.")

        # Still save the keep classes list
        final_gloss_list = output_dir / f"gloss_list_{len(keep_classes)}_class.json"
        save_keep_classes_list(keep_classes, final_gloss_list)
        print(f"\nSaved final gloss list to: {final_gloss_list}")
        return 0

    print(f"\n  Gap to target: {args.target} - {len(keep_classes)} = {num_to_add} classes needed")

    # Save keep classes for smart selector
    save_keep_classes_list(keep_classes, keep_classes_file)

    # Step 3: Run smart gloss selector
    success = run_step(
        "smart_gloss_selector.py",
        [
            "--model-dir", str(args.model_dir),
            "--keep-classes", str(keep_classes_file),
            "--accuracy-report", str(accuracy_report),
            "--num-to-select", str(num_to_add),
            "--min-samples", str(args.min_samples),
            "--max-candidates", str(args.max_candidates),
            "--output-dir", str(output_dir)
        ],
        f"Smart selection of {num_to_add} new candidates"
    )
    if not success:
        return 1

    # Load results
    final_gloss_list = output_dir / f"gloss_list_{args.target}_class.json"
    selection_report = output_dir / "smart_selection_report.json"

    if not final_gloss_list.exists():
        print(f"ERROR: Expected output not found: {final_gloss_list}")
        return 1

    with open(final_gloss_list, 'r') as f:
        new_glosses = json.load(f)

    with open(selection_report, 'r') as f:
        selection = json.load(f)

    # Print final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)

    print(f"\nIteration Summary:")
    print(f"  Classes kept (above {args.threshold}%): {len(keep_classes)}")
    print(f"  New candidates selected: {len(selection['selected_glosses'])}")
    print(f"  Total for next training: {len(new_glosses)}")

    print(f"\nClasses dropped (below {args.threshold}% accuracy):")
    for gloss in drop_classes[:10]:
        print(f"  - {gloss}")
    if len(drop_classes) > 10:
        print(f"  ... and {len(drop_classes) - 10} more")

    print(f"\nNew candidates added:")
    for item in selection['selected_glosses'][:10]:
        rec = item.get('recommendation', 'N/A')
        dist = item.get('distinctiveness', 0)
        print(f"  + {item['gloss']} (distinctiveness: {dist:.3f}, {rec})")
    if len(selection['selected_glosses']) > 10:
        print(f"  ... and {len(selection['selected_glosses']) - 10} more")

    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print("="*70)
    print(f"""
1. Review the gloss list: {final_gloss_list}

2. Run pre-processing on the other machine:
   python training_pre_processing_setup.py \\
       --gloss-file "{final_gloss_list}" \\
       --setup --force-fresh

3. Train the new model:
   python train_asl.py --num-classes {args.target} --model-size small --dropout 0.3

4. Run this pipeline again on the new model:
   python incremental_trainer.py \\
       --model-dir <new_model_path> \\
       --num-classes {args.target} \\
       --target {args.target} \\
       --threshold {args.threshold}

5. Repeat until accuracy stabilizes (most classes above threshold)
""")

    # Cleanup temp file
    if keep_classes_file.exists():
        keep_classes_file.unlink()

    return 0


if __name__ == "__main__":
    exit(main())
