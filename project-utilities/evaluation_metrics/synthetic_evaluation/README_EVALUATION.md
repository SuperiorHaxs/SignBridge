# Synthetic Dataset Evaluation

## Overview

The `evaluate_synthetic_dataset.py` script provides comprehensive batch evaluation of synthetic ASL sentence datasets. It creates concatenated pose files from glosses, runs model predictions, and compares BLEU scores to measure translation quality.

## Pipeline Flow

For each entry in the synthetic dataset:

1. **Create Concatenated Pose**: Calls `sentence_to_pickle.py` with glosses to create pose file + segment metadata
2. **Calculate Baseline BLEU**: Compares simple gloss concatenation vs reference sentence
3. **Run Prediction**: Calls `predict_sentence.py` with pose + metadata to get model predictions
4. **Calculate Model BLEU**: Compares model-generated sentence vs reference sentence
5. **Calculate Improvement**: Model BLEU - Baseline BLEU

## Requirements

### Files Required

- **Synthetic Dataset**: `../../datasets/synthetic_sentences/synthetic_gloss_to_sentence_llm_dataset_50_glosses.json`
  - Format: `[{"glosses": ["WORD1", "WORD2", ...], "sentence": "Reference sentence."}, ...]`

- **Model Checkpoint** (recommended): Path to trained OpenHands checkpoint
  - Without checkpoint: predictions will fail but baseline BLEU is still calculated

- **Pose Files**: WLASL pose dataset (automatically located via config)

### Dependencies

- Python 3.11+
- All dependencies from `requirements.txt`
- `sentence_to_pickle.py` (in parent directory: `project-utilities/`)
- `predict_sentence.py` (in `../../applications/`)
- `calculate_sent_bleu.py` (in parent directory: `project-utilities/`)

## Usage

### Basic Usage (Test Run)

Test with first 3 entries without checkpoint:

```bash
cd project-utilities/synthetic_evaluation
python evaluate_synthetic_dataset.py --limit 3 --output-dir test_evaluation
```

### Full Evaluation with Checkpoint

Evaluate all entries with trained model:

```bash
python evaluate_synthetic_dataset.py \
  --checkpoint ../../models/openhands-modernized/checkpoints/checkpoint_epoch_10 \
  --output-dir evaluation_results_epoch10 \
  --num-glosses 50
```

### With Gemini API (for LLM sentence construction)

```bash
python evaluate_synthetic_dataset.py \
  --checkpoint ../../models/openhands-modernized/checkpoints/checkpoint_epoch_10 \
  --gemini-api-key YOUR_API_KEY \
  --output-dir evaluation_results_with_llm \
  --num-glosses 50
```

Or set environment variable:

```bash
export GEMINI_API_KEY=YOUR_API_KEY
python evaluate_synthetic_dataset.py --checkpoint <path> --output-dir <dir>
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset PATH` | Path to synthetic dataset JSON | `../../datasets/synthetic_sentences/synthetic_gloss_to_sentence_llm_dataset_50_glosses.json` |
| `--checkpoint PATH` | Path to model checkpoint directory | None (uses default model) |
| `--gemini-api-key KEY` | Gemini API key for sentence construction | From `GEMINI_API_KEY` env var |
| `--output-dir PATH` | Output directory for results | `./evaluation_results` |
| `--limit N` | Limit evaluation to first N entries | All entries |
| `--num-glosses N` | Number of glosses in model (20 or 50) | 50 |
| `--skip-existing` | Skip entries with existing results | False |

## Output Files

The script generates three main output files in the output directory:

### 1. `evaluation_results.json`

Detailed results for each entry in JSON format:

```json
[
  {
    "entry_id": 0,
    "glosses": ["DOG", "LIKE", "WALK"],
    "reference_sentence": "My dog likes to walk.",
    "baseline_bleu": 17.80,
    "model_bleu": 45.32,
    "predicted_sentence": "The dog likes to walk.",
    "predicted_glosses": ["DOG", "LIKE", "WALK"],
    "improvement": 27.52,
    "status": "success"
  },
  ...
]
```

### 2. `comparison_table.md`

Markdown table comparing baseline vs model performance:

| Entry | Glosses | Reference Sentence | Predicted Sentence | Baseline BLEU | Model BLEU | Improvement | Status |
|-------|---------|-------------------|-------------------|---------------|------------|-------------|--------|
| 1 | DOG, LIKE, WALK | My dog likes to walk. | The dog likes to walk. | 17.80 | 45.32 | +27.52 | success |
| ... | ... | ... | ... | ... | ... | ... | ... |

### 3. `evaluation_report.txt`

Summary statistics and detailed results:

```
SYNTHETIC DATASET EVALUATION REPORT
================================================================================

Dataset: ../datasets/synthetic_sentences/synthetic_gloss_to_sentence_llm_dataset_50_glosses.json
Total entries: 300
Successful: 285 (95.0%)
Failed: 15 (5.0%)

BASELINE BLEU: 22.45
MODEL BLEU: 48.73
IMPROVEMENT: +26.28

DETAILED RESULTS:
--------------------------------------------------------------------------------
...
```

## Generated Pose Files

In addition to the evaluation reports, the script creates:

- **Concatenated pose files**: `sentence_<glosses>_<timestamp>.pose`
- **Segment metadata files**: `sentence_<glosses>_<timestamp>_segments.json`

These files are saved in the output directory and can be reused for manual testing.

## Example Workflow

### 1. Quick Test (3 entries, no checkpoint)

```bash
python evaluate_synthetic_dataset.py --limit 3 --output-dir test_run
```

**Expected output:**
- Baseline BLEU calculated
- Model predictions fail (no checkpoint)
- Files created in `test_run/`

### 2. Full Evaluation (all entries, with checkpoint)

```bash
python evaluate_synthetic_dataset.py \
  --checkpoint ../models/openhands-modernized/checkpoints/best_model \
  --output-dir full_evaluation_best_model \
  --num-glosses 50
```

**Expected output:**
- All 300 entries evaluated
- Baseline and model BLEU scores
- Improvement statistics
- Comparison table

### 3. Incremental Evaluation (resume after failure)

If evaluation fails partway through, results are saved incrementally. You can continue by:

```bash
python evaluate_synthetic_dataset.py \
  --checkpoint <path> \
  --output-dir <same_dir> \
  --skip-existing
```

## Performance Notes

- **Time per entry**: ~10-20 seconds (depends on sentence length and checkpoint size)
- **Estimated total time**:
  - 3 entries: ~1 minute
  - 100 entries: ~30 minutes
  - 300 entries: ~1.5 hours

## Troubleshooting

### "No valid predictions available"

**Cause**: No checkpoint provided or checkpoint failed to load

**Solution**: Provide valid checkpoint path:
```bash
python evaluate_synthetic_dataset.py --checkpoint <path_to_checkpoint>
```

### "Glosses not found in reference dataset"

**Cause**: Gloss combination not in WLASL reference dataset

**Effect**: BLEU score will be `null` for that entry

**Solution**: This is expected for some synthetic combinations

### Script times out

**Cause**: Large checkpoint or long sentence

**Solution**: Increase timeout in script (currently 120s for prediction)

### Unicode encoding errors

**Cause**: Windows console encoding (cp1252) doesn't support special characters

**Solution**: Already handled - script uses ASCII arrows (->)

## Integration with Other Scripts

### Using with sentence_to_pickle.py

The evaluation script calls `sentence_to_pickle.py` with:
```bash
python sentence_to_pickle.py \
  --sentence "WORD1 WORD2 WORD3" \
  --pose-mode \
  --num-glosses 50 \
  --output-dir <output_dir>
```

### Using with predict_sentence.py

The evaluation script calls `predict_sentence.py` with:
```bash
python predict_sentence.py \
  --pose-file <pose_file> \
  --segment-metadata <metadata_file> \
  --checkpoint <checkpoint_path> \
  --num-glosses 50
```

## Next Steps

After running evaluation:

1. **Analyze results**: Check `comparison_table.md` for per-entry performance
2. **Review failures**: Look for patterns in failed entries
3. **Compare checkpoints**: Run evaluation on multiple checkpoints to find best model
4. **Identify improvements**: Look for glosses combinations where model significantly outperforms baseline

## Example Results Interpretation

Given this output:

```
BASELINE BLEU: 22.45
MODEL BLEU: 48.73
IMPROVEMENT: +26.28
```

**Interpretation:**
- **Baseline (22.45)**: Simple gloss concatenation is 22.45% similar to reference
- **Model (48.73)**: Model-generated sentences are 48.73% similar to reference
- **Improvement (+26.28)**: Model improves translation quality by 26.28 BLEU points

**Good performance indicators:**
- Model BLEU > 40: Strong translation quality
- Improvement > +20: Significant improvement over baseline
- Success rate > 90%: Robust model

## File Locations

```
project-utilities/
├── synthetic_evaluation/
│   ├── evaluate_synthetic_dataset.py    # Main evaluation script
│   └── README_EVALUATION.md             # This file
├── sentence_to_pickle.py                # Called to create poses
└── calculate_sent_bleu.py               # BLEU calculation

applications/
└── predict_sentence.py                  # Called for predictions

datasets/synthetic_sentences/
└── synthetic_gloss_to_sentence_llm_dataset_50_glosses.json  # Input dataset
```
