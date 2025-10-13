# BLEU Score Calculator for ASL Sentence Construction

Calculate BLEU scores for predicted ASL sentences by comparing them against reference sentences from the synthetic dataset.

## Overview

This tool:
1. Takes a comma-separated list of glosses
2. Looks up the reference sentence for those glosses in the synthetic dataset
3. Compares your predicted sentence against the reference
4. Returns a BLEU score (0-100)

## Configuration

All file paths are configured at the top of `calculate_sent_bleu.py`:

```python
# CONFIGURATION
REFERENCE_DATASET_DIR = "datasets/synthetic_sentences"  # Relative to project root
REFERENCE_FILENAME_TEMPLATE = "synthetic_gloss_to_sentence_llm_dataset_{num_glosses}_glosses.json"
```

## Prerequisites

1. **sacrebleu package** - Already added to `requirements.txt`
   ```bash
   pip install sacrebleu>=2.3.0
   ```

2. **Reference dataset** - Generate using the sentence construction tool:
   ```bash
   cd dataset-utilities/sentence-construction
   python synthetic-sentence-generator.py --num-glosses 20
   ```

## Usage

### Basic Usage

```bash
python project-utilities/calculate_sent_bleu.py \
    --glosses "I,WANT,BOOK" \
    --sentence "I want to read a book" \
    --num-glosses 20
```

Output:
```
100.00
```

### Verbose Mode

```bash
python project-utilities/calculate_sent_bleu.py \
    --glosses "I,WANT,BOOK" \
    --sentence "I want a book" \
    --num-glosses 20 \
    --verbose
```

Output:
```
Input glosses: ['I', 'WANT', 'BOOK']
Predicted sentence: I want a book
Loading reference dataset (20 glosses)...
Loaded 300 reference sentences
Reference sentence: I want to read a book

======================================================================
BLEU Score Results
======================================================================
Glosses:            I, WANT, BOOK
Predicted:          I want a book
Reference:          I want to read a book
BLEU Score:         54.23
======================================================================
```

### Command Line Arguments

- `--glosses` (required): Comma-separated list of glosses (e.g., "I,WANT,BOOK")
- `--sentence` (required): Predicted sentence to evaluate
- `--num-glosses` (required): Number of glosses in reference dataset (20, 50, etc.)
- `--verbose`: Print detailed information

## Integration with Prediction Scripts

You can call this from your prediction scripts:

```python
import subprocess

def calculate_bleu(glosses_list, predicted_sentence, num_glosses=20):
    """Calculate BLEU score for predicted sentence."""
    glosses_str = ",".join(glosses_list)

    result = subprocess.run([
        "python", "project-utilities/calculate_sent_bleu.py",
        "--glosses", glosses_str,
        "--sentence", predicted_sentence,
        "--num-glosses", str(num_glosses)
    ], capture_output=True, text=True)

    if result.returncode == 0:
        bleu_score = float(result.stdout.strip())
        return bleu_score
    else:
        print(f"Error: {result.stderr}")
        return None

# Example usage
glosses = ["I", "WANT", "BOOK"]
predicted = "I want a book"
score = calculate_bleu(glosses, predicted, num_glosses=20)
print(f"BLEU Score: {score:.2f}")
```

Or import directly:

```python
import sys
from pathlib import Path

# Add project-utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent / "project-utilities"))

from calculate_sent_bleu import (
    load_reference_dataset,
    find_reference_sentence,
    calculate_bleu_score
)
from config import get_config

# Load reference data
config = get_config()
reference_data = load_reference_dataset(num_glosses=20, config=config)

# Find reference
glosses = ["I", "WANT", "BOOK"]
reference = find_reference_sentence(glosses, reference_data)

# Calculate BLEU
predicted = "I want a book"
bleu_score = calculate_bleu_score(predicted, reference)
print(f"BLEU Score: {bleu_score:.2f}")
```

## How BLEU Score Works

BLEU (Bilingual Evaluation Understudy) measures similarity between predicted and reference sentences:

- **100.00**: Perfect match
- **50-99**: High similarity with some differences
- **20-49**: Moderate similarity
- **0-19**: Low similarity

BLEU considers:
- Word overlap (n-grams)
- Word order
- Sentence length

## Examples

### Perfect Match
```bash
python calculate_sent_bleu.py \
    --glosses "DOCTOR,HELP,DEAF" \
    --sentence "The doctor helps deaf people" \
    --num-glosses 20
```
Output: `100.00`

### Partial Match
```bash
python calculate_sent_bleu.py \
    --glosses "DOCTOR,HELP,DEAF" \
    --sentence "Doctor helps people" \
    --num-glosses 20
```
Output: `~54.00` (missing "The" and "deaf")

### No Match
```bash
python calculate_sent_bleu.py \
    --glosses "DOCTOR,HELP,DEAF" \
    --sentence "Completely different sentence" \
    --num-glosses 20
```
Output: `~0.00`

## Error Handling

### Reference Dataset Not Found
```
Error: Reference dataset not found: .../synthetic_gloss_to_sentence_llm_dataset_20_glosses.json
Make sure you've generated the dataset with:
  python dataset-utilities/sentence-construction/synthetic-sentence-generator.py --num-glosses 20
```

**Solution**: Generate the reference dataset first.

### Glosses Not Found
```
Error: No reference sentence found for glosses: ['UNKNOWN', 'GLOSSES']
These glosses may not exist in the reference dataset.
```

**Solution**:
- Check gloss spelling
- Ensure glosses exist in the dataset
- Use the correct `--num-glosses` parameter

## Output Formats

### Programmatic (default)
Returns just the score for easy parsing:
```
54.23
```

### Verbose (--verbose)
Returns detailed information:
```
======================================================================
BLEU Score Results
======================================================================
Glosses:            I, WANT, BOOK
Predicted:          I want a book
Reference:          I want to read a book
BLEU Score:         54.23
======================================================================
```

## Testing

Test with known sentences from your reference dataset:

```bash
# First, check what's in your dataset
python -c "
import json
with open('datasets/synthetic_sentences/synthetic_gloss_to_sentence_llm_dataset_20_glosses.json') as f:
    data = json.load(f)
    print(data[0])
"

# Then test with that exact sentence (should get 100.00)
python calculate_sent_bleu.py \
    --glosses "GLOSS1,GLOSS2,..." \
    --sentence "exact sentence from dataset" \
    --num-glosses 20 \
    --verbose
```

## Files

- `calculate_sent_bleu.py` - Main BLEU calculator script
- `README_bleu_calculator.md` - This file
