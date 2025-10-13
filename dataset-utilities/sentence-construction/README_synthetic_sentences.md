# Synthetic Sentence Generator for ASL Testing

This tool generates synthetic ASL sentences using glosses from the WLASL dataset and the Gemini API. It creates a test dataset with natural English sentences paired with their corresponding ASL glosses.

## Features

- **Configurable Gloss Count**: Specify number of glosses, automatically selects closest available dataset
- **Gemini-Powered Generation**: Uses Google Gemini API to generate natural English sentences
- **Customizable Prompts**: Edit the prompt template file without touching code
- **Controlled Output**: 300 sentences, 5-10 words each, evenly distributed
- **Organized Output**: Saves to configurable directory with gloss count in filename

## Configuration

All configuration is at the top of `synthetic-sentence-generator.py`:

```python
# CONFIGURATION
PROMPT_TEMPLATE_FILE = "syn_sentence_gen_prompt.txt"
OUTPUT_FILENAME_TEMPLATE = "synthetic_gloss_to_sentence_llm_dataset_{class_count}_glosses.json"
OUTPUT_REF_SENTENCE_DATASET_DIR = "datasets/synthetic_sentences"  # Relative to project root
```

### Customizing the Prompt

Edit `syn_sentence_gen_prompt.txt` to modify how Gemini generates sentences. The template uses these placeholders:
- `{num_sentences}` - Number of sentences to generate
- `{gloss_list}` - Comma-separated list of available glosses

No code changes needed - just edit the text file!

## Setup

### 1. Get Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create an API key
3. Set it as an environment variable:

**Windows:**
```bash
set GEMINI_API_KEY=your_api_key_here
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY=your_api_key_here
```

### 2. Install Dependencies

The required package (`google-generativeai`) is already in `requirements.txt`. If you ran `setup.py`, you're all set.

## Usage

### Basic Usage

```bash
# From the sentence-construction directory
cd dataset-utilities/sentence-construction

# Generate dataset with 20 glosses (will use 20_classes dataset)
python synthetic-sentence-generator.py --num-glosses 20

# Generate dataset with 50 glosses (will use 50_classes dataset)
python synthetic-sentence-generator.py --num-glosses 50
```

### Custom Output Directory

```bash
python synthetic-sentence-generator.py --num-glosses 20 --output-dir /path/to/custom/dir
```

## Output Format

The script generates files named `synthetic_gloss_to_sentence_llm_dataset_{N}_glosses.json`:

```json
[
  {
    "glosses": ["I", "WANT", "BOOK"],
    "sentence": "I want to read a book"
  },
  {
    "glosses": ["DOCTOR", "HELP", "DEAF"],
    "sentence": "The doctor helps deaf people"
  }
]
```

### Output Location

By default, files are saved to:
```
{project_root}/datasets/synthetic_sentences/synthetic_gloss_to_sentence_llm_dataset_{N}_glosses.json
```

Examples:
- `synthetic_gloss_to_sentence_llm_dataset_20_glosses.json` (20 glosses)
- `synthetic_gloss_to_sentence_llm_dataset_50_glosses.json` (50 glosses)

### Fields

- **glosses**: Array of ASL glosses from the dataset used to sign the sentence
- **sentence**: Natural, grammatically correct English sentence

## How It Works

### 1. Dataset Selection

Scans `datasets/wlasl_poses_complete/dataset_splits/` and finds the dataset with the closest number of classes **≤** your specified `--num-glosses`.

Available datasets:
- `20_classes` - 20 glosses
- `50_classes` - 50 glosses

### 2. Gloss Extraction

Extracts gloss names from the selected dataset's `train/` folder (folder names become glosses).

### 3. Sentence Generation

1. Loads prompt template from `syn_sentence_gen_prompt.txt`
2. Fills in `{num_sentences}` and `{gloss_list}` placeholders
3. Sends to Gemini API
4. Validates output:
   - Ensures all glosses are from the available set
   - Checks JSON format
   - Filters invalid sentences

### 4. Output

Saves validated sentences to JSON file with gloss count in the filename.

## Example Output

```
Searching for dataset with <= 50 glosses...
Selected dataset: 50 classes
Dataset path: .../datasets/wlasl_poses_complete/dataset_splits/50_classes/.../train

Extracting glosses...
Found 50 glosses: ACCIDENT, APPLE, BATH, BEFORE, BLUE, BOOK, ...

Generating sentences with Gemini API...

Checking available models...
  Found: models/gemini-2.5-pro-preview-03-25
  Found: models/gemini-2.5-flash-preview-05-20
  ...
Using model: models/gemini-2.5-flash-preview-05-20

Calling Gemini API to generate sentences...
Generated 300 valid sentences

======================================================================
Dataset Generation Complete!
======================================================================
Output file: C:\...\datasets\synthetic_sentences\synthetic_gloss_to_sentence_llm_dataset_50_glosses.json
Total sentences: 300
Glosses used: 50

Sentence length distribution:
  5 words: 50 sentences
  6 words: 50 sentences
  7 words: 50 sentences
  8 words: 50 sentences
  9 words: 50 sentences
  10 words: 50 sentences

Sample output (first 3 sentences):

1. Sentence: I want to read a book
   Glosses: ['I', 'WANT', 'BOOK']

2. Sentence: The doctor helps deaf people
   Glosses: ['DOCTOR', 'HELP', 'DEAF']

3. Sentence: My cousin wears blue clothes
   Glosses: ['COUSIN', 'BLUE', 'CLOTHES']
```

## Troubleshooting

### "GEMINI_API_KEY environment variable not set"

Set your API key:
```bash
# Windows
set GEMINI_API_KEY=your_key

# Linux/Mac
export GEMINI_API_KEY=your_key
```

### "No dataset found with <= N glosses"

Make sure you have the dataset splits in:
```
datasets/wlasl_poses_complete/dataset_splits/
├── 20_classes/original/pickle_from_pose_split_20_class/train/
└── 50_classes/original/pickle_from_pose_split_50_class/train/
```

### "Prompt file not found"

Ensure `syn_sentence_gen_prompt.txt` exists in the `dataset-utilities/sentence-construction/` directory.

### API Rate Limits / Quota Exceeded

If you hit Gemini API rate limits:
- The script automatically tries alternative models
- Wait a moment and try again
- Check your API quota at [Google AI Studio](https://makersuite.google.com)

### Invalid Glosses in Output

The script validates all generated glosses against the input set and automatically filters out invalid sentences.

## Integration with Testing

Use the generated dataset for:
- **Gloss-to-Text translation testing**
- **Sentence construction evaluation**
- **Model robustness testing**
- **Reference sentence generation**

Load in your test code:
```python
import json

with open('datasets/synthetic_sentences/synthetic_gloss_to_sentence_llm_dataset_50_glosses.json') as f:
    test_data = json.load(f)

for item in test_data:
    glosses = item['glosses']
    reference_sentence = item['sentence']
    # Your test logic here
```

## Files

- `synthetic-sentence-generator.py` - Main script
- `syn_sentence_gen_prompt.txt` - Customizable prompt template
- `README.md` - This file
