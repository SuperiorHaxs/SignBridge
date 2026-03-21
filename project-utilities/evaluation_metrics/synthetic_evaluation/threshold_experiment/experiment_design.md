# Threshold Experiment Design

## Objective
Find the minimum Top-3 confidence below which LLM sentence construction
does NOT add statistically significant CTQI v3 improvement.

## Methodology
### 1. Multi-Pickle Profiling
Profiled all 43 signs using all val pickle files (original + augmented)
to get robust per-sign confidence distributions.

### 2. Variable-Length Sentences
Used 3-5 gloss sentences mixing hard/medium/easy signs to target
specific Top-3 accuracy bands.

### 3. Stratified Design
5 bands x 30 sentences = 150 total, targeting:
- Band A: 0-33% Top-3 accuracy
- Band B: 33-50%
- Band C: 50-67%
- Band D: 67-83%
- Band E: 83-100%

## Sign Difficulty Classification
| Difficulty | Signs |
|---|---|
| HARD | BIRTHDAY, COW, DANCE, FULL, LATER, TABLE, WHAT |
| MEDIUM | COMPUTER, NEED |
| MEDIUM-EASY | FINE, NO, TELL, THANKSGIVING, WALK |
| EASY | AFRICA, APPLE, BASKETBALL, BEFORE, BOWLING, BUT, CHAIR, CHANGE, DEAF, DOCTOR, DRINK, ENJOY, FAMILY, GIVE, HELP, HOT, JACKET, MAN, ORANGE, PAPER, PLAY, SON, STUDY, TALL, THURSDAY, TIME, WATER, WRONG, YEAR |

## How to Run
### Step 1: Rewrite reference sentences
Auto-generated sentences are gloss-joins. Rewrite to natural English.

### Step 2: Run evaluation
```bash
python evaluate_synthetic_dataset.py \
  --dataset C:\Users\ashwi\Projects\WLASL-proj\asl-v1\project-utilities\evaluation_metrics\synthetic_evaluation\threshold_experiment\threshold_experiment_dataset.json \
  --use-manifest --use-llm --use-gemini-plausibility \
  --output-dir C:\Users\ashwi\Projects\WLASL-proj\asl-v1\project-utilities\evaluation_metrics\synthetic_evaluation\threshold_experiment/results
```

### Step 3: Analyze results
Run top3_threshold_analysis.py on the results directory.
