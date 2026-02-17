# Synthetic Dataset Evaluation Summary (n=53, v4 - Gemini Plausibility)

**Generated:** 2026-02-16
**Dataset:** `synthetic_gloss_to_sentence_llm_dataset_43_glosses_50.json`

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Entries** | 53 |
| **Total Glosses** | 164 |
| **Success Rate** | 100% (53/53) |

### Gloss Accuracy Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Top-1 Accuracy** | 83.5% (137/164) | Model's raw prediction accuracy |
| **Effective Accuracy** | 89.0% (146/164) | After LLM selection from top-3 |
| **Top-3 Accuracy** | 95.1% (156/164) | Upper bound (correct in top-3) |
| **LLM Improvement** | +5.5% | 9 additional glosses recovered |

### Translation Quality Metrics (CTQI v2 with Gemini Plausibility)

| Metric | Baseline | Model | Improvement |
|--------|----------|-------|-------------|
| **CTQI v2** | 46.51 | 76.05 | **+29.55** |
| **Entries Improved** | - | 50/53 | 94.3% |

**CTQI v2 Formula:** `(GA/100) × (CF1/100) × (0.5 + 0.5 × P/100) × 100`

Where P = Plausibility (Gemini LLM-based scoring)

---

## Comparison: GPT-2 vs Gemini Plausibility

| Metric | v3 (GPT-2) | v4 (Gemini) | Difference |
|--------|------------|-------------|------------|
| Baseline CTQI v2 | 46.04 | 46.51 | +0.47 |
| Model CTQI v2 | 67.79 | **76.05** | **+8.26** |
| Improvement | +21.75 | **+29.55** | **+7.80** |

**Why Gemini scores higher:**
- Gemini gives 100 for fluent sentences (GPT-2 gives ~80-90)
- Gemini penalizes broken grammar more severely (7-43 vs 45-60)
- Better discrimination between good and bad translations

---

## Key Findings

### 1. Model Performance (Top-1 = 83.5%)

The ASL recognition model achieves 83.5% accuracy on top-1 predictions, with known confusion patterns:

| Confusion Pattern | Confidence | Frequency |
|-------------------|------------|-----------|
| NEED → BOWLING | 91.5% | 16 instances |
| COMPUTER → SON | 67.2% | 3 instances |
| LATER → DRINK | 45.1% | 8 instances |

### 2. LLM Recovery Analysis

**Successful result:** LLM improved accuracy by +5.5% (9 glosses recovered out of 27 errors).

**Mismatch Breakdown:**
- **Model Prediction Errors:** 8 (44.4%) - Correct answer NOT in top-3
- **LLM Selection Errors:** 10 (55.6%) - Correct answer WAS in top-3, but LLM chose wrong

**Recovery Rate for NEED:** 10/16 (62.5%) - improved from previous evaluations

### 3. Gemini Plausibility Scoring Examples

| Sentence | Gemini | GPT-2 | Delta |
|----------|--------|-------|-------|
| "My son plays basketball." | 100 | 87 | +13 |
| "The deaf son plays basketball." | 100 | 34 | +66 |
| "The doctor helps with bowling." | 7 | 44 | -37 |
| "The deaf study." | 75 | 5 | +70 |
| "My son enjoys family." | 43 | 80 | -37 |

Gemini better captures semantic plausibility, not just grammatical correctness.

---

## CTQI v2 Score Distribution

| CTQI v2 Range | Baseline | Model |
|---------------|----------|-------|
| 90-100 | 1 | 32 |
| 70-90 | 4 | 4 |
| 50-70 | 9 | 5 |
| 30-50 | 15 | 7 |
| 0-30 | 24 | 5 |

### Top Performing Entries (Highest CTQI v2 Improvement)

| # | Glosses | Baseline | Model | Improvement |
|---|---------|----------|-------|-------------|
| 31 | FAMILY, NEED, APPLE | 23.8 | 100.0 | **+76.2** |
| 33 | DOCTOR, NEED, PAPER | 23.8 | 100.0 | **+76.2** |
| 20 | DOCTOR, NEED, WATER | 24.0 | 100.0 | **+76.0** |
| 34 | DOCTOR, NEED, CHAIR | 24.0 | 100.0 | **+76.0** |
| 32 | FAMILY, NEED, WATER | 26.7 | 100.0 | **+73.3** |

### Entries with Negative CTQI v2 (Model < Baseline)

| # | Glosses | Baseline | Model | Delta | Reason |
|---|---------|----------|-------|-------|--------|
| 9 | CHAIR, FINE, DOCTOR | 53.5 | 44.4 | -9.1 | FINE→PAPER LLM error |
| 39 | DOCTOR, NEED, HELP | 29.6 | 23.8 | -5.8 | NEED→BOWLING, Plaus=7 |
| 2 | STUDY, BEFORE, PLAY | 99.0 | 95.5 | -3.5 | Baseline was near-perfect |
| 6 | BASKETBALL, PLAY, TIME | 76.0 | 75.0 | -1.0 | Quality dropped |

---

## Hierarchy Verification

**Expected:** Top-3 > Effective > Top-1

**Actual:**
- Top-3: 95.1% (156/164)
- Effective: 89.0% (146/164)
- Top-1: 83.5% (137/164)

**Result:** Top-3 > Effective > Top-1 ✓ **HIERARCHY CORRECT**

The LLM adds value by recovering 9 glosses from top-3 predictions.

---

## Files Generated

1. **evaluation_results.json** - Full per-entry metrics in JSON format
2. **comparison_table.md** - Side-by-side comparison of all entries
3. **evaluation_report.txt** - Detailed breakdown with mismatch analysis
4. **evaluation_summary.md** - This summary document
5. **failure_scenario_analysis.txt** - Detailed failure scenario breakdown

---

## Conclusion

Using **Gemini plausibility scoring** instead of GPT-2 quality:

1. **Model CTQI v2 increased from 67.79 to 76.05** (+8.26 points)
2. **Overall improvement increased from +21.75 to +29.55** (+7.80 points)
3. **32 entries now score 90-100** (vs 24 with GPT-2)

Gemini provides more discriminative plausibility scoring that better reflects human judgment of sentence quality, resulting in higher scores for fluent translations and lower scores for semantically broken ones.
