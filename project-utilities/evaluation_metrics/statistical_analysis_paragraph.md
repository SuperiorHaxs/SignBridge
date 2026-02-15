# Statistical Analysis for Research Paper

## Gloss Selection and Translation Quality Evaluation

Statistical analysis of SignBridge results using paired t-tests (n=34 sentence pairs) demonstrates significant improvements across all evaluation metrics. For gloss-level selection accuracy, Coverage F1—which measures the overlap of content words between the generated and reference sentences—improved from 74.64 to 87.62 (t(33) = 4.944, p < 0.001, Cohen's d = 0.848), representing a large effect size. This improvement indicates that the LLM pipeline more accurately selects contextually appropriate glosses from the model's top-k predictions, resulting in translations that better capture the intended meaning.

For overall translation quality, the Quality Score (a reference-free grammaticality measure based on GPT-2 perplexity) improved substantially from 39.38 to 74.56 (t(33) = 6.700, p < 0.001, Cohen's d = 1.149), representing a large effect size. Additionally, the Perfect Translation Rate—a binary metric indicating whether all glosses in a sentence were correctly predicted—increased from 41.2% (14/34) to 67.6% (23/34), with p=0.004, confirming this improvement is statistically significant. The Composite Translation Quality Index (CTQI v2, introduced by SignBridge) uses a prerequisite chain formula: CTQI = (GA/100) × (CF1/100) × (0.5 + 0.5 × P/100) × 100, where GA is Gloss Accuracy, CF1 is Coverage F1 (with lemmatization), and P is Plausibility. This multiplicative design ensures that all three dimensions—recognition accuracy, semantic coverage, and translation coherence—must be strong for a high score. CTQI v2 improved from 55.56 to 78.16 (t(33) = 6.403, p < 0.001, Cohen's d = 1.098). Overall, 88.2% of test entries (30/34) showed improvement in CTQI, demonstrating consistent gains across the evaluation dataset.

## Summary Table

| Metric | Baseline | Model | Improvement | p-value | Cohen's d |
|--------|----------|-------|-------------|---------|-----------|
| Coverage F1 | 74.64 | 87.62 | +12.98 | 2.18e-05*** | 0.848 |
| Quality Score | 39.38 | 74.56 | +35.18 | 1.25e-07*** | 1.149 |
| CTQI v2 | 55.56 | 78.16 | +22.60 | 2.96e-07*** | 1.098 |
| BLEU Score | 20.62 | 56.53 | +35.91 | 9.04e-06*** | 0.899 |
| BERTScore | 91.64 | 96.30 | +4.65 | 1.47e-06*** | 1.004 |

Significance levels: \*p < 0.05, \*\*p < 0.01, \*\*\*p < 0.001

*Note: All metrics show statistically significant improvements with large effect sizes (d > 0.8).*

## CTQI v2 Formula (Prerequisite Chain)

The Composite Translation Quality Index v2 (CTQI v2) is calculated as:

```
CTQI = (GA/100) × (CF1/100) × (0.5 + 0.5 × P/100) × 100
```

Where:
- **GA (Gloss Accuracy)**: Sign recognition accuracy (continuous, 0-100)
- **CF1 (Coverage F1)**: Semantic content coverage with lemmatization (continuous, 0-100)
- **P (Plausibility)**: Grammar + semantic sense + naturalness (continuous, 0-100)

The multiplicative design creates a prerequisite chain: accuracy gates meaning, meaning gates fluency. Both GA and CF1 must be high or the score collapses. Plausibility acts as a modifier (0.5x to 1.0x) rather than a gate, so even unreadable but correct output retains value.
