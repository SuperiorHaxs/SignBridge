#!/usr/bin/env python3
"""
Statistical significance analysis for ASL gloss-to-sentence translation evaluation.
Computes p-values using paired t-tests and Wilcoxon signed-rank tests.

This file consolidates all statistical analysis for the research paper.
"""

import numpy as np
from scipy import stats

# =============================================================================
# SYNTHETIC EVALUATION DATA (Gloss-to-Sentence Translation)
# =============================================================================
# Per-entry data extracted from evaluation_report_best.txt
# n = 34 sentence pairs

# BLEU Scores
baseline_bleu = [24.84, 24.84, 19.72, 19.72, 12.75, 12.75, 55.03, 16.70, 9.93, 9.14,
                 30.33, 14.60, 27.53, 12.75, 19.72, 7.25, 34.67, 17.80, 14.60, 17.80,
                 17.80, 17.80, 17.80, 17.80, 14.60, 14.60, 19.72, 19.72, 32.34, 17.80,
                 10.46, 14.79, 28.44, 55.03]

model_bleu = [100.00, 100.00, 10.68, 15.97, 20.80, 100.00, 55.03, 84.09, 32.16, 67.03,
              100.00, 100.00, 66.87, 32.56, 10.68, 8.97, 34.67, 100.00, 100.00, 10.68,
              100.00, 16.37, 32.47, 100.00, 100.00, 100.00, 19.00, 19.72, 100.00, 14.79,
              33.52, 26.27, 100.00, 9.65]

# BERTScore
baseline_bert = [89.07, 94.97, 93.14, 94.52, 91.55, 88.31, 95.96, 89.04, 88.84, 89.94,
                 94.76, 89.85, 95.90, 88.14, 94.10, 89.21, 94.82, 88.65, 88.94, 91.59,
                 91.67, 94.55, 88.30, 90.26, 89.35, 91.05, 96.36, 90.89, 95.91, 89.19,
                 89.53, 89.79, 91.68, 96.01]

model_bert = [100.00, 100.00, 93.71, 90.13, 91.31, 100.00, 96.77, 98.04, 94.76, 95.53,
              100.00, 100.00, 99.91, 90.85, 94.37, 88.52, 95.68, 100.00, 100.00, 96.09,
              100.00, 97.14, 92.99, 97.77, 100.00, 100.00, 95.13, 90.75, 100.00, 95.21,
              95.08, 94.84, 100.00, 89.53]

# Quality Score
baseline_quality = [69.56, 58.19, 58.62, 53.83, 88.85, 19.19, 59.91, 5.49, 30.22, 58.91,
                    44.80, 46.54, 28.47, 54.65, 19.70, 36.94, 74.94, 7.39, 14.45, 9.01,
                    59.78, 12.81, 42.61, 47.80, 16.87, 5.99, 70.82, 52.55, 45.10, 0.00,
                    26.08, 32.17, 20.63, 66.10]

model_quality = [64.85, 86.56, 55.96, 98.17, 89.75, 99.28, 59.79, 80.64, 74.30, 88.38,
                 79.08, 90.18, 34.37, 95.93, 75.63, 40.69, 30.24, 83.30, 76.96, 83.84,
                 74.50, 66.93, 83.67, 83.91, 79.08, 91.57, 74.35, 41.16, 97.10, 56.14,
                 50.48, 89.22, 69.31, 89.65]

# Perfect Translation Rate (PTR) - binary: 100 if all glosses correct, 0 otherwise
# Baseline PTR (from top-1 predictions)
baseline_ptr = [0, 100, 0, 0, 0, 0, 100, 0, 0, 0,
                0, 100, 100, 0, 0, 0, 100, 0, 100, 100,
                0, 0, 0, 0, 100, 100, 0, 0, 100, 100,
                100, 0, 100, 100]  # 14/34 perfect

# Model PTR (from LLM-selected glosses)
model_ptr = [100, 100, 0, 0, 0, 100, 100, 100, 100, 100,
             100, 100, 100, 0, 0, 0, 100, 100, 100, 100,
             100, 0, 0, 100, 100, 100, 0, 0, 100, 100,
             100, 0, 100, 100]  # 23/34 perfect

# CTQI (Composite Translation Quality Index)
# Formula: 0.4 * gloss_accuracy + 0.4 * quality + 0.2 * PTR
# Calculated from gloss_acc (as %), quality, and PTR
baseline_ctqi = [54.50, 83.28, 50.11, 48.20, 62.21, 34.35, 83.96, 44.20, 52.09, 50.23,
                 37.92, 76.61, 71.19, 48.53, 34.55, 29.44, 89.98, 29.62, 65.78, 63.60,
                 50.58, 31.79, 43.71, 45.79, 66.75, 62.40, 56.59, 47.69, 71.04, 60.00,
                 70.43, 32.87, 68.25, 86.44]

model_ctqi = [85.94, 94.62, 49.05, 65.94, 62.57, 99.71, 83.92, 92.26, 89.72, 95.35,
              91.63, 96.07, 73.75, 65.04, 56.92, 42.94, 72.10, 93.32, 90.78, 93.54,
              89.80, 40.11, 60.14, 93.56, 91.63, 96.63, 56.41, 43.13, 98.84, 82.46,
              80.19, 45.69, 87.72, 95.86]

# Gloss Accuracy (Top-1 vs LLM-selected from Top-K)
baseline_gloss_acc = [2/3, 3/3, 2/3, 2/3, 2/3, 2/3, 3/3, 3/4, 3/4, 2/3,
                      1/2, 3/3, 4/4, 2/3, 2/3, 1/3, 3/3, 2/3, 3/3, 3/3,
                      2/3, 2/3, 2/3, 2/3, 3/3, 3/3, 2/3, 2/3, 3/3, 3/3,
                      3/3, 2/4, 3/3, 3/3]

model_gloss_acc = [3/3, 3/3, 2/3, 2/3, 2/3, 3/3, 3/3, 4/4, 4/4, 3/3,
                   2/2, 3/3, 4/4, 2/3, 2/3, 2/3, 3/3, 3/3, 3/3, 3/3,
                   3/3, 1/3, 2/3, 3/3, 3/3, 3/3, 2/3, 2/3, 3/3, 3/3,
                   3/3, 1/4, 3/3, 3/3]

# Coverage F1 Scores
baseline_f1 = [80.0, 85.7, 57.1, 66.7, 80.0, 66.7, 100.0, 75.0, 85.7, 40.0,
               50.0, 85.7, 88.9, 66.7, 57.1, 0.0, 100.0, 66.7, 100.0, 85.7,
               66.7, 57.1, 66.7, 66.7, 85.7, 85.7, 66.7, 66.7, 85.7, 85.7,
               100.0, 66.7, 100.0, 100.0]

model_f1 = [100.0, 100.0, 75.0, 66.7, 80.0, 100.0, 100.0, 88.9, 100.0, 80.0,
            100.0, 100.0, 88.9, 57.1, 75.0, 40.0, 100.0, 100.0, 100.0, 100.0,
            100.0, 66.7, 66.7, 100.0, 100.0, 100.0, 66.7, 66.7, 100.0, 75.0,
            100.0, 85.7, 100.0, 100.0]


def analyze_metric(name, baseline, model, n):
    """Perform paired statistical tests and compute effect size."""
    baseline = np.array(baseline)
    model = np.array(model)
    diff = model - baseline

    mean_baseline = np.mean(baseline)
    mean_model = np.mean(model)
    mean_improvement = np.mean(diff)
    std_improvement = np.std(diff, ddof=1)

    # Paired t-test
    t_stat, p_ttest = stats.ttest_rel(model, baseline)

    # Wilcoxon signed-rank test (non-parametric)
    w_stat, p_wilcox = stats.wilcoxon(model, baseline, alternative='two-sided')

    # Effect size (Cohen's d for paired samples)
    cohens_d = mean_improvement / std_improvement if std_improvement > 0 else 0

    # 95% CI for mean difference
    se = std_improvement / np.sqrt(n)
    ci_low = mean_improvement - 1.96 * se
    ci_high = mean_improvement + 1.96 * se

    print(f"{name}:")
    print(f"  Baseline Mean: {mean_baseline:.2f}")
    print(f"  Model Mean: {mean_model:.2f}")
    print(f"  Mean Improvement: +{mean_improvement:.2f} (95% CI: [{ci_low:.2f}, {ci_high:.2f}])")
    print(f"  Paired t-test: t({n-1}) = {t_stat:.3f}, p = {p_ttest:.2e}")
    print(f"  Wilcoxon signed-rank: W = {w_stat:.0f}, p = {p_wilcox:.2e}")
    effect_label = "negligible" if abs(cohens_d) < 0.2 else "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
    print(f"  Effect size (Cohen's d): {cohens_d:.3f} ({effect_label})")
    print()

    return p_ttest, p_wilcox, cohens_d, mean_baseline, mean_model


def binomial_test(improved, total, metric_name):
    """Test if proportion improved is significantly greater than 50%."""
    result = stats.binomtest(improved, total, p=0.5, alternative='greater')
    print(f"{metric_name}: {improved}/{total} ({100*improved/total:.1f}%) improved")
    print(f"  Binomial test (H0: p=0.5): p = {result.pvalue:.4f}")
    return result.pvalue


def run_synthetic_evaluation_analysis():
    """Run statistical analysis on synthetic evaluation data."""
    n = len(baseline_bleu)

    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS - SYNTHETIC EVALUATION")
    print("=" * 80)
    print(f"Sample size: n = {n} sentence pairs")
    print()

    print("-" * 80)
    print("1. BLEU SCORE IMPROVEMENT")
    print("-" * 80)
    p1_t, p1_w, d1, b1, m1 = analyze_metric("BLEU Score", baseline_bleu, model_bleu, n)

    print("-" * 80)
    print("2. BERTScore IMPROVEMENT")
    print("-" * 80)
    p2_t, p2_w, d2, b2, m2 = analyze_metric("BERTScore", baseline_bert, model_bert, n)

    print("-" * 80)
    print("3. QUALITY SCORE IMPROVEMENT")
    print("-" * 80)
    p3_t, p3_w, d3, b3, m3 = analyze_metric("Quality Score", baseline_quality, model_quality, n)

    print("-" * 80)
    print("4. CTQI (COMPOSITE TRANSLATION QUALITY INDEX) IMPROVEMENT")
    print("-" * 80)
    p4_t, p4_w, d4, b4, m4 = analyze_metric("CTQI", baseline_ctqi, model_ctqi, n)

    print("-" * 80)
    print("5. COVERAGE F1 IMPROVEMENT")
    print("-" * 80)
    p5_t, p5_w, d5, b5, m5 = analyze_metric("Coverage F1", baseline_f1, model_f1, n)

    print("-" * 80)
    print("6. GLOSS ACCURACY IMPROVEMENT")
    print("-" * 80)
    p6_t, p6_w, d6, b6, m6 = analyze_metric("Gloss Accuracy", [x*100 for x in baseline_gloss_acc], [x*100 for x in model_gloss_acc], n)

    # Summary table
    print("=" * 80)
    print("SUMMARY TABLE FOR RESEARCH PAPER")
    print("=" * 80)
    print()
    print("Table: Statistical Significance of System Improvements (n=34)")
    print("-" * 80)
    header = f"{'Metric':<20} {'Baseline':<12} {'Model':<12} {'Improvement':<12} {'p-value':<14} {'d':<8}"
    print(header)
    print("-" * 80)

    metrics = [
        ("BLEU Score", b1, m1, p1_t, d1),
        ("BERTScore", b2, m2, p2_t, d2),
        ("Quality Score", b3, m3, p3_t, d3),
        ("CTQI", b4, m4, p4_t, d4),
        ("Coverage F1", b5, m5, p5_t, d5),
        ("Gloss Accuracy (%)", b6, m6, p6_t, d6),
    ]

    for name, base, model, p, d in metrics:
        delta = model - base
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{name:<20} {base:<12.2f} {model:<12.2f} +{delta:<11.2f} {p:<14.2e} {d:<8.3f}{sig}")

    print("-" * 80)
    print("Significance levels: * p < 0.05, ** p < 0.01, *** p < 0.001")
    print()

    # Bonferroni correction
    print("=" * 80)
    print("BONFERRONI CORRECTION (for multiple comparisons)")
    print("=" * 80)
    p_values = [p1_t, p2_t, p3_t, p4_t, p5_t, p6_t]
    bonferroni_threshold = 0.05 / len(p_values)
    print(f"Number of tests: {len(p_values)}")
    print(f"Bonferroni-corrected alpha: {bonferroni_threshold:.4f}")
    print()
    for name, p in zip(["BLEU", "BERTScore", "Quality", "CTQI", "Coverage F1", "Gloss Accuracy"], p_values):
        sig = "SIGNIFICANT" if p < bonferroni_threshold else "not significant"
        print(f"  {name}: p = {p:.2e} -> {sig}")

    print()
    print("=" * 80)
    print("LATEX TABLE FOR PAPER")
    print("=" * 80)
    print(r"""
\begin{table}[h]
\centering
\caption{Statistical significance of improvements from baseline gloss concatenation to the proposed two-stage LLM pipeline (n=34 sentence pairs).}
\label{tab:statistical_significance}
\begin{tabular}{lcccccc}
\toprule
\textbf{Metric} & \textbf{Baseline} & \textbf{Model} & \textbf{$\Delta$} & \textbf{p-value} & \textbf{Cohen's d} \\
\midrule""")

    for name, base, model, p, d in metrics:
        delta = model - base
        sig = "^{***}" if p < 0.001 else "^{**}" if p < 0.01 else "^{*}" if p < 0.05 else ""
        latex_name = name.replace("%", r"\%")
        print(f"{latex_name} & {base:.2f} & {model:.2f} & +{delta:.2f} & ${p:.2e}${sig} & {d:.3f} \\\\")

    print(r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Significance levels: $^{*}p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$
\item All tests are paired t-tests with 33 degrees of freedom
\end{tablenotes}
\end{table}
""")

    # Additional chi-squared test for proportion of improved entries
    print("=" * 80)
    print("PROPORTION OF IMPROVED ENTRIES (Binomial Tests)")
    print("=" * 80)
    print()
    # From the report - CTQI improved in 29/34 entries
    binomial_test(23, 34, "BLEU Score")       # 67.6%
    binomial_test(28, 34, "BERTScore")        # 82.4%
    binomial_test(29, 34, "Quality Score")    # 85.3%
    binomial_test(29, 34, "CTQI")             # 85.3%
    binomial_test(20, 34, "Coverage Recall")  # 58.8%

    # PTR improvement: 14 -> 23 perfect translations (+9)
    print()
    print("Perfect Translation Rate (PTR) - Binary Metric:")
    print(f"  Baseline: 14/34 (41.2%) perfect translations")
    print(f"  Model:    23/34 (67.6%) perfect translations")
    print(f"  Improvement: +9 entries achieving perfect translation")
    # McNemar's test for paired nominal data
    # Table:
    #                 Model Correct  Model Incorrect
    # Baseline Correct     14           0
    # Baseline Incorrect    9           11
    # Only cases where baseline and model disagree matter
    # 9 changes from incorrect->correct, 0 changes from correct->incorrect
    # Under null, changes should be equally likely in both directions
    mcnemar_result = stats.binomtest(9, 9, 0.5, alternative='two-sided')
    print(f"  McNemar's test (exact): p = {mcnemar_result.pvalue:.4f}")

    return metrics


# =============================================================================
# TRAINING-RELATED P-VALUES (placeholder for future additions)
# =============================================================================
# Add training comparison data here (e.g., comparing different model architectures,
# hyperparameter configurations, etc.)

def run_training_analysis():
    """Placeholder for training-related statistical analysis."""
    print("=" * 80)
    print("TRAINING-RELATED STATISTICAL ANALYSIS")
    print("=" * 80)
    print("(Add training comparison data here)")
    print()


if __name__ == "__main__":
    run_synthetic_evaluation_analysis()
    # run_training_analysis()  # Uncomment when data is added
