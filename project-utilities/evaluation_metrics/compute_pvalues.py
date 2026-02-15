#!/usr/bin/env python3
"""
Statistical significance analysis for ASL gloss-to-sentence translation evaluation.
Computes p-values using paired t-tests and Wilcoxon signed-rank tests.

This file consolidates all statistical analysis for the research paper.

Usage:
    python compute_pvalues.py
    python compute_pvalues.py --input path/to/evaluation_results.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from scipy import stats

# Default input file path
DEFAULT_INPUT = Path(__file__).parent / "synthetic_evaluation" / "evaluation_results" / "evaluation_results.json"

# CTQI v2 prerequisite chain formula
def _compute_ctqi_v2_chain(ga, cf1, plausibility):
    """Compute CTQI v2 using prerequisite chain: (GA/100) × (CF1/100) × (0.5 + 0.5 × P/100) × 100"""
    ga_clamped = max(0.0, min(100.0, ga)) / 100.0
    cf1_clamped = max(0.0, min(100.0, cf1)) / 100.0
    p_clamped = max(0.0, min(100.0, plausibility)) / 100.0
    plausibility_modifier = 0.5 + 0.5 * p_clamped
    return ga_clamped * cf1_clamped * plausibility_modifier * 100.0

# CTQI v2 geometric mean (legacy)
CTQI_V2_EPSILON = 1.0  # Floor to prevent zero-product collapse

def _compute_ctqi_v2_geomean(ga, q, cf1, epsilon=CTQI_V2_EPSILON):
    """Compute CTQI v2 geometric mean (legacy). Kept for comparison."""
    import math
    ga_safe = max(ga, epsilon) / 100.0
    q_safe = max(q, epsilon) / 100.0
    cf1_safe = max(cf1, epsilon) / 100.0
    w = 1/3  # Equal weights
    return math.exp(w * math.log(ga_safe) + w * math.log(q_safe) + w * math.log(cf1_safe)) * 100.0


def load_evaluation_data(json_path):
    """
    Load evaluation data from JSON file and extract metrics.

    Args:
        json_path: Path to evaluation_results.json

    Returns:
        Dictionary with baseline and model metric arrays
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    n = len(data)

    # Initialize arrays
    metrics = {
        'baseline_bleu': [],
        'model_bleu': [],
        'baseline_bert': [],
        'model_bert': [],
        'baseline_quality': [],
        'model_quality': [],
        'baseline_f1': [],
        'model_f1': [],
        'baseline_gloss_acc': [],
        'model_gloss_acc': [],
        'baseline_ptr': [],
        'model_ptr': [],
        'baseline_ctqi': [],
        'model_ctqi': [],
        'baseline_ctqi_v2': [],
        'model_ctqi_v2': [],
    }

    for entry in data:
        # BLEU
        metrics['baseline_bleu'].append(entry['baseline_bleu'])
        metrics['model_bleu'].append(entry['model_bleu'])

        # BERTScore
        metrics['baseline_bert'].append(entry['baseline_bertscore'])
        metrics['model_bert'].append(entry['model_bertscore'])

        # Quality Score
        metrics['baseline_quality'].append(entry['baseline_quality'])
        metrics['model_quality'].append(entry['model_quality'])

        # Coverage F1
        metrics['baseline_f1'].append(entry['baseline_coverage_f1'])
        metrics['model_f1'].append(entry['model_coverage_f1'])

        # Gloss Accuracy (as percentage)
        baseline_gloss = entry['gloss_accuracy']  # Already percentage
        model_gloss = entry['effective_gloss_accuracy']  # Already percentage
        metrics['baseline_gloss_acc'].append(baseline_gloss)
        metrics['model_gloss_acc'].append(model_gloss)

        # Perfect Translation Rate (PTR) - binary: 100 if all glosses correct, 0 otherwise
        baseline_ptr = 100.0 if entry['gloss_correct'] == entry['gloss_total'] else 0.0
        model_ptr = 100.0 if entry['effective_gloss_correct'] == entry['effective_gloss_total'] else 0.0
        metrics['baseline_ptr'].append(baseline_ptr)
        metrics['model_ptr'].append(model_ptr)

        # Calculate CTQI v2 (prerequisite chain: GA × CF1 × plausibility modifier)
        baseline_ctqi = _compute_ctqi_v2_chain(
            baseline_gloss, entry['baseline_coverage_f1'], entry['baseline_quality']
        )
        model_ctqi = _compute_ctqi_v2_chain(
            model_gloss, entry['model_coverage_f1'], entry['model_quality']
        )
        metrics['baseline_ctqi'].append(baseline_ctqi)
        metrics['model_ctqi'].append(model_ctqi)

        # Calculate CTQI v2 geometric mean (legacy, for comparison)
        baseline_ctqi_v2 = _compute_ctqi_v2_geomean(
            baseline_gloss, entry['baseline_quality'], entry['baseline_coverage_f1']
        )
        model_ctqi_v2 = _compute_ctqi_v2_geomean(
            model_gloss, entry['model_quality'], entry['model_coverage_f1']
        )
        metrics['baseline_ctqi_v2'].append(baseline_ctqi_v2)
        metrics['model_ctqi_v2'].append(model_ctqi_v2)

    metrics['n'] = n
    return metrics


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
    try:
        w_stat, p_wilcox = stats.wilcoxon(model, baseline, alternative='two-sided')
    except ValueError:
        # All differences are zero
        w_stat, p_wilcox = 0, 1.0

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


def count_improved(baseline, model):
    """Count entries where model > baseline."""
    return sum(1 for b, m in zip(baseline, model) if m > b)


def run_synthetic_evaluation_analysis(metrics):
    """Run statistical analysis on synthetic evaluation data."""
    n = metrics['n']

    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS - SYNTHETIC EVALUATION")
    print("=" * 80)
    print(f"Sample size: n = {n} sentence pairs")
    print()

    print("-" * 80)
    print("1. BLEU SCORE IMPROVEMENT")
    print("-" * 80)
    p1_t, p1_w, d1, b1, m1 = analyze_metric("BLEU Score", metrics['baseline_bleu'], metrics['model_bleu'], n)

    print("-" * 80)
    print("2. BERTScore IMPROVEMENT")
    print("-" * 80)
    p2_t, p2_w, d2, b2, m2 = analyze_metric("BERTScore", metrics['baseline_bert'], metrics['model_bert'], n)

    print("-" * 80)
    print("3. QUALITY SCORE IMPROVEMENT")
    print("-" * 80)
    p3_t, p3_w, d3, b3, m3 = analyze_metric("Quality Score", metrics['baseline_quality'], metrics['model_quality'], n)

    print("-" * 80)
    print("4. CTQI v2 (PREREQUISITE CHAIN) IMPROVEMENT")
    print("-" * 80)
    p4_t, p4_w, d4, b4, m4 = analyze_metric("CTQI v2", metrics['baseline_ctqi'], metrics['model_ctqi'], n)

    print("-" * 80)
    print("4b. CTQI V2 GEOMETRIC MEAN (LEGACY) IMPROVEMENT")
    print("-" * 80)
    p4b_t, p4b_w, d4b, b4b, m4b = analyze_metric("CTQI v2 Geomean", metrics['baseline_ctqi_v2'], metrics['model_ctqi_v2'], n)

    print("-" * 80)
    print("5. COVERAGE F1 IMPROVEMENT")
    print("-" * 80)
    p5_t, p5_w, d5, b5, m5 = analyze_metric("Coverage F1", metrics['baseline_f1'], metrics['model_f1'], n)

    print("-" * 80)
    print("6. GLOSS ACCURACY IMPROVEMENT")
    print("-" * 80)
    p6_t, p6_w, d6, b6, m6 = analyze_metric("Gloss Accuracy", metrics['baseline_gloss_acc'], metrics['model_gloss_acc'], n)

    # Summary table
    print("=" * 80)
    print(f"SUMMARY TABLE FOR RESEARCH PAPER")
    print("=" * 80)
    print()
    print(f"Table: Statistical Significance of System Improvements (n={n})")
    print("-" * 80)
    header = f"{'Metric':<20} {'Baseline':<12} {'Model':<12} {'Improvement':<12} {'p-value':<14} {'d':<8}"
    print(header)
    print("-" * 80)

    results = [
        ("BLEU Score", b1, m1, p1_t, d1),
        ("BERTScore", b2, m2, p2_t, d2),
        ("Quality Score", b3, m3, p3_t, d3),
        ("CTQI v2", b4, m4, p4_t, d4),
        ("CTQI v2 (geomean)", b4b, m4b, p4b_t, d4b),
        ("Coverage F1", b5, m5, p5_t, d5),
        ("Gloss Accuracy (%)", b6, m6, p6_t, d6),
    ]

    for name, base, model, p, d in results:
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
    p_values = [p1_t, p2_t, p3_t, p4_t, p4b_t, p5_t, p6_t]
    bonferroni_threshold = 0.05 / len(p_values)
    print(f"Number of tests: {len(p_values)}")
    print(f"Bonferroni-corrected alpha: {bonferroni_threshold:.4f}")
    print()
    for name, p in zip(["BLEU", "BERTScore", "Quality", "CTQI v2", "CTQI v2 Geomean", "Coverage F1", "Gloss Accuracy"], p_values):
        sig = "SIGNIFICANT" if p < bonferroni_threshold else "not significant"
        print(f"  {name}: p = {p:.2e} -> {sig}")

    print()
    print("=" * 80)
    print("LATEX TABLE FOR PAPER")
    print("=" * 80)
    print(r"""
\begin{table}[h]
\centering
\caption{Statistical significance of improvements from baseline gloss concatenation to the proposed two-stage LLM pipeline (n=""" + str(n) + r""" sentence pairs).}
\label{tab:statistical_significance}
\begin{tabular}{lcccccc}
\toprule
\textbf{Metric} & \textbf{Baseline} & \textbf{Model} & \textbf{$\Delta$} & \textbf{p-value} & \textbf{Cohen's d} \\
\midrule""")

    for name, base, model, p, d in results:
        delta = model - base
        sig = "^{***}" if p < 0.001 else "^{**}" if p < 0.01 else "^{*}" if p < 0.05 else ""
        latex_name = name.replace("%", r"\%")
        print(f"{latex_name} & {base:.2f} & {model:.2f} & +{delta:.2f} & ${p:.2e}${sig} & {d:.3f} \\\\")

    print(r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Significance levels: $^{*}p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$
\item All tests are paired t-tests with """ + str(n-1) + r""" degrees of freedom
\end{tablenotes}
\end{table}
""")

    # Proportion of improved entries (calculated from data)
    print("=" * 80)
    print("PROPORTION OF IMPROVED ENTRIES (Binomial Tests)")
    print("=" * 80)
    print()

    bleu_improved = count_improved(metrics['baseline_bleu'], metrics['model_bleu'])
    bert_improved = count_improved(metrics['baseline_bert'], metrics['model_bert'])
    quality_improved = count_improved(metrics['baseline_quality'], metrics['model_quality'])
    ctqi_improved = count_improved(metrics['baseline_ctqi'], metrics['model_ctqi'])
    ctqi_v2_improved = count_improved(metrics['baseline_ctqi_v2'], metrics['model_ctqi_v2'])
    f1_improved = count_improved(metrics['baseline_f1'], metrics['model_f1'])

    binomial_test(bleu_improved, n, "BLEU Score")
    binomial_test(bert_improved, n, "BERTScore")
    binomial_test(quality_improved, n, "Quality Score")
    binomial_test(ctqi_improved, n, "CTQI v2")
    binomial_test(ctqi_v2_improved, n, "CTQI v2 Geomean")
    binomial_test(f1_improved, n, "Coverage F1")

    # PTR improvement
    baseline_perfect = sum(1 for p in metrics['baseline_ptr'] if p == 100.0)
    model_perfect = sum(1 for p in metrics['model_ptr'] if p == 100.0)

    print()
    print("Perfect Translation Rate (PTR) - Binary Metric:")
    print(f"  Baseline: {baseline_perfect}/{n} ({100*baseline_perfect/n:.1f}%) perfect translations")
    print(f"  Model:    {model_perfect}/{n} ({100*model_perfect/n:.1f}%) perfect translations")
    print(f"  Improvement: +{model_perfect - baseline_perfect} entries achieving perfect translation")

    # McNemar's test for paired nominal data
    # Count transitions
    both_perfect = sum(1 for b, m in zip(metrics['baseline_ptr'], metrics['model_ptr']) if b == 100.0 and m == 100.0)
    baseline_only = sum(1 for b, m in zip(metrics['baseline_ptr'], metrics['model_ptr']) if b == 100.0 and m == 0.0)
    model_only = sum(1 for b, m in zip(metrics['baseline_ptr'], metrics['model_ptr']) if b == 0.0 and m == 100.0)
    neither = sum(1 for b, m in zip(metrics['baseline_ptr'], metrics['model_ptr']) if b == 0.0 and m == 0.0)

    print(f"  Transitions: {model_only} improved (0->100), {baseline_only} regressed (100->0)")

    # McNemar's test using discordant pairs
    discordant_total = model_only + baseline_only
    if discordant_total > 0:
        mcnemar_result = stats.binomtest(model_only, discordant_total, 0.5, alternative='two-sided')
        print(f"  McNemar's test (exact): p = {mcnemar_result.pvalue:.4f}")
    else:
        print("  McNemar's test: No discordant pairs")

    return results


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


def main():
    parser = argparse.ArgumentParser(
        description="Statistical significance analysis for ASL translation evaluation"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to evaluation_results.json (default: {DEFAULT_INPUT})"
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        return 1

    print(f"Loading data from: {args.input}")
    print()

    metrics = load_evaluation_data(args.input)
    run_synthetic_evaluation_analysis(metrics)
    # run_training_analysis()  # Uncomment when data is added

    return 0


if __name__ == "__main__":
    exit(main())
