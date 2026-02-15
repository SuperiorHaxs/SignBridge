#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ctqi_v2_analysis.py - CTQI v2 Analysis Toolkit

Provides advanced analysis for the CTQI v2 (Composite Translation Quality Index v2):
- PCA-based weight derivation (data-driven weights)
- Bootstrap confidence intervals (BCa method)
- Ablation study (component necessity)
- Human correlation analysis framework
- V1 vs V2 comparison

Usage:
    python ctqi_v2_analysis.py
    python ctqi_v2_analysis.py --input path/to/evaluation_results.json
    python ctqi_v2_analysis.py --human-ratings path/to/human_ratings.json
"""

import argparse
import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from scipy import stats

# Default paths
DEFAULT_INPUT = Path(__file__).parent / "synthetic_evaluation" / "evaluation_results" / "evaluation_results.json"

# CTQI v2 configuration
CTQI_V2_EPSILON = 1.0  # Score floor to prevent zero-product collapse
DEFAULT_WEIGHTS = {'gloss_accuracy': 1/3, 'quality': 1/3, 'coverage_f1': 1/3}

# CTQI v1 weights (for comparison)
CTQI_V1_WEIGHTS = {'gloss_accuracy': 0.4, 'quality': 0.4, 'perfect_translation_rate': 0.2}

COMPONENT_NAMES = ['gloss_accuracy', 'quality', 'coverage_f1']


# =============================================================================
# CORE COMPUTATION
# =============================================================================

def compute_ctqi_v2(
    gloss_accuracy: float,
    quality: float,
    coverage_f1: float,
    weights: Optional[Dict[str, float]] = None,
    epsilon: float = CTQI_V2_EPSILON
) -> float:
    """
    Compute a single CTQI v2 score using weighted geometric mean.

    Args:
        gloss_accuracy: Gloss accuracy (0-100)
        quality: Quality score (0-100)
        coverage_f1: Coverage F1 (0-100)
        weights: Dict with keys 'gloss_accuracy', 'quality', 'coverage_f1' summing to 1.0
        epsilon: Floor value to prevent log(0)

    Returns:
        CTQI v2 score (0-100)
    """
    w = weights or DEFAULT_WEIGHTS
    ga = max(gloss_accuracy, epsilon) / 100.0
    q = max(quality, epsilon) / 100.0
    cf1 = max(coverage_f1, epsilon) / 100.0
    log_sum = (w['gloss_accuracy'] * math.log(ga) +
               w['quality'] * math.log(q) +
               w['coverage_f1'] * math.log(cf1))
    return math.exp(log_sum) * 100.0


def compute_ctqi_v1(gloss_accuracy: float, quality: float, ptr: float) -> float:
    """Compute CTQI v1 score (weighted arithmetic mean)."""
    return (CTQI_V1_WEIGHTS['gloss_accuracy'] * gloss_accuracy +
            CTQI_V1_WEIGHTS['quality'] * quality +
            CTQI_V1_WEIGHTS['perfect_translation_rate'] * ptr)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_component_arrays(json_path: Path) -> Dict[str, np.ndarray]:
    """
    Load evaluation results and extract component metric arrays.

    Args:
        json_path: Path to evaluation_results.json

    Returns:
        Dict with baseline/model arrays for each component plus PTR and sample size
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    n = len(data)
    baseline_ga, model_ga = [], []
    baseline_q, model_q = [], []
    baseline_cf1, model_cf1 = [], []
    baseline_ptr, model_ptr = [], []

    for entry in data:
        baseline_ga.append(entry['gloss_accuracy'])
        model_ga.append(entry['effective_gloss_accuracy'])
        baseline_q.append(entry['baseline_quality'])
        model_q.append(entry['model_quality'])
        baseline_cf1.append(entry['baseline_coverage_f1'])
        model_cf1.append(entry['model_coverage_f1'])
        # PTR reconstruction
        baseline_ptr.append(100.0 if entry['gloss_correct'] == entry['gloss_total'] else 0.0)
        model_ptr.append(100.0 if entry['effective_gloss_correct'] == entry['effective_gloss_total'] else 0.0)

    return {
        'baseline_ga': np.array(baseline_ga),
        'model_ga': np.array(model_ga),
        'baseline_q': np.array(baseline_q),
        'model_q': np.array(model_q),
        'baseline_cf1': np.array(baseline_cf1),
        'model_cf1': np.array(model_cf1),
        'baseline_ptr': np.array(baseline_ptr),
        'model_ptr': np.array(model_ptr),
        'n': n
    }


# =============================================================================
# PCA WEIGHT DERIVATION
# =============================================================================

def derive_pca_weights(
    evaluation_data: Dict[str, np.ndarray],
    use_model: bool = True
) -> Dict[str, Any]:
    """
    Derive CTQI v2 weights from PCA on the three component metrics.

    Methodology: Standardize the three components, compute the covariance matrix,
    and use the first principal component's loadings (absolute values, normalized
    to sum to 1) as the weights. This identifies the dimension of maximum variance
    in the data, producing empirically grounded weights.

    Args:
        evaluation_data: Dict from load_component_arrays()
        use_model: If True, use model scores; if False, use baseline scores

    Returns:
        Dict with PCA weights, explained variance, loadings, and component names
    """
    prefix = 'model' if use_model else 'baseline'
    ga = evaluation_data[f'{prefix}_ga']
    q = evaluation_data[f'{prefix}_q']
    cf1 = evaluation_data[f'{prefix}_cf1']

    # Build matrix (n x 3)
    X = np.column_stack([ga, q, cf1])

    # Standardize (zero mean, unit variance)
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    # Handle zero std (constant column)
    stds[stds == 0] = 1.0
    X_std = (X - means) / stds

    # Compute correlation matrix and eigen-decomposition
    corr_matrix = np.corrcoef(X_std.T)
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

    # Sort by eigenvalue descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Explained variance ratios
    total_var = eigenvalues.sum()
    explained_variance_ratio = eigenvalues / total_var

    # PC1 loadings -> absolute values -> normalize to sum=1
    pc1_loadings = eigenvectors[:, 0]
    abs_loadings = np.abs(pc1_loadings)
    pca_weights_arr = abs_loadings / abs_loadings.sum()

    pca_weights = {
        'gloss_accuracy': float(pca_weights_arr[0]),
        'quality': float(pca_weights_arr[1]),
        'coverage_f1': float(pca_weights_arr[2]),
    }

    return {
        'pca_weights': pca_weights,
        'explained_variance_ratio': explained_variance_ratio.tolist(),
        'loadings': pc1_loadings.tolist(),
        'component_names': COMPONENT_NAMES,
        'correlation_matrix': corr_matrix.tolist(),
        'eigenvalues': eigenvalues.tolist(),
        'data_source': prefix,
    }


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_ctqi_ci(
    scores: np.ndarray,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval for the mean of CTQI v2 scores.

    Uses the percentile method with bias-correction for small samples (n=34).

    Args:
        scores: Array of CTQI v2 scores
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level (default 0.95 for 95% CI)
        random_seed: Random seed for reproducibility

    Returns:
        Dict with mean, CI bounds, bootstrap SE, and configuration
    """
    rng = np.random.default_rng(random_seed)
    n = len(scores)
    observed_mean = np.mean(scores)

    # Generate bootstrap resamples
    bootstrap_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resample = rng.choice(scores, size=n, replace=True)
        bootstrap_means[i] = np.mean(resample)

    # Bias-corrected percentile method
    # Bias correction factor
    z0 = stats.norm.ppf(np.mean(bootstrap_means < observed_mean))

    # Jackknife for acceleration factor
    jackknife_means = np.empty(n)
    for i in range(n):
        jackknife_sample = np.delete(scores, i)
        jackknife_means[i] = np.mean(jackknife_sample)

    jack_mean = np.mean(jackknife_means)
    numerator = np.sum((jack_mean - jackknife_means) ** 3)
    denominator = 6.0 * (np.sum((jack_mean - jackknife_means) ** 2) ** 1.5)
    a = numerator / denominator if denominator != 0 else 0.0

    # BCa adjusted percentiles
    alpha = 1 - confidence_level
    z_alpha_low = stats.norm.ppf(alpha / 2)
    z_alpha_high = stats.norm.ppf(1 - alpha / 2)

    # Adjusted percentile positions
    p_low = stats.norm.cdf(z0 + (z0 + z_alpha_low) / (1 - a * (z0 + z_alpha_low)))
    p_high = stats.norm.cdf(z0 + (z0 + z_alpha_high) / (1 - a * (z0 + z_alpha_high)))

    # Clamp to valid percentile range
    p_low = max(0.001, min(0.999, p_low))
    p_high = max(0.001, min(0.999, p_high))

    ci_lower = float(np.percentile(bootstrap_means, p_low * 100))
    ci_upper = float(np.percentile(bootstrap_means, p_high * 100))

    return {
        'mean': float(observed_mean),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se_bootstrap': float(np.std(bootstrap_means, ddof=1)),
        'confidence_level': confidence_level,
        'n_bootstrap': n_bootstrap,
        'n_samples': n,
        'bias_correction': float(z0),
        'acceleration': float(a),
    }


# =============================================================================
# ABLATION STUDY
# =============================================================================

def ablation_study(
    evaluation_data: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Ablation study: remove each component and measure discriminative power degradation.

    For each of the 3 components, compute CTQI v2 with the remaining 2 components
    only (re-normalizing their weights). The component whose removal causes the
    largest degradation in Cohen's d (baseline vs model separation) is the most
    necessary.

    Args:
        evaluation_data: Dict from load_component_arrays()
        weights: Optional weights (default: equal 1/3)

    Returns:
        Dict with full model results, per-ablation results, and component ranking
    """
    w = weights or DEFAULT_WEIGHTS
    n = evaluation_data['n']

    # Compute full model CTQI v2
    baseline_full = np.array([
        compute_ctqi_v2(ga, q, cf1, w)
        for ga, q, cf1 in zip(
            evaluation_data['baseline_ga'],
            evaluation_data['baseline_q'],
            evaluation_data['baseline_cf1']
        )
    ])
    model_full = np.array([
        compute_ctqi_v2(ga, q, cf1, w)
        for ga, q, cf1 in zip(
            evaluation_data['model_ga'],
            evaluation_data['model_q'],
            evaluation_data['model_cf1']
        )
    ])
    diff_full = model_full - baseline_full
    full_d = np.mean(diff_full) / np.std(diff_full, ddof=1) if np.std(diff_full, ddof=1) > 0 else 0

    result = {
        'full_model': {
            'cohens_d': float(full_d),
            'mean_improvement': float(np.mean(diff_full)),
            'baseline_mean': float(np.mean(baseline_full)),
            'model_mean': float(np.mean(model_full)),
        },
        'ablations': {},
        'component_necessity_ranking': [],
    }

    # Ablate each component
    for component in COMPONENT_NAMES:
        # Create weights with this component zeroed out, renormalize
        ablated_weights = {k: v for k, v in w.items() if k != component}
        total = sum(ablated_weights.values())
        ablated_weights = {k: v / total for k, v in ablated_weights.items()}
        # Add the removed component with weight 0 (won't be used)
        ablated_weights[component] = 0.0

        # Compute ablated scores using only the two remaining components
        remaining = [c for c in COMPONENT_NAMES if c != component]
        baseline_ablated = np.array([
            _compute_ablated_v2(evaluation_data, i, remaining, ablated_weights, 'baseline')
            for i in range(n)
        ])
        model_ablated = np.array([
            _compute_ablated_v2(evaluation_data, i, remaining, ablated_weights, 'model')
            for i in range(n)
        ])

        diff_ablated = model_ablated - baseline_ablated
        ablated_d = np.mean(diff_ablated) / np.std(diff_ablated, ddof=1) if np.std(diff_ablated, ddof=1) > 0 else 0
        degradation = full_d - ablated_d

        result['ablations'][f'without_{component}'] = {
            'cohens_d': float(ablated_d),
            'mean_improvement': float(np.mean(diff_ablated)),
            'degradation': float(degradation),
            'baseline_mean': float(np.mean(baseline_ablated)),
            'model_mean': float(np.mean(model_ablated)),
        }

    # Rank by degradation (higher = more necessary)
    ranking = sorted(
        [(comp, result['ablations'][f'without_{comp}']['degradation'])
         for comp in COMPONENT_NAMES],
        key=lambda x: x[1],
        reverse=True
    )
    result['component_necessity_ranking'] = ranking

    return result


def _compute_ablated_v2(data, i, remaining, weights, prefix):
    """Compute ablated CTQI v2 for a single entry using only remaining components."""
    component_map = {
        'gloss_accuracy': f'{prefix}_ga',
        'quality': f'{prefix}_q',
        'coverage_f1': f'{prefix}_cf1',
    }
    # Compute geometric mean over remaining components with renormalized weights
    total_w = sum(weights[c] for c in remaining)
    if total_w == 0:
        return 0.0
    log_sum = 0.0
    for comp in remaining:
        val = max(float(data[component_map[comp]][i]), CTQI_V2_EPSILON) / 100.0
        normalized_w = weights[comp] / total_w
        log_sum += normalized_w * math.log(val)
    return math.exp(log_sum) * 100.0


# =============================================================================
# HUMAN CORRELATION ANALYSIS
# =============================================================================

def human_correlation_analysis(
    ctqi_scores: np.ndarray,
    human_ratings: np.ndarray
) -> Dict[str, Any]:
    """
    Correlate CTQI v2 scores with human quality ratings.

    Computes Pearson (linear) and Spearman (rank) correlations to validate
    that CTQI v2 aligns with human judgments of translation quality.

    Args:
        ctqi_scores: Array of CTQI v2 scores
        human_ratings: Array of human quality ratings (same length, averaged across raters)

    Returns:
        Dict with correlation coefficients, p-values, and interpretation
    """
    n = len(ctqi_scores)
    assert len(human_ratings) == n, "Score arrays must have same length"

    pearson_r, pearson_p = stats.pearsonr(ctqi_scores, human_ratings)
    spearman_rho, spearman_p = stats.spearmanr(ctqi_scores, human_ratings)

    # Interpretation of correlation strength
    abs_r = abs(pearson_r)
    if abs_r >= 0.7:
        interpretation = "strong"
    elif abs_r >= 0.5:
        interpretation = "moderate"
    elif abs_r >= 0.3:
        interpretation = "weak"
    else:
        interpretation = "negligible"

    return {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_rho': float(spearman_rho),
        'spearman_p': float(spearman_p),
        'n': n,
        'interpretation': interpretation,
    }


# =============================================================================
# V1 vs V2 COMPARISON
# =============================================================================

def compare_v1_v2(
    evaluation_data: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """
    Side-by-side comparison of CTQI v1 and v2.

    Computes both metrics for all entries and compares discriminative power
    (Cohen's d), rank agreement (Spearman rho), and disagreements.

    Args:
        evaluation_data: Dict from load_component_arrays()

    Returns:
        Dict with v1 stats, v2 stats, rank correlation, and disagreement count
    """
    n = evaluation_data['n']

    # Compute v1 for each entry
    baseline_v1 = np.array([
        compute_ctqi_v1(ga, q, ptr)
        for ga, q, ptr in zip(
            evaluation_data['baseline_ga'],
            evaluation_data['baseline_q'],
            evaluation_data['baseline_ptr']
        )
    ])
    model_v1 = np.array([
        compute_ctqi_v1(ga, q, ptr)
        for ga, q, ptr in zip(
            evaluation_data['model_ga'],
            evaluation_data['model_q'],
            evaluation_data['model_ptr']
        )
    ])

    # Compute v2 for each entry
    baseline_v2 = np.array([
        compute_ctqi_v2(ga, q, cf1)
        for ga, q, cf1 in zip(
            evaluation_data['baseline_ga'],
            evaluation_data['baseline_q'],
            evaluation_data['baseline_cf1']
        )
    ])
    model_v2 = np.array([
        compute_ctqi_v2(ga, q, cf1)
        for ga, q, cf1 in zip(
            evaluation_data['model_ga'],
            evaluation_data['model_q'],
            evaluation_data['model_cf1']
        )
    ])

    # Cohen's d for each version
    diff_v1 = model_v1 - baseline_v1
    diff_v2 = model_v2 - baseline_v2
    d_v1 = np.mean(diff_v1) / np.std(diff_v1, ddof=1) if np.std(diff_v1, ddof=1) > 0 else 0
    d_v2 = np.mean(diff_v2) / np.std(diff_v2, ddof=1) if np.std(diff_v2, ddof=1) > 0 else 0

    # Rank correlation between v1 and v2 model scores
    rho, rho_p = stats.spearmanr(model_v1, model_v2)

    # Count disagreements on improvement direction
    improvements_v1 = diff_v1 > 0
    improvements_v2 = diff_v2 > 0
    disagreements = int(np.sum(improvements_v1 != improvements_v2))

    # Paired t-test comparing v1 and v2 model scores
    t_stat, t_p = stats.ttest_rel(model_v1, model_v2)

    return {
        'v1': {
            'baseline_mean': float(np.mean(baseline_v1)),
            'baseline_std': float(np.std(baseline_v1, ddof=1)),
            'model_mean': float(np.mean(model_v1)),
            'model_std': float(np.std(model_v1, ddof=1)),
            'mean_improvement': float(np.mean(diff_v1)),
            'cohens_d': float(d_v1),
            'pct_improved': float(np.mean(improvements_v1) * 100),
        },
        'v2': {
            'baseline_mean': float(np.mean(baseline_v2)),
            'baseline_std': float(np.std(baseline_v2, ddof=1)),
            'model_mean': float(np.mean(model_v2)),
            'model_std': float(np.std(model_v2, ddof=1)),
            'mean_improvement': float(np.mean(diff_v2)),
            'cohens_d': float(d_v2),
            'pct_improved': float(np.mean(improvements_v2) * 100),
        },
        'rank_correlation': {
            'spearman_rho': float(rho),
            'p_value': float(rho_p),
        },
        'disagreements': disagreements,
        'n': n,
        'paired_ttest': {
            't_stat': float(t_stat),
            'p_value': float(t_p),
        },
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CTQI v2 Analysis Toolkit for ASL Translation Evaluation"
    )
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT,
                        help=f"Path to evaluation_results.json (default: {DEFAULT_INPUT})")
    parser.add_argument("--human-ratings", type=Path, default=None,
                        help="Path to human ratings JSON file (optional)")
    parser.add_argument("--n-bootstrap", type=int, default=10000,
                        help="Number of bootstrap resamples (default: 10000)")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        return 1

    print(f"Loading data from: {args.input}")
    data = load_component_arrays(args.input)
    n = data['n']
    print(f"Loaded {n} evaluation entries\n")

    # =========================================================================
    # 1. PCA WEIGHT DERIVATION
    # =========================================================================
    print("=" * 80)
    print("1. PCA WEIGHT DERIVATION")
    print("=" * 80)
    pca_result = derive_pca_weights(data, use_model=True)

    print(f"\nData source: {pca_result['data_source']} scores (n={n})")
    print(f"\nCorrelation Matrix:")
    corr = np.array(pca_result['correlation_matrix'])
    print(f"  {'':>20} {'Gloss Acc':>12} {'Quality':>12} {'Coverage F1':>12}")
    for i, name in enumerate(COMPONENT_NAMES):
        row = '  '.join(f"{corr[i][j]:>12.3f}" for j in range(3))
        print(f"  {name:>20} {row}")

    print(f"\nEigenvalues: {', '.join(f'{e:.3f}' for e in pca_result['eigenvalues'])}")
    print(f"Explained variance: {', '.join(f'{e:.1%}' for e in pca_result['explained_variance_ratio'])}")
    print(f"PC1 loadings: {', '.join(f'{l:.3f}' for l in pca_result['loadings'])}")

    print(f"\n  PCA-Derived Weights:")
    for name in COMPONENT_NAMES:
        w = pca_result['pca_weights'][name]
        print(f"    {name}: {w:.3f} ({w:.1%})")

    print(f"\n  Comparison with defaults:")
    print(f"    {'Component':>20} {'Default':>10} {'PCA':>10} {'Difference':>12}")
    for name in COMPONENT_NAMES:
        default_w = DEFAULT_WEIGHTS[name]
        pca_w = pca_result['pca_weights'][name]
        diff = pca_w - default_w
        print(f"    {name:>20} {default_w:>10.3f} {pca_w:>10.3f} {diff:>+12.3f}")

    # =========================================================================
    # 2. BOOTSTRAP CONFIDENCE INTERVALS
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("2. BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 80)

    # Compute v2 scores for baseline and model
    baseline_v2 = np.array([
        compute_ctqi_v2(ga, q, cf1)
        for ga, q, cf1 in zip(data['baseline_ga'], data['baseline_q'], data['baseline_cf1'])
    ])
    model_v2 = np.array([
        compute_ctqi_v2(ga, q, cf1)
        for ga, q, cf1 in zip(data['model_ga'], data['model_q'], data['model_cf1'])
    ])
    improvement_v2 = model_v2 - baseline_v2

    ci_baseline = bootstrap_ctqi_ci(baseline_v2, n_bootstrap=args.n_bootstrap)
    ci_model = bootstrap_ctqi_ci(model_v2, n_bootstrap=args.n_bootstrap)
    ci_improvement = bootstrap_ctqi_ci(improvement_v2, n_bootstrap=args.n_bootstrap)

    print(f"\n  Bootstrap resamples: {args.n_bootstrap}")
    print(f"  Confidence level: {ci_model['confidence_level']:.0%}")
    print(f"  BCa bias correction: z0={ci_model['bias_correction']:.4f}, a={ci_model['acceleration']:.4f}")

    print(f"\n  Baseline CTQI v2:     {ci_baseline['mean']:.2f}  (95% CI: [{ci_baseline['ci_lower']:.2f}, {ci_baseline['ci_upper']:.2f}])")
    print(f"  Model CTQI v2:        {ci_model['mean']:.2f}  (95% CI: [{ci_model['ci_lower']:.2f}, {ci_model['ci_upper']:.2f}])")
    print(f"  Improvement:         {ci_improvement['mean']:+.2f}  (95% CI: [{ci_improvement['ci_lower']:.2f}, {ci_improvement['ci_upper']:.2f}])")
    print(f"  Bootstrap SE:         {ci_model['se_bootstrap']:.3f}")

    # =========================================================================
    # 3. ABLATION STUDY
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("3. ABLATION STUDY (Component Necessity)")
    print("=" * 80)
    ablation = ablation_study(data)

    print(f"\n  Full model (all 3 components):")
    print(f"    Cohen's d: {ablation['full_model']['cohens_d']:.3f}")
    print(f"    Mean improvement: {ablation['full_model']['mean_improvement']:+.2f}")

    print(f"\n  {'Ablation':<25} {'Cohen d':>10} {'Degradation':>14} {'Necessity':>12}")
    print(f"  {'-' * 61}")
    print(f"  {'Full model (baseline)':<25} {ablation['full_model']['cohens_d']:>10.3f} {'---':>14} {'---':>12}")
    for comp in COMPONENT_NAMES:
        abl = ablation['ablations'][f'without_{comp}']
        necessity = "HIGH" if abl['degradation'] > 0.1 else "MODERATE" if abl['degradation'] > 0.01 else "LOW"
        print(f"  Without {comp:<16} {abl['cohens_d']:>10.3f} {abl['degradation']:>+14.3f} {necessity:>12}")

    print(f"\n  Component Necessity Ranking (by degradation when removed):")
    for rank, (comp, deg) in enumerate(ablation['component_necessity_ranking'], 1):
        print(f"    {rank}. {comp}: d degradation = {deg:+.3f}")

    # =========================================================================
    # 4. V1 vs V2 COMPARISON
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("4. CTQI V1 vs V2 COMPARISON")
    print("=" * 80)
    comparison = compare_v1_v2(data)

    print(f"\n  {'Metric':<20} {'CTQI v1':>12} {'CTQI v2':>12}")
    print(f"  {'-' * 44}")
    print(f"  {'Baseline Mean':<20} {comparison['v1']['baseline_mean']:>12.2f} {comparison['v2']['baseline_mean']:>12.2f}")
    print(f"  {'Model Mean':<20} {comparison['v1']['model_mean']:>12.2f} {comparison['v2']['model_mean']:>12.2f}")
    print(f"  {'Mean Improvement':<20} {comparison['v1']['mean_improvement']:>+12.2f} {comparison['v2']['mean_improvement']:>+12.2f}")
    cohens_d_label = "Cohen's d"
    print(f"  {cohens_d_label:<20} {comparison['v1']['cohens_d']:>12.3f} {comparison['v2']['cohens_d']:>12.3f}")
    print(f"  {'% Improved':<20} {comparison['v1']['pct_improved']:>11.1f}% {comparison['v2']['pct_improved']:>11.1f}%")

    print(f"\n  Rank agreement (Spearman rho): {comparison['rank_correlation']['spearman_rho']:.3f} (p={comparison['rank_correlation']['p_value']:.2e})")
    print(f"  Direction disagreements: {comparison['disagreements']}/{n} entries")

    # Better discriminative power?
    if comparison['v2']['cohens_d'] > comparison['v1']['cohens_d']:
        print(f"\n  >> CTQI v2 has BETTER discriminative power (d={comparison['v2']['cohens_d']:.3f} vs d={comparison['v1']['cohens_d']:.3f})")
    else:
        print(f"\n  >> CTQI v1 has better discriminative power (d={comparison['v1']['cohens_d']:.3f} vs d={comparison['v2']['cohens_d']:.3f})")

    # =========================================================================
    # 5. HUMAN CORRELATION (optional)
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("5. HUMAN CORRELATION ANALYSIS")
    print("=" * 80)

    if args.human_ratings and args.human_ratings.exists():
        with open(args.human_ratings, 'r', encoding='utf-8') as f:
            human_data = json.load(f)

        # Expected format: {"ratings": [4.2, 3.8, 5.0, ...]}
        # where each value is the average human rating for that entry
        human_scores = np.array(human_data['ratings'])

        if len(human_scores) != n:
            print(f"\n  ERROR: Human ratings ({len(human_scores)}) don't match evaluation entries ({n})")
        else:
            # Correlate with CTQI v2
            corr_v2 = human_correlation_analysis(model_v2, human_scores)
            print(f"\n  CTQI v2 vs Human Ratings:")
            print(f"    Pearson r = {corr_v2['pearson_r']:.3f} (p = {corr_v2['pearson_p']:.4f})")
            print(f"    Spearman rho = {corr_v2['spearman_rho']:.3f} (p = {corr_v2['spearman_p']:.4f})")
            print(f"    Correlation strength: {corr_v2['interpretation']}")

            # Also correlate v1 for comparison
            corr_v1 = human_correlation_analysis(
                np.array([compute_ctqi_v1(ga, q, ptr)
                          for ga, q, ptr in zip(data['model_ga'], data['model_q'], data['model_ptr'])]),
                human_scores
            )
            print(f"\n  CTQI v1 vs Human Ratings (comparison):")
            print(f"    Pearson r = {corr_v1['pearson_r']:.3f} (p = {corr_v1['pearson_p']:.4f})")
            print(f"    Spearman rho = {corr_v1['spearman_rho']:.3f} (p = {corr_v1['spearman_p']:.4f})")

            # Also correlate individual components
            print(f"\n  Individual Component Correlations with Human Ratings:")
            for name, arr in [('Gloss Accuracy', data['model_ga']),
                              ('Quality Score', data['model_q']),
                              ('Coverage F1', data['model_cf1'])]:
                r, p = stats.pearsonr(arr, human_scores)
                print(f"    {name}: r = {r:.3f} (p = {p:.4f})")
    else:
        print(f"\n  No human ratings file provided.")
        print(f"  To run human correlation analysis:")
        print(f"    1. Create a JSON file with format: {{\"ratings\": [4.2, 3.8, 5.0, ...]}}")
        print(f"       (one average rating per evaluation entry, {n} values total)")
        print(f"    2. Run: python ctqi_v2_analysis.py --human-ratings path/to/ratings.json")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"""
  CTQI v2 = ((GA/100)^w1 * (Q/100)^w2 * (CF1/100)^w3) * 100

  Default weights: equal 1/3 each
  PCA weights: GA={pca_result['pca_weights']['gloss_accuracy']:.3f}, Q={pca_result['pca_weights']['quality']:.3f}, CF1={pca_result['pca_weights']['coverage_f1']:.3f}

  Model CTQI v2: {ci_model['mean']:.2f} (95% CI: [{ci_model['ci_lower']:.2f}, {ci_model['ci_upper']:.2f}])
  Improvement:  {ci_improvement['mean']:+.2f} (95% CI: [{ci_improvement['ci_lower']:.2f}, {ci_improvement['ci_upper']:.2f}])
  Cohen's d:    {comparison['v2']['cohens_d']:.3f}

  Most necessary component: {ablation['component_necessity_ranking'][0][0]}
  (removing it causes d degradation of {ablation['component_necessity_ranking'][0][1]:+.3f})
""")

    return 0


if __name__ == "__main__":
    exit(main())
