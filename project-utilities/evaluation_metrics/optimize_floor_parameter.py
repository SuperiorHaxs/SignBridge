#!/usr/bin/env python3
"""
Optimize the floor parameter (α) in the CTQI v3 plausibility modifier.

Formula: CTQI = (GA/100) × (CF1/100) × (α + (1−α) × P/100 × GA/100) × 100

Sweeps α from 0.0 to 1.0 and computes Pearson and Spearman correlation
with human survey ratings for each value. Produces a plot showing both
correlation curves and identifies the optimal α for each.
"""

import json
import csv
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# =========================================================================
# Data paths (same as human_survey_analysis.py)
# =========================================================================
EVAL_RESULTS_PATH = Path(__file__).parent / "synthetic_evaluation" / "evaluation_results_gemini_t1_n53_v4" / "evaluation_results.json"
SURVEY_RESPONSES_PATH = Path(r"C:\Users\ashwi\Downloads\SignBridge Translation Quality Survey .csv\SignBridge Translation Quality Survey .csv")


def load_evaluation_results():
    with open(EVAL_RESULTS_PATH, 'r') as f:
        return json.load(f)


def load_human_scores():
    """Load survey responses and compute mean human score per sentence."""
    responses = []
    with open(SURVEY_RESPONSES_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            ratings = row[4:57]  # 53 sentence ratings
            responses.append(ratings)

    human_scores = []
    for sent_idx in range(53):
        ratings = []
        for resp in responses:
            if sent_idx < len(resp) and resp[sent_idx]:
                try:
                    ratings.append(int(resp[sent_idx]))
                except ValueError:
                    pass
        human_scores.append(np.mean(ratings) if ratings else np.nan)

    return human_scores


def compute_ctqi_v3(ga, cf1, p, alpha):
    """Compute CTQI v3 with parameterized floor α."""
    ga_n = max(0.0, min(100.0, ga)) / 100.0
    cf1_n = max(0.0, min(100.0, cf1)) / 100.0
    p_n = max(0.0, min(100.0, p)) / 100.0
    modifier = alpha + (1 - alpha) * p_n * ga_n
    return ga_n * cf1_n * modifier * 100.0


def main():
    print("Loading data...")
    eval_results = load_evaluation_results()
    human_scores = load_human_scores()

    # Extract per-entry component values
    entries = []
    for entry in eval_results:
        entries.append({
            'ega': entry.get('effective_gloss_accuracy', 0) or 0,
            'cf1': entry.get('model_coverage_f1', 0) or 0,
            'p': entry.get('model_plausibility', 0) or 0,
        })

    # Filter NaN human scores
    valid = [(i, h) for i, h in enumerate(human_scores) if not np.isnan(h)]
    valid_idx = [v[0] for v in valid]
    human_valid = np.array([v[1] for v in valid])

    print(f"Valid entries: {len(valid_idx)}")

    # Sweep α from 0.0 to 1.0
    alphas = np.arange(0.0, 1.001, 0.01)
    pearson_rs = []
    spearman_rs = []

    for alpha in alphas:
        ctqi_scores = []
        for i in valid_idx:
            e = entries[i]
            ctqi_scores.append(compute_ctqi_v3(e['ega'], e['cf1'], e['p'], alpha))
        ctqi_arr = np.array(ctqi_scores)

        r_pearson, _ = stats.pearsonr(human_valid, ctqi_arr)
        r_spearman, _ = stats.spearmanr(human_valid, ctqi_arr)

        pearson_rs.append(r_pearson)
        spearman_rs.append(r_spearman)

    pearson_rs = np.array(pearson_rs)
    spearman_rs = np.array(spearman_rs)

    # Find optimal α for each
    best_pearson_idx = np.argmax(pearson_rs)
    best_spearman_idx = np.argmax(spearman_rs)

    best_alpha_pearson = alphas[best_pearson_idx]
    best_alpha_spearman = alphas[best_spearman_idx]
    best_r_pearson = pearson_rs[best_pearson_idx]
    best_r_spearman = spearman_rs[best_spearman_idx]

    # Current α=0.5 values
    idx_050 = np.argmin(np.abs(alphas - 0.50))
    r_pearson_050 = pearson_rs[idx_050]
    r_spearman_050 = spearman_rs[idx_050]

    # Print results
    print(f"\n{'='*60}")
    print(f"FLOOR PARAMETER OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"\nPearson:")
    print(f"  Optimal alpha = {best_alpha_pearson:.2f}  (r = {best_r_pearson:.4f})")
    print(f"  Current alpha = 0.50  (r = {r_pearson_050:.4f})")
    print(f"  Difference: {best_r_pearson - r_pearson_050:+.4f}")
    print(f"\nSpearman:")
    print(f"  Optimal alpha = {best_alpha_spearman:.2f}  (rho = {best_r_spearman:.4f})")
    print(f"  Current alpha = 0.50  (rho = {r_spearman_050:.4f})")
    print(f"  Difference: {best_r_spearman - r_spearman_050:+.4f}")

    # =====================================================================
    # Plot
    # =====================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(alphas, pearson_rs, color='#2E7D32', linewidth=2.5, label='Pearson r')
    ax.plot(alphas, spearman_rs, color='#1565C0', linewidth=2.5, label='Spearman ρ')

    # Mark optimal points
    ax.scatter(best_alpha_pearson, best_r_pearson, color='#2E7D32', s=120, zorder=5, edgecolors='black', linewidths=1.5)
    ax.scatter(best_alpha_spearman, best_r_spearman, color='#1565C0', s=120, zorder=5, edgecolors='black', linewidths=1.5)

    # Mark current α=0.5
    ax.axvline(x=0.50, color='#999999', linestyle='--', linewidth=1.5, alpha=0.7, label='Current α = 0.50')

    # Annotate optimal Pearson
    ax.annotate(f'α = {best_alpha_pearson:.2f}\nr = {best_r_pearson:.4f}',
                (best_alpha_pearson, best_r_pearson),
                textcoords="offset points", xytext=(50, -30),
                fontsize=12, fontweight='bold', color='#2E7D32',
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))

    # Annotate optimal Spearman
    ax.annotate(f'α = {best_alpha_spearman:.2f}\nρ = {best_r_spearman:.4f}',
                (best_alpha_spearman, best_r_spearman),
                textcoords="offset points", xytext=(50, -25),
                fontsize=12, fontweight='bold', color='#1565C0',
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))

    ax.set_xlabel('Floor Parameter (α)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Correlation with Human Ratings', fontsize=15, fontweight='bold')
    ax.set_title('CTQI v3 Floor Parameter Optimization\n(n=53 sentences)',
                 fontsize=17, fontweight='bold', pad=15)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(fontsize=13, loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    output = str(Path(__file__).parent / 'chart_floor_optimization')
    plt.savefig(output + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(output + '.pdf', bbox_inches='tight')
    print(f"\nSaved chart to {output}.png and {output}.pdf")


if __name__ == "__main__":
    main()
