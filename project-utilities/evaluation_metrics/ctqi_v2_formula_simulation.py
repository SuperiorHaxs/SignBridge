"""
CTQI v2 Formula Simulation

Tests different formula variations to find one that correlates better with human ratings
while not degrading performance on cases that already correlate well.
"""

import json
import csv
import numpy as np
from scipy import stats
from pathlib import Path

# File paths
EVAL_RESULTS_PATH = Path(__file__).parent / "synthetic_evaluation" / "evaluation_results_gemini_t1_n53_v4" / "evaluation_results.json"
SURVEY_RESPONSES_PATH = Path(r"C:\Users\ashwi\Downloads\SignBridge Translation Quality Survey .csv\SignBridge Translation Quality Survey .csv")


def load_data():
    """Load evaluation results and human survey data."""
    with open(EVAL_RESULTS_PATH, 'r') as f:
        eval_results = json.load(f)

    # Load survey responses
    responses = []
    with open(SURVEY_RESPONSES_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            ratings = row[4:57]
            responses.append(ratings)

    # Compute human scores (mean of 5 raters)
    human_scores = []
    for sent_idx in range(53):
        ratings = []
        for resp in responses:
            if sent_idx < len(resp) and resp[sent_idx]:
                try:
                    ratings.append(int(resp[sent_idx]))
                except ValueError:
                    pass
        human_scores.append(np.mean(ratings) * 20 if ratings else np.nan)  # Convert to 0-100

    return eval_results, human_scores


def current_v2_formula(ga, cf1, p):
    """Current CTQI v2 prerequisite chain formula."""
    ga_norm = ga / 100.0
    cf1_norm = cf1 / 100.0
    p_norm = p / 100.0
    p_modifier = 0.5 + 0.5 * p_norm
    return ga_norm * cf1_norm * p_modifier * 100.0


def option_a_formula(ga, cf1, p):
    """Option A: P modifier scaled by GA."""
    ga_norm = ga / 100.0
    cf1_norm = cf1 / 100.0
    p_norm = p / 100.0
    # P's contribution scales with GA
    p_modifier = 0.5 + 0.5 * p_norm * ga_norm
    return ga_norm * cf1_norm * p_modifier * 100.0


def option_b_formula(ga, cf1, p):
    """Option B: P modifier scaled by GA squared."""
    ga_norm = ga / 100.0
    cf1_norm = cf1 / 100.0
    p_norm = p / 100.0
    # P's contribution scales with GA squared
    p_modifier = 0.5 + 0.5 * p_norm * (ga_norm ** 2)
    return ga_norm * cf1_norm * p_modifier * 100.0


def option_c_formula(ga, cf1, p):
    """Option C: Threshold-based P weight."""
    ga_norm = ga / 100.0
    cf1_norm = cf1 / 100.0
    p_norm = p / 100.0

    # P weight based on GA threshold
    if ga >= 90:
        p_weight = 1.0
    elif ga >= 70:
        p_weight = 0.5
    else:
        p_weight = 0.2

    p_modifier = 0.5 + 0.5 * p_norm * p_weight
    return ga_norm * cf1_norm * p_modifier * 100.0


def option_d_formula(ga, cf1, p):
    """Option D: P weight depends on BOTH GA and CF1 being low."""
    ga_norm = ga / 100.0
    cf1_norm = cf1 / 100.0
    p_norm = p / 100.0

    # Only reduce P weight if BOTH GA and CF1 are low (indicates real problem)
    # If CF1 is high, the translation captured the meaning despite GA mismatch
    if ga < 80 and cf1 < 80:
        p_weight = 0.3  # Both low = likely bad translation
    else:
        p_weight = 1.0  # At least one is good = trust P

    p_modifier = 0.5 + 0.5 * p_norm * p_weight
    return ga_norm * cf1_norm * p_modifier * 100.0


def option_e_formula(ga, cf1, p):
    """Option E: Use min(GA, CF1) to gate P."""
    ga_norm = ga / 100.0
    cf1_norm = cf1 / 100.0
    p_norm = p / 100.0

    # P contribution scales with the weakest link
    weak_link = min(ga_norm, cf1_norm)
    p_modifier = 0.5 + 0.5 * p_norm * weak_link

    return ga_norm * cf1_norm * p_modifier * 100.0


def option_f_formula(ga, cf1, p):
    """Option F: Harsh penalty only when GA < 70."""
    ga_norm = ga / 100.0
    cf1_norm = cf1 / 100.0
    p_norm = p / 100.0

    # If GA < 70, P can't save the score
    if ga < 70:
        p_modifier = 0.5 + 0.5 * p_norm * 0.2
    else:
        p_modifier = 0.5 + 0.5 * p_norm

    return ga_norm * cf1_norm * p_modifier * 100.0


def option_g_formula(ga, cf1, p):
    """Option G: Weighted average hybrid (like V1 but with CF1)."""
    # More like V1's averaging approach
    return 0.35 * ga + 0.35 * cf1 + 0.30 * p


def option_h_formula(ga, cf1, p):
    """Option H: GA as hard gate, then average rest."""
    ga_norm = ga / 100.0

    # GA gates, then average CF1 and P
    return ga_norm * (0.5 * cf1 + 0.5 * p)


def chain_with_ega(ega, cf1, p):
    """Chain formula using effective_gloss_accuracy (matches stored model_composite)."""
    ga_norm = ega / 100.0
    cf1_norm = cf1 / 100.0
    p_norm = p / 100.0
    p_modifier = 0.5 + 0.5 * p_norm
    return ga_norm * cf1_norm * p_modifier * 100.0


def geometric_mean_v2(ega, quality, cf1):
    """Geometric mean formula (matches stored model_composite_v2)."""
    import math
    if ega > 0 and quality > 0 and cf1 > 0:
        return math.pow(ega * quality * cf1, 1/3)
    return 0


def option_i_formula(ega, cf1, p):
    """Option I: Chain with EGA, P scaled by EGA."""
    ga_norm = ega / 100.0
    cf1_norm = cf1 / 100.0
    p_norm = p / 100.0
    # P's contribution scales with effective GA
    p_modifier = 0.5 + 0.5 * p_norm * ga_norm
    return ga_norm * cf1_norm * p_modifier * 100.0


def option_j_formula(ega, cf1, p):
    """Option J: Chain with EGA, P scaled when BOTH EGA and CF1 low."""
    ga_norm = ega / 100.0
    cf1_norm = cf1 / 100.0
    p_norm = p / 100.0

    # Only reduce P weight if BOTH EGA and CF1 are low
    if ega < 80 and cf1 < 80:
        p_weight = 0.3
    else:
        p_weight = 1.0

    p_modifier = 0.5 + 0.5 * p_norm * p_weight
    return ga_norm * cf1_norm * p_modifier * 100.0


def run_simulation():
    """Run simulation comparing formula options."""
    eval_results, human_scores = load_data()

    # Extract component scores and calculate variants
    results = []
    for i, entry in enumerate(eval_results):
        # Note: V1 (model_composite) uses effective_gloss_accuracy, not gloss_accuracy
        ga = entry.get('gloss_accuracy', 0) or 0
        ega = entry.get('effective_gloss_accuracy', 0) or 0  # Used by stored model_composite
        cf1 = entry.get('model_coverage_f1', 0) or 0
        p = entry.get('model_plausibility', 0) or 0
        quality = entry.get('model_quality', 0) or 0  # For geometric mean formula

        current_v2 = entry.get('model_composite_v2', 0)  # Geometric mean formula
        v1 = entry.get('model_composite', 0)  # Chain formula with effective_gloss_accuracy

        # Calculate formula variants
        # Verify stored values
        calc_v1 = chain_with_ega(ega, cf1, p)  # Should match stored model_composite
        calc_v2 = geometric_mean_v2(ega, quality, cf1)  # Should match stored model_composite_v2

        # New formula options using effective_gloss_accuracy
        option_i = option_i_formula(ega, cf1, p)  # Chain + P scaled by EGA
        option_j = option_j_formula(ega, cf1, p)  # Chain + P reduced when both low

        # Original options for comparison
        chain_current = current_v2_formula(ga, cf1, p)
        option_g = option_g_formula(ga, cf1, p)

        results.append({
            'id': i + 1,
            'human': human_scores[i],
            'v1_stored': v1,
            'v1_calc': calc_v1,
            'v2_stored': current_v2,
            'v2_calc': calc_v2,
            'option_i': option_i,
            'option_j': option_j,
            'option_g': option_g,
            'ga': ga,
            'ega': ega,
            'cf1': cf1,
            'p': p,
            'quality': quality
        })

    # Filter valid entries (non-NaN human scores)
    valid = [r for r in results if not np.isnan(r['human'])]

    # Calculate correlations
    human = [r['human'] for r in valid]

    correlations = {}
    for metric in ['v1_stored', 'v1_calc', 'v2_stored', 'v2_calc', 'option_i', 'option_j', 'option_g']:
        values = [r[metric] for r in valid]
        pearson_r, pearson_p = stats.pearsonr(human, values)
        spearman_r, spearman_p = stats.spearmanr(human, values)
        correlations[metric] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p
        }

    # Print report
    print("=" * 100)
    print("CTQI V2 FORMULA SIMULATION RESULTS")
    print("=" * 100)
    print()

    print("CORRELATION COMPARISON (sorted by Pearson r)")
    print("-" * 80)
    print(f"{'Formula':<20} {'Pearson r':<12} {'p-value':<12} {'Spearman r':<12} {'vs V1':<12}")
    print("-" * 80)

    sorted_corr = sorted(correlations.items(), key=lambda x: x[1]['pearson_r'], reverse=True)
    v1_r = correlations['v1_stored']['pearson_r']

    for metric, corr in sorted_corr:
        diff = corr['pearson_r'] - v1_r
        diff_str = f"{diff:+.4f}" if metric != 'v1' else "baseline"
        print(f"{metric:<20} {corr['pearson_r']:>10.4f}   {corr['pearson_p']:>10.4f}   {corr['spearman_r']:>10.4f}   {diff_str:<12}")

    print()
    print("=" * 100)
    print("IMPACT ON INDIVIDUAL CASES")
    print("=" * 100)

    # Analyze impact on overestimated cases (where V2 was too high)
    print()
    print("PREVIOUSLY OVERESTIMATED CASES (V2_stored > Human by >10 pts)")
    print("-" * 130)
    print(f"{'ID':<4} {'Human':<8} {'V1':<8} {'V2':<8} {'OptI':<8} {'OptJ':<8} {'EGA':<6} {'CF1':<6} {'P':<6} {'Best':<8}")
    print("-" * 130)

    overestimated = [r for r in valid if r['v2_stored'] - r['human'] > 10]

    for r in sorted(overestimated, key=lambda x: x['v2_stored'] - x['human'], reverse=True):
        # Find which is closest to human
        errors = {
            'V1': abs(r['v1_stored'] - r['human']),
            'V2': abs(r['v2_stored'] - r['human']),
            'OptI': abs(r['option_i'] - r['human']),
            'OptJ': abs(r['option_j'] - r['human']),
        }
        best = min(errors, key=errors.get)

        print(f"{r['id']:<4} {r['human']:<8.1f} {r['v1_stored']:<8.1f} {r['v2_stored']:<8.1f} {r['option_i']:<8.1f} {r['option_j']:<8.1f} {r['ega']:<6.0f} {r['cf1']:<6.0f} {r['p']:<6.0f} {best:<8}")

    # Analyze impact on well-correlated cases (where V2 was close)
    print()
    print("PREVIOUSLY WELL-CORRELATED CASES (V2_stored within 10 pts of Human)")
    print("-" * 130)
    print(f"{'ID':<4} {'Human':<8} {'V1':<8} {'V2':<8} {'OptI':<8} {'OptJ':<8} {'EGA':<6} {'CF1':<6} {'P':<6} {'Impact':<10}")
    print("-" * 130)

    well_correlated = [r for r in valid if abs(r['v2_stored'] - r['human']) <= 10]

    degraded_count = {'option_i': 0, 'option_j': 0}

    for r in sorted(well_correlated, key=lambda x: x['id']):
        v2_error = abs(r['v2_stored'] - r['human'])
        opt_i_error = abs(r['option_i'] - r['human'])
        opt_j_error = abs(r['option_j'] - r['human'])

        # Check if options make it worse
        impact = ""
        if opt_i_error > v2_error + 5:
            impact = "I worse"
            degraded_count['option_i'] += 1
        elif opt_j_error > v2_error + 5:
            impact = "J worse"
            degraded_count['option_j'] += 1
        else:
            impact = "OK"

        print(f"{r['id']:<4} {r['human']:<8.1f} {r['v1_stored']:<8.1f} {r['v2_stored']:<8.1f} {r['option_i']:<8.1f} {r['option_j']:<8.1f} {r['ega']:<6.0f} {r['cf1']:<6.0f} {r['p']:<6.0f} {impact:<10}")

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()
    print(f"Total valid cases: {len(valid)}")
    print(f"Overestimated cases (V2 > Human by >10): {len(overestimated)}")
    print(f"Well-correlated cases (within 10): {len(well_correlated)}")
    print()
    print(f"Cases where Option I degrades well-correlated: {degraded_count['option_i']}")
    print(f"Cases where Option J degrades well-correlated: {degraded_count['option_j']}")
    print()

    # Verify calculations match stored values
    print("CALCULATION VERIFICATION:")
    print(f"  V1 stored vs calculated match: {sum(1 for r in valid if abs(r['v1_stored'] - r['v1_calc']) < 0.1)}/{len(valid)}")
    print(f"  V2 stored vs calculated match: {sum(1 for r in valid if abs(r['v2_stored'] - r['v2_calc']) < 0.1)}/{len(valid)}")
    print()

    # Final recommendation
    best_formula = sorted_corr[0][0]
    best_r = sorted_corr[0][1]['pearson_r']
    v2_r = correlations['v2_stored']['pearson_r']

    print("RECOMMENDATION:")
    print(f"  Current V1 (chain formula): r={v1_r:.4f}")
    print(f"  Current V2 (geometric mean): r={v2_r:.4f}")
    print()

    # Find best option that beats V1
    for metric, corr in sorted_corr:
        if metric.startswith('option') and corr['pearson_r'] > v1_r:
            print(f"  {metric} BEATS V1! r={corr['pearson_r']:.4f} (diff: {corr['pearson_r'] - v1_r:+.4f})")
            break
    else:
        print(f"  No formula option beats V1's correlation of {v1_r:.4f}")

    return correlations, results


if __name__ == "__main__":
    run_simulation()
