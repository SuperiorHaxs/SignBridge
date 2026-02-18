"""
Divergence Analysis: Why CTQI v1 correlates better with human ratings than CTQI v2

This script analyzes specific cases where the two formulas diverge to understand
why the weighted average (v1) outperforms the prerequisite chain (v2).
"""

import json
import csv
import numpy as np
from scipy import stats
from pathlib import Path

# File paths
EVAL_RESULTS_PATH = Path(__file__).parent / "synthetic_evaluation" / "evaluation_results_gemini_t1_n53_v4" / "evaluation_results.json"
SURVEY_SENTENCES_PATH = Path(__file__).parent / "synthetic_evaluation" / "evaluation_results_gemini_t1_n53_v4" / "human_survey_sentences.csv"
SURVEY_RESPONSES_PATH = Path(r"C:\Users\ashwi\Downloads\SignBridge Translation Quality Survey .csv\SignBridge Translation Quality Survey .csv")
OUTPUT_DIR = Path(__file__).parent / "synthetic_evaluation" / "evaluation_results_gemini_t1_n53_v4"


def load_data():
    """Load all data sources."""
    # Load evaluation results
    with open(EVAL_RESULTS_PATH, 'r') as f:
        eval_results = json.load(f)

    # Load survey responses
    responses = []
    with open(SURVEY_RESPONSES_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            ratings = row[4:57]
            responses.append(ratings)

    # Load sentence info
    sentences = []
    with open(SURVEY_SENTENCES_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            sentences.append({
                'id': int(row[0]),
                'glosses': row[1],
                'sentence': row[2],
                'num_glosses': int(row[3])
            })

    return eval_results, responses, sentences


def compute_human_scores(responses):
    """Compute mean human score for each sentence."""
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


def analyze_divergence(eval_results, human_scores, sentences):
    """Analyze where CTQI v1 and v2 diverge from human judgment."""

    analysis = []

    for i, (entry, human_score, sent_info) in enumerate(zip(eval_results, human_scores, sentences)):
        if np.isnan(human_score):
            continue

        # Normalize human score to 0-100 scale (from 0-5)
        human_normalized = human_score * 20

        # Get scores
        ctqi_v1 = entry.get('model_composite', 0)
        ctqi_v2 = entry.get('model_composite_v2', 0)

        # Get component scores for CTQI v2
        ga = entry.get('gloss_accuracy', 0)
        cf1 = entry.get('model_coverage_f1', 0)
        plausibility = entry.get('model_plausibility', 0)
        quality = entry.get('model_quality', 0)
        bleu = entry.get('model_bleu', 0)
        bert = entry.get('model_bertscore', 0)

        # Calculate residuals (how far each metric is from human judgment)
        v1_residual = ctqi_v1 - human_normalized
        v2_residual = ctqi_v2 - human_normalized

        # Calculate which is closer to human
        v1_error = abs(v1_residual)
        v2_error = abs(v2_residual)

        analysis.append({
            'id': i + 1,
            'glosses': sent_info['glosses'],
            'sentence': sent_info['sentence'],
            'human_raw': human_score,
            'human_normalized': human_normalized,
            'ctqi_v1': ctqi_v1,
            'ctqi_v2': ctqi_v2,
            'v1_residual': v1_residual,
            'v2_residual': v2_residual,
            'v1_error': v1_error,
            'v2_error': v2_error,
            'v1_closer': v1_error < v2_error,
            'ga': ga,
            'cf1': cf1,
            'plausibility': plausibility,
            'quality': quality,
            'bleu': bleu,
            'bert': bert,
            # Identify the "weak link" in v2
            'min_component': min(ga, cf1, plausibility),
            'weak_link': 'GA' if ga == min(ga, cf1, plausibility) else ('CF1' if cf1 == min(ga, cf1, plausibility) else 'P')
        })

    return analysis


def generate_divergence_report(analysis):
    """Generate a detailed report on divergence patterns."""

    report = []
    report.append("=" * 100)
    report.append("DIVERGENCE ANALYSIS: Why CTQI v1 (Weighted Average) Outperforms CTQI v2 (Prerequisite Chain)")
    report.append("=" * 100)
    report.append("")

    # Summary statistics
    v1_wins = sum(1 for a in analysis if a['v1_closer'])
    v2_wins = len(analysis) - v1_wins

    report.append("SUMMARY")
    report.append("-" * 50)
    report.append(f"Cases where CTQI v1 is closer to human: {v1_wins} ({v1_wins/len(analysis)*100:.1f}%)")
    report.append(f"Cases where CTQI v2 is closer to human: {v2_wins} ({v2_wins/len(analysis)*100:.1f}%)")
    report.append("")

    # Analyze the pattern of divergence
    report.append("PATTERN ANALYSIS")
    report.append("-" * 50)

    # Cases where V2 underestimated (humans liked it more)
    v2_underestimated = [a for a in analysis if a['v2_residual'] < -10]
    # Cases where V2 overestimated (humans liked it less)
    v2_overestimated = [a for a in analysis if a['v2_residual'] > 10]

    report.append(f"\nCases where CTQI v2 significantly UNDERESTIMATED human rating (>10 pts below): {len(v2_underestimated)}")
    report.append(f"Cases where CTQI v2 significantly OVERESTIMATED human rating (>10 pts above): {len(v2_overestimated)}")
    report.append("")

    # Analyze weak links in v2 underestimated cases
    if v2_underestimated:
        report.append("=" * 100)
        report.append("CASES WHERE CTQI v2 UNDERESTIMATED HUMAN RATING")
        report.append("(Humans liked these sentences MORE than v2 predicted)")
        report.append("=" * 100)

        weak_link_counts = {'GA': 0, 'CF1': 0, 'P': 0}

        for a in sorted(v2_underestimated, key=lambda x: x['v2_residual']):
            weak_link_counts[a['weak_link']] += 1
            report.append("")
            report.append(f"ID {a['id']}: {a['glosses']}")
            report.append(f"   Sentence: \"{a['sentence']}\"")
            report.append(f"   Human Rating: {a['human_raw']:.1f}/5 ({a['human_normalized']:.0f}/100)")
            report.append(f"   CTQI v1: {a['ctqi_v1']:.1f}  |  CTQI v2: {a['ctqi_v2']:.1f}")
            report.append(f"   V2 Components: GA={a['ga']:.0f}, CF1={a['cf1']:.0f}, P={a['plausibility']:.0f}")
            report.append(f"   Weak Link: {a['weak_link']} (min={a['min_component']:.0f})")
            report.append(f"   V1 error: {a['v1_error']:.1f}  |  V2 error: {a['v2_error']:.1f}")

        report.append("")
        report.append("WEAK LINK DISTRIBUTION (what's dragging v2 down):")
        for link, count in sorted(weak_link_counts.items(), key=lambda x: -x[1]):
            report.append(f"   {link}: {count} cases ({count/len(v2_underestimated)*100:.1f}%)")

    # Look at cases with low plausibility but high human ratings
    report.append("")
    report.append("=" * 100)
    report.append("CASES WITH LOW PLAUSIBILITY BUT HIGH HUMAN RATINGS")
    report.append("(Plausibility < 70, Human Rating >= 4/5)")
    report.append("=" * 100)

    low_p_high_human = [a for a in analysis if a['plausibility'] < 70 and a['human_raw'] >= 4]

    for a in low_p_high_human:
        report.append("")
        report.append(f"ID {a['id']}: {a['glosses']}")
        report.append(f"   Sentence: \"{a['sentence']}\"")
        report.append(f"   Human: {a['human_raw']:.1f}/5  |  Plausibility: {a['plausibility']:.0f}")
        report.append(f"   Full Components: GA={a['ga']:.0f}, CF1={a['cf1']:.0f}, P={a['plausibility']:.0f}")
        report.append(f"   CTQI v1: {a['ctqi_v1']:.1f}  |  CTQI v2: {a['ctqi_v2']:.1f}")

    # Look at cases with low coverage but high human ratings
    report.append("")
    report.append("=" * 100)
    report.append("CASES WITH LOW COVERAGE F1 BUT HIGH HUMAN RATINGS")
    report.append("(Coverage F1 < 90, Human Rating >= 4/5)")
    report.append("=" * 100)

    low_cf1_high_human = [a for a in analysis if a['cf1'] < 90 and a['human_raw'] >= 4]

    for a in low_cf1_high_human:
        report.append("")
        report.append(f"ID {a['id']}: {a['glosses']}")
        report.append(f"   Sentence: \"{a['sentence']}\"")
        report.append(f"   Human: {a['human_raw']:.1f}/5  |  Coverage F1: {a['cf1']:.0f}")
        report.append(f"   Full Components: GA={a['ga']:.0f}, CF1={a['cf1']:.0f}, P={a['plausibility']:.0f}")
        report.append(f"   CTQI v1: {a['ctqi_v1']:.1f}  |  CTQI v2: {a['ctqi_v2']:.1f}")

    # Cases where V2 OVERESTIMATED (humans didn't like it as much)
    report.append("")
    report.append("=" * 100)
    report.append("CASES WHERE CTQI v2 OVERESTIMATED HUMAN RATING (22 cases)")
    report.append("(Humans liked these sentences LESS than v2 predicted)")
    report.append("These are PROBLEMATIC translations that got HIGH automated scores!")
    report.append("=" * 100)

    v2_overest = sorted([a for a in analysis if a['v2_residual'] > 10],
                        key=lambda x: -x['v2_residual'])

    for a in v2_overest:
        report.append("")
        report.append(f"ID {a['id']}: {a['glosses']}")
        report.append(f"   Sentence: \"{a['sentence']}\"")
        report.append(f"   Human Rating: {a['human_raw']:.1f}/5 ({a['human_normalized']:.0f}/100)")
        report.append(f"   CTQI v1: {a['ctqi_v1']:.1f}  |  CTQI v2: {a['ctqi_v2']:.1f}")
        report.append(f"   V2 Components: GA={a['ga']:.0f}, CF1={a['cf1']:.0f}, P={a['plausibility']:.0f}")
        report.append(f"   V2 overestimated by: +{a['v2_residual']:.1f} points")

    # Key insight section
    report.append("")
    report.append("=" * 100)
    report.append("KEY INSIGHTS")
    report.append("=" * 100)
    report.append("")

    # Calculate average component scores for high human / low v2 cases
    if v2_underestimated:
        avg_ga = np.mean([a['ga'] for a in v2_underestimated])
        avg_cf1 = np.mean([a['cf1'] for a in v2_underestimated])
        avg_p = np.mean([a['plausibility'] for a in v2_underestimated])

        report.append("For cases where v2 underestimated human judgment:")
        report.append(f"   Average Gloss Accuracy: {avg_ga:.1f}")
        report.append(f"   Average Coverage F1: {avg_cf1:.1f}")
        report.append(f"   Average Plausibility: {avg_p:.1f}")
        report.append("")

        if avg_p < 70:
            report.append("FINDING: Low Plausibility scores are dragging down CTQI v2,")
            report.append("         but humans are more forgiving of sentences that aren't perfectly fluent.")

        if avg_cf1 < 90:
            report.append("FINDING: Low Coverage F1 scores are penalizing CTQI v2,")
            report.append("         but humans may not care as much about exact word coverage.")

    report.append("")
    report.append("HYPOTHESIS:")
    report.append("The multiplicative nature of CTQI v2 (prerequisite chain) is too harsh.")
    report.append("When ANY component is low, the entire score collapses.")
    report.append("Humans seem to use a more 'averaging' mental model where good qualities")
    report.append("can partially compensate for weaknesses, similar to CTQI v1.")
    report.append("")
    report.append("POTENTIAL ISSUES WITH SURVEY:")
    report.append("1. Raters may not have clear criteria for what constitutes a 'failed' translation")
    report.append("2. Raters may focus on 'does this make sense' rather than 'is this faithful to glosses'")
    report.append("3. The 0-5 scale may not capture the nuance that v2 is designed to measure")
    report.append("")

    return "\n".join(report)


def main():
    print("Loading data...")
    eval_results, responses, sentences = load_data()

    print("Computing human scores...")
    human_scores = compute_human_scores(responses)

    print("Analyzing divergence...")
    analysis = analyze_divergence(eval_results, human_scores, sentences)

    print("Generating report...")
    report = generate_divergence_report(analysis)

    # Save report
    output_path = OUTPUT_DIR / 'divergence_analysis_report.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
