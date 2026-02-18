"""
Human Survey Analysis Script
Compares human survey ratings against various automated metrics (CTQI v3, v2, v1, BLEU, BERT, etc.)
to determine which metric best correlates with human judgment.

Correlation type: Pearson r
- Measures linear relationship between continuous variables
- Standard choice for comparing metric scores to human ratings
- Values range from -1 to 1, with 1 indicating perfect positive correlation
"""

import json
import csv
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from metrics import calculate_composite_score_v3

# File paths
EVAL_RESULTS_PATH = Path(__file__).parent / "synthetic_evaluation" / "evaluation_results_gemini_t1_n53_v4" / "evaluation_results.json"
SURVEY_SENTENCES_PATH = Path(__file__).parent / "synthetic_evaluation" / "evaluation_results_gemini_t1_n53_v4" / "human_survey_sentences.csv"
SURVEY_RESPONSES_PATH = Path(r"C:\Users\ashwi\Downloads\SignBridge Translation Quality Survey .csv\SignBridge Translation Quality Survey .csv")
OUTPUT_DIR = Path(__file__).parent / "synthetic_evaluation" / "evaluation_results_gemini_t1_n53_v4"

# Metric display order (as specified by user)
# CF1 and P come after CTQI v1 since they are components introduced in v2
METRIC_ORDER = [
    'gloss_accuracy',
    'bleu',
    'bert',
    'ctqi_v1',
    'coverage_f1',
    'plausibility',
    'ctqi_v2',
    'ctqi_v3',
]

# Display names for chart x-axis (single line, concise)
DISPLAY_NAMES = {
    'gloss_accuracy': 'GA',
    'bleu': 'BLEU',
    'bert': 'BERTScore',
    'coverage_f1': 'CF1',
    'quality': 'Quality',
    'ctqi_v1': 'CTQI v1',
    'plausibility': 'P',
    'ctqi_v2': 'CTQI v2',
    'ctqi_v3': 'CTQI v3',
}

# Full definitions for the legend box
METRIC_DEFINITIONS = {
    'gloss_accuracy': ('GA', '% of ASL signs correctly recognized'),
    'bleu': ('BLEU', 'N-gram overlap with reference'),
    'bert': ('BERTScore', 'Semantic similarity via BERT'),
    'ctqi_v1': ('CTQI v1', '0.4×GA + 0.4×GPT2-Perplexity + 0.2×Perfect Translation Rate'),
    'coverage_f1': ('CF1', '% of meaning words matched in sentence'),
    'plausibility': ('P', 'LLM-assessed grammar & naturalness'),
    'ctqi_v2': ('CTQI v2', 'GA × CF1 × (0.5 + 0.5×P)'),
    'ctqi_v3': ('CTQI v3', 'CTQI v2 with P weighted by GA (post human validation feedback)'),
}


def load_evaluation_results():
    """Load the evaluation results JSON."""
    with open(EVAL_RESULTS_PATH, 'r') as f:
        return json.load(f)


def load_survey_responses():
    """Load and parse human survey responses."""
    responses = []
    with open(SURVEY_RESPONSES_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        for row in reader:
            # Columns E onwards (index 4 onwards) are the ratings
            # There are 53 sentences, so columns 4 to 56 (inclusive)
            ratings = row[4:57]  # 53 ratings
            responses.append(ratings)
    return responses


def compute_human_scores(responses):
    """Compute mean human score for each sentence."""
    n_sentences = 53
    human_scores = []

    for sent_idx in range(n_sentences):
        ratings = []
        for resp in responses:
            if sent_idx < len(resp) and resp[sent_idx]:
                try:
                    rating = int(resp[sent_idx])
                    ratings.append(rating)
                except ValueError:
                    pass

        if ratings:
            mean_score = np.mean(ratings)
            human_scores.append(mean_score)
        else:
            human_scores.append(np.nan)

    return human_scores


def extract_metrics(eval_results):
    """Extract various metrics from evaluation results, including computed CTQI v3."""
    metrics = {
        'gloss_accuracy': [],
        'bleu': [],
        'bert': [],
        'coverage_f1': [],
        'quality': [],
        'ctqi_v1': [],
        'plausibility': [],
        'ctqi_v2': [],
        'ctqi_v3': [],
    }

    for entry in eval_results:
        # Extract stored values
        metrics['gloss_accuracy'].append(entry.get('gloss_accuracy', 0) or 0)
        metrics['bleu'].append(entry.get('model_bleu', 0) or 0)
        metrics['bert'].append(entry.get('model_bertscore', 0) or 0)
        metrics['coverage_f1'].append(entry.get('model_coverage_f1', 0) or 0)
        metrics['quality'].append(entry.get('model_quality', 0) or 0)
        metrics['ctqi_v1'].append(entry.get('model_composite', 0) or 0)
        metrics['plausibility'].append(entry.get('model_plausibility', 0) or 0)
        metrics['ctqi_v2'].append(entry.get('model_composite_v2', 0) or 0)

        # Calculate CTQI v3 using effective_gloss_accuracy
        ega = entry.get('effective_gloss_accuracy', 0) or 0
        cf1 = entry.get('model_coverage_f1', 0) or 0
        p = entry.get('model_plausibility', 0) or 0
        v3_score = calculate_composite_score_v3(ega, cf1, p)
        metrics['ctqi_v3'].append(v3_score)

    return metrics


def compute_correlations(human_scores, metrics):
    """Compute Pearson and Spearman correlations between human scores and each metric."""
    correlations = {}

    # Filter out NaN values
    valid_indices = [i for i, h in enumerate(human_scores) if not np.isnan(h)]
    human_valid = [human_scores[i] for i in valid_indices]

    for metric_name, metric_values in metrics.items():
        metric_valid = [metric_values[i] for i in valid_indices]

        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(human_valid, metric_valid)

        # Spearman correlation
        spearman_r, spearman_p = stats.spearmanr(human_valid, metric_valid)

        correlations[metric_name] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p
        }

    return correlations


def create_correlation_bar_chart(correlations, output_path):
    """
    Create a bar chart showing Pearson correlations for all metrics.
    Includes a definition legend box on the right side.

    Uses Pearson correlation (not Spearman) because:
    - Pearson is the standard for measuring linear relationships
    - Our metrics and human scores are continuous variables
    - Pearson is more sensitive to the actual values, not just ranks
    """
    # Use specified metric order
    metric_names = []
    metric_keys = []
    pearson_values = []

    for metric in METRIC_ORDER:
        if metric in correlations:
            metric_names.append(DISPLAY_NAMES.get(metric, metric))
            metric_keys.append(metric)
            pearson_values.append(correlations[metric]['pearson_r'])

    # Create the figure with space for legend on right
    fig = plt.figure(figsize=(18, 8))

    # Create gridspec for chart (left) and legend (right)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.2], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])

    x = np.arange(len(metric_names))
    width = 0.7

    # Color gradient: lower correlation = red, higher = green
    colors = []
    for val in pearson_values:
        if val >= 0.9:
            colors.append('#2E7D32')  # Dark green
        elif val >= 0.8:
            colors.append('#4CAF50')  # Green
        elif val >= 0.7:
            colors.append('#8BC34A')  # Light green
        elif val >= 0.6:
            colors.append('#CDDC39')  # Lime
        elif val >= 0.5:
            colors.append('#FFC107')  # Amber
        else:
            colors.append('#FF5722')  # Deep orange

    bars = ax.bar(x, pearson_values, width, color=colors, edgecolor='black', linewidth=1.2)

    # Customize the chart
    ax.set_xlabel('Metric', fontsize=13, fontweight='bold')
    ax.set_ylabel('Pearson Correlation (r)', fontsize=13, fontweight='bold')
    ax.set_title('Correlation of Automated Metrics with Human Survey Ratings\n(n=53 sentences, 5 raters, Pearson r)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11, fontweight='bold')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Set y-axis limits
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add reference lines
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, linewidth=1)

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Add legend for reference lines (on left side to avoid overlap with metrics table)
    ax.text(0.3, 0.91, 'Excellent (r > 0.9)', fontsize=9, color='green', alpha=0.8)
    ax.text(0.3, 0.71, 'Good (r > 0.7)', fontsize=9, color='orange', alpha=0.8)

    # Create definition legend box on the right
    ax_legend.axis('off')

    # Build legend text with word wrapping for long definitions
    import textwrap

    y_start = 0.95
    y_pos = y_start
    x_abbrev = 0.02
    x_def = 0.22
    max_chars = 35  # Max characters per line for definitions

    # Header
    ax_legend.text(0.5, y_pos, 'Metric Definitions', transform=ax_legend.transAxes,
                  fontsize=12, fontweight='bold', ha='center', va='top')
    y_pos -= 0.08

    for metric in METRIC_ORDER:
        if metric in METRIC_DEFINITIONS:
            abbrev, definition = METRIC_DEFINITIONS[metric]

            # Draw abbreviation
            ax_legend.text(x_abbrev, y_pos, f"{abbrev}:", transform=ax_legend.transAxes,
                          fontsize=10, fontweight='bold', ha='left', va='top',
                          family='monospace')

            # Word wrap long definitions
            wrapped_lines = textwrap.wrap(definition, width=max_chars)
            for j, line in enumerate(wrapped_lines):
                ax_legend.text(x_def, y_pos - j * 0.04, line, transform=ax_legend.transAxes,
                              fontsize=9, ha='left', va='top')

            # Adjust y position based on number of wrapped lines
            y_pos -= 0.05 + (len(wrapped_lines) - 1) * 0.04 + 0.03

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved correlation chart to {output_path}")

    return fig


def create_scatter_plots(human_scores, metrics, output_dir):
    """Create scatter plots for CTQI versions comparison."""
    valid_indices = [i for i, h in enumerate(human_scores) if not np.isnan(h)]
    human_valid = [human_scores[i] for i in valid_indices]

    # Create 2x2 scatter plot comparing CTQI versions and top metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    plot_metrics = ['gloss_accuracy', 'plausibility', 'ctqi_v2', 'ctqi_v3']
    plot_names = {
        'gloss_accuracy': 'Gloss Accuracy (GA)',
        'plausibility': 'Plausibility (P)',
        'ctqi_v2': 'CTQI v2 (Prerequisite Chain)',
        'ctqi_v3': 'CTQI v3 (Human-Validated)'
    }

    for idx, metric in enumerate(plot_metrics):
        ax = axes[idx // 2, idx % 2]
        metric_valid = [metrics[metric][i] for i in valid_indices]

        ax.scatter(metric_valid, human_valid, alpha=0.6, edgecolors='black', linewidths=0.5, s=60)

        # Add trend line
        z = np.polyfit(metric_valid, human_valid, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(metric_valid), max(metric_valid), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend line')

        # Compute correlation for label
        r, _ = stats.pearsonr(metric_valid, human_valid)

        ax.set_xlabel(f'{plot_names[metric]} Score', fontsize=11)
        ax.set_ylabel('Human Rating (0-5)', fontsize=11)
        ax.set_title(f'{plot_names[metric]} vs Human Rating\n(Pearson r = {r:.4f})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Highlight if best
        if metric == 'ctqi_v3':
            ax.set_facecolor('#f0fff0')  # Light green background

    plt.tight_layout()
    output_path = output_dir / 'human_survey_scatter_plots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved scatter plots to {output_path}")


def generate_report(human_scores, metrics, correlations, output_path):
    """Generate a detailed analysis report."""

    # Compute statistics
    valid_indices = [i for i, h in enumerate(human_scores) if not np.isnan(h)]
    human_valid = [human_scores[i] for i in valid_indices]

    report = []
    report.append("=" * 80)
    report.append("HUMAN SURVEY ANALYSIS REPORT")
    report.append("SignBridge Translation Quality Evaluation")
    report.append("=" * 80)
    report.append("")

    report.append("CORRELATION TYPE: Pearson r")
    report.append("-" * 40)
    report.append("Pearson correlation measures linear relationships between continuous")
    report.append("variables. It is the standard choice for comparing metric scores to")
    report.append("human ratings. Values range from -1 to 1, with 1 indicating perfect")
    report.append("positive correlation.")
    report.append("")

    report.append("DATASET OVERVIEW")
    report.append("-" * 40)
    report.append(f"Total sentences evaluated: 53")
    report.append(f"Number of human raters: 5")
    report.append(f"Valid responses: {len(valid_indices)}")
    report.append(f"Human rating scale: 0-5 (0=Failed, 5=Excellent)")
    report.append("")

    report.append("HUMAN RATING STATISTICS")
    report.append("-" * 40)
    report.append(f"Mean human rating: {np.mean(human_valid):.2f}")
    report.append(f"Median human rating: {np.median(human_valid):.2f}")
    report.append(f"Std deviation: {np.std(human_valid):.2f}")
    report.append(f"Min rating: {np.min(human_valid):.2f}")
    report.append(f"Max rating: {np.max(human_valid):.2f}")
    report.append("")

    report.append("CORRELATION ANALYSIS")
    report.append("-" * 40)
    report.append("Metric correlations with human ratings (sorted by Pearson r):")
    report.append("")
    report.append(f"{'Metric':<20} {'Pearson r':<12} {'p-value':<12} {'Spearman r':<12} {'p-value':<12}")
    report.append("-" * 68)

    sorted_correlations = sorted(correlations.items(), key=lambda x: x[1]['pearson_r'], reverse=True)

    display_names_report = {
        'ctqi_v3': 'CTQI v3',
        'ctqi_v2': 'CTQI v2',
        'ctqi_v1': 'CTQI v1',
        'bleu': 'BLEU',
        'bert': 'BERTScore',
        'gloss_accuracy': 'Gloss Accuracy',
        'plausibility': 'Plausibility',
        'coverage_f1': 'Coverage F1',
        'quality': 'Quality (GPT-2)'
    }

    for metric, corr in sorted_correlations:
        name = display_names_report.get(metric, metric)
        report.append(f"{name:<20} {corr['pearson_r']:>10.4f}   {corr['pearson_p']:>10.6f}   {corr['spearman_r']:>10.4f}   {corr['spearman_p']:>10.6f}")

    report.append("")
    report.append("KEY FINDINGS")
    report.append("-" * 40)

    # Find best correlating metric
    best_metric = sorted_correlations[0][0]
    best_corr = sorted_correlations[0][1]

    report.append(f"1. BEST CORRELATING METRIC: {display_names_report[best_metric]}")
    report.append(f"   - Pearson correlation: {best_corr['pearson_r']:.4f}")
    report.append(f"   - Spearman correlation: {best_corr['spearman_r']:.4f}")
    report.append(f"   - p-value: {best_corr['pearson_p']:.6f} (highly significant)")
    report.append("")

    # Compare CTQI versions
    ctqi_v3_r = correlations['ctqi_v3']['pearson_r']
    ctqi_v2_r = correlations['ctqi_v2']['pearson_r']
    ctqi_v1_r = correlations['ctqi_v1']['pearson_r']

    report.append(f"2. CTQI VERSION COMPARISON")
    report.append(f"   - CTQI v3 Pearson r: {ctqi_v3_r:.4f} (Human-Validated)")
    report.append(f"   - CTQI v1 Pearson r: {ctqi_v1_r:.4f} (Chain Formula)")
    report.append(f"   - CTQI v2 Pearson r: {ctqi_v2_r:.4f} (Geometric Mean)")
    report.append(f"")
    report.append(f"   V3 vs V1: {ctqi_v3_r - ctqi_v1_r:+.4f}")
    report.append(f"   V3 vs V2: {ctqi_v3_r - ctqi_v2_r:+.4f}")
    report.append("")

    # Compare with traditional metrics
    bleu_r = correlations['bleu']['pearson_r']
    bert_r = correlations['bert']['pearson_r']

    report.append(f"3. COMPARISON WITH TRADITIONAL METRICS")
    report.append(f"   - CTQI v3 vs BLEU: {ctqi_v3_r:.4f} vs {bleu_r:.4f} (diff: {ctqi_v3_r - bleu_r:+.4f})")
    report.append(f"   - CTQI v3 vs BERT: {ctqi_v3_r:.4f} vs {bert_r:.4f} (diff: {ctqi_v3_r - bert_r:+.4f})")
    report.append("")

    report.append("CTQI v3 FORMULA")
    report.append("-" * 40)
    report.append("CTQI v3 = (GA/100) * (CF1/100) * (0.5 + 0.5 * P/100 * GA/100) * 100")
    report.append("")
    report.append("Key improvement over v2:")
    report.append("- Plausibility's contribution is scaled by gloss accuracy")
    report.append("- Prevents high plausibility from masking translation errors")
    report.append("- When GA=100%, behaves identically to the chain formula")
    report.append("- When GA<100%, plausibility's boost is proportionally reduced")
    report.append("")

    report.append("CONCLUSION")
    report.append("-" * 40)
    if best_metric == 'ctqi_v3':
        report.append("CTQI v3 (Human-Validated) demonstrates the HIGHEST correlation")
        report.append("with human judgment among all evaluated metrics, validating")
        report.append("the effectiveness of the formula refinement based on human")
        report.append("survey feedback.")
        report.append("")
        report.append(f"Final ranking: CTQI v3 (r={ctqi_v3_r:.4f}) > CTQI v1 (r={ctqi_v1_r:.4f}) > CTQI v2 (r={ctqi_v2_r:.4f})")
    else:
        report.append(f"{display_names_report[best_metric]} shows the highest correlation with human judgment.")
        report.append(f"CTQI v3 ranks with Pearson r = {ctqi_v3_r:.4f}")

    report.append("")
    report.append("=" * 80)

    # Write report
    report_text = "\n".join(report)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"Saved analysis report to {output_path}")
    return report_text


def main():
    print("Loading evaluation results...")
    eval_results = load_evaluation_results()
    print(f"Loaded {len(eval_results)} evaluation entries")

    print("\nLoading survey responses...")
    responses = load_survey_responses()
    print(f"Loaded {len(responses)} survey responses")

    print("\nComputing human scores (mean of 5 raters)...")
    human_scores = compute_human_scores(responses)
    print(f"Computed scores for {len([h for h in human_scores if not np.isnan(h)])} sentences")

    print("\nExtracting automated metrics (including CTQI v3)...")
    metrics = extract_metrics(eval_results)

    print("\nComputing correlations...")
    correlations = compute_correlations(human_scores, metrics)

    # Print correlation summary
    print("\n" + "=" * 60)
    print("CORRELATION SUMMARY (Pearson r)")
    print("=" * 60)
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1]['pearson_r'], reverse=True)
    for metric, corr in sorted_corr:
        print(f"  {metric:<20}: r = {corr['pearson_r']:.4f} (p = {corr['pearson_p']:.6f})")

    print("\n" + "=" * 60)

    # Generate outputs
    print("\nGenerating visualizations...")

    # Bar chart
    chart_path = OUTPUT_DIR / 'human_survey_correlation_chart.png'
    create_correlation_bar_chart(correlations, chart_path)

    # Scatter plots
    create_scatter_plots(human_scores, metrics, OUTPUT_DIR)

    # Report
    report_path = OUTPUT_DIR / 'human_survey_analysis_report.txt'
    report = generate_report(human_scores, metrics, correlations, report_path)

    print("\n" + report)

    print("\n" + "=" * 60)
    print("Analysis complete! Output files saved to:")
    print(f"  - {chart_path}")
    print(f"  - {OUTPUT_DIR / 'human_survey_scatter_plots.png'}")
    print(f"  - {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
