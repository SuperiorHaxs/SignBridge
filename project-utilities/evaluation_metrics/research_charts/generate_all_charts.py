"""
Research Charts Generator for ASL-to-English Translation Study
==============================================================
Generates charts and CSVs for three experiments:
  1. Metric Validity (BLEU/BERT vs CTQI v3 vs Human Ratings)
  2. Error Propagation / Phase Transition (Top-3 Accuracy vs CTQI v3 with/without LLM)
  3. Domain Vocabulary Scaling (Vocab size vs % words meeting threshold)

All experiments use the n=53 evaluation dataset with CTQI v3 and real human ratings
from 5 raters via Google Forms survey.
"""

import json
import csv
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Paths
BASE = Path(__file__).resolve().parent.parent.parent.parent
EVAL_DIR = BASE / "project-utilities" / "evaluation_metrics" / "synthetic_evaluation"
EVAL_N53_DIR = EVAL_DIR / "evaluation_results_gemini_t1_n53_v4"
PROFILES_DIR = EVAL_DIR / "profiles"
OUTPUT_DIR = Path(__file__).resolve().parent
CHARTS_DIR = OUTPUT_DIR / "charts"
CSV_DIR = OUTPUT_DIR / "csvs"

CHARTS_DIR.mkdir(exist_ok=True)
CSV_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'bleu': '#e74c3c',
    'bert': '#3498db',
    'ctqi': '#27ae60',
    'human': '#9b59b6',
    'baseline': '#95a5a6',
    'model': '#27ae60',
    'accent': '#e67e22',
    'phase': '#e74c3c',
}


# =====================================================================
# DATA LOADING
# =====================================================================

def load_n53_data():
    """Load the n=53 evaluation dataset with CTQI v3 and human ratings (5 raters)."""
    with open(EVAL_N53_DIR / "evaluation_results.json") as f:
        eval_data = json.load(f)

    # Load human ratings from survey CSV
    SURVEY_PATH = Path(r"C:\Users\ashwi\Downloads\SignBridge Translation Quality Survey .csv\SignBridge Translation Quality Survey .csv")
    human_scores = []
    if SURVEY_PATH.exists():
        responses = []
        with open(SURVEY_PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                responses.append(row[4:57])

        for sent_idx in range(53):
            ratings = []
            for resp in responses:
                if sent_idx < len(resp) and resp[sent_idx]:
                    try:
                        ratings.append(int(resp[sent_idx]))
                    except ValueError:
                        pass
            human_scores.append(np.mean(ratings) if ratings else np.nan)
    else:
        print(f"  WARNING: Survey CSV not found at {SURVEY_PATH}")
        with open(EVAL_N53_DIR / "divergence_analysis_report.txt") as f:
            text = f.read()
        pattern = r'ID (\d+):.*?\n\s+Sentence:.*?\n\s+Human Rating: ([\d.]+)/5'
        matches = re.findall(pattern, text)
        human_map = {int(m[0]): float(m[1]) for m in matches}
        human_scores = [human_map.get(i, np.nan) for i in range(53)]

    rows = []
    for e in eval_data:
        eid = e['entry_id']
        h = human_scores[eid] if eid < len(human_scores) else np.nan
        rows.append({
            'entry_id': eid,
            'glosses': ' '.join(e['glosses']),
            'gloss_list': e['glosses'],
            'reference': e['reference_sentence'],
            'predicted': e['predicted_sentence'],
            'baseline_bleu': e['baseline_bleu'],
            'model_bleu': e['model_bleu'],
            'baseline_bert': e['baseline_bertscore'],
            'model_bert': e['model_bertscore'],
            'baseline_v3': e.get('baseline_composite_v3', e['baseline_composite_v2']),
            'model_v3': e.get('model_composite_v3', e['model_composite_v2']),
            'gloss_accuracy': e['gloss_accuracy'],
            'model_coverage_f1': e['model_coverage_f1'],
            'human_rating': h,
            'has_human': not np.isnan(h),
        })
    return rows


def load_profiles(filename):
    path = PROFILES_DIR / filename
    with open(path, 'r') as f:
        return json.load(f)


# =====================================================================
# EXPERIMENT 1: METRIC VALIDITY (n=53, 5 human raters, CTQI v3)
# =====================================================================

def exp1_metric_correlation_scatter(n53_data):
    """Scatter plots: real BLEU, BERTScore, CTQI v3 vs human ratings."""
    rated = [r for r in n53_data if r['has_human']]
    n = len(rated)

    humans_100 = [r['human_rating'] * 20 for r in rated]
    bleu_scores = [r['model_bleu'] for r in rated]
    bert_scores = [r['model_bert'] for r in rated]
    ctqi_v3 = [r['model_v3'] for r in rated]

    r_bleu_p, p_bleu_p = stats.pearsonr(humans_100, bleu_scores)
    r_bert_p, p_bert_p = stats.pearsonr(humans_100, bert_scores)
    r_ctqi_p, p_ctqi_p = stats.pearsonr(humans_100, ctqi_v3)

    r_bleu_s, _ = stats.spearmanr(humans_100, bleu_scores)
    r_bert_s, _ = stats.spearmanr(humans_100, bert_scores)
    r_ctqi_s, _ = stats.spearmanr(humans_100, ctqi_v3)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, scores, color, name, rp, pp, rs in [
        (axes[0], bleu_scores, COLORS['bleu'], 'BLEU', r_bleu_p, p_bleu_p, r_bleu_s),
        (axes[1], bert_scores, COLORS['bert'], 'BERTScore', r_bert_p, p_bert_p, r_bert_s),
        (axes[2], ctqi_v3, COLORS['ctqi'], 'CTQI v3', r_ctqi_p, p_ctqi_p, r_ctqi_s),
    ]:
        ax.scatter(humans_100, scores, c=color, alpha=0.6, edgecolors='white', s=60)
        z = np.polyfit(humans_100, scores, 1)
        xs = np.linspace(min(humans_100) - 5, max(humans_100) + 5, 100)
        ax.plot(xs, np.poly1d(z)(xs), '--', color=color, alpha=0.5)
        ax.set_xlabel('Human Rating (scaled 0-100)')
        ax.set_ylabel(f'{name} Score')
        sig = '***' if pp < 0.001 else ('**' if pp < 0.01 else ('*' if pp < 0.05 else 'ns'))
        ax.set_title(f'{name} vs Human (n={n})\nPearson r={rp:.3f}{sig}, Spearman r={rs:.3f}')
        ax.set_xlim(10, 110)
        ax.set_ylim(-5, 110)

    fig.suptitle('Experiment 1: Metric-Human Correlation (Real Data)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / 'exp1_metric_correlation_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    with open(CSV_DIR / 'exp1_metric_correlation.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Entry_ID', 'Glosses', 'Human_Rating_1to5', 'Human_Rating_0to100',
                         'Model_BLEU', 'Model_BERTScore', 'Model_CTQI_v3',
                         'Gloss_Accuracy', 'Reference', 'Predicted'])
        for r in rated:
            writer.writerow([r['entry_id'], r['glosses'], f"{r['human_rating']:.1f}",
                             f"{r['human_rating']*20:.0f}", f"{r['model_bleu']:.1f}",
                             f"{r['model_bert']:.1f}", f"{r['model_v3']:.1f}",
                             f"{r['gloss_accuracy']:.1f}", r['reference'], r['predicted']])

    return (r_bleu_p, r_bert_p, r_ctqi_p), (r_bleu_s, r_bert_s, r_ctqi_s), n


def exp1_correlation_bar(pearson_rs, spearman_rs, n):
    """Bar chart comparing Pearson and Spearman r for each metric."""
    metrics = ['BLEU', 'BERTScore', 'CTQI v3']
    colors = [COLORS['bleu'], COLORS['bert'], COLORS['ctqi']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, rs, ylabel, subtitle in [
        (ax1, pearson_rs, "Pearson r", "Pearson Correlation\n(linear relationship)"),
        (ax2, spearman_rs, "Spearman r", "Spearman Correlation\n(monotonic relationship)"),
    ]:
        bars = ax.bar(metrics, rs, color=colors, edgecolor='white', width=0.5)
        for bar, val in zip(bars, rs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'r={val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.axhline(y=0.7, color='gray', linestyle=':', alpha=0.5, label='Strong threshold')
        ax.legend(loc='upper left', fontsize=8)

    fig.suptitle(f'Experiment 1: Metric Validity \u2014 Correlation with Human Judgment (n={n}, 5 raters)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / 'exp1_correlation_bar.png', dpi=150, bbox_inches='tight')
    plt.close()

    with open(CSV_DIR / 'exp1_correlation_comparison.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Pearson_r', 'Spearman_r', 'Interpretation'])
        interps = [
            'Moderate \u2014 penalizes valid ASL reordering and paraphrase',
            'Strong \u2014 captures semantics but compressed range (floor ~88)',
            'Strongest \u2014 prerequisite chain: accuracy gates coverage gates plausibility',
        ]
        for m, pr, sr, interp in zip(metrics, pearson_rs, spearman_rs, interps):
            writer.writerow([m, f'{pr:.4f}', f'{sr:.4f}', interp])


def exp1_failure_cases(n53_data):
    """Table of real failure cases where BLEU diverges from human judgment."""
    rated = [r for r in n53_data if r['has_human']]

    cases = []
    for r in rated:
        h100 = r['human_rating'] * 20
        bleu_err = r['model_bleu'] - h100
        ctqi_err = r['model_v3'] - h100
        cases.append({**r, 'h100': h100, 'bleu_err': bleu_err, 'ctqi_err': ctqi_err,
                       'bleu_abs_err': abs(bleu_err), 'ctqi_abs_err': abs(ctqi_err)})

    cases.sort(key=lambda x: x['bleu_abs_err'], reverse=True)
    top_cases = cases[:6]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis('off')
    col_labels = ['Glosses', 'Translation', 'Human\n(0-100)', 'BLEU', 'CTQI v3',
                  'BLEU Error', 'Why BLEU Fails']
    table_data = []
    for c in top_cases:
        bleu_reason = 'Overestimates' if c['bleu_err'] > 0 else 'Underestimates'
        if c['model_bleu'] > 80 and c['h100'] < 50:
            why = f'{bleu_reason}: n-gram match despite wrong meaning'
        elif c['model_bleu'] < 30 and c['h100'] > 70:
            why = f'{bleu_reason}: valid paraphrase penalized'
        elif c['model_bleu'] > 50 and c['h100'] < 40:
            why = f'{bleu_reason}: surface overlap masks errors'
        else:
            why = f'{bleu_reason}: ASL reordering/insertion penalty'

        table_data.append([
            c['glosses'][:25], c['predicted'][:35],
            f"{c['h100']:.0f}", f"{c['model_bleu']:.1f}", f"{c['model_v3']:.1f}",
            f"{c['bleu_err']:+.1f}", why
        ])

    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center',
                     cellLoc='center', colColours=['#d5e8d4']*7)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)

    for i in range(len(top_cases)):
        cell = table[i+1, 5]
        val = top_cases[i]['bleu_err']
        cell.set_facecolor('#fde8e8' if abs(val) > 30 else '#fff3cd')

    ax.set_title('Experiment 1: Cases Where BLEU Diverges Most from Human Judgment (Real Data)',
                 fontweight='bold', fontsize=13, pad=20)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / 'exp1_failure_cases_table.png', dpi=150, bbox_inches='tight')
    plt.close()

    with open(CSV_DIR / 'exp1_failure_cases.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Entry_ID', 'Glosses', 'Reference', 'Predicted',
                         'Human_0to100', 'BLEU', 'BERTScore', 'CTQI_v3',
                         'BLEU_Error', 'CTQI_Error'])
        for c in cases:
            writer.writerow([c['entry_id'], c['glosses'], c['reference'], c['predicted'],
                             f"{c['h100']:.0f}", f"{c['model_bleu']:.1f}", f"{c['model_bert']:.1f}",
                             f"{c['model_v3']:.1f}", f"{c['bleu_err']:+.1f}", f"{c['ctqi_err']:+.1f}"])


# =====================================================================
# EXPERIMENT 2: ERROR PROPAGATION / PHASE TRANSITION (n=53, CTQI v3)
# =====================================================================

def _build_sign_lookup(profiles_38, profiles_77):
    sign_top3, sign_top1 = {}, {}
    for sign, data in profiles_38.items():
        sign_top3[sign.upper()] = data['top3_hit_rate']
        sign_top1[sign.upper()] = data['top1_accuracy']
    for sign, data in profiles_77.items():
        if sign.upper() not in sign_top3:
            sign_top3[sign.upper()] = data['top3_hit_rate']
            sign_top1[sign.upper()] = data['top1_accuracy']
    return sign_top1, sign_top3


def _compute_ctqi_v3(ga, cf1, plausibility):
    """Compute CTQI v3 from raw components."""
    return (ga / 100) * (cf1 / 100) * (0.5 + 0.5 * plausibility / 100 * ga / 100) * 100


def load_n150_data():
    """Load the n=150 threshold experiment dataset, computing CTQI v3 from components."""
    with open(EVAL_DIR / "threshold_experiment" / "results" / "evaluation_results.json") as f:
        eval_data = json.load(f)

    rows = []
    for e in eval_data:
        ga = e.get('gloss_accuracy', 0) or 0
        b_cf1 = e.get('baseline_coverage_f1', 0) or 0
        m_cf1 = e.get('model_coverage_f1', 0) or 0
        b_p = e.get('baseline_plausibility', 0) or 0
        m_p = e.get('model_plausibility', 0) or 0

        baseline_v3 = _compute_ctqi_v3(ga, b_cf1, b_p)
        model_v3 = _compute_ctqi_v3(ga, m_cf1, m_p)

        rows.append({
            'entry_id': e['entry_id'],
            'glosses': ' '.join(e['glosses']),
            'gloss_list': e['glosses'],
            'baseline_v3': baseline_v3,
            'model_v3': model_v3,
            'gloss_accuracy': ga,
        })
    return rows


def _plot_phase_transition(df, n_label, filename, csv_filename):
    """Reusable phase transition chart for any dataset."""
    bin_edges = [0, 70, 80, 85, 90, 101]
    bin_labels = ['<70%', '70-80%', '80-85%', '85-90%', '90-100%']
    df['bin'] = pd.cut(df['avg_top3'], bins=bin_edges, labels=bin_labels, right=False)

    bin_stats = df.groupby('bin', observed=False).agg(
        mean_improvement=('improvement', 'mean'),
        std_improvement=('improvement', 'std'),
        mean_baseline=('baseline_ctqi', 'mean'),
        mean_model=('model_ctqi', 'mean'),
        count=('improvement', 'size'),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # LEFT: Scatter of individual points colored by bin, with bin means overlaid
    bin_colors = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#27ae60']
    for i, (label, color) in enumerate(zip(bin_labels, bin_colors)):
        subset = df[df['bin'] == label]
        ax1.scatter(subset['avg_top3'], subset['improvement'], c=color,
                    alpha=0.5, s=50, edgecolors='white', zorder=3, label=f'{label} (n={len(subset)})')

    # Connect bin means with a line to show trend
    bin_centers = [60, 75, 82.5, 87.5, 95]
    valid_mask = bin_stats['count'] > 0
    ax1.plot([c for c, v in zip(bin_centers, valid_mask) if v],
             bin_stats.loc[valid_mask, 'mean_improvement'].tolist(),
             'ko-', markersize=8, linewidth=2.5, zorder=5, label='Bin mean')

    ax1.axhline(y=0, color='black', linewidth=0.8, alpha=0.4)
    ax1.axvline(x=85, color=COLORS['phase'], linestyle='--', alpha=0.6, linewidth=1.5)
    ax1.annotate('Top-3 ≥ 85%', xy=(85.5, ax1.get_ylim()[0] if ax1.get_ylim()[0] != 0 else -15),
                 fontsize=9, color=COLORS['phase'], ha='left', va='bottom',
                 fontstyle='italic', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    ax1.set_xlabel('Average Top-3 Accuracy (%)')
    ax1.set_ylabel('LLM Improvement (ΔCTQI v3)')
    ax1.set_title('Per-Sentence LLM Improvement\nvs Recognition Accuracy', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_xlim(max(df['avg_top3'].min() - 5, 0), 105)

    # RIGHT: Grouped bars — baseline vs model per bin
    x = np.arange(len(bin_stats))
    width = 0.35
    ax2.bar(x - width/2, bin_stats['mean_baseline'], width, color=COLORS['baseline'],
            label='Baseline (no LLM)', edgecolor='white')
    ax2.bar(x + width/2, bin_stats['mean_model'], width, color=COLORS['model'],
            label='With LLM', edgecolor='white')

    # Find the bin with the largest mean improvement for p-value annotation
    valid_bins = bin_stats[bin_stats['count'] >= 2]
    best_bin_idx = valid_bins['mean_improvement'].idxmax() if len(valid_bins) > 0 else None

    for i, row in bin_stats.iterrows():
        gap = row['mean_model'] - row['mean_baseline']
        top = max(row['mean_baseline'], row['mean_model'])
        if row['count'] > 0:
            label_text = f'Δ{gap:+.0f}'
            # Add p-value for the bin with largest improvement
            if i == best_bin_idx and row['count'] >= 2:
                subset = df[df['bin'] == row['bin']]
                t_stat, p_val = stats.ttest_rel(subset['model_ctqi'], subset['baseline_ctqi'])
                sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
                label_text += f'\np={p_val:.1e} {sig}'
                ax2.annotate(label_text, xy=(x[i], top + 2), ha='center', fontsize=9,
                             fontweight='bold', color=COLORS['model'],
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='#d5e8d4', alpha=0.8))
            else:
                ax2.annotate(label_text, xy=(x[i], top + 2), ha='center', fontsize=9,
                             fontweight='bold', color=COLORS['model'] if gap > 0 else COLORS['bleu'])
        ax2.text(x[i], -4, f'n={int(row["count"])}', ha='center', fontsize=8, color='gray')

    ax2.axvspan(2.5, 4.5, alpha=0.06, color=COLORS['ctqi'], zorder=0)
    ax2.axvline(x=2.5, color=COLORS['phase'], linestyle='--', alpha=0.5, linewidth=1.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels, fontsize=10)
    ax2.set_xlabel('Average Top-3 Accuracy Band')
    ax2.set_ylabel('Mean CTQI v3 Score')
    ax2.set_title('Baseline vs LLM by Accuracy Band\n(gap = LLM rescue effect)', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_ylim(-8, 110)

    fig.suptitle(f'Experiment 2: Error Propagation — LLM Rescue Effect Across Recognition Accuracy ({n_label})',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()

    df_out = df[['glosses', 'avg_top3', 'avg_top1', 'baseline_ctqi', 'model_ctqi', 'improvement']].copy()
    df_out.columns = ['Glosses', 'Avg_Top3_Accuracy', 'Avg_Top1_Accuracy',
                       'Baseline_CTQI_v3', 'Model_CTQI_v3', 'CTQI_v3_Improvement']
    df_out.round(2).to_csv(CSV_DIR / csv_filename, index=False)


def _build_phase_df(data, sign_top1, sign_top3):
    """Build a DataFrame with avg_top1, avg_top3, and improvement for phase transition charts."""
    entries = []
    for row in data:
        glosses = row['gloss_list']
        avg_top3 = np.mean([sign_top3.get(g.upper(), 50.0) for g in glosses])
        avg_top1 = np.mean([sign_top1.get(g.upper(), 30.0) for g in glosses])
        entries.append({
            'glosses': row['glosses'],
            'avg_top3': avg_top3, 'avg_top1': avg_top1,
            'baseline_ctqi': row['baseline_v3'], 'model_ctqi': row['model_v3'],
            'improvement': row['model_v3'] - row['baseline_v3'],
        })
    return pd.DataFrame(entries).sort_values('avg_top3')


def exp2_phase_transition(profiles_38, profiles_77, n53_data):
    """Phase transition chart for n=53 dataset."""
    sign_top1, sign_top3 = _build_sign_lookup(profiles_38, profiles_77)
    df = _build_phase_df(n53_data, sign_top1, sign_top3)
    _plot_phase_transition(df, 'n=53', 'exp2_phase_transition.png', 'exp2_phase_transition.csv')
    return df


def exp2_phase_transition_n150(profiles_38, profiles_77):
    """Phase transition chart for n=150 threshold experiment dataset."""
    sign_top1, sign_top3 = _build_sign_lookup(profiles_38, profiles_77)
    n150_data = load_n150_data()
    df = _build_phase_df(n150_data, sign_top1, sign_top3)
    _plot_phase_transition(df, 'n=150', 'exp2_phase_transition_n150.png', 'exp2_phase_transition_n150.csv')
    return df


def exp2_llm_rescue_effect(n53_data):
    """Horizontal bar chart of per-sentence LLM improvement on CTQI v3."""
    df = pd.DataFrame(n53_data)
    df['improvement'] = df['model_v3'] - df['baseline_v3']
    df = df.sort_values('improvement', ascending=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    y = range(len(df))
    ax.barh(y, df['improvement'],
            color=[COLORS['model'] if v > 0 else COLORS['bleu'] for v in df['improvement']],
            alpha=0.7, edgecolor='white')
    ax.set_yticks(y)
    ax.set_yticklabels([g[:25] for g in df['glosses']], fontsize=6)
    ax.set_xlabel('CTQI v3 Improvement (Model - Baseline)')
    ax.set_title('Experiment 2: LLM Rescue Effect per Sentence (n=53)\n(Positive = LLM improved translation quality)',
                 fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)

    pos = (df['improvement'] > 0).sum()
    neg = (df['improvement'] <= 0).sum()
    mean_imp = df['improvement'].mean()
    ax.annotate(f'Improved: {pos}/{len(df)} ({pos/len(df)*100:.0f}%)\n'
                f'Degraded: {neg}/{len(df)}\nMean: {mean_imp:+.1f}',
                xy=(0.98, 0.05), xycoords='axes fraction', ha='right', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='#d5e8d4', alpha=0.8))

    plt.tight_layout()
    fig.savefig(CHARTS_DIR / 'exp2_llm_rescue_effect.png', dpi=150, bbox_inches='tight')
    plt.close()

    df_out = df[['glosses', 'baseline_v3', 'model_v3', 'improvement']].copy()
    df_out.columns = ['Glosses', 'Baseline_CTQI_v3', 'Model_CTQI_v3', 'Improvement']
    df_out.round(2).to_csv(CSV_DIR / 'exp2_llm_rescue_effect.csv', index=False)


def _plot_heatmap(data, sign_top1, sign_top3, n_label, filename, csv_filename):
    """Reusable heatmap: Top-1 vs Top-3 accuracy, colored by CTQI v3."""
    entries = []
    for row in data:
        glosses = row['gloss_list']
        entries.append({
            'avg_top1': np.mean([sign_top1.get(g.upper(), 30.0) for g in glosses]),
            'avg_top3': np.mean([sign_top3.get(g.upper(), 50.0) for g in glosses]),
            'ctqi': row['model_v3'],
        })
    df = pd.DataFrame(entries)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Shade the "effective zone" above both thresholds
    ax.fill_between([70, 105], 85, 105, color=COLORS['ctqi'], alpha=0.08, zorder=1,
                    label='Effective zone (Top-1≥70, Top-3≥85)')
    # Draw threshold lines
    ax.axvline(x=70, color=COLORS['phase'], linestyle='--', alpha=0.6, linewidth=1.5)
    ax.axhline(y=85, color=COLORS['phase'], linestyle='--', alpha=0.6, linewidth=1.5)
    # Mark the intersection point
    ax.plot(70, 85, 's', color=COLORS['phase'], markersize=10, zorder=5, label='Threshold intersection (70, 85)')

    scatter = ax.scatter(df['avg_top1'], df['avg_top3'], c=df['ctqi'],
                         cmap='RdYlGn', s=80, edgecolors='white', vmin=0, vmax=100, zorder=3)
    plt.colorbar(scatter, ax=ax, label='CTQI v3 Score')

    # Annotate threshold lines
    ax.annotate('Top-1 ≥ 70%', xy=(71, 103), fontsize=9, color=COLORS['phase'],
                ha='left', va='top', fontstyle='italic', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    ax.annotate('Top-3 ≥ 85%', xy=(34, 85), fontsize=9, color=COLORS['phase'],
                ha='left', va='center', fontstyle='italic', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax.set_xlabel('Average Top-1 Accuracy (%)')
    ax.set_ylabel('Average Top-3 Accuracy (%)')
    ax.set_title(f'Experiment 2: Recognition Confidence vs Translation Quality ({n_label})\n'
                 'Shaded region = both thresholds met → high CTQI v3', fontweight='bold')
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.2, label='Top1 = Top3')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(min(df['avg_top1'].min() - 5, 30), 105)
    ax.set_ylim(min(df['avg_top3'].min() - 5, 50), 105)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()

    df.round(2).to_csv(CSV_DIR / csv_filename, index=False)


def exp2_heatmap(profiles_38, profiles_77, n53_data):
    """Heatmap for n=53 dataset."""
    sign_top1, sign_top3 = _build_sign_lookup(profiles_38, profiles_77)
    _plot_heatmap(n53_data, sign_top1, sign_top3, 'n=53',
                  'exp2_heatmap_top1_top3_ctqi.png', 'exp2_heatmap_data.csv')


def exp2_heatmap_n150(profiles_38, profiles_77):
    """Heatmap for n=150 threshold experiment dataset."""
    sign_top1, sign_top3 = _build_sign_lookup(profiles_38, profiles_77)
    n150_data = load_n150_data()
    _plot_heatmap(n150_data, sign_top1, sign_top3, 'n=150',
                  'exp2_heatmap_top1_top3_ctqi_n150.png', 'exp2_heatmap_data_n150.csv')


def exp2_statistical_significance(n53_data):
    """Statistical significance of LLM improvement grouped by baseline CTQI v3 band."""
    df = pd.DataFrame(n53_data)
    bins = [(0, 40), (40, 60), (60, 80), (80, 101)]
    bin_labels = ['<40', '40-60', '60-80', '80+']

    results = []
    for (lo, hi), label in zip(bins, bin_labels):
        mask = (df['baseline_v3'] >= lo) & (df['baseline_v3'] < hi)
        subset = df[mask]
        if len(subset) >= 2:
            t_stat, p_val = stats.ttest_rel(subset['model_v3'], subset['baseline_v3'])
            std = subset['baseline_v3'].std()
            d = (subset['model_v3'].mean() - subset['baseline_v3'].mean()) / std if std > 0 else 0
            results.append({
                'bin': label, 'n': len(subset),
                'baseline_mean': subset['baseline_v3'].mean(),
                'model_mean': subset['model_v3'].mean(),
                'improvement': subset['model_v3'].mean() - subset['baseline_v3'].mean(),
                't_stat': t_stat, 'p_value': p_val, 'cohens_d': d,
            })

    res_df = pd.DataFrame(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(res_df))
    width = 0.3
    ax1.bar(x - width/2, res_df['baseline_mean'], width, color=COLORS['baseline'],
            label='Baseline', edgecolor='white')
    ax1.bar(x + width/2, res_df['model_mean'], width, color=COLORS['model'],
            label='With LLM', edgecolor='white')
    for i, row in res_df.iterrows():
        sig = '***' if row['p_value'] < 0.001 else ('**' if row['p_value'] < 0.01 else ('*' if row['p_value'] < 0.05 else 'ns'))
        ax1.annotate(sig, xy=(x[i], max(row['baseline_mean'], row['model_mean']) + 3),
                     ha='center', fontsize=12, fontweight='bold',
                     color=COLORS['phase'] if sig != 'ns' else 'gray')
    ax1.set_xticks(x)
    ax1.set_xticklabels(res_df['bin'])
    ax1.set_xlabel('Baseline CTQI v3 Range')
    ax1.set_ylabel('Mean CTQI v3')
    ax1.set_title('LLM Improvement by Baseline Quality Band', fontweight='bold')
    ax1.legend()
    ax1.set_ylim(0, 120)

    bars = ax2.bar(res_df['bin'], res_df['cohens_d'],
                   color=[COLORS['ctqi'] if d > 0.8 else COLORS['accent'] if d > 0.5 else COLORS['baseline']
                          for d in res_df['cohens_d']], edgecolor='white')
    ax2.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label="Large effect (d=0.8)")
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label="Medium effect (d=0.5)")
    ax2.set_xlabel('Baseline CTQI v3 Range')
    ax2.set_ylabel("Cohen's d (effect size)")
    ax2.set_title('Effect Size of LLM Improvement', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)

    fig.suptitle('Experiment 2: Statistical Significance of LLM Rescue Effect (n=53, CTQI v3)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / 'exp2_statistical_significance.png', dpi=150, bbox_inches='tight')
    plt.close()

    res_df.round(4).to_csv(CSV_DIR / 'exp2_statistical_significance.csv', index=False)


# =====================================================================
# EXPERIMENT 3: DOMAIN VOCABULARY SCALING
# =====================================================================

def discover_all_profiles():
    """Auto-discover all per-sign profile JSONs in the profiles directory.

    Returns a list of dicts sorted by vocab size:
        [{'name': '38-class healthcare', 'vocab_size': 38, 'profiles': {...}}, ...]
    """
    import re as _re
    results = []
    for path in sorted(PROFILES_DIR.glob('per_sign_profiles_wlasl_*.json')):
        with open(path) as f:
            profiles = json.load(f)
        vocab_size = len(profiles)
        # Extract a friendly name from the filename
        # e.g. per_sign_profiles_wlasl_38_class_healthcare_model.json -> "38-class healthcare"
        stem = path.stem.replace('per_sign_profiles_wlasl_', '').replace('_model', '')
        # Try to extract the class count and descriptor
        m = _re.match(r'(\d+)_class(?:_(.+))?', stem)
        if m:
            name = f"{m.group(1)}-class"
            if m.group(2):
                name += f" {m.group(2).replace('_', ' ')}"
        else:
            name = stem.replace('_', ' ')
        results.append({
            'name': name,
            'vocab_size': vocab_size,
            'profiles': profiles,
            'path': path,
        })
    results.sort(key=lambda x: x['vocab_size'])
    return results


def _project_metric(real_sizes, real_values, target_sizes):
    """Project metrics to target vocab sizes using log-entropy degradation model.

    Uses logarithmic interpolation/extrapolation: accuracy degrades proportionally
    to log(vocab_size), reflecting information-theoretic entropy increase.
    Interpolates between real points; extrapolates beyond with the same log-rate.
    """
    # Fit: value = a - b * ln(vocab_size)
    log_real = np.log(real_sizes)
    # Linear regression on log-space
    b, a = np.polyfit(log_real, real_values, 1)
    projected = a + b * np.log(target_sizes)
    # Clamp to [0, 100]
    return np.clip(projected, 0, 100)


def exp3_multipoint_scaling(all_profiles):
    """Multi-point scaling curve with real data + proportional projections up to 107.

    Real data points are shown as filled markers; projected (mocked) points
    are shown as open markers with dashed connecting lines.
    """
    if len(all_profiles) < 1:
        print("  Skipping multi-point scaling: need at least 1 profile file.")
        return

    thresholds = [90, 85, 80, 75]

    # --- Collect real data ---
    real_sizes, real_labels = [], []
    real_top1s, real_top3s = [], []
    real_pct = {t: [] for t in thresholds}

    for prof in all_profiles:
        vs = prof['vocab_size']
        real_sizes.append(vs)
        real_labels.append(prof['name'])
        top3_vals = [d['top3_hit_rate'] for d in prof['profiles'].values()]
        top1_vals = [d['top1_accuracy'] for d in prof['profiles'].values()]
        real_top3s.append(np.mean(top3_vals))
        real_top1s.append(np.mean(top1_vals))
        for t in thresholds:
            pct = sum(1 for v in top3_vals if v >= t) / len(top3_vals) * 100
            real_pct[t].append(pct)

    real_sizes = np.array(real_sizes, dtype=float)
    real_top1s = np.array(real_top1s)
    real_top3s = np.array(real_top3s)

    # --- Define the full target range including mocked sizes ---
    # Include all real sizes + standard projected sizes up to 107
    target_sizes_set = set([38, 45, 55, 65, 77, 90, 107])
    for vs in real_sizes:
        target_sizes_set.add(int(vs))
    all_sizes = np.array(sorted(target_sizes_set), dtype=float)
    real_set = set(real_sizes.astype(int))

    # --- Project all metrics ---
    proj_top1 = _project_metric(real_sizes, real_top1s, all_sizes)
    proj_top3 = _project_metric(real_sizes, real_top3s, all_sizes)
    proj_pct = {}
    for t in thresholds:
        proj_pct[t] = _project_metric(real_sizes, np.array(real_pct[t]), all_sizes)

    # Override projected values with actual values at real data points
    for i, vs in enumerate(all_sizes):
        if int(vs) in real_set:
            idx = list(real_sizes.astype(int)).index(int(vs))
            proj_top1[i] = real_top1s[idx]
            proj_top3[i] = real_top3s[idx]
            for t in thresholds:
                proj_pct[t][i] = real_pct[t][idx]

    is_real = [int(vs) in real_set for vs in all_sizes]

    # --- Build labels for x-axis ---
    mock_label_map = {45: '45-class\n(projected)', 55: '55-class\n(projected)',
                      65: '65-class\n(projected)', 90: '90-class\n(projected)',
                      107: '107-class\n(projected)'}
    real_label_map = {int(vs): lbl for vs, lbl in zip(real_sizes, real_labels)}
    x_labels = []
    for vs in all_sizes:
        ivs = int(vs)
        if ivs in real_label_map:
            x_labels.append(f'{ivs}\n({real_label_map[ivs]})')
        elif ivs in mock_label_map:
            x_labels.append(mock_label_map[ivs])
        else:
            x_labels.append(f'{ivs}\n(projected)')

    # ================================================================
    # PLOT
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # LEFT: % meeting each threshold vs vocab size
    colors_t = ['#27ae60', '#3498db', '#e67e22', '#e74c3c']
    for t, color in zip(thresholds, colors_t):
        vals = proj_pct[t]
        # Dashed line through all points (projection envelope)
        ax1.plot(all_sizes, vals, '--', color=color, linewidth=1.5, alpha=0.5)
        # Real points: filled markers
        real_x = [all_sizes[i] for i in range(len(all_sizes)) if is_real[i]]
        real_y = [vals[i] for i in range(len(all_sizes)) if is_real[i]]
        ax1.plot(real_x, real_y, 'o', color=color, markersize=9, markeredgecolor='white',
                 markeredgewidth=1.5, zorder=5, label=f'Top-3 ≥ {t}% (real)')
        # Mock points: open markers
        mock_x = [all_sizes[i] for i in range(len(all_sizes)) if not is_real[i]]
        mock_y = [vals[i] for i in range(len(all_sizes)) if not is_real[i]]
        if mock_x:
            ax1.plot(mock_x, mock_y, 'o', color=color, markersize=7, markerfacecolor='white',
                     markeredgecolor=color, markeredgewidth=1.5, zorder=4)
        # Annotate each point
        for x, y, real in zip(all_sizes, vals, is_real):
            weight = 'bold' if real else 'normal'
            ax1.annotate(f'{y:.0f}%', xy=(x, y), textcoords='offset points',
                         xytext=(0, 9), ha='center', fontsize=7, color=color, fontweight=weight)

    ax1.set_xlabel('Vocabulary Size (# signs)')
    ax1.set_ylabel('% of Signs Meeting Threshold')
    ax1.set_title('Translation Reliability vs Vocabulary Size', fontweight='bold')
    ax1.legend(loc='lower left', fontsize=8, ncol=1)
    ax1.set_ylim(max(min(proj_pct[90]) - 15, 0), 105)
    ax1.set_xticks(all_sizes)
    ax1.set_xticklabels(x_labels, fontsize=7, ha='center')

    # RIGHT: Mean Top-1 and Top-3 accuracy vs vocab size
    # Dashed projection lines
    ax2.plot(all_sizes, proj_top3, '--', color=COLORS['ctqi'], linewidth=1.5, alpha=0.5)
    ax2.plot(all_sizes, proj_top1, '--', color=COLORS['bert'], linewidth=1.5, alpha=0.5)
    # Real points
    r_x = [all_sizes[i] for i in range(len(all_sizes)) if is_real[i]]
    r_top3 = [proj_top3[i] for i in range(len(all_sizes)) if is_real[i]]
    r_top1 = [proj_top1[i] for i in range(len(all_sizes)) if is_real[i]]
    ax2.plot(r_x, r_top3, 'o', color=COLORS['ctqi'], markersize=9, markeredgecolor='white',
             markeredgewidth=1.5, zorder=5, label='Mean Top-3 (real)')
    ax2.plot(r_x, r_top1, 's', color=COLORS['bert'], markersize=9, markeredgecolor='white',
             markeredgewidth=1.5, zorder=5, label='Mean Top-1 (real)')
    # Mock points
    m_x = [all_sizes[i] for i in range(len(all_sizes)) if not is_real[i]]
    m_top3 = [proj_top3[i] for i in range(len(all_sizes)) if not is_real[i]]
    m_top1 = [proj_top1[i] for i in range(len(all_sizes)) if not is_real[i]]
    if m_x:
        ax2.plot(m_x, m_top3, 'o', color=COLORS['ctqi'], markersize=7, markerfacecolor='white',
                 markeredgecolor=COLORS['ctqi'], markeredgewidth=1.5, zorder=4)
        ax2.plot(m_x, m_top1, 's', color=COLORS['bert'], markersize=7, markerfacecolor='white',
                 markeredgecolor=COLORS['bert'], markeredgewidth=1.5, zorder=4)
    # Annotate
    for x, y3, y1, real in zip(all_sizes, proj_top3, proj_top1, is_real):
        weight = 'bold' if real else 'normal'
        ax2.annotate(f'{y3:.1f}%', xy=(x, y3), textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=8, color=COLORS['ctqi'], fontweight=weight)
        ax2.annotate(f'{y1:.1f}%', xy=(x, y1), textcoords='offset points',
                     xytext=(0, -13), ha='center', fontsize=8, color=COLORS['bert'], fontweight=weight)

    ax2.set_xlabel('Vocabulary Size (# signs)')
    ax2.set_ylabel('Mean Accuracy (%)')
    ax2.set_title('Mean Recognition Accuracy vs Vocabulary Size', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_ylim(max(min(proj_top1) - 15, 0), 105)
    ax2.set_xticks(all_sizes)
    ax2.set_xticklabels(x_labels, fontsize=7, ha='center')

    # Add a legend note about filled vs open markers
    for ax in (ax1, ax2):
        ax.annotate('● = real data   ○ = projected (log-entropy model)',
                     xy=(0.5, -0.18), xycoords='axes fraction', ha='center', fontsize=8,
                     color='gray', fontstyle='italic')

    n_real = sum(is_real)
    n_proj = len(all_sizes) - n_real
    fig.suptitle(f'Experiment 3: Vocabulary Scaling — {n_real} Trained + {n_proj} Projected (up to 107 signs)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / 'exp3_multipoint_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()

    # CSV
    with open(CSV_DIR / 'exp3_multipoint_scaling.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Vocab_Size', 'Source', 'Model_Name', 'Mean_Top1', 'Mean_Top3'] +
                         [f'Pct_Top3_GE_{t}' for t in thresholds])
        for i, vs in enumerate(all_sizes):
            ivs = int(vs)
            source = 'real' if is_real[i] else 'projected'
            name = real_label_map.get(ivs, f'{ivs}-class (projected)')
            writer.writerow([
                ivs, source, name,
                f'{proj_top1[i]:.1f}', f'{proj_top3[i]:.1f}',
            ] + [f'{proj_pct[t][i]:.1f}' for t in thresholds])

    print(f"  Multi-point scaling: {n_real} real + {n_proj} projected points")


def exp3_accuracy_by_vocab_size():
    """Top-1 / Top-3 accuracy and % signs meeting threshold by vocabulary size."""
    # All data: total_classes = 25 + incremental domain signs
    # 25 common words included in every set

    # Healthcare scaling: (top1, top3, pct_meeting_threshold)
    healthcare = {
        60:  (80.24, 91.99, 87),   # 25 + 35
        68:  (79.07, 89.22, 86),   # 25 + 43
        82:  (76.48, 87.45, 84),   # 25 + 57
        102: (72.28, 86.00, 80),   # 25 + 77
        125: (69.16, 79.28, 65),   # 25 + 100
        150: (68.23, 74.66, 49),   # 25 + 125
    }

    # Other domains at +77 (total 102 classes each): (top1, top3, pct)
    domains_102 = {
        'Banking':       (69.57, 85.45, 80),
        'Restaurant':    (71.88, 86.56, 79),
        'Education':     (70.63, 85.54, 80),
        'Emergency':     (70.33, 85.45, 79),
        'Travel':        (68.56, 84.96, 81),
        'Shopping':      (70.68, 85.45, 82),
        'Job Interview': (71.39, 86.33, 78),
        'Gov Services':  (70.55, 86.67, 79),
    }

    # Average at 102: healthcare + 8 other domains
    all_102 = [healthcare[102]] + list(domains_102.values())
    avg_102_top1 = np.mean([v[0] for v in all_102])
    avg_102_top3 = np.mean([v[1] for v in all_102])
    avg_102_pct  = np.mean([v[2] for v in all_102])

    vocab_sizes = [25, 60, 68, 82, 102, 125, 150]
    top1_vals = [82.33, 80.24, 79.07, 76.48, avg_102_top1, 69.16, 68.23]
    top3_vals = [94.28, 91.99, 89.22, 87.45, avg_102_top3, 79.28, 74.66]
    pct_vals  = [90, 87, 86, 84, avg_102_pct, 65, 49]

    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()

    # Top-1 and Top-3 lines on left axis
    l1, = ax1.plot(vocab_sizes, top1_vals, 'o-', color='#3498db', markersize=12, linewidth=3,
                   label='Top-1 Accuracy', markeredgecolor='white', markeredgewidth=2, zorder=5)
    l2, = ax1.plot(vocab_sizes, top3_vals, 's-', color='#27ae60', markersize=12, linewidth=3,
                   label='Top-3 Accuracy', markeredgecolor='white', markeredgewidth=2, zorder=5)

    # % meeting threshold on right axis
    l3, = ax2.plot(vocab_sizes, pct_vals, 'D--', color='#8e44ad', markersize=10, linewidth=2.5,
                   label='% Signs Meeting Threshold', markeredgecolor='white', markeredgewidth=2, zorder=5)

    # Value labels for Top-1
    for i, vs in enumerate(vocab_sizes):
        ax1.text(vs, top1_vals[i] - 2.2, f'{top1_vals[i]:.1f}%', ha='center', va='top',
                 fontsize=13, fontweight='bold', fontfamily='Arial', color='#3498db')

    # Value labels for Top-3
    for i, vs in enumerate(vocab_sizes):
        ax1.text(vs, top3_vals[i] + 1.5, f'{top3_vals[i]:.1f}%', ha='center', va='bottom',
                 fontsize=13, fontweight='bold', fontfamily='Arial', color='#27ae60')

    # Value labels for % threshold
    for i, vs in enumerate(vocab_sizes):
        ax2.text(vs + 2.5, pct_vals[i], f'{pct_vals[i]:.0f}%', ha='left', va='center',
                 fontsize=13, fontweight='bold', fontfamily='Arial', color='#8e44ad')

    # Mark 102-class as averaged
    ax1.annotate('avg across\n9 domains', xy=(102, avg_102_top1), xytext=(115, avg_102_top1 + 6),
                 fontsize=11, fontweight='bold', fontfamily='Arial', color='gray',
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                 ha='left', va='bottom')

    # Exp 2 recovery thresholds
    ax1.axhline(y=70, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(22, 70.8, 'Top-1 Recovery Threshold (70%)', fontsize=12, fontweight='bold',
             fontfamily='Arial', color='#e74c3c', ha='left', va='bottom')

    ax1.axhline(y=85, color='#e74c3c', linestyle=':', linewidth=2, alpha=0.5)
    ax1.text(22, 85.8, 'Top-3 Recovery Threshold (85%)', fontsize=12, fontweight='bold',
             fontfamily='Arial', color='#e74c3c', ha='left', va='bottom')

    # Golden star at 80% threshold line on right axis
    ax2.axhline(y=80, color='#f39c12', linestyle='-', linewidth=1.5, alpha=0.4)
    ax2.plot([], [], '*', color='#f39c12', markersize=20, label='80% Signs Target')  # legend entry
    # Place stars where pct_vals crosses or is near 80
    for i, vs in enumerate(vocab_sizes):
        if abs(pct_vals[i] - 80) <= 2:
            ax2.plot(vs, pct_vals[i], '*', color='#f39c12', markersize=25, zorder=10,
                     markeredgecolor='#d68910', markeredgewidth=1)

    ax1.set_xlabel('Total Vocabulary Size (# Classes)', fontsize=16, fontweight='bold', fontfamily='Arial')
    ax1.set_ylabel('Accuracy (%)', fontsize=16, fontweight='bold', fontfamily='Arial', color='#2c3e50')
    ax2.set_ylabel('% Signs Meeting Threshold', fontsize=16, fontweight='bold', fontfamily='Arial', color='#8e44ad')

    ax1.set_title('Top-1 / Top-3 Accuracy and Threshold Coverage by Vocabulary Size',
                  fontsize=18, fontweight='bold', fontfamily='Arial', pad=15)

    # Combined legend
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=13, loc='upper right',
               prop={'weight': 'bold', 'family': 'Arial'})

    ax1.set_ylim(45, 100)
    ax2.set_ylim(40, 100)
    ax1.set_xlim(20, 160)
    ax1.tick_params(axis='both', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('Arial')
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('Arial')
        label.set_color('#8e44ad')
    ax1.grid(axis='y', alpha=0.3)

    fig.text(0.5, -0.02,
             'Each vocabulary includes 25 common signs shared across all domains.\n'
             'At 102 classes, metrics averaged across 9 domain-constrained vocabularies (25 common + 77 domain-specific).',
             ha='center', fontsize=14, fontweight='bold', fontfamily='Arial', fontstyle='italic')

    plt.tight_layout()
    fig.savefig(CHARTS_DIR / 'exp3_accuracy_by_vocab_size.png', dpi=150, bbox_inches='tight')
    plt.close()

    with open(CSV_DIR / 'exp3_accuracy_by_vocab_size.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Vocab_Size', 'Top1', 'Top3', 'Pct_Meeting_Threshold'])
        for vs, t1, t3, p in zip(vocab_sizes, top1_vals, top3_vals, pct_vals):
            writer.writerow([vs, f'{t1:.2f}', f'{t3:.2f}', f'{p:.1f}'])

    print(f"  Accuracy by vocab size: {len(vocab_sizes)} data points")


def exp3_domain_threshold_table():
    """Horizontal bar chart showing % signs meeting threshold per domain at 102 classes."""
    domains = [
        ('Job Interview', 85),
        ('Healthcare',    84),
        ('Restaurant',    83),
        ('Emergency',     83),
        ('Shopping',      81),
        ('Gov Services',  77),
        ('Education',     76),
        ('Banking',       73),
        ('Travel',        71),
    ]

    names = [d[0] for d in domains]
    pcts = [d[1] for d in domains]
    avg_pct = np.mean(pcts)

    fig, ax = plt.subplots(figsize=(7, 8))

    colors = ['#27ae60' if p >= 80 else '#f39c12' if p >= 75 else '#e74c3c' for p in pcts]

    bars = ax.barh(range(len(names)), pcts, color=colors, edgecolor='white', height=0.6)

    # Value labels
    for i, (bar, pct) in enumerate(zip(bars, pcts)):
        ax.text(pct + 0.8, i, f'{pct}%', va='center', ha='left',
                fontsize=16, fontweight='bold', fontfamily='Arial')

    # 80% target line
    ax.axvline(x=80, color='#f39c12', linestyle='--', linewidth=2.5, alpha=0.8)
    ax.text(80.5, -0.8, '80% Target', fontsize=14, fontweight='bold',
            fontfamily='Arial', color='#f39c12')

    # Average line
    ax.axvline(x=avg_pct, color='#3498db', linestyle=':', linewidth=2, alpha=0.7)
    ax.text(avg_pct + 0.5, len(names) - 0.3, f'Avg: {avg_pct:.0f}%', fontsize=13, fontweight='bold',
            fontfamily='Arial', color='#3498db')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=15, fontweight='bold', fontfamily='Arial')
    ax.set_xlabel('% Signs Meeting Threshold', fontsize=16, fontweight='bold', fontfamily='Arial')
    ax.set_title('Per-Domain Sign Recovery\nat 102 Classes',
                 fontsize=18, fontweight='bold', fontfamily='Arial', pad=15)
    ax.set_xlim(60, 95)
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelsize=14)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('Arial')
    ax.grid(axis='x', alpha=0.3)

    plt.subplots_adjust(bottom=0.18)
    fig.text(0.5, 0.03,
             '102 classes = 25 common signs + 77 domain-specific signs.\n'
             'Green >= 80%  |  Orange >= 75%  |  Red < 75%',
             ha='center', fontsize=13, fontweight='bold', fontfamily='Arial', fontstyle='italic')

    fig.savefig(CHARTS_DIR / 'exp3_domain_threshold_table.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Domain threshold table: {len(domains)} domains, avg {avg_pct:.0f}% meeting threshold")


def intro_accessibility_gap():
    """Infographic showing the accessibility gap statistics."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)

    # Title
    ax.text(5, 11.5, 'The Accessibility Gap', ha='center', va='top',
            fontsize=24, fontweight='bold', fontfamily='Arial', color='#2c3e50')

    stats = [
        (9.2, '72M', '#2980b9', '#ebf5fb', '#3498db',
         'Deaf individuals worldwide',
         '1.5B with hearing loss — projected 2.5B by 2050'),
        (6.8, '50:1', '#e67e22', '#fef9e7', '#f39c12',
         'Deaf-to-interpreter ratio in the US',
         '500K ASL users share ~10,000 certified interpreters'),
        (4.4, '< 20%', '#c0392b', '#fdedec', '#e74c3c',
         'of digital content accurately captioned',
         'Current systems translate words, not meaning'),
        (2.0, '300+', '#7d3c98', '#f4ecf7', '#8e44ad',
         'sign languages worldwide',
         'Each with unique grammar — no universal standard'),
    ]

    for y, big_num, num_color, bg_color, edge_color, line1, line2 in stats:
        box = FancyBboxPatch((0.3, y), 9.4, 2.0, boxstyle="round,pad=0.15",
                             facecolor=bg_color, edgecolor=edge_color, linewidth=2)
        ax.add_patch(box)
        ax.text(1.2, y + 1.3, big_num, ha='left', va='center',
                fontsize=42, fontweight='bold', fontfamily='Arial', color=num_color)
        ax.text(1.2, y + 0.5, f'{line1}\n{line2}',
                ha='left', va='center', fontsize=13, fontweight='bold', fontfamily='Arial', color='#2c3e50')

    # Source line
    ax.text(5, 1.2, 'Sources: World Federation of the Deaf, National Deaf Center,\nBoston University School of Public Health',
            ha='center', va='center', fontsize=10, fontfamily='Arial', color='gray', fontstyle='italic')

    fig.savefig(CHARTS_DIR / 'intro_accessibility_gap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Accessibility gap infographic saved")


def exp3_ablation_study():
    """Ablation study chart: keypoint subsets and augmentation volumes."""
    configs = [
        ('[1.0] Published Ours2', '27', 'Runtime', 71.07, None),
        ('[1.1] OpenHands Baseline', '27', 'Runtime', 41.23, 63.24),
        ('[1.3] + 75 keypoints', '75', 'Runtime', 43.56, 63.58),
        ('[1.4] + 83 keypoints', '83', 'Runtime', 50.33, 71.26),
        ('[1.5] + 100 keypoints', '100', 'Runtime', 49.54, 66.45),
        ('[1.6] + Fingermarks', '83/f', 'Runtime', 55.23, 75.62),
        ('[2.0] + 16x Aug', '83/f', '16x', 57.28, 77.48),
        ('[2.1] + 50x Aug', '83/f', '50x', 62.74, 79.64),
        ('[2.2] + 65x Aug', '83/f', '65x', 71.47, 82.56),
        ('[2.3] + 100x Aug', '83/f', '100x', 71.60, 81.86),
        ('[3.0] + Domain-constrained', '83/f', '65x', 80.97, 91.62),
    ]

    labels = [c[0] for c in configs]
    top1 = [c[3] for c in configs]
    top3 = [c[4] for c in configs]
    n = len(configs)

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(n)
    width = 0.35

    # Color phases
    phase_colors_top1 = []
    phase_colors_top3 = []
    for c in configs:
        if c[2] == 'Runtime' and c[1] in ('27', '75', '83', '100', '83/f'):
            phase_colors_top1.append('#3498db')
            phase_colors_top3.append('#85c1e9')
        elif c[2] != 'Runtime':
            phase_colors_top1.append('#e67e22')
            phase_colors_top3.append('#f0b27a')
        else:
            phase_colors_top1.append('#27ae60')
            phase_colors_top3.append('#7dcea0')

    bars1 = ax.bar(x - width/2, top1, width, color=phase_colors_top1, edgecolor='white', linewidth=1.5)

    # Top-3 bars (skip None)
    for i in range(n):
        if top3[i] is not None:
            ax.bar(x[i] + width/2, top3[i], width, color=phase_colors_top3[i], edgecolor='white', linewidth=1.5)

    # Value labels
    for i in range(n):
        ax.text(x[i] - width/2, top1[i] + 1, f'{top1[i]:.1f}%', ha='center', va='bottom',
                fontsize=12, fontweight='bold', fontfamily='Arial')
        if top3[i] is not None:
            ax.text(x[i] + width/2, top3[i] + 1, f'{top3[i]:.1f}%', ha='center', va='bottom',
                    fontsize=12, fontweight='bold', fontfamily='Arial')

    # Phase separators
    ax.axvline(x=5.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=9.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(2.5, 97, 'Keypoint Selection', ha='center', fontsize=14, fontweight='bold',
            fontfamily='Arial', color='#3498db')
    ax.text(7.5, 97, 'Augmentation Volume', ha='center', fontsize=14, fontweight='bold',
            fontfamily='Arial', color='#e67e22')
    ax.text(10, 97, 'Domain', ha='center', fontsize=14, fontweight='bold',
            fontfamily='Arial', color='#27ae60')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Top-1 Accuracy'),
        Patch(facecolor='#85c1e9', label='Top-3 Accuracy'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=14, prop={'weight': 'bold', 'family': 'Arial'})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=12, fontweight='bold', fontfamily='Arial')
    ax.set_ylabel('Accuracy (%)', fontsize=16, fontweight='bold', fontfamily='Arial')
    ax.set_title('Experiment 3: Recognition Design Parameter Ablation',
                 fontsize=18, fontweight='bold', fontfamily='Arial', pad=20)
    ax.set_ylim(0, 105)
    ax.tick_params(axis='y', labelsize=14)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('Arial')

    # Description below chart
    fig.text(0.5, -0.02,
             'Progressive ablation from baseline (27 keypoints) through keypoint expansion, augmentation scaling,\n'
             'and domain-constrained vocabulary. Each configuration builds on the previous best.',
             ha='center', fontsize=13, fontweight='bold', fontfamily='Arial', fontstyle='italic')

    plt.tight_layout()
    fig.savefig(CHARTS_DIR / 'exp3_ablation_study.png', dpi=150, bbox_inches='tight')
    plt.close()

    # CSV
    with open(CSV_DIR / 'exp3_ablation_study.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Config', 'Keypoints', 'Augmentation', 'Top1', 'Top3'])
        for c in configs:
            writer.writerow([c[0], c[1], c[2], c[3], c[4] if c[4] else ''])

    print(f"  Ablation study: {n} configurations")


def dataset_domain_breakdown():
    """Horizontal bar chart of domain-constrained vocabularies with gloss counts and overlap."""
    domain_dir = BASE / "datasets" / "domain-specific"

    domains = {}
    for f in sorted(domain_dir.glob("*.json")):
        if f.name == 'generation_report.json':
            continue
        with open(f) as fh:
            data = json.load(fh)
        name = data.get('scenario', f.stem.replace('_', ' ').title())
        classes = set(data.get('classes', []))
        domains[name] = classes

    # Sort by size descending
    sorted_domains = sorted(domains.items(), key=lambda x: len(x[1]), reverse=True)
    names = [d[0] for d in sorted_domains]
    counts = [len(d[1]) for d in sorted_domains]

    # Compute overlap: for each domain, how many glosses are shared with at least one other domain
    all_other = {}
    for name, glosses in domains.items():
        others = set()
        for other_name, other_glosses in domains.items():
            if other_name != name:
                others.update(other_glosses)
        all_other[name] = len(glosses & others)

    overlaps = [all_other[n] for n in names]
    unique_counts = [c - o for c, o in zip(counts, overlaps)]

    # Color by domain type
    healthcare_color = '#e74c3c'
    general_color = '#3498db'
    colors = [healthcare_color if 'Doctor' in n or 'Emergency' in n else general_color for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(range(len(names)), unique_counts, color=colors, alpha=0.85, edgecolor='white', label='Unique to domain')
    ax.barh(range(len(names)), overlaps, left=unique_counts, color=colors, alpha=0.35, edgecolor='white', label='Shared with other domains')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Number of Signs (Glosses)')
    ax.set_title('Domain-Constrained Vocabulary Breakdown\n10 Domains, 307 Unique Signs from WLASL 2000',
                 fontweight='bold')
    ax.invert_yaxis()

    # Annotate counts
    for i, (c, o) in enumerate(zip(counts, overlaps)):
        ax.text(c + 0.3, i, f'{c}', va='center', fontsize=9, fontweight='bold')
        if o > 0:
            ax.text(c + 0.3, i + 0.3, f'({o} shared)', va='center', fontsize=7, color='gray')

    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / 'dataset_domain_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()

    # CSV
    with open(CSV_DIR / 'dataset_domain_breakdown.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Domain', 'Total_Glosses', 'Unique_Glosses', 'Shared_Glosses'])
        for name, count, unique, overlap in zip(names, counts, unique_counts, overlaps):
            writer.writerow([name, count, unique, overlap])

    total_unique = len(set().union(*domains.values()))
    print(f"  Domain breakdown: {len(domains)} domains, {total_unique} unique glosses, avg {sum(counts)/len(counts):.0f}/domain")


def exp3_vocabulary_scaling(profiles_38, profiles_77):
    """Vocabulary size vs % words meeting Top-3 accuracy threshold."""
    all_signs = {}
    for sign, data in profiles_38.items():
        all_signs[sign] = {'top1': data['top1_accuracy'], 'top3': data['top3_hit_rate'], 'model': '38-class'}
    for sign, data in profiles_77.items():
        if sign not in all_signs:
            all_signs[sign] = {'top1': data['top1_accuracy'], 'top3': data['top3_hit_rate'], 'model': '77-class'}

    sorted_signs = sorted(all_signs.items(), key=lambda x: x[1]['top3'], reverse=True)

    thresholds = [90, 85, 80, 75]
    results = {t: [] for t in thresholds}
    for vocab_size in range(5, len(sorted_signs) + 1):
        current = sorted_signs[:vocab_size]
        for t in thresholds:
            pct = sum(1 for _, d in current if d['top3'] >= t) / vocab_size * 100
            results[t].append((vocab_size, pct))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_t = ['#27ae60', '#3498db', '#e67e22', '#e74c3c']
    for t, color in zip(thresholds, colors_t):
        xs = [r[0] for r in results[t]]
        ys = [r[1] for r in results[t]]
        ax.plot(xs, ys, '-o', color=color, markersize=3, label=f'Top-3 >= {t}%', linewidth=2)

    ax.axhline(y=80, color='black', linestyle=':', alpha=0.4, label='80% words target')
    for t, color in zip(thresholds, colors_t):
        for vocab_size, pct in results[t]:
            if pct < 80:
                ax.annotate(f'{vocab_size-1}w', xy=(vocab_size-1, 80), fontsize=8,
                           color=color, fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color=color),
                           xytext=(vocab_size + 3, 83 + thresholds.index(t) * 4))
                break

    ax.set_xlabel('Domain Vocabulary Size (# words)')
    ax.set_ylabel('% of Words Meeting Top-3 Threshold')
    ax.set_title('Experiment 3: Domain Vocabulary Scaling\nTranslation Reliability Degrades with Vocabulary Size',
                 fontweight='bold')
    ax.legend(loc='lower left')
    ax.set_ylim(0, 105)
    ax.set_xlim(3, len(sorted_signs) + 2)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / 'exp3_vocabulary_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()

    with open(CSV_DIR / 'exp3_per_sign_accuracy.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Sign', 'Top1_Accuracy', 'Top3_Accuracy', 'Source_Model'])
        for i, (sign, data) in enumerate(sorted_signs, 1):
            writer.writerow([i, sign, data['top1'], data['top3'], data['model']])

    with open(CSV_DIR / 'exp3_scaling_curve.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Vocab_Size'] + [f'Pct_Meeting_Top3_{t}' for t in thresholds])
        for idx in range(len(results[thresholds[0]])):
            row = [results[thresholds[0]][idx][0]]
            for t in thresholds:
                row.append(f"{results[t][idx][1]:.1f}")
            writer.writerow(row)

    return sorted_signs


def exp3_domain_comparison(profiles_38, profiles_77):
    """Compare healthcare (38) vs expanded (77) for shared signs."""
    common = set(profiles_38.keys()) & set(profiles_77.keys())
    comparison = []
    for sign in sorted(common):
        comparison.append({
            'sign': sign, 'top1_38': profiles_38[sign]['top1_accuracy'],
            'top3_38': profiles_38[sign]['top3_hit_rate'],
            'top1_77': profiles_77[sign]['top1_accuracy'],
            'top3_77': profiles_77[sign]['top3_hit_rate'],
            'top1_change': profiles_77[sign]['top1_accuracy'] - profiles_38[sign]['top1_accuracy'],
            'top3_change': profiles_77[sign]['top3_hit_rate'] - profiles_38[sign]['top3_hit_rate'],
        })
    df = pd.DataFrame(comparison).sort_values('top3_change')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    y = range(len(df))
    ax1.barh(y, df['top3_38'], height=0.4, color=COLORS['model'], alpha=0.7,
             label='38-class (healthcare)', align='edge')
    ax1.barh([i - 0.4 for i in y], df['top3_77'], height=0.4, color=COLORS['accent'], alpha=0.7,
             label='77-class (expanded)', align='edge')
    ax1.set_yticks(y)
    ax1.set_yticklabels(df['sign'], fontsize=7)
    ax1.set_xlabel('Top-3 Accuracy (%)')
    ax1.set_title('Top-3 Accuracy: 38 vs 77 Class', fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_xlim(0, 105)

    colors = [COLORS['model'] if v >= 0 else COLORS['bleu'] for v in df['top3_change']]
    ax2.barh(y, df['top3_change'], color=colors, alpha=0.7, edgecolor='white')
    ax2.set_yticks(y)
    ax2.set_yticklabels(df['sign'], fontsize=7)
    ax2.set_xlabel('Change in Top-3 Accuracy (pp)')
    ax2.set_title('Accuracy Change: 38 -> 77 Classes', fontweight='bold')
    ax2.axvline(x=0, color='black', linewidth=0.8)

    mean_change = df['top3_change'].mean()
    degraded = (df['top3_change'] < 0).sum()
    ax2.annotate(f'Mean change: {mean_change:+.1f}pp\nDegraded: {degraded}/{len(df)}',
                 xy=(0.98, 0.05), xycoords='axes fraction', ha='right', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='#fde8e8', alpha=0.8))

    fig.suptitle('Experiment 3: Domain Expansion Impact on Per-Sign Accuracy',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / 'exp3_domain_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    df.round(2).to_csv(CSV_DIR / 'exp3_domain_comparison.csv', index=False)


def exp3_accuracy_distribution(profiles_38, profiles_77):
    """Histogram of Top-3 accuracy for 38 vs 77 class."""
    top3_38 = [d['top3_hit_rate'] for d in profiles_38.values()]
    top3_77 = [d['top3_hit_rate'] for d in profiles_77.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bins = np.arange(0, 105, 5)

    for ax, data, color, name in [
        (ax1, top3_38, COLORS['model'], '38-Class Healthcare'),
        (ax2, top3_77, COLORS['accent'], '77-Class Expanded'),
    ]:
        ax.hist(data, bins=bins, color=color, alpha=0.7, edgecolor='white')
        ax.axvline(x=np.mean(data), color=color, linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(data):.1f}%')
        ax.axvline(x=90, color=COLORS['phase'], linestyle=':', alpha=0.5, label='90% threshold')
        pct_above = sum(1 for v in data if v >= 90) / len(data) * 100
        ax.set_xlabel('Top-3 Accuracy (%)')
        ax.set_ylabel('Number of Signs')
        ax.set_title(f'{name}\n{pct_above:.0f}% of signs >= 90% Top-3', fontweight='bold')
        ax.legend(fontsize=8)

    fig.suptitle('Experiment 3: Top-3 Accuracy Distribution Shift with Vocabulary Size',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / 'exp3_accuracy_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    with open(CSV_DIR / 'exp3_accuracy_distribution.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Statistic', '38_Class_Healthcare', '77_Class_Expanded'])
        writer.writerow(['Vocab Size', 38, 77])
        writer.writerow(['Mean Top-3 Accuracy', f'{np.mean(top3_38):.1f}', f'{np.mean(top3_77):.1f}'])
        writer.writerow(['Median Top-3 Accuracy', f'{np.median(top3_38):.1f}', f'{np.median(top3_77):.1f}'])
        writer.writerow(['Std Dev', f'{np.std(top3_38):.1f}', f'{np.std(top3_77):.1f}'])
        writer.writerow(['% Signs >= 90% Top-3',
                         f'{sum(1 for v in top3_38 if v >= 90)/len(top3_38)*100:.1f}',
                         f'{sum(1 for v in top3_77 if v >= 90)/len(top3_77)*100:.1f}'])
        writer.writerow(['% Signs >= 80% Top-3',
                         f'{sum(1 for v in top3_38 if v >= 80)/len(top3_38)*100:.1f}',
                         f'{sum(1 for v in top3_77 if v >= 80)/len(top3_77)*100:.1f}'])
        writer.writerow(['# Signs < 50% Top-3',
                         sum(1 for v in top3_38 if v < 50),
                         sum(1 for v in top3_77 if v < 50)])


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("Loading data...")
    n53_data = load_n53_data()
    profiles_38 = load_profiles('per_sign_profiles_wlasl_38_class_healthcare_model.json')
    profiles_77 = load_profiles('per_sign_profiles_wlasl_77_class_model.json')

    n_rated = sum(1 for r in n53_data if r['has_human'])
    print(f"  n=53 dataset: {len(n53_data)} entries, {n_rated} with human ratings, CTQI v3 throughout")

    print("\n=== EXPERIMENT 1: Metric Validity (n=53, CTQI v3) ===")
    pearson_rs, spearman_rs, n = exp1_metric_correlation_scatter(n53_data)
    print(f"  Pearson r:  BLEU={pearson_rs[0]:.3f}, BERT={pearson_rs[1]:.3f}, CTQI v3={pearson_rs[2]:.3f}")
    print(f"  Spearman r: BLEU={spearman_rs[0]:.3f}, BERT={spearman_rs[1]:.3f}, CTQI v3={spearman_rs[2]:.3f}")
    exp1_correlation_bar(pearson_rs, spearman_rs, n)
    exp1_failure_cases(n53_data)
    print("  -> 3 charts saved")

    print("\n=== EXPERIMENT 2: Error Propagation / Phase Transition (n=53, CTQI v3) ===")
    exp2_phase_transition(profiles_38, profiles_77, n53_data)
    exp2_llm_rescue_effect(n53_data)
    exp2_heatmap(profiles_38, profiles_77, n53_data)
    exp2_statistical_significance(n53_data)
    print("  -> 4 charts saved")

    print("\n=== EXPERIMENT 2b: Phase Transition + Heatmap (n=150 threshold experiment) ===")
    exp2_phase_transition_n150(profiles_38, profiles_77)
    exp2_heatmap_n150(profiles_38, profiles_77)
    print("  -> 2 charts saved")

    print("\n=== EXPERIMENT 3: Accuracy by Vocab Size ===")
    exp3_accuracy_by_vocab_size()
    exp3_domain_threshold_table()
    print("  -> 2 charts saved")

    print("\n=== EXPERIMENT 3: Ablation Study ===")
    exp3_ablation_study()
    print("  -> 1 chart saved")

    print("\n=== INTRO: Accessibility Gap ===")
    intro_accessibility_gap()
    print("  -> 1 chart saved")

    print("\n=== DATASET: Domain Breakdown ===")
    dataset_domain_breakdown()
    print("  -> 1 chart saved")

    print("\n=== EXPERIMENT 3: Domain Vocabulary Scaling ===")
    all_profiles = discover_all_profiles()
    print(f"  Discovered {len(all_profiles)} profile files: {[p['name'] for p in all_profiles]}")
    exp3_multipoint_scaling(all_profiles)
    exp3_vocabulary_scaling(profiles_38, profiles_77)
    exp3_domain_comparison(profiles_38, profiles_77)
    exp3_accuracy_distribution(profiles_38, profiles_77)
    print("  -> 4 charts saved")

    print(f"\nCharts saved to: {CHARTS_DIR}")
    print(f"CSVs saved to:   {CSV_DIR}")
    print(f"Total: 13 charts (.png), 12 CSVs (.csv)")


if __name__ == '__main__':
    main()
