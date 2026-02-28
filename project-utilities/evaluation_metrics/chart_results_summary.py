#!/usr/bin/env python3
"""
Results Summary and Statistical Analysis - Trifold Poster Panel.
Combines all 7 statistical rigor questions with brief answers and mini charts.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

fig = plt.figure(figsize=(11, 17))
gs = GridSpec(7, 2, figure=fig, width_ratios=[1.1, 1], hspace=0.45, wspace=0.3,
              left=0.06, right=0.97, top=0.95, bottom=0.02)

# Title
fig.suptitle('Results Summary and Statistical Analysis',
             fontsize=20, fontweight='bold', y=0.98)

# =========================================================================
# Helpers
# =========================================================================
def text_panel(ax, q_num, question, answer):
    """Create a text-only Q&A panel."""
    ax.axis('off')
    ax.text(0, 1.0, f'Q{q_num}', fontsize=16, fontweight='bold', color='#2E7D32',
            transform=ax.transAxes, va='top')
    ax.text(0.08, 1.0, question, fontsize=11, fontweight='bold', color='#333333',
            transform=ax.transAxes, va='top', wrap=True,
            bbox=dict(boxstyle='square,pad=0', facecolor='none', edgecolor='none'))
    ax.text(0.0, 0.55, answer, fontsize=10, color='#444444',
            transform=ax.transAxes, va='top', wrap=True, linespacing=1.4)


# =========================================================================
# Q1: LLM Pipeline Improvement (mini dumbbell)
# =========================================================================
ax1_text = fig.add_subplot(gs[0, 0])
ax1_text.axis('off')
ax1_text.text(0, 1.0, 'Q1', fontsize=16, fontweight='bold', color='#2E7D32',
              transform=ax1_text.transAxes, va='top')
ax1_text.text(0.08, 1.0, 'Is the LLM pipeline improvement\nstatistically significant?',
              fontsize=11, fontweight='bold', color='#333333',
              transform=ax1_text.transAxes, va='top')
ax1_text.text(0.0, 0.45, 'Yes. All 5 metrics show significant\nimprovement (all p < 0.01).\nCTQI v3: 45.5 \u2192 74.0 (+28.6, p < 0.001)',
              fontsize=10, color='#444444', transform=ax1_text.transAxes, va='top', linespacing=1.4)

ax1_chart = fig.add_subplot(gs[0, 1])
metrics_q1 = ['CTQI v3', 'PTR', 'P', 'CF1', 'GA']
base_q1 = [45.5, 49.1, 32.3, 81.0, 81.9]
llm_q1 = [74.0, 66.0, 88.8, 87.7, 87.6]
y_q1 = range(len(metrics_q1))
for i, (y, b, l) in enumerate(zip(y_q1, base_q1, llm_q1)):
    ax1_chart.plot([b, l], [y, y], color='#999999', linewidth=2.5, zorder=1)
    ax1_chart.scatter(b, y, color='#999999', s=60, zorder=2)
    ax1_chart.scatter(l, y, color='#2E7D32', s=60, zorder=2)
ax1_chart.set_yticks(y_q1)
ax1_chart.set_yticklabels(metrics_q1, fontsize=9, fontweight='bold')
ax1_chart.set_xlim(25, 100)
ax1_chart.set_xlabel('Score', fontsize=9)
ax1_chart.tick_params(axis='x', labelsize=8)
ax1_chart.xaxis.grid(True, linestyle='--', alpha=0.3)
ax1_chart.set_axisbelow(True)
ax1_chart.spines['top'].set_visible(False)
ax1_chart.spines['right'].set_visible(False)
legend_q1 = [Line2D([0],[0],marker='o',color='w',markerfacecolor='#999999',markersize=6,label='Baseline'),
             Line2D([0],[0],marker='o',color='w',markerfacecolor='#2E7D32',markersize=6,label='With LLM')]
ax1_chart.legend(handles=legend_q1, fontsize=8, loc='lower right')

# =========================================================================
# Q2: CTQI v3 vs BLEU/BERT (mini bar)
# =========================================================================
ax2_text = fig.add_subplot(gs[1, 0])
ax2_text.axis('off')
ax2_text.text(0, 1.0, 'Q2', fontsize=16, fontweight='bold', color='#2E7D32',
              transform=ax2_text.transAxes, va='top')
ax2_text.text(0.08, 1.0, 'Does CTQI v3 beat BLEU and\nBERTScore on human correlation?',
              fontsize=11, fontweight='bold', color='#333333',
              transform=ax2_text.transAxes, va='top')
ax2_text.text(0.0, 0.45, 'Yes. Hotelling-Williams test:\nvs BLEU: +0.18 (p < 0.0001 ***)\nvs BERT: +0.14 (p < 0.0001 ***)',
              fontsize=10, color='#444444', transform=ax2_text.transAxes, va='top', linespacing=1.4)

ax2_chart = fig.add_subplot(gs[1, 1])
metrics_q2 = ['BLEU', 'BERTScore', 'CTQI v3']
vals_q2 = [0.7584, 0.8073, 0.9427]
colors_q2 = ['#999999', '#999999', '#2E7D32']
bars_q2 = ax2_chart.barh(metrics_q2, vals_q2, color=colors_q2, height=0.5, edgecolor='white')
for bar, val in zip(bars_q2, vals_q2):
    ax2_chart.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'r = {val:.3f}', va='center', fontsize=9, fontweight='bold')
ax2_chart.set_xlim(0.6, 1.05)
ax2_chart.set_xlabel('Pearson r with Human Ratings', fontsize=9)
ax2_chart.tick_params(axis='both', labelsize=9)
ax2_chart.xaxis.grid(True, linestyle='--', alpha=0.3)
ax2_chart.set_axisbelow(True)
ax2_chart.spines['top'].set_visible(False)
ax2_chart.spines['right'].set_visible(False)

# =========================================================================
# Q3: Plausibility contribution
# =========================================================================
ax3_text = fig.add_subplot(gs[2, 0])
ax3_text.axis('off')
ax3_text.text(0, 1.0, 'Q3', fontsize=16, fontweight='bold', color='#2E7D32',
              transform=ax3_text.transAxes, va='top')
ax3_text.text(0.08, 1.0, 'Does plausibility significantly\nimprove CTQI v3 over GA \u00d7 CF1?',
              fontsize=11, fontweight='bold', color='#333333',
              transform=ax3_text.transAxes, va='top')
ax3_text.text(0.0, 0.45, 'Yes. GA\u00d7CF1 alone: r = 0.9206.\nWith plausibility: r = 0.9427.\nHotelling-Williams: t(50) = 2.68, p = 0.0098.',
              fontsize=10, color='#444444', transform=ax3_text.transAxes, va='top', linespacing=1.4)

ax3_chart = fig.add_subplot(gs[2, 1])
bars_q3 = ax3_chart.barh(['GA \u00d7 CF1', 'CTQI v3\n(+Plausibility)'],
                          [0.9206, 0.9427],
                          color=['#999999', '#2E7D32'], height=0.45, edgecolor='white')
for bar, val in zip(bars_q3, [0.9206, 0.9427]):
    ax3_chart.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
ax3_chart.set_xlim(0.90, 0.96)
ax3_chart.set_xlabel('Pearson r', fontsize=9)
ax3_chart.tick_params(axis='both', labelsize=9)
ax3_chart.xaxis.grid(True, linestyle='--', alpha=0.3)
ax3_chart.set_axisbelow(True)
ax3_chart.spines['top'].set_visible(False)
ax3_chart.spines['right'].set_visible(False)
# Add p-value annotation
ax3_chart.annotate('p = 0.0098 **', xy=(0.935, 0.85), fontsize=9, fontweight='bold',
                   color='#2E7D32', xycoords='axes fraction', ha='center')

# =========================================================================
# Q4: Floor parameter robustness (mini curve)
# =========================================================================
ax4_text = fig.add_subplot(gs[3, 0])
ax4_text.axis('off')
ax4_text.text(0, 1.0, 'Q4', fontsize=16, fontweight='bold', color='#2E7D32',
              transform=ax4_text.transAxes, va='top')
ax4_text.text(0.08, 1.0, 'Is the floor parameter (\u03b1 = 0.5)\nrobust or overfit?',
              fontsize=11, fontweight='bold', color='#333333',
              transform=ax4_text.transAxes, va='top')
ax4_text.text(0.0, 0.45, 'Robust. Optimal \u03b1 = 0.35 gains only\n+0.001 over \u03b1 = 0.50. Broad plateau\nfrom 0.2\u20130.6. Not sensitive to exact value.',
              fontsize=10, color='#444444', transform=ax4_text.transAxes, va='top', linespacing=1.4)

ax4_chart = fig.add_subplot(gs[3, 1])
# Simplified curve (approximate shape from actual data)
alphas = np.linspace(0, 1, 101)
# Approximate the Pearson curve shape
pearson_approx = 0.9437 - 0.004 * (alphas - 0.35)**2 - 0.015 * (alphas - 0.35)**4
pearson_approx = np.clip(pearson_approx, 0.92, 0.9437)
ax4_chart.plot(alphas, pearson_approx, color='#2E7D32', linewidth=2)
ax4_chart.axvline(x=0.50, color='#999999', linestyle='--', linewidth=1.5, alpha=0.7)
ax4_chart.scatter(0.35, 0.9437, color='#2E7D32', s=50, zorder=5, edgecolors='black', linewidths=1)
ax4_chart.annotate('optimal\n\u03b1=0.35', (0.35, 0.9437), textcoords="offset points",
                   xytext=(-35, -20), fontsize=8, fontweight='bold', color='#2E7D32')
ax4_chart.annotate('\u03b1=0.50', (0.50, 0.9427), textcoords="offset points",
                   xytext=(5, -15), fontsize=8, color='#666666')
ax4_chart.set_xlabel('Floor Parameter (\u03b1)', fontsize=9)
ax4_chart.set_ylabel('Pearson r', fontsize=9)
ax4_chart.tick_params(axis='both', labelsize=8)
ax4_chart.set_xlim(-0.02, 1.02)
ax4_chart.set_ylim(0.918, 0.948)
ax4_chart.yaxis.grid(True, linestyle='--', alpha=0.3)
ax4_chart.set_axisbelow(True)
ax4_chart.spines['top'].set_visible(False)
ax4_chart.spines['right'].set_visible(False)

# =========================================================================
# Q5: 83 vs 27 keypoints (mini bar)
# =========================================================================
ax5_text = fig.add_subplot(gs[4, 0])
ax5_text.axis('off')
ax5_text.text(0, 1.0, 'Q5', fontsize=16, fontweight='bold', color='#2E7D32',
              transform=ax5_text.transAxes, va='top')
ax5_text.text(0.08, 1.0, 'Do 83 keypoints improve accuracy\nover the 27-keypoint baseline?',
              fontsize=11, fontweight='bold', color='#333333',
              transform=ax5_text.transAxes, va='top')
ax5_text.text(0.0, 0.45, 'Yes. 71.6% \u2192 81.0% (+9.4 pp).\nTwo-proportion z-test:\nz = 4.14, p < 0.0001.',
              fontsize=10, color='#444444', transform=ax5_text.transAxes, va='top', linespacing=1.4)

ax5_chart = fig.add_subplot(gs[4, 1])
bars_q5 = ax5_chart.bar(['27-pt\nBaseline', '83-pt\nOpenHands-HD'], [71.57, 80.97],
                         color=['#999999', '#2E7D32'], width=0.5, edgecolor='white')
for bar in bars_q5:
    ax5_chart.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                   f'{bar.get_height():.1f}%', ha='center', fontsize=10, fontweight='bold')
ax5_chart.set_ylabel('Top-1 Accuracy (%)', fontsize=9)
ax5_chart.set_ylim(0, 95)
ax5_chart.tick_params(axis='both', labelsize=9)
ax5_chart.yaxis.grid(True, linestyle='--', alpha=0.3)
ax5_chart.set_axisbelow(True)
ax5_chart.spines['top'].set_visible(False)
ax5_chart.spines['right'].set_visible(False)
# Bracket
ax5_chart.plot([0, 0, 1, 1], [84, 86, 86, 84], color='#333333', linewidth=1.2)
ax5_chart.text(0.5, 87, '+9.4 pp ***', ha='center', fontsize=9, fontweight='bold')

# =========================================================================
# Q6: LLM recovery rate (text + highlight)
# =========================================================================
ax6_text = fig.add_subplot(gs[5, 0])
ax6_text.axis('off')
ax6_text.text(0, 1.0, 'Q6', fontsize=16, fontweight='bold', color='#2E7D32',
              transform=ax6_text.transAxes, va='top')
ax6_text.text(0.08, 1.0, 'Is the LLM recovery rate\nsignificantly better than chance?',
              fontsize=11, fontweight='bold', color='#333333',
              transform=ax6_text.transAxes, va='top')
ax6_text.text(0.0, 0.45, 'Yes. NEED recovered 10/16 (62.5%)\nfrom 0.8% confidence behind 91.5%\nBOWLING. Binomial test vs random\ntop-3 (33%): p = 0.016.',
              fontsize=10, color='#444444', transform=ax6_text.transAxes, va='top', linespacing=1.4)

ax6_chart = fig.add_subplot(gs[5, 1])
bars_q6 = ax6_chart.bar(['Random\nTop-3', 'LLM\nRecovery'], [33.3, 62.5],
                         color=['#999999', '#2E7D32'], width=0.45, edgecolor='white')
for bar, val in zip(bars_q6, [33.3, 62.5]):
    ax6_chart.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
ax6_chart.set_ylabel('Recovery Rate (%)', fontsize=9)
ax6_chart.set_ylim(0, 80)
ax6_chart.tick_params(axis='both', labelsize=9)
ax6_chart.yaxis.grid(True, linestyle='--', alpha=0.3)
ax6_chart.set_axisbelow(True)
ax6_chart.spines['top'].set_visible(False)
ax6_chart.spines['right'].set_visible(False)
ax6_chart.plot([0, 0, 1, 1], [66, 69, 69, 66], color='#333333', linewidth=1.2)
ax6_chart.text(0.5, 70, 'p = 0.016 *', ha='center', fontsize=9, fontweight='bold')

# =========================================================================
# Q7: Inter-rater agreement (mini bar)
# =========================================================================
ax7_text = fig.add_subplot(gs[6, 0])
ax7_text.axis('off')
ax7_text.text(0, 1.0, 'Q7', fontsize=16, fontweight='bold', color='#2E7D32',
              transform=ax7_text.transAxes, va='top')
ax7_text.text(0.08, 1.0, 'Does inter-rater agreement support\nusing mean ratings as ground truth?',
              fontsize=11, fontweight='bold', color='#333333',
              transform=ax7_text.transAxes, va='top')
ax7_text.text(0.0, 0.45, 'Yes. ICC(2,k) = 0.960 (excellent).\nWithin-1 agreement: 91.7%.\n5 raters, means range 3.30\u20134.00.',
              fontsize=10, color='#444444', transform=ax7_text.transAxes, va='top', linespacing=1.4)

ax7_chart = fig.add_subplot(gs[6, 1])
icc_labels = ['ICC(2,k)', 'ICC(2,1)', 'Within-1']
icc_vals = [0.960, 0.829, 0.917]
colors_q7 = ['#2E7D32', '#4CAF50', '#66BB6A']
bars_q7 = ax7_chart.barh(icc_labels, icc_vals, color=colors_q7, height=0.45, edgecolor='white')
for bar, val in zip(bars_q7, icc_vals):
    ax7_chart.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
ax7_chart.axvline(x=0.9, color='#2E7D32', linestyle='--', alpha=0.5, linewidth=1)
ax7_chart.text(0.91, 2.3, 'Excellent', fontsize=8, color='#2E7D32', fontstyle='italic')
ax7_chart.set_xlim(0.7, 1.05)
ax7_chart.set_xlabel('Score', fontsize=9)
ax7_chart.tick_params(axis='both', labelsize=9)
ax7_chart.xaxis.grid(True, linestyle='--', alpha=0.3)
ax7_chart.set_axisbelow(True)
ax7_chart.spines['top'].set_visible(False)
ax7_chart.spines['right'].set_visible(False)

# Save
from pathlib import Path
output = str(Path(__file__).parent / 'chart_results_summary')
plt.savefig(output + '.png', dpi=300, bbox_inches='tight')
plt.savefig(output + '.pdf', bbox_inches='tight')
print(f"Saved chart to {output}.png and {output}.pdf")
