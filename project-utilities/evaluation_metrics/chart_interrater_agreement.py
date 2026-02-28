#!/usr/bin/env python3
"""
Chart: Inter-rater agreement visualization for human survey validation.
Shows per-rater distributions and ICC summary.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

SURVEY_PATH = Path(r"C:\Users\ashwi\Downloads\SignBridge Translation Quality Survey .csv\SignBridge Translation Quality Survey .csv")

# Load ratings
responses = []
with open(SURVEY_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        ratings = []
        for r in row[4:57]:
            try:
                ratings.append(int(r))
            except (ValueError, IndexError):
                ratings.append(np.nan)
        responses.append(ratings)

matrix = np.array(responses)  # (5 raters, 53 sentences)
n_raters, n_sentences = matrix.shape

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# =========================================================================
# LEFT: Per-rater rating distributions (box plot)
# =========================================================================
rater_data = []
for r in range(n_raters):
    vals = [matrix[r, s] for s in range(n_sentences) if not np.isnan(matrix[r, s])]
    rater_data.append(vals)

bp = ax1.boxplot(rater_data, patch_artist=True, widths=0.5,
                 medianprops=dict(color='black', linewidth=2))

colors = ['#66BB6A', '#42A5F5', '#FFA726', '#AB47BC', '#EF5350']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_xticklabels([f'Rater {i+1}' for i in range(n_raters)], fontsize=12, fontweight='bold')
ax1.set_ylabel('Rating (0-5)', fontsize=15, fontweight='bold')
ax1.set_title('Per-Rater Rating Distributions\n(53 sentences)', fontsize=15, fontweight='bold', pad=15)
ax1.set_ylim(-0.5, 5.5)
ax1.tick_params(axis='y', labelsize=13)
ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add mean markers
means = [np.nanmean(d) for d in rater_data]
ax1.scatter(range(1, n_raters+1), means, color='black', s=60, zorder=5, marker='D')
for i, m in enumerate(means):
    ax1.annotate(f'{m:.2f}', (i+1, m), textcoords="offset points", xytext=(12, 0),
                 fontsize=11, fontweight='bold', color='#333333')

# =========================================================================
# RIGHT: Agreement summary bar chart
# =========================================================================
metrics = ['ICC(2,k)\n(Average)', 'ICC(2,1)\n(Single)', 'Within-1\nAgreement', 'Exact\nAgreement']
values = [0.9604, 0.8292, 0.917, 0.591]
colors_right = ['#2E7D32', '#4CAF50', '#66BB6A', '#A5D6A7']
thresholds = {'Excellent': 0.9, 'Good': 0.75, 'Moderate': 0.5}

bars = ax2.bar(metrics, values, color=colors_right, edgecolor='white', linewidth=1.5, width=0.6)

for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.1%}' if val < 1 else f'{val:.4f}',
             ha='center', fontsize=13, fontweight='bold', color='#333333')

# Threshold lines
ax2.axhline(y=0.9, color='#2E7D32', linestyle='--', alpha=0.5, linewidth=1.5)
ax2.axhline(y=0.75, color='#F57C00', linestyle='--', alpha=0.5, linewidth=1.5)
ax2.text(3.5, 0.91, 'Excellent', fontsize=10, color='#2E7D32', ha='right', fontstyle='italic')
ax2.text(3.5, 0.76, 'Good', fontsize=10, color='#F57C00', ha='right', fontstyle='italic')

ax2.set_ylabel('Score', fontsize=15, fontweight='bold')
ax2.set_title('Inter-Rater Reliability\n(5 raters, ICC)', fontsize=15, fontweight='bold', pad=15)
ax2.set_ylim(0, 1.08)
ax2.tick_params(axis='x', labelsize=11)
ax2.tick_params(axis='y', labelsize=13)
ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
ax2.set_axisbelow(True)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()

output = str(Path(__file__).parent / 'chart_interrater_agreement')
plt.savefig(output + '.png', dpi=300, bbox_inches='tight')
plt.savefig(output + '.pdf', bbox_inches='tight')
print(f"Saved chart to {output}.png and {output}.pdf")
