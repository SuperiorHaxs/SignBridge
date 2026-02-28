#!/usr/bin/env python3
"""
Generate failure scenario analysis charts as two separate files.
Chart 1: Donut of all 164 glosses (correct vs error types).
Chart 2: Breakdown of the 18 mismatches by category.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

output_dir = Path(__file__).parent

# =========================================================================
# CHART 1: Donut chart - Overall gloss outcomes (164 total)
# =========================================================================
fig1, ax1 = plt.subplots(figsize=(7, 6))

labels_left = ['Correct\n(146)', 'Model Error\n(8)', 'LLM Error\n(10)']
sizes_left = [146, 8, 10]
colors_left = ['#2E7D32', '#D32F2F', '#F57C00']
explode_left = (0.02, 0.05, 0.05)

wedges1, texts1, autotexts1 = ax1.pie(
    sizes_left, explode=explode_left, labels=labels_left, colors=colors_left,
    autopct='%1.1f%%', startangle=90, pctdistance=0.78,
    wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
    textprops=dict(fontsize=13, fontweight='bold')
)
for at in autotexts1:
    at.set_fontsize(12)
    at.set_fontweight('bold')
    at.set_color('white')

ax1.set_title('Gloss-Level Outcomes\n(164 total glosses)', fontsize=15, fontweight='bold', pad=15)
ax1.text(0, 0, '89.0%\nAccuracy', ha='center', va='center',
         fontsize=16, fontweight='bold', color='#333333')

plt.tight_layout()
out1 = str(output_dir / 'chart_failure_donut')
fig1.savefig(out1 + '.png', dpi=300, bbox_inches='tight')
fig1.savefig(out1 + '.pdf', bbox_inches='tight')
print(f"Saved {out1}.png and {out1}.pdf")

# =========================================================================
# CHART 2: Horizontal bar chart - Mismatch category breakdown (18 total)
# =========================================================================
fig2, ax2 = plt.subplots(figsize=(8, 5))

categories = [
    'D. Semantic Preference Bias',
    'C. Plausibility-Driven\nMisselection',
    'B. Failed Recovery Despite\nError Detection',
    'A. Sign Misclassification',
]
counts = [1, 3, 6, 8]
colors_right = ['#F57C00', '#F57C00', '#F57C00', '#D32F2F']

bars = ax2.barh(categories, counts, color=colors_right, edgecolor='white', linewidth=1.5, height=0.6)

for bar, count in zip(bars, counts):
    ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
             f'{count}', va='center', ha='left', fontsize=14, fontweight='bold', color='#333333')

ax2.set_xlim(0, 11)
ax2.set_xlabel('Number of Mismatches', fontsize=13, fontweight='bold')
ax2.set_title('Mismatch Breakdown by Category\n(18 total mismatches)', fontsize=15, fontweight='bold', pad=15)
ax2.tick_params(axis='y', labelsize=12)
ax2.tick_params(axis='x', labelsize=12)
ax2.xaxis.grid(True, linestyle='--', alpha=0.3)
ax2.set_axisbelow(True)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
out2 = str(output_dir / 'chart_failure_breakdown')
fig2.savefig(out2 + '.png', dpi=300, bbox_inches='tight')
fig2.savefig(out2 + '.pdf', bbox_inches='tight')
print(f"Saved {out2}.png and {out2}.pdf")
