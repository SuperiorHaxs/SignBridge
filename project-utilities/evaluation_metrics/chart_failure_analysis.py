#!/usr/bin/env python3
"""
Generate failure scenario analysis charts.
Left: Donut of all 164 glosses (correct vs error types).
Right: Breakdown of the 18 mismatches by category.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# =========================================================================
# LEFT: Donut chart - Overall gloss outcomes (164 total)
# =========================================================================
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

# Center text
ax1.text(0, 0, '89.0%\nAccuracy', ha='center', va='center',
         fontsize=16, fontweight='bold', color='#333333')

# =========================================================================
# RIGHT: Horizontal bar chart - Mismatch category breakdown (18 total)
# =========================================================================
categories = [
    'D. NEED Recovery\n(LLM Error)',
    'C. COMPUTER\u2192SON\n(LLM Error)',
    'B. FINE\u2192PAPER\n(LLM Error)',
    'A. LATER\u2192DRINK\n(Model Prediction Error)',
]
counts = [6, 3, 1, 8]
colors_right = ['#F57C00', '#F57C00', '#F57C00', '#D32F2F']
recovery = ['10/16 recovered', '0/3 recovered', '0/1 recovered', 'Unrecoverable']

bars = ax2.barh(categories, counts, color=colors_right, edgecolor='white', linewidth=1.5, height=0.6)

# Add count labels and recovery info
for bar, count, rec in zip(bars, counts, recovery):
    # Count on bar
    ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
             f'{count}', va='center', ha='left', fontsize=14, fontweight='bold', color='#333333')
    # Recovery info
    ax2.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2,
             f'({rec})', va='center', ha='left', fontsize=11, color='#666666', fontstyle='italic')

ax2.set_xlim(0, 14)
ax2.set_xlabel('Number of Mismatches', fontsize=13, fontweight='bold')
ax2.set_title('Mismatch Breakdown by Category\n(18 total mismatches)', fontsize=15, fontweight='bold', pad=15)
ax2.tick_params(axis='y', labelsize=12)
ax2.tick_params(axis='x', labelsize=12)
ax2.xaxis.grid(True, linestyle='--', alpha=0.3)
ax2.set_axisbelow(True)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()

output = __file__.replace('.py', '')
plt.savefig(output + '.png', dpi=300, bbox_inches='tight')
plt.savefig(output + '.pdf', bbox_inches='tight')
print(f"Saved chart to {output}.png and {output}.pdf")
