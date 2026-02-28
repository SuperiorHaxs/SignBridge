#!/usr/bin/env python3
"""
Chart: Plausibility contribution to CTQI v3.
Shows that adding plausibility significantly improves human correlation.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

output_dir = Path(__file__).parent

fig, ax = plt.subplots(figsize=(7, 5.5))

categories = ['GA x CF1\n(Without Plausibility)', 'CTQI v3\n(With Plausibility)']
values = [0.9206, 0.9427]
colors = ['#999999', '#2E7D32']

bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor='white', linewidth=2)

# Value labels on bars
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f'r = {val:.4f}', ha='center', va='bottom', fontsize=16, fontweight='bold',
            color='#333333')

# Significance bracket
bracket_y = max(values) + 0.015
ax.plot([0, 0, 1, 1], [bracket_y, bracket_y + 0.004, bracket_y + 0.004, bracket_y],
        color='#333333', linewidth=1.5)
ax.text(0.5, bracket_y + 0.006, '+0.022  (p = 0.0098 **)',
        ha='center', va='bottom', fontsize=13, fontweight='bold', color='#2E7D32')

# Styling
ax.set_ylabel('Pearson r (vs Human Ratings)', fontsize=13, fontweight='bold')
ax.set_ylim(0.88, 0.98)
ax.set_title('Does Plausibility Improve CTQI v3?\n(53 sentences, Hotelling-Williams test)',
             fontsize=14, fontweight='bold', pad=18)
ax.tick_params(axis='both', labelsize=12)
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = str(output_dir / 'chart_plausibility_contribution')
fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
fig.savefig(out + '.pdf', bbox_inches='tight')
print(f"Saved {out}.png and {out}.pdf")
