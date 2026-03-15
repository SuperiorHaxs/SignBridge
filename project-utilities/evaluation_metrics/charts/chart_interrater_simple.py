#!/usr/bin/env python3
"""
Simple inter-rater reliability chart.
Single horizontal bar showing ICC(2,k) = 0.96 against the interpretation scale.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

output_dir = Path(__file__).parent

fig, ax = plt.subplots(figsize=(8, 3.5))

# ICC interpretation zones (Koo & Li, 2016)
zones = [
    (0.00, 0.50, '#D32F2F', 'Poor'),
    (0.50, 0.75, '#F57C00', 'Moderate'),
    (0.75, 0.90, '#FDD835', 'Good'),
    (0.90, 1.00, '#2E7D32', 'Excellent'),
]

bar_y = 0
bar_h = 0.5

for x0, x1, color, label in zones:
    ax.barh(bar_y, x1 - x0, left=x0, height=bar_h, color=color, alpha=0.25,
            edgecolor='white', linewidth=1.5)
    ax.text((x0 + x1) / 2, bar_y - 0.38, label, ha='center', va='center',
            fontsize=11, fontweight='bold', color=color)

# Our ICC value
icc_val = 0.960
ax.plot(icc_val, bar_y, marker='v', markersize=18, color='#2E7D32', zorder=5)
ax.annotate(f'ICC = {icc_val:.2f}', (icc_val, bar_y),
            textcoords="offset points", xytext=(0, 22),
            fontsize=16, fontweight='bold', color='#2E7D32', ha='center')

# Within-1 callout
ax.text(0.5, 0.55, '91.7% of ratings within 1 point of group mean (5-point scale)',
        transform=ax.transAxes, ha='center', va='center',
        fontsize=11, color='#444444', fontstyle='italic')

# Styling
ax.set_xlim(0, 1.0)
ax.set_ylim(-0.7, 0.7)
ax.set_xlabel('ICC Value', fontsize=12, fontweight='bold')
ax.set_yticks([])
ax.set_title('Human Survey Reliability\n(5 raters, 53 sentences)',
             fontsize=14, fontweight='bold', pad=12)
ax.tick_params(axis='x', labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()
out = str(output_dir / 'chart_interrater_simple')
fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
fig.savefig(out + '.pdf', bbox_inches='tight')
print(f"Saved {out}.png and {out}.pdf")
