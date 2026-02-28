#!/usr/bin/env python3
"""
Chart: CTQI v3 improvement with LLM pipeline.
Baseline (Top-1 only) vs With LLM (Top-3 contextual selection).
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

output_dir = Path(__file__).parent

fig, ax = plt.subplots(figsize=(7, 5.5))

categories = ['Baseline\n(Top-1 Only)', 'With LLM\n(Top-3 Selection)']
values = [45.5, 74.0]
colors = ['#999999', '#2E7D32']

bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor='white', linewidth=2)

# Value labels on bars
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f'{val:.1f}', ha='center', va='bottom', fontsize=18, fontweight='bold',
            color='#333333')

# Significance bracket
bracket_y = max(values) + 7
ax.plot([0, 0, 1, 1], [bracket_y, bracket_y + 1.5, bracket_y + 1.5, bracket_y],
        color='#333333', linewidth=1.5)
ax.text(0.5, bracket_y + 2.5, '+28.5  (p < 0.001 ***, d = 1.53)',
        ha='center', va='bottom', fontsize=13, fontweight='bold', color='#2E7D32')

# Styling
ax.set_ylabel('CTQI v3 Score', fontsize=13, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_title('Translation Quality: Baseline vs LLM Pipeline\n(53 sentences, paired t-test)',
             fontsize=14, fontweight='bold', pad=18)
ax.tick_params(axis='both', labelsize=12)
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Effect size context
ax.text(0.98, 0.02,
        "Cohen's d = 1.53 (large effect)\n92% of sentences improved",
        transform=ax.transAxes, fontsize=9, color='#666666',
        ha='right', va='bottom', fontstyle='italic')

plt.tight_layout()
out = str(output_dir / 'chart_ctqi_improvement')
fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
fig.savefig(out + '.pdf', bbox_inches='tight')
print(f"Saved {out}.png and {out}.pdf")
