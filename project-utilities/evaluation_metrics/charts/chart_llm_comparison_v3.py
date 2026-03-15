#!/usr/bin/env python3
"""
Generate dumbbell chart: Sentence-Level Translation Baseline vs LLM Pipeline.
Colors: grey baseline dots, green LLM dots, grey rods.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Data from evaluation results (n=53)
metrics = ['CTQI v3', 'Perfect Translation Rate', 'Plausibility', 'Coverage F1', 'Effective Gloss Accuracy']
baseline = [45.5, 49.1, 32.3, 81.0, 81.9]
llm = [74.0, 66.0, 88.8, 87.7, 87.6]
deltas = ['+28.6  (p < 0.001)', '+17.0  (p = 0.006)', '+56.5  (p < 0.001)', '+6.7  (p = 0.002)', '+5.7  (p = 0.006)']

fig, ax = plt.subplots(figsize=(12, 7))

y_positions = range(len(metrics))

for i, (y, b, l) in enumerate(zip(y_positions, baseline, llm)):
    # Grey rod connecting baseline to LLM
    ax.plot([b, l], [y, y], color='#999999', linewidth=3.5, zorder=1)

    # Grey baseline dot
    ax.scatter(b, y, color='#999999', s=200, zorder=2, edgecolors='white', linewidths=0.5)

    # Green LLM dot
    ax.scatter(l, y, color='#2E7D32', s=200, zorder=2, edgecolors='white', linewidths=0.5)

    # Baseline value label (grey, above dot)
    ax.annotate(f'{b}', (b, y), textcoords="offset points", xytext=(0, 16),
                ha='center', fontsize=14, color='#666666', fontweight='bold')

    # LLM value label (green, above dot)
    ax.annotate(f'{l}', (l, y), textcoords="offset points", xytext=(0, 16),
                ha='center', fontsize=14, color='#2E7D32', fontweight='bold')

    # Delta + p-value label (green, below rod)
    midpoint = (b + l) / 2
    ax.annotate(deltas[i], (midpoint, y), textcoords="offset points", xytext=(0, -20),
                ha='center', fontsize=13, color='#2E7D32', fontstyle='italic')

ax.set_yticks(y_positions)
ax.set_yticklabels(metrics, fontsize=15, fontweight='bold')
ax.set_xlim(25, 100)
ax.set_xlabel('Score', fontsize=15, fontweight='bold')
ax.set_title('Sentence-Level Translation: Baseline vs LLM Pipeline (n=53)',
             fontsize=17, fontweight='bold', pad=15)

ax.tick_params(axis='x', labelsize=13)
ax.set_ylim(-0.7, len(metrics) - 0.4)

# Grid
ax.xaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#999999',
           markersize=12, label='Baseline (No LLM)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E7D32',
           markersize=12, label='With LLM'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=13,
          framealpha=0.95, edgecolor='#cccccc')

plt.subplots_adjust(bottom=0.15)
plt.tight_layout()

output_dir = __file__.replace('.py', '')
plt.savefig(output_dir + '.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir + '.pdf', bbox_inches='tight')
print(f"Saved chart to {output_dir}.png and {output_dir}.pdf")
