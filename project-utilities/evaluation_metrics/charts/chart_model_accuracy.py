#!/usr/bin/env python3
"""
Chart: Isolated Word-Level Sign Recognition Accuracy (WLASL-100).
Compares baseline models against OpenHands-HD (Ours).
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path

fig, ax = plt.subplots(figsize=(10, 7))

# ── Data ──────────────────────────────────────────────────────────────
models = [
    'I3D\nBaseline',
    'Multi-stream\nCNN (SOTA)',
    'Ours2',
    'OpenHands\nBaseline\n(replicated)',
    'OpenHands-HD\n(Ours)',
]
accuracies = [65.89, 81.38, 71.07, 41.23, 80.97]
modalities = ['Video', 'Video', 'Pose', 'Pose', 'Pose']
top3_final = 91.62  # Only for OpenHands-HD

n = len(models)
x = np.arange(n)

# ── Colors ────────────────────────────────────────────────────────────
baseline_color = '#C0C0C0'
ours_t1_color = '#2E7D32'
ours_t3_color = '#66BB6A'

# ── Draw bars ─────────────────────────────────────────────────────────
for i in range(n):
    is_ours = (i == n - 1)
    if is_ours:
        # Top-3 bar behind (taller, lighter green)
        ax.bar(x[i], top3_final, 0.6, color=ours_t3_color, edgecolor='white', linewidth=1.2)
        # Top-1 bar in front (shorter, darker green)
        ax.bar(x[i], accuracies[i], 0.6, color=ours_t1_color, edgecolor='white', linewidth=1.2)
        # Top-1 label
        ax.text(x[i], accuracies[i] - 4, f'{accuracies[i]:.2f}%',
                ha='center', va='top', fontsize=12, fontweight='bold', color='white')
        # Top-3 label
        ax.text(x[i], top3_final + 1.5, f'{top3_final}%',
                ha='center', fontsize=12, fontweight='bold', color=ours_t3_color)
    else:
        ax.bar(x[i], accuracies[i], 0.6, color=baseline_color, edgecolor='white', linewidth=1.2)
        ax.text(x[i], accuracies[i] + 1.5, f'{accuracies[i]:.2f}%',
                ha='center', fontsize=12, fontweight='bold', color='#444444')

    # Modality label inside bar near bottom
    ax.text(x[i], 8, modalities[i], ha='center', fontsize=11,
            fontweight='bold', fontstyle='italic',
            color='white' if is_ours else '#666666')

# ── Bold the "Ours" x-label ──────────────────────────────────────────
ax.set_xticks(x)
tick_labels = ax.set_xticklabels(models, fontsize=11)
tick_labels[-1].set_fontweight('bold')

# ── Legend ─────────────────────────────────────────────────────────────
from matplotlib.patches import Patch
legend_items = [
    Patch(facecolor=baseline_color, edgecolor='white', label='Baseline Models'),
    Patch(facecolor=ours_t1_color, edgecolor='white', label='OpenHands-HD Top-1 Accuracy'),
    Patch(facecolor=ours_t3_color, edgecolor='white', label='OpenHands-HD Top-3 Accuracy'),
]
ax.legend(handles=legend_items, fontsize=10, loc='upper left')

# ── Axes ──────────────────────────────────────────────────────────────
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 105)
ax.tick_params(axis='y', labelsize=12)
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_title('Isolated Word-Level Sign Recognition Accuracy (WLASL-100)',
             fontsize=15, fontweight='bold', pad=15)

plt.tight_layout()

output = str(Path(__file__).parent / 'chart_model_accuracy')
plt.savefig(output + '.png', dpi=300, bbox_inches='tight')
plt.savefig(output + '.pdf', bbox_inches='tight')
print(f"Saved chart to {output}.png and {output}.pdf")
