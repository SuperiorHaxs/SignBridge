#!/usr/bin/env python3
"""
Chart: Ablation Study - Progressive Component Contributions.
Each configuration adds one variable to isolate its impact on sign recognition accuracy.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path

fig, ax = plt.subplots(figsize=(14, 7.5))

# ── Data ──────────────────────────────────────────────────────────────
# Labels under each group (with line breaks for readability)
labels = [
    '[1.0]\nPublished Ours2\nLiterature Reference\n(27 pts)',
    '[1.1]\nReplicated\nOpenHands Baseline\n(27 pts)',
    '[1.2]\n+16x\naugment\n(27pts)',
    '[1.3]\n+75\nkeypoints\n(75pts+16x)',
    '[1.4]\n+50x\naugment\n(75pts+50x)',
    '[1.5/2.0]\n+Fingermarks\nOpenHands-HD\n(83pts+50x)',
]

# Top-1 and Top-3 for each configuration
top1 = [71.07, 41.23, 43.56, 47.27, 68.47, 80.97]
top3 = [None,   63.24, 63.58, 67.25, 82.56, 91.62]

n = len(labels)
x = np.arange(n)
width = 0.32

# ── Colors ────────────────────────────────────────────────────────────
ref_color = '#A0A0A0'       # reference bar (hatched)
t1_color = '#707070'        # top-1 gray
t3_color = '#B0B0B0'        # top-3 light gray
final_t1 = '#2E7D32'        # final top-1 green
final_t3 = '#43A047'        # final top-3 green

# ── Draw bars ─────────────────────────────────────────────────────────
for i in range(n):
    is_ref = (i == 0)
    is_final = (i == n - 1)

    if is_ref:
        # Reference bar – single hatched bar (no top-3)
        ax.bar(x[i], top1[i], width * 2, color=ref_color, edgecolor='white',
               linewidth=1.2, hatch='///', alpha=0.7)
        ax.text(x[i], top1[i] + 1.0, f'{top1[i]:.2f}',
                ha='center', fontsize=10, fontweight='bold', color='#555555')
    else:
        c1 = final_t1 if is_final else t1_color
        c3 = final_t3 if is_final else t3_color

        # Top-1
        b1 = ax.bar(x[i] - width / 2, top1[i], width, color=c1,
                     edgecolor='white', linewidth=1.2)
        ax.text(x[i] - width / 2, top1[i] + 1.0, f'{top1[i]:.2f}',
                ha='center', fontsize=9, fontweight='bold',
                color=c1 if is_final else '#555555')

        # Top-3
        if top3[i] is not None:
            b3 = ax.bar(x[i] + width / 2, top3[i], width, color=c3,
                         edgecolor='white', linewidth=1.2)
            ax.text(x[i] + width / 2, top3[i] + 1.0, f'{top3[i]:.2f}',
                    ha='center', fontsize=9, fontweight='bold',
                    color=c3 if is_final else '#777777')

# ── Separator line before final group ─────────────────────────────────
sep_x = x[-2] + 0.55
ax.axvline(sep_x, color='#CCCCCC', linestyle='--', linewidth=1)

# ── Axis labels on left/right ─────────────────────────────────────────
ax.text(-0.02, -0.14, 'REFERENCE', transform=ax.get_xaxis_transform(),
        ha='left', fontsize=8, fontstyle='italic', color='#888888')
ax.text(1.02, -0.14, 'FINAL', transform=ax.get_xaxis_transform(),
        ha='right', fontsize=8, fontstyle='italic', color='#888888')

# ── Bottom annotation ─────────────────────────────────────────────────
improvement = top1[-1] - top1[1]
ax.text(0.5, -0.17, f'PROGRESSIVE IMPROVEMENTS (+{improvement:.2f} TOP-1)',
        transform=ax.transAxes, ha='center', fontsize=9,
        fontstyle='italic', color='#888888')

# ── Legend ─────────────────────────────────────────────────────────────
from matplotlib.patches import Patch
legend_items = [
    Patch(facecolor=ref_color, edgecolor='white', hatch='///', alpha=0.7,
          label='Published (reference)'),
    Patch(facecolor=t1_color, edgecolor='white', label='Top-1 Accuracy'),
    Patch(facecolor=t3_color, edgecolor='white', label='Top-3 Accuracy'),
    Patch(facecolor=final_t1, edgecolor='white', label='OpenHands-HD Top-1'),
    Patch(facecolor=final_t3, edgecolor='white', label='OpenHands-HD Top-3'),
]
ax.legend(handles=legend_items, fontsize=9, loc='upper left', ncol=3,
          framealpha=0.9)

# ── Axes formatting ───────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8, linespacing=1.1)
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_ylim(0, 100)
ax.tick_params(axis='y', labelsize=11)
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_title('Ablation Study: Progressive Component Contributions\n'
             'Each configuration adds one variable to isolate its impact on sign recognition accuracy',
             fontsize=14, fontweight='bold', pad=12)

plt.tight_layout()

output = str(Path(__file__).parent / 'chart_ablation_study')
plt.savefig(output + '.png', dpi=300, bbox_inches='tight')
plt.savefig(output + '.pdf', bbox_inches='tight')
print(f"Saved chart to {output}.png and {output}.pdf")
