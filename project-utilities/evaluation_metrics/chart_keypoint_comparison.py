#!/usr/bin/env python3
"""
Chart: 27-keypoint baseline vs 83-keypoint OpenHands-HD accuracy comparison.
Includes statistical significance annotation.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

fig, ax = plt.subplots(figsize=(8, 5.5))

# Data
categories = ['Top-1 Accuracy']
baseline_27 = [71.57]
openhands_hd = [80.97]

x = [0]
width = 0.3

bars1 = ax.bar([xi - width/2 for xi in x], baseline_27, width, color='#999999',
               edgecolor='white', linewidth=1.5, label='27 Keypoints (Baseline)')
bars2 = ax.bar([xi + width/2 for xi in x], openhands_hd, width, color='#2E7D32',
               edgecolor='white', linewidth=1.5, label='83 Keypoints (OpenHands-HD)')

# Value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{bar.get_height():.1f}%', ha='center', fontsize=14, fontweight='bold', color='#666666')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{bar.get_height():.1f}%', ha='center', fontsize=14, fontweight='bold', color='#2E7D32')

# Significance bracket
bracket_y = 85
ax.plot([-width/2, -width/2, width/2, width/2], [83, bracket_y, bracket_y, 83],
        color='#333333', linewidth=1.5)
ax.text(0, bracket_y + 0.5, '+9.4 pp\n(z = 4.14, p < 0.0001 ***)',
        ha='center', fontsize=12, fontweight='bold', color='#333333')

ax.set_ylabel('Accuracy (%)', fontsize=15, fontweight='bold')
ax.set_title('Sign Recognition: 27 vs 83 Keypoints\n(WLASL-100 Test Set)',
             fontsize=17, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(['Top-1 Accuracy'], fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.tick_params(axis='y', labelsize=13)
ax.legend(fontsize=13, loc='lower right')
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

output = str(Path(__file__).parent / 'chart_keypoint_comparison')
plt.savefig(output + '.png', dpi=300, bbox_inches='tight')
plt.savefig(output + '.pdf', bbox_inches='tight')
print(f"Saved chart to {output}.png and {output}.pdf")
