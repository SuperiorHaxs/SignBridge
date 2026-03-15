#!/usr/bin/env python3
"""
Generate a poster-ready graphic showing all 9 data augmentation techniques.
Each card: original -> one transformed example + text listing other variations.

Output: research_papers/augmentation_poster.png (and .pdf)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os


def create_mini_pose():
    pts = {
        'nose': (0.50, 0.15),
        'lsh': (0.38, 0.30), 'rsh': (0.62, 0.30),
        'lel': (0.30, 0.42), 'rel': (0.70, 0.42),
        'lwr': (0.34, 0.30), 'rwr': (0.72, 0.26),
        'lhip': (0.42, 0.58), 'rhip': (0.58, 0.58),
    }
    conns = [
        ('nose', 'lsh'), ('nose', 'rsh'), ('lsh', 'rsh'),
        ('lsh', 'lel'), ('lel', 'lwr'), ('rsh', 'rel'), ('rel', 'rwr'),
        ('lsh', 'lhip'), ('rsh', 'rhip'), ('lhip', 'rhip'),
    ]
    return pts, conns


def draw_mini_pose(ax, pts, conns, color='#2196F3', lw=1.8, ms=12):
    for a, b in conns:
        if pts.get(a) is not None and pts.get(b) is not None:
            ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
                    color=color, linewidth=lw, solid_capstyle='round')
    for name, val in pts.items():
        if val is not None:
            ax.plot(val[0], val[1], 'o', color=color, markersize=ms,
                    markeredgecolor='white', markeredgewidth=0.8)


def rotate_pts(pts, angle_deg):
    angle = np.radians(angle_deg)
    cx, cy = 0.5, 0.38
    new = {}
    for name, (x, y) in pts.items():
        dx, dy = x - cx, y - cy
        new[name] = (np.cos(angle)*dx - np.sin(angle)*dy + cx,
                     np.sin(angle)*dx + np.cos(angle)*dy + cy)
    return new


def scale_pts(pts, s):
    cx, cy = 0.5, 0.38
    return {name: ((x-cx)*s+cx, (y-cy)*s+cy) for name, (x, y) in pts.items()}


def translate_pts(pts, tx, ty):
    return {name: (x+tx, y+ty) for name, (x, y) in pts.items()}


def noise_pts(pts, std, seed):
    np.random.seed(seed)
    new = {}
    for name, val in pts.items():
        if val is None:
            new[name] = None
        else:
            new[name] = (val[0] + np.random.normal(0, std), val[1] + np.random.normal(0, std))
    return new


def flip_pts(pts):
    return {name: (1.0 - x, y) for name, (x, y) in pts.items()}


def speed_pts(pts, factor):
    new = {name: (x, y) for name, (x, y) in pts.items()}
    shift = (factor - 1.0) * 0.08
    new['lel'] = (pts['lel'][0] - shift, pts['lel'][1] - shift * 1.2)
    new['rel'] = (pts['rel'][0] + shift * 0.8, pts['rel'][1] + shift)
    new['lwr'] = (pts['lwr'][0] - shift * 1.3, pts['lwr'][1] - shift * 1.5)
    new['rwr'] = (pts['rwr'][0] + shift, pts['rwr'][1] + shift * 1.2)
    return new


def occlude_pts(pts, names_to_drop):
    new = {name: (x, y) for name, (x, y) in pts.items()}
    for n in names_to_drop:
        new[n] = None
    return new


TECHNIQUES = [
    {
        'name': 'Geometric\n(Rotation + Shear)',
        'weight': 3,
        'color': '#E53935',
        'why': 'Camera angle variation',
        'shown': ('\u03b8 = +15\u00b0', lambda p: rotate_pts(p, 15)),
        'others': '\u03b8 = \u221212\u00b0, +6\u00b0, +18\u00b0, ...\nshear std = 0.12',
    },
    {
        'name': 'Horizontal\nFlip',
        'weight': 2,
        'color': '#FB8C00',
        'why': 'Left/right hand ambiguity',
        'shown': ('mirrored', lambda p: flip_pts(p)),
        'others': 'flip + rot +5\u00b0\nflip + rot \u22125\u00b0',
    },
    {
        'name': 'Spatial\nNoise',
        'weight': 1,
        'color': '#7CB342',
        'why': 'Keypoint detector jitter',
        'shown': ('\u03c3 = 0.02', lambda p: noise_pts(p, 0.020, 2)),
        'others': '\u03c3 = 0.01 (low)\n\u03c3 = 0.03 (high)',
    },
    {
        'name': 'Translation',
        'weight': 1,
        'color': '#00ACC1',
        'why': 'Signer position in frame',
        'shown': ('\u0394 = +0.2', lambda p: translate_pts(p, 0.10, 0.06)),
        'others': '\u0394 = \u00b10.1 (small)\n\u0394 = \u00b10.3 (large)',
    },
    {
        'name': 'Scaling',
        'weight': 2,
        'color': '#5C6BC0',
        'why': 'Camera distance variation',
        'shown': ('0.75x', lambda p: scale_pts(p, 0.75)),
        'others': '0.7x \u2013 1.5x range\ne.g. 0.9x, 1.2x, 1.5x',
    },
    {
        'name': 'Speed\nVariation',
        'weight': 2,
        'color': '#AB47BC',
        'why': 'Signing speed differences',
        'shown': ('0.5x slow', lambda p: speed_pts(p, 0.5)),
        'others': '0.6x, 0.75x (slower)\n1.25x, 1.5x, 1.8x (faster)',
    },
    {
        'name': 'Keypoint\nOcclusion',
        'weight': 1,
        'color': '#EC407A',
        'why': 'Missing/blocked landmarks',
        'shown': ('25% dropped', lambda p: occlude_pts(p, ['lel', 'lwr'])),
        'others': '15% dropped\n35% dropped',
    },
    {
        'name': 'Hand\nDropout',
        'weight': 1,
        'color': '#8D6E63',
        'why': 'One-handed signs & partial views',
        'shown': ('drop left hand', lambda p: occlude_pts(p, ['lel', 'lwr'])),
        'others': 'drop right hand\ndrop left + noise',
    },
    {
        'name': 'Combinations',
        'weight': 2,
        'color': '#546E7A',
        'why': 'Real-world has multiple factors',
        'shown': ('rot + scale', lambda p: scale_pts(rotate_pts(p, 10), 0.85)),
        'others': 'rot + noise\nscale + shift + noise',
    },
]


def main():
    fig = plt.figure(figsize=(16, 10.5))
    fig.patch.set_facecolor('white')

    # Title
    fig.text(0.5, 0.975, 'Data Augmentation Pipeline',
             ha='center', fontsize=22, fontweight='bold')
    fig.text(0.5, 0.950,
             '9 techniques with weighted sampling \u2014 '
             'each class is balanced to 50 training samples',
             ha='center', fontsize=12, color='#555555')

    orig_pts, conns = create_mini_pose()

    for idx, tech in enumerate(TECHNIQUES):
        row = idx // 3
        col = idx % 3

        x0 = 0.015 + col * 0.335
        y0 = 0.635 - row * 0.30
        card_w = 0.32
        card_h = 0.295

        # Card background
        card_ax = fig.add_axes([x0, y0, card_w, card_h])
        card_ax.set_xlim(0, 1)
        card_ax.set_ylim(0, 1)
        card_ax.axis('off')

        rect = FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                               boxstyle="round,pad=0.02",
                               facecolor='#FAFAFA', edgecolor=tech['color'],
                               linewidth=2.5)
        card_ax.add_patch(rect)

        # Weight badge
        weight_circle = plt.Circle((0.93, 0.91), 0.045, color=tech['color'],
                                    transform=card_ax.transAxes, zorder=5)
        card_ax.add_artist(weight_circle)
        card_ax.text(0.93, 0.91, f"{tech['weight']}x", ha='center', va='center',
                     fontsize=9, fontweight='bold', color='white', zorder=6)

        # Title
        card_ax.text(0.50, 0.96, tech['name'], ha='center', va='top',
                     fontsize=11.5, fontweight='bold', color=tech['color'],
                     linespacing=1.1)

        # "Why" text
        card_ax.text(0.50, 0.76, tech['why'], ha='center', va='top',
                     fontsize=9, color='#666', fontstyle='italic')

        # --- Original pose (left) ---
        fig_h = card_h * 0.58
        pose_ax1 = fig.add_axes([x0 + 0.01, y0 + 0.04, 0.10, fig_h])
        pose_ax1.set_xlim(0.05, 0.95)
        pose_ax1.set_ylim(0.75, 0.0)
        pose_ax1.axis('off')
        draw_mini_pose(pose_ax1, orig_pts, conns, color='#90CAF9', lw=1.5, ms=5)
        pose_ax1.set_title('original', fontsize=7.5, color='#999', pad=2)

        # Arrow
        arr_ax = fig.add_axes([x0 + 0.105, y0 + 0.04 + fig_h * 0.30, 0.025, fig_h * 0.2])
        arr_ax.set_xlim(0, 1); arr_ax.set_ylim(0, 1)
        arr_ax.axis('off')
        arr_ax.annotate('', xy=(0.95, 0.5), xytext=(0.05, 0.5),
                        arrowprops=dict(arrowstyle='->', color='#999', lw=1.5))

        # --- Transformed pose (middle) ---
        label, transform_fn = tech['shown']
        transformed = transform_fn(orig_pts)
        pose_ax2 = fig.add_axes([x0 + 0.125, y0 + 0.04, 0.10, fig_h])
        pose_ax2.set_xlim(0.05, 0.95)
        pose_ax2.set_ylim(0.75, 0.0)
        pose_ax2.axis('off')
        draw_mini_pose(pose_ax2, transformed, conns, color=tech['color'], lw=1.5, ms=5)
        pose_ax2.set_title(label, fontsize=7.5, color=tech['color'], fontweight='bold', pad=2)

        # --- Other variations text box (right) ---
        card_ax.text(0.72, 0.64, 'Other variations:', ha='center', va='top',
                     fontsize=8, color='#555', fontweight='bold')
        card_ax.text(0.72, 0.55, tech['others'], ha='center', va='top',
                     fontsize=8, fontfamily='monospace', color='#444',
                     linespacing=1.4,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor='#DDD', linewidth=0.8))

    # --- Bottom summary bar ---
    summary_ax = fig.add_axes([0.015, 0.005, 0.97, 0.04])
    summary_ax.set_xlim(0, 1)
    summary_ax.set_ylim(0, 1)
    summary_ax.axis('off')

    summary_rect = FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                                   boxstyle="round,pad=0.02",
                                   facecolor='#263238', edgecolor='none')
    summary_ax.add_patch(summary_rect)

    total_weight = sum(t['weight'] for t in TECHNIQUES)

    summary_ax.text(0.50, 0.50,
                    f'Target: 50 samples/class  |  '
                    f'Distributed by weight (total {total_weight}): '
                    f'Geometric 20%  \u2022  Flip 13%  \u2022  Scale 13%  \u2022  Speed 13%  \u2022  '
                    f'Combos 13%  \u2022  Noise 7%  \u2022  Translate 7%  \u2022  Occlude 7%  \u2022  Dropout 7%',
                    ha='center', va='center', fontsize=9.5, color='#B0BEC5')

    # Save
    out_dir = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(out_dir, 'augmentation_poster.png')
    pdf_path = os.path.join(out_dir, 'augmentation_poster.pdf')

    fig.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    plt.close()


if __name__ == '__main__':
    main()
