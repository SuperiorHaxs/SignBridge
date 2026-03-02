#!/usr/bin/env python3
"""
Generate a one-page worked example of the rotation augmentation algorithm.
Shows 3 keypoints (nose, left shoulder, right shoulder) before and after
a 15-degree rotation, with all math steps shown.

Output: research_papers/rotation_example_onepager.png (and .pdf)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


def draw_three_points(ax, nose, lsh, rsh, title, color, label_offset=0.015):
    """Draw 3 keypoints with skeleton connections and labels."""
    # Connections: nose-lsh, nose-rsh, lsh-rsh
    ax.plot([lsh[0], rsh[0]], [lsh[1], rsh[1]], color=color, linewidth=2.5, zorder=1)
    ax.plot([nose[0], lsh[0]], [nose[1], lsh[1]], color=color, linewidth=2.5, zorder=1)
    ax.plot([nose[0], rsh[0]], [nose[1], rsh[1]], color=color, linewidth=2.5, zorder=1)

    # Points
    ax.scatter(*nose, c=color, s=80, zorder=3, edgecolors='white', linewidths=1.5)
    ax.scatter(*lsh, c=color, s=80, zorder=3, edgecolors='white', linewidths=1.5)
    ax.scatter(*rsh, c=color, s=80, zorder=3, edgecolors='white', linewidths=1.5)

    # Labels
    ax.annotate(f'nose\n({nose[0]:.3f}, {nose[1]:.3f})', nose,
                textcoords="offset points", xytext=(0, -22),
                ha='center', fontsize=8, fontweight='bold', color=color)
    ax.annotate(f'L shoulder\n({lsh[0]:.3f}, {lsh[1]:.3f})', lsh,
                textcoords="offset points", xytext=(-40, 12),
                ha='center', fontsize=8, fontweight='bold', color=color)
    ax.annotate(f'R shoulder\n({rsh[0]:.3f}, {rsh[1]:.3f})', rsh,
                textcoords="offset points", xytext=(40, 12),
                ha='center', fontsize=8, fontweight='bold', color=color)


def main():
    # --- The example data ---
    theta_deg = 15
    theta = np.radians(theta_deg)
    cos_a = np.cos(theta)
    sin_a = np.sin(theta)

    # Original points
    nose_orig = np.array([0.500, 0.180])
    lsh_orig = np.array([0.400, 0.300])
    rsh_orig = np.array([0.600, 0.300])

    # Step 2: Center
    center = (nose_orig + lsh_orig + rsh_orig) / 3  # (0.500, 0.260)

    # Step 3: Center then rotate
    def rotate_point(pt):
        cx = pt[0] - center[0]
        cy = pt[1] - center[1]
        rx = cos_a * cx - sin_a * cy
        ry = sin_a * cx + cos_a * cy
        return np.array([rx + center[0], ry + center[1]])

    nose_rot = rotate_point(nose_orig)
    lsh_rot = rotate_point(lsh_orig)
    rsh_rot = rotate_point(rsh_orig)

    # --- BUILD FIGURE ---
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')

    fig.suptitle('Rotation Augmentation: Worked Example (3 Keypoints, 15\u00b0)',
                 fontsize=16, fontweight='bold', y=0.97)
    fig.text(0.5, 0.935,
             'Image coordinates: x \u2192 right (0\u20131), y \u2193 down (0\u20131)  |  '
             'Smaller y = higher in frame',
             ha='center', fontsize=10, color='#555555')

    # --- LEFT PLOT: Original ---
    ax1 = fig.add_axes([0.04, 0.45, 0.38, 0.46])
    ax1.set_xlim(0.30, 0.70)
    ax1.set_ylim(0.40, 0.10)  # inverted y
    ax1.set_aspect('equal')
    ax1.set_title('Original Pose', fontsize=13, fontweight='bold', pad=8)
    ax1.set_facecolor('#FAFAFA')
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.set_xlabel('x \u2192', fontsize=9)
    ax1.set_ylabel('y  (0 = top)', fontsize=9)
    draw_three_points(ax1, nose_orig, lsh_orig, rsh_orig, 'Original', '#2196F3')
    # Mark center
    ax1.scatter(*center, c='gray', s=50, marker='+', zorder=4, linewidths=2)
    ax1.annotate(f'center\n({center[0]:.3f}, {center[1]:.3f})', center,
                 textcoords="offset points", xytext=(30, -8),
                 ha='center', fontsize=8, color='gray', fontstyle='italic')

    # --- RIGHT PLOT: Both overlaid ---
    ax2 = fig.add_axes([0.58, 0.45, 0.38, 0.46])
    ax2.set_xlim(0.30, 0.70)
    ax2.set_ylim(0.40, 0.10)
    ax2.set_aspect('equal')
    ax2.set_title('After 15\u00b0 Rotation (overlaid)', fontsize=13, fontweight='bold', pad=8)
    ax2.set_facecolor('#FAFAFA')
    ax2.grid(True, alpha=0.2, linestyle='--')
    ax2.set_xlabel('x \u2192', fontsize=9)
    ax2.set_ylabel('y  (0 = top)', fontsize=9)
    # Draw original faded
    draw_three_points(ax2, nose_orig, lsh_orig, rsh_orig, 'Orig', '#2196F3')
    for line in ax2.get_lines():
        line.set_alpha(0.25)
    for coll in ax2.collections:
        coll.set_alpha(0.25)
    for ann in ax2.texts:
        ann.set_alpha(0.25)
    # Draw rotated on top
    draw_three_points(ax2, nose_rot, lsh_rot, rsh_rot, 'Rotated', '#E53935')
    # Center
    ax2.scatter(*center, c='gray', s=50, marker='+', zorder=4, linewidths=2)

    # Arrow between plots
    arrow_ax = fig.add_axes([0.40, 0.56, 0.20, 0.22])
    arrow_ax.set_xlim(0, 1); arrow_ax.set_ylim(0, 1)
    arrow_ax.axis('off')
    arrow_ax.annotate('', xy=(0.85, 0.5), xytext=(0.15, 0.5),
                      arrowprops=dict(arrowstyle='->', color='#333', lw=2.5))
    arrow_ax.text(0.5, 0.72, f'\u03b8 = {theta_deg}\u00b0', ha='center',
                  fontsize=13, fontstyle='italic', color='#333', fontweight='bold')
    arrow_ax.text(0.5, 0.28, f'cos={cos_a:.4f}\nsin={sin_a:.4f}', ha='center',
                  fontsize=9, fontfamily='monospace', color='#555')

    # --- BOTTOM: Step-by-step math ---
    math_ax = fig.add_axes([0.04, 0.03, 0.92, 0.40])
    math_ax.axis('off')
    math_ax.set_xlim(0, 1); math_ax.set_ylim(0, 1)

    steps_text = (
        "Step 1: Pick angle                                           "
        "Step 2: Find center of keypoints\n"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"  \u03b8 = 15\u00b0  \u2192  cos(15\u00b0) = {cos_a:.4f},  sin(15\u00b0) = {sin_a:.4f}"
        f"          center = mean of 3 points = ({center[0]:.3f}, {center[1]:.3f})\n"
        "\n"
        "Step 3: Center coordinates, apply rotation matrix, translate back\n"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    )

    # Nose calculation
    n_cx = nose_orig[0] - center[0]
    n_cy = nose_orig[1] - center[1]
    n_rx = cos_a * n_cx - sin_a * n_cy
    n_ry = sin_a * n_cx + cos_a * n_cy

    l_cx = lsh_orig[0] - center[0]
    l_cy = lsh_orig[1] - center[1]
    l_rx = cos_a * l_cx - sin_a * l_cy
    l_ry = sin_a * l_cx + cos_a * l_cy

    r_cx = rsh_orig[0] - center[0]
    r_cy = rsh_orig[1] - center[1]
    r_rx = cos_a * r_cx - sin_a * r_cy
    r_ry = sin_a * r_cx + cos_a * r_cy

    steps_text += (
        f"                       center (subtract)       rotate [ cos\u03b8 -sin\u03b8 ]        translate back (add center)\n"
        f"                       \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500       "
        f"\u2500\u2500\u2500\u2500\u2500\u2500 [ sin\u03b8  cos\u03b8 ]        \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\n"
        f"  nose     (0.500, 0.180)  \u2192  ({n_cx:+.3f}, {n_cy:+.3f})  \u2192  ({n_rx:+.4f}, {n_ry:+.4f})  \u2192  ({nose_rot[0]:.3f}, {nose_rot[1]:.3f})\n"
        f"  L shld   (0.400, 0.300)  \u2192  ({l_cx:+.3f}, {l_cy:+.3f})  \u2192  ({l_rx:+.4f}, {l_ry:+.4f})  \u2192  ({lsh_rot[0]:.3f}, {lsh_rot[1]:.3f})\n"
        f"  R shld   (0.600, 0.300)  \u2192  ({r_cx:+.3f}, {r_cy:+.3f})  \u2192  ({r_rx:+.4f}, {r_ry:+.4f})  \u2192  ({rsh_rot[0]:.3f}, {rsh_rot[1]:.3f})\n"
    )

    # Summary
    steps_text += (
        "\n"
        "Result\n"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"  Shoulders tilted: L shoulder moved UP (y {lsh_orig[1]:.3f} \u2192 {lsh_rot[1]:.3f}), "
        f"R shoulder moved DOWN (y {rsh_orig[1]:.3f} \u2192 {rsh_rot[1]:.3f})\n"
        f"  Center preserved: ({center[0]:.3f}, {center[1]:.3f}) \u2192 still ({center[0]:.3f}, {center[1]:.3f}) \u2014 "
        f"pose rotates in place, doesn't drift\n"
        f"  Distances preserved: shoulder width = {np.linalg.norm(rsh_orig - lsh_orig):.4f} \u2192 "
        f"{np.linalg.norm(rsh_rot - lsh_rot):.4f} (rigid transform)"
    )

    math_ax.text(0.01, 0.97, steps_text, transform=math_ax.transAxes,
                 fontsize=8.2, fontfamily='monospace', verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                           edgecolor='#CCCCCC', linewidth=1))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#2196F3', label='Original'),
        mpatches.Patch(facecolor='#E53935', label='After rotation'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10,
               frameon=True, fancybox=True, bbox_to_anchor=(0.96, 0.935),
               edgecolor='#CCCCCC')

    # Save
    out_dir = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(out_dir, 'rotation_example_onepager.png')
    pdf_path = os.path.join(out_dir, 'rotation_example_onepager.pdf')

    fig.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    plt.close()


if __name__ == '__main__':
    main()
