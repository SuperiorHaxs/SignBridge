#!/usr/bin/env python3
"""
Generate a one-page visual showing data augmentation (rotation) on ASL pose data.
Shows: original stick figure -> rotation transform -> augmented stick figure,
with the rotation math and code excerpt.

Output: research_papers/data_augmentation_onepager.png (and .pdf)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

# ============================================================================
# SIMPLIFIED 83-POINT POSE (one frame of a person signing)
# We'll create a realistic upper-body pose for illustration.
# Format: (x, y) normalized coordinates, origin at shoulder center
# ============================================================================

# Landmark indices (83pt format):
#   0-32:  Body pose (33 points)
#   33-53: Left hand (21 points)
#   54-74: Right hand (21 points)
#   75-82: Face (8 points)

# For visual clarity, we'll show key body + hand + face landmarks
# and draw skeleton connections

def create_sample_pose():
    """Create a realistic ASL signing pose (one frame, 83 keypoints)."""
    pose = np.zeros((83, 2), dtype=np.float32)

    # --- Body pose (indices 0-32) ---
    pose[0]  = [0.50, 0.18]   # nose
    pose[1]  = [0.48, 0.16]   # left_eye_inner
    pose[2]  = [0.47, 0.15]   # left_eye
    pose[3]  = [0.46, 0.16]   # left_eye_outer
    pose[4]  = [0.52, 0.16]   # right_eye_inner
    pose[5]  = [0.53, 0.15]   # right_eye
    pose[6]  = [0.54, 0.16]   # right_eye_outer
    pose[7]  = [0.43, 0.17]   # left_ear
    pose[8]  = [0.57, 0.17]   # right_ear
    pose[9]  = [0.48, 0.21]   # mouth_left
    pose[10] = [0.52, 0.21]   # mouth_right
    pose[11] = [0.40, 0.30]   # left_shoulder
    pose[12] = [0.60, 0.30]   # right_shoulder
    pose[13] = [0.32, 0.42]   # left_elbow (arm raised for signing)
    pose[14] = [0.68, 0.38]   # right_elbow
    pose[15] = [0.35, 0.32]   # left_wrist (hand up near face)
    pose[16] = [0.72, 0.28]   # right_wrist (hand up)
    pose[23] = [0.45, 0.55]   # left_hip
    pose[24] = [0.55, 0.55]   # right_hip

    # --- Left hand (indices 33-53, 21 points) ---
    lw = pose[15]  # left wrist position
    # Thumb
    pose[33] = lw                          # wrist
    pose[34] = lw + [-0.02, -0.01]        # thumb_cmc
    pose[35] = lw + [-0.03, -0.02]        # thumb_mcp
    pose[36] = lw + [-0.04, -0.03]        # thumb_ip
    pose[37] = lw + [-0.05, -0.04]        # thumb_tip
    # Index
    pose[38] = lw + [-0.01, -0.02]        # index_mcp
    pose[39] = lw + [-0.01, -0.04]        # index_pip
    pose[40] = lw + [-0.01, -0.05]        # index_dip
    pose[41] = lw + [-0.01, -0.06]        # index_tip
    # Middle
    pose[42] = lw + [0.00, -0.02]         # middle_mcp
    pose[43] = lw + [0.00, -0.04]         # middle_pip
    pose[44] = lw + [0.00, -0.055]        # middle_dip
    pose[45] = lw + [0.00, -0.065]        # middle_tip
    # Ring
    pose[46] = lw + [0.01, -0.02]         # ring_mcp
    pose[47] = lw + [0.01, -0.035]        # ring_pip
    pose[48] = lw + [0.01, -0.048]        # ring_dip
    pose[49] = lw + [0.01, -0.058]        # ring_tip
    # Pinky
    pose[50] = lw + [0.02, -0.015]        # pinky_mcp
    pose[51] = lw + [0.02, -0.028]        # pinky_pip
    pose[52] = lw + [0.02, -0.038]        # pinky_dip
    pose[53] = lw + [0.02, -0.046]        # pinky_tip

    # --- Right hand (indices 54-74, 21 points) ---
    rw = pose[16]  # right wrist position
    pose[54] = rw
    pose[55] = rw + [0.02, -0.01]
    pose[56] = rw + [0.03, -0.02]
    pose[57] = rw + [0.04, -0.03]
    pose[58] = rw + [0.05, -0.04]
    pose[59] = rw + [0.01, -0.02]
    pose[60] = rw + [0.01, -0.04]
    pose[61] = rw + [0.01, -0.05]
    pose[62] = rw + [0.01, -0.06]
    pose[63] = rw + [0.00, -0.02]
    pose[64] = rw + [0.00, -0.04]
    pose[65] = rw + [0.00, -0.055]
    pose[66] = rw + [0.00, -0.065]
    pose[67] = rw + [-0.01, -0.02]
    pose[68] = rw + [-0.01, -0.035]
    pose[69] = rw + [-0.01, -0.048]
    pose[70] = rw + [-0.01, -0.058]
    pose[71] = rw + [-0.02, -0.015]
    pose[72] = rw + [-0.02, -0.028]
    pose[73] = rw + [-0.02, -0.038]
    pose[74] = rw + [-0.02, -0.046]

    # --- Face (indices 75-82, 8 minimal points) ---
    pose[75] = [0.50, 0.19]   # nose_tip
    pose[76] = [0.48, 0.21]   # mouth_left
    pose[77] = [0.52, 0.21]   # mouth_right
    pose[78] = [0.50, 0.23]   # chin
    pose[79] = [0.47, 0.14]   # left_eyebrow
    pose[80] = [0.53, 0.14]   # right_eyebrow
    pose[81] = [0.46, 0.16]   # left_eye_outer
    pose[82] = [0.54, 0.16]   # right_eye_outer

    return pose


def apply_rotation_demo(pose, angle_degrees):
    """Apply rotation (same algorithm as augment_pose_file.py)."""
    rotated = pose.copy()
    angle = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    valid_mask = (pose != 0).any(axis=1)
    center = pose[valid_mask].mean(axis=0)

    centered_x = rotated[:, 0] - center[0]
    centered_y = rotated[:, 1] - center[1]

    rotated[:, 0] = cos_a * centered_x - sin_a * centered_y + center[0]
    rotated[:, 1] = sin_a * centered_x + cos_a * centered_y + center[1]

    # Zero out originally-zero keypoints
    rotated[~valid_mask] = 0
    return rotated


# Skeleton connections for drawing
BODY_CONNECTIONS = [
    (0, 2), (0, 5),          # nose to eyes
    (2, 7), (5, 8),          # eyes to ears
    (9, 10),                  # mouth
    (11, 12),                 # shoulders
    (11, 13), (13, 15),      # left arm
    (12, 14), (14, 16),      # right arm
    (11, 23), (12, 24),      # torso
    (23, 24),                 # hips
]

HAND_CONNECTIONS_LEFT = [
    (33, 34), (34, 35), (35, 36), (36, 37),  # thumb
    (33, 38), (38, 39), (39, 40), (40, 41),  # index
    (33, 42), (42, 43), (43, 44), (44, 45),  # middle
    (33, 46), (46, 47), (47, 48), (48, 49),  # ring
    (33, 50), (50, 51), (51, 52), (52, 53),  # pinky
]

HAND_CONNECTIONS_RIGHT = [
    (54, 55), (55, 56), (56, 57), (57, 58),
    (54, 59), (59, 60), (60, 61), (61, 62),
    (54, 63), (63, 64), (64, 65), (65, 66),
    (54, 67), (67, 68), (68, 69), (69, 70),
    (54, 71), (71, 72), (72, 73), (73, 74),
]

FACE_CONNECTIONS = [
    (75, 76), (75, 77), (76, 78), (77, 78),  # nose-mouth-chin
    (79, 81), (80, 82),                        # eyebrow-eye
]


def draw_pose(ax, pose, title, color_body='#2196F3', color_hand='#4CAF50', color_face='#FF9800', alpha=1.0):
    """Draw a stick figure from 83-point pose data."""
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(0.65, 0.05)  # inverted y (image coords)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_facecolor('#FAFAFA')
    ax.grid(True, alpha=0.15, linestyle='--')
    ax.set_xlabel('x', fontsize=9, alpha=0.6)
    ax.set_ylabel('y', fontsize=9, alpha=0.6)

    # Draw body skeleton
    for (i, j) in BODY_CONNECTIONS:
        if (pose[i] != 0).any() and (pose[j] != 0).any():
            ax.plot([pose[i][0], pose[j][0]], [pose[i][1], pose[j][1]],
                    color=color_body, linewidth=2.2, alpha=alpha, zorder=1)

    # Draw hands
    for (i, j) in HAND_CONNECTIONS_LEFT:
        if (pose[i] != 0).any() and (pose[j] != 0).any():
            ax.plot([pose[i][0], pose[j][0]], [pose[i][1], pose[j][1]],
                    color=color_hand, linewidth=1.5, alpha=alpha, zorder=1)

    for (i, j) in HAND_CONNECTIONS_RIGHT:
        if (pose[i] != 0).any() and (pose[j] != 0).any():
            ax.plot([pose[i][0], pose[j][0]], [pose[i][1], pose[j][1]],
                    color=color_hand, linewidth=1.5, alpha=alpha, zorder=1)

    # Draw face
    for (i, j) in FACE_CONNECTIONS:
        if (pose[i] != 0).any() and (pose[j] != 0).any():
            ax.plot([pose[i][0], pose[j][0]], [pose[i][1], pose[j][1]],
                    color=color_face, linewidth=1.2, alpha=alpha, zorder=1)

    # Draw keypoints
    valid = (pose != 0).any(axis=1)
    # Body points
    body_mask = valid.copy(); body_mask[33:] = False
    ax.scatter(pose[body_mask, 0], pose[body_mask, 1],
               c=color_body, s=20, zorder=2, alpha=alpha, edgecolors='white', linewidths=0.5)
    # Hand points
    hand_mask = valid.copy(); hand_mask[:33] = False; hand_mask[75:] = False
    ax.scatter(pose[hand_mask, 0], pose[hand_mask, 1],
               c=color_hand, s=14, zorder=2, alpha=alpha, edgecolors='white', linewidths=0.3)
    # Face points
    face_mask = valid.copy(); face_mask[:75] = False
    ax.scatter(pose[face_mask, 0], pose[face_mask, 1],
               c=color_face, s=16, zorder=2, alpha=alpha, edgecolors='white', linewidths=0.3)


def main():
    # Create poses
    original = create_sample_pose()
    angle = 15  # degrees — clear visual difference for the demo
    rotated = apply_rotation_demo(original, angle)

    # --- BUILD THE FIGURE ---
    fig = plt.figure(figsize=(11, 8.5))  # letter size
    fig.patch.set_facecolor('white')

    # Title
    fig.suptitle('Data Augmentation: Rotation Transform on ASL Pose Data',
                 fontsize=16, fontweight='bold', y=0.97)
    fig.text(0.5, 0.935,
             '83-point skeleton (33 body + 21 left hand + 21 right hand + 8 face)  |  '
             f'Rotation angle: {angle}\u00b0',
             ha='center', fontsize=10, color='#555555')

    # --- Top row: Original -> Arrow -> Rotated ---
    ax1 = fig.add_axes([0.04, 0.44, 0.38, 0.48])
    draw_pose(ax1, original, 'Original Pose')

    ax2 = fig.add_axes([0.58, 0.44, 0.38, 0.48])
    draw_pose(ax2, rotated, f'After Rotation ({angle}\u00b0)',
              color_body='#E53935', color_hand='#AB47BC', color_face='#FF9800')

    # Arrow between the two
    arrow_ax = fig.add_axes([0.40, 0.55, 0.20, 0.25])
    arrow_ax.set_xlim(0, 1); arrow_ax.set_ylim(0, 1)
    arrow_ax.axis('off')
    arrow_ax.annotate('', xy=(0.85, 0.5), xytext=(0.15, 0.5),
                      arrowprops=dict(arrowstyle='->', color='#333', lw=2.5,
                                      connectionstyle='arc3,rad=0'))
    arrow_ax.text(0.5, 0.72, 'apply_rotation()', ha='center', va='center',
                  fontsize=10, fontfamily='monospace', fontweight='bold', color='#333')
    arrow_ax.text(0.5, 0.30, f'\u03b8 = {angle}\u00b0', ha='center', va='center',
                  fontsize=12, fontstyle='italic', color='#555')

    # --- Bottom section: Code + Math + Legend ---
    # Math box
    math_ax = fig.add_axes([0.04, 0.05, 0.44, 0.34])
    math_ax.axis('off')
    math_ax.set_xlim(0, 1); math_ax.set_ylim(0, 1)

    math_text = (
        "Rotation Algorithm\n"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "1. Find center of valid keypoints:\n"
        "     center = mean(keypoints[valid])\n\n"
        "2. Center the coordinates:\n"
        "     x\u2032 = x \u2212 center_x\n"
        "     y\u2032 = y \u2212 center_y\n\n"
        "3. Apply 2D rotation matrix:\n"
        "     x\u2033 = cos(\u03b8)\u00b7x\u2032 \u2212 sin(\u03b8)\u00b7y\u2032\n"
        "     y\u2033 = sin(\u03b8)\u00b7x\u2032 + cos(\u03b8)\u00b7y\u2032\n\n"
        "4. Translate back:\n"
        "     x_final = x\u2033 + center_x\n"
        "     y_final = y\u2033 + center_y"
    )
    math_ax.text(0.02, 0.97, math_text, transform=math_ax.transAxes,
                 fontsize=9, fontfamily='monospace', verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                           edgecolor='#CCCCCC', linewidth=1))

    # Code box
    code_ax = fig.add_axes([0.52, 0.05, 0.44, 0.34])
    code_ax.axis('off')
    code_ax.set_xlim(0, 1); code_ax.set_ylim(0, 1)

    code_text = (
        "Source: augment_pose_file.py\n"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "def apply_rotation(pose_data, rotation_std=0.1):\n"
        "  angle_deg = np.random.normal(0, rotation_std * 50)\n"
        "  angle = np.radians(angle_deg)\n"
        "  cos_a, sin_a = np.cos(angle), np.sin(angle)\n"
        "\n"
        "  xy = pose_data[frame, person, :, :2]\n"
        "  valid = (xy != 0).any(axis=1)\n"
        "  center = xy[valid].mean(axis=0)\n"
        "\n"
        "  cx = pose_data[..., 0] - center[0]\n"
        "  cy = pose_data[..., 1] - center[1]\n"
        "\n"
        "  pose_data[..., 0] = cos_a*cx - sin_a*cy + center[0]\n"
        "  pose_data[..., 1] = sin_a*cx + cos_a*cy + center[1]\n"
        "\n"
        "Other augmentations used:\n"
        "  \u2022 Shear (std=0.12)    \u2022 Horizontal flip\n"
        "  \u2022 Spatial noise       \u2022 Translation\n"
        "  \u2022 Scaling (0.7-1.5x)  \u2022 Speed variation\n"
        "  \u2022 Keypoint occlusion  \u2022 Hand dropout"
    )
    code_ax.text(0.02, 0.97, code_text, transform=code_ax.transAxes,
                 fontsize=8.5, fontfamily='monospace', verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E1',
                           edgecolor='#FFB74D', linewidth=1))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#2196F3', label='Body (33 pts)'),
        mpatches.Patch(facecolor='#4CAF50', label='Hands (42 pts)'),
        mpatches.Patch(facecolor='#FF9800', label='Face (8 pts)'),
        mpatches.Patch(facecolor='#E53935', label='Rotated body'),
        mpatches.Patch(facecolor='#AB47BC', label='Rotated hands'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               fontsize=9, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, -0.02), edgecolor='#CCCCCC')

    # Save
    out_dir = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(out_dir, 'data_augmentation_onepager.png')
    pdf_path = os.path.join(out_dir, 'data_augmentation_onepager.pdf')

    fig.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    plt.close()


if __name__ == '__main__':
    main()
