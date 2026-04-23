#!/usr/bin/env python3
"""
Framework figure: CNN (deploy) + LR (explain with raw stats + CNN logits).
All on-canvas text is ASCII/English so Matplotlib never shows "tofu" boxes
when CJK fonts are missing. Add Chinese captions in Word/LaTeX if needed.
"""

import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib_lena')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(18, 13))
ax.set_xlim(0, 19.2)
ax.set_ylim(0, 13)
ax.axis('off')
fig.patch.set_facecolor('#fafbfc')

COL_TEACHER = '#f59e0b'
COL_CNN = '#3b82f6'
COL_LR = '#8b5cf6'
COL_FEAT = '#22c55e'
COL_LOSS = '#ef4444'
COL_TRAFFIC = '#64748b'


def box(ax, x, y, w, h, text, subtext=None, color='#e2e8f0', edgecolor='#94a3b8',
        fontsize=11, fontweight='normal', textcolor='#334155', radius=0.15):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=color, edgecolor=edgecolor, linewidth=1.5, zorder=3,
    )
    ax.add_patch(rect)
    if subtext:
        ax.text(x, y + 0.08, text, ha='center', va='center', fontsize=fontsize,
                fontweight=fontweight, color=textcolor, zorder=4)
        ax.text(x, y - 0.22, subtext, ha='center', va='center', fontsize=fontsize - 2,
                color='#94a3b8', zorder=4)
    else:
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight=fontweight, color=textcolor, zorder=4)


def arrow(ax, x1, y1, x2, y2, color='#64748b', lw=1.8, dashed=False,
          label=None, offset=(0, 0.18)):
    ls = (0, (5, 3)) if dashed else '-'
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=color, lw=lw, linestyle=ls),
        zorder=2,
    )
    if label:
        mx, my = (x1 + x2) / 2 + offset[0], (y1 + y2) / 2 + offset[1]
        ax.text(mx, my, label, ha='center', va='center', fontsize=9, color=color,
                zorder=5, bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.1))


def dashed_line(ax, x1, y1, x2, y2, color='#94a3b8', lw=1.2):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, linestyle='--', zorder=1)


# Title (English only)
ax.text(9, 12.55, 'Dual-Student Distillation Framework', ha='center', va='center',
        fontsize=17, fontweight='bold', color='#1e293b')
ax.text(9, 12.15,
        'CNN: lightweight deploy  |  LR: explain (raw stats + CNN logits)',
        ha='center', va='center', fontsize=12, color='#64748b')

ax.axvline(x=10.6, ymin=0.04, ymax=0.92, color='#cbd5e1', lw=2, linestyle='--', zorder=1)

phase1 = FancyBboxPatch((0.2, 11.8), 1.3, 0.45,
                        boxstyle="round,pad=0.05,rounding_size=0.12",
                        facecolor='#e0f2fe', edgecolor='#7dd3fc', lw=1.2, zorder=3)
ax.add_patch(phase1)
ax.text(0.85, 12.02, 'Training', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#0369a1', zorder=4)

phase2 = FancyBboxPatch((11.3, 11.8), 1.3, 0.45,
                        boxstyle="round,pad=0.05,rounding_size=0.12",
                        facecolor='#dcfce7', edgecolor='#86efac', lw=1.2, zorder=3)
ax.add_patch(phase2)
ax.text(11.95, 12.02, 'Deploy', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#15803d', zorder=4)

# Training: traffic
box(ax, 1.5, 11.0, 2.2, 0.75,
    'Encrypted traffic', 'N samples + labels',
    color='white', edgecolor=COL_TRAFFIC, radius=0.18)

ax.annotate('', xy=(2.6, 10.62), xytext=(2.3, 11.0),
            arrowprops=dict(arrowstyle='->', color=COL_TRAFFIC, lw=1.5), zorder=4)
ax.text(2.3, 10.48, 'split', ha='center', va='center', fontsize=9, color=COL_TRAFFIC,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.1))

box(ax, 3.4, 10.45, 2.5, 0.85,
    'Transformer Teacher', 'reference, train only',
    color='#fffbeb', edgecolor='#fcd34d', radius=0.18)
ax.text(3.4, 10.45, 'Transformer Teacher', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#92400e', zorder=4)

arrow(ax, 3.4, 10.02, 3.4, 9.45, color=COL_TEACHER, lw=1.8,
      label='soft', offset=(0.4, 0))

soft_box = FancyBboxPatch((2.6, 8.95), 1.6, 0.5,
                          boxstyle="round,pad=0.03,rounding_size=0.1",
                          facecolor='#fef9c3', edgecolor='#fde047', lw=1.5, zorder=3)
ax.add_patch(soft_box)
ax.text(3.4, 9.18, 'Soft targets', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#854d0e', zorder=4)
ax.text(3.4, 9.0, '[p_1, p_2, ..., p_C]', ha='center', va='center',
        fontsize=8.5, color='#a16207', zorder=4)

# CNN student
cnn_box = FancyBboxPatch((2.15, 7.6), 2.5, 0.95,
                         boxstyle="round,pad=0.03,rounding_size=0.15",
                         facecolor='white', edgecolor=COL_CNN, lw=2.5, zorder=4)
ax.add_patch(cnn_box)
ax.text(3.4, 7.95, 'CNN student', ha='center', va='center',
        fontsize=13, fontweight='bold', color='#1d4ed8', zorder=5)
ax.text(3.4, 7.7, 'KD: KL div + CE, weights absorb teacher', ha='center', va='center',
        fontsize=9.5, color='#64748b', zorder=5)

arrow(ax, 2.55, 10.28, 2.8, 8.12, color=COL_TRAFFIC, lw=1.5, dashed=True,
      label='packets', offset=(0, 0.18))
arrow(ax, 3.4, 8.95, 3.4, 8.56, color=COL_TEACHER, lw=2,
      label='soft', offset=(0.45, 0))
arrow(ax, 4.65, 8.08, 5.2, 8.08, color=COL_CNN, lw=2,
      label='logits', offset=(0, 0.2))

logits_box = FancyBboxPatch((5.2, 7.65), 1.9, 0.85,
                            boxstyle="round,pad=0.03,rounding_size=0.12",
                            facecolor='#eff6ff', edgecolor='#93c5fd', lw=1.5, zorder=3)
ax.add_patch(logits_box)
ax.text(6.15, 7.96, 'CNN logits', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#1e40af', zorder=4)
ax.text(6.15, 7.73, '[z_1, z_2, ..., z_C]', ha='center', va='center',
        fontsize=8.5, color='#3b82f6', zorder=4)

kl_box = FancyBboxPatch((2.25, 6.15), 2.3, 0.7,
                          boxstyle="round,pad=0.03,rounding_size=0.12",
                          facecolor='#fef2f2', edgecolor='#fca5a5', lw=1.5, zorder=3)
ax.add_patch(kl_box)
ax.text(3.4, 6.42, 'KL Div loss', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#991b1b', zorder=4)
ax.text(3.4, 6.22, 'softmax(CNN/T) vs softmax(Teacher/T)', ha='center', va='center',
        fontsize=8.5, color='#b91c1c', zorder=4)

ax.plot([3.4, 3.4], [8.95, 6.85], color=COL_TEACHER, lw=1.2, linestyle=':', zorder=1)
ax.annotate('', xy=(3.4, 6.85), xytext=(3.4, 7.6),
            arrowprops=dict(arrowstyle='->', color=COL_TEACHER, lw=1.2, linestyle=':'),
            zorder=2)
ax.plot([3.4, 3.4], [7.6, 6.85], color=COL_CNN, lw=1.2, linestyle=':', zorder=1)
ax.annotate('', xy=(3.4, 6.85), xytext=(3.4, 7.6),
            arrowprops=dict(arrowstyle='->', color=COL_CNN, lw=1.2, linestyle=':'),
            zorder=2)

arrow(ax, 3.4, 6.15, 3.4, 5.65, color=COL_LOSS, lw=2,
      label='backprop', offset=(0.5, 0))
box(ax, 3.4, 5.4, 2.3, 0.5,
    'Update CNN weights', color='#f0fdf4', edgecolor='#86efac', radius=0.12)

# LR path
feat_box = FancyBboxPatch((6.8, 10.05), 2.1, 0.85,
                          boxstyle="round,pad=0.03,rounding_size=0.15",
                          facecolor='white', edgecolor=COL_FEAT, lw=2, zorder=4)
ax.add_patch(feat_box)
ax.text(7.85, 10.35, 'Raw flow stats', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#166534', zorder=5)
ax.text(7.85, 10.1, 'length / IAT / dir stats (16-d)', ha='center', va='center',
        fontsize=9.5, color='#64748b', zorder=5)

dashed_line(ax, 2.7, 10.62, 6.25, 10.62, color=COL_TRAFFIC, lw=1.3)
ax.annotate('', xy=(6.25, 10.62), xytext=(6.8, 10.48),
            arrowprops=dict(arrowstyle='->', color=COL_FEAT, lw=1.5, linestyle='--'),
            zorder=4)
ax.text(4.7, 10.5, 'extract', ha='center', va='center', fontsize=9, color=COL_TRAFFIC,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.1))

arrow(ax, 7.85, 9.63, 7.85, 9.05, color=COL_FEAT, lw=1.8,
      label='raw', offset=(0.5, 0))
arrow(ax, 7.05, 8.08, 7.85, 9.0, color=COL_CNN, lw=1.8,
      label='CNN logits', offset=(0, -0.25))

concat_box = FancyBboxPatch((7.1, 8.5), 1.5, 0.5,
                            boxstyle="round,pad=0.03,rounding_size=0.1",
                            facecolor='#f5f3ff', edgecolor='#c4b5fd', lw=1.5, zorder=3)
ax.add_patch(concat_box)
ax.text(7.85, 8.73, 'Concat', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#5b21b6', zorder=4)
ax.text(7.85, 8.55, '[raw (16d) | CNN logits (2d)]', ha='center', va='center',
        fontsize=8, color='#7c3aed', zorder=4)

arrow(ax, 7.85, 8.5, 7.85, 7.85, color=COL_LR, lw=2)

lr_box = FancyBboxPatch((6.9, 6.75), 2.3, 0.95,
                        boxstyle="round,pad=0.03,rounding_size=0.15",
                        facecolor='white', edgecolor=COL_LR, lw=2.5, zorder=4)
ax.add_patch(lr_box)
ax.text(8.05, 7.1, 'LR (explain)', ha='center', va='center',
        fontsize=13, fontweight='bold', color='#6d28d9', zorder=5)
ax.text(8.05, 6.85, 'Logistic Regression: coef = importance', ha='center', va='center',
        fontsize=9.5, color='#64748b', zorder=5)

arrow(ax, 9.2, 7.22, 9.8, 7.22, color=COL_LR, lw=2,
      label='pred', offset=(0, 0.2))

ce_box = FancyBboxPatch((9.8, 6.98), 1.4, 0.48,
                        boxstyle="round,pad=0.03,rounding_size=0.1",
                        facecolor='#fdf4ff', edgecolor='#e879f9', lw=1.5, zorder=3)
ax.add_patch(ce_box)
ax.text(10.5, 7.18, 'CE loss', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#a21caf', zorder=4)
ax.text(10.5, 7.02, 'cross-entropy(LR, y_true)', ha='center', va='center',
        fontsize=8, color='#a21caf', zorder=4)

ax.annotate('', xy=(10.5, 7.22), xytext=(1.5, 10.62),
            arrowprops=dict(arrowstyle='->', color=COL_TRAFFIC, lw=1.5,
                            linestyle=(0, (5, 3))), zorder=2)
ax.text(10.8, 8.8, 'hard label', ha='center', va='center', fontsize=9, color=COL_TRAFFIC,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.1))

arrow(ax, 10.5, 6.98, 10.5, 6.5, color=COL_LOSS, lw=2,
      label='backprop', offset=(0.45, 0))
box(ax, 10.5, 6.25, 1.4, 0.45,
    'Update LR', color='#f0fdf4', edgecolor='#86efac', radius=0.1)

# Deploy
box(ax, 12.2, 10.5, 2.2, 0.75,
    'New traffic', 'unlabeled',
    color='white', edgecolor=COL_TRAFFIC, radius=0.18)

arrow(ax, 13.2, 10.12, 13.2, 9.38, color=COL_TRAFFIC, lw=2,
      label='packets', offset=(0.5, 0))

cnn_deploy = FancyBboxPatch((11.95, 8.75), 2.0, 0.65,
                            boxstyle="round,pad=0.03,rounding_size=0.15",
                            facecolor='white', edgecolor=COL_CNN, lw=2.5, zorder=4)
ax.add_patch(cnn_deploy)
ax.text(12.95, 9.05, 'CNN inference', ha='center', va='center',
        fontsize=14, fontweight='bold', color='#1d4ed8', zorder=5)
ax.text(12.95, 8.82, 'no teacher at runtime', ha='center', va='center',
        fontsize=9.5, color='#64748b', zorder=5)

arrow(ax, 13.95, 9.08, 14.6, 9.08, color=COL_CNN, lw=2.5,
      label='class', offset=(0, 0.22))

pred_box = FancyBboxPatch((14.6, 8.55), 1.6, 1.05,
                          boxstyle="round,pad=0.03,rounding_size=0.15",
                          facecolor='#dbeafe', edgecolor='#3b82f6', lw=2, zorder=4)
ax.add_patch(pred_box)
ax.text(15.4, 9.24, 'Prediction', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#1e40af', zorder=5)
ax.text(15.4, 9.02, 'chat / voip / ...', ha='center', va='center',
        fontsize=9.5, color='#3b82f6', zorder=5)

arrow(ax, 12.95, 8.75, 12.95, 8.35, color=COL_CNN, lw=1.8, dashed=True,
      label='logits', offset=(0, 0))

cache_box = FancyBboxPatch((11.95, 7.75), 2.0, 0.55,
                           boxstyle="round,pad=0.03,rounding_size=0.1",
                           facecolor='#eff6ff', edgecolor='#93c5fd', lw=1.5, zorder=3)
ax.add_patch(cache_box)
ax.text(12.95, 8.0, 'CNN logits (buffer)', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#1e40af', zorder=4)

dashed_line(ax, 13.2, 10.12, 13.2, 6.75, color=COL_TRAFFIC, lw=1.3)
dashed_line(ax, 13.2, 6.75, 14.4, 6.75, color=COL_TRAFFIC, lw=1.3)
ax.annotate('', xy=(14.4, 6.75), xytext=(14.8, 6.9),
            arrowprops=dict(arrowstyle='->', color=COL_FEAT, lw=1.8, linestyle='--'),
            zorder=4)
ax.text(13.8, 6.6, 'stats', ha='center', va='center', fontsize=9, color=COL_TRAFFIC,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.1))

ax.annotate('', xy=(14.4, 6.68), xytext=(12.95, 7.75),
            arrowprops=dict(arrowstyle='->', color=COL_CNN, lw=1.8), zorder=4)

concat2 = FancyBboxPatch((14.4, 6.3), 1.3, 0.5,
                           boxstyle="round,pad=0.03,rounding_size=0.1",
                           facecolor='#f5f3ff', edgecolor='#c4b5fd', lw=1.5, zorder=3)
ax.add_patch(concat2)
ax.text(15.05, 6.53, 'Concat', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#5b21b6', zorder=4)
ax.text(15.05, 6.36, 'raw(16d)+logits(2d)', ha='center', va='center',
        fontsize=8, color='#7c3aed', zorder=4)

arrow(ax, 15.7, 6.55, 16.1, 6.3, color=COL_LR, lw=2)

lr_deploy = FancyBboxPatch((16.1, 5.85), 1.5, 0.75,
                           boxstyle="round,pad=0.03,rounding_size=0.15",
                           facecolor='white', edgecolor=COL_LR, lw=2.5, zorder=4)
ax.add_patch(lr_deploy)
ax.text(16.85, 6.17, 'LR explain', ha='center', va='center',
        fontsize=13, fontweight='bold', color='#6d28d9', zorder=5)
ax.text(16.85, 5.95, 'offline: coef / bar chart', ha='center', va='center',
        fontsize=9, color='#64748b', zorder=5)

arrow(ax, 17.6, 6.22, 18.25, 6.22, color=COL_LR, lw=2)

out_box = FancyBboxPatch((18.25, 5.6), 0.75, 1.25,
                         boxstyle="round,pad=0.03,rounding_size=0.12",
                         facecolor='#ede9fe', edgecolor='#a78bfa', lw=1.5, zorder=4)
ax.add_patch(out_box)
ax.text(18.62, 6.22, 'explain', ha='center', va='center', fontsize=8, color='#5b21b6',
        rotation=90, fontweight='bold')

# Notes
note1 = FancyBboxPatch((0.3, 4.5), 5.0, 1.8,
                       boxstyle="round,pad=0.05,rounding_size=0.15",
                       facecolor='#fefce8', edgecolor='#fcd34d', lw=1.5, zorder=3)
ax.add_patch(note1)
ax.text(0.55, 6.12, 'Training notes:', ha='left', va='center',
        fontsize=11, fontweight='bold', color='#92400e', zorder=4)
ax.text(0.55, 5.85, '(1) CNN: KL distillation from teacher', ha='left', va='center',
        fontsize=10, color='#a16207', zorder=4)
ax.text(0.55, 5.55, '(2) LR: fit on concat[raw stats (16-d), CNN logits (2-d)]', ha='left', va='center',
        fontsize=10, color='#a16207', zorder=4)
ax.text(0.55, 5.25, '(3) LR learns how raw + logits predict y', ha='left', va='center',
        fontsize=10, color='#a16207', zorder=4)
ax.text(0.55, 4.97, '    (coefficients = interpretable)', ha='left', va='center',
        fontsize=10, color='#a16207', zorder=4)

note2 = FancyBboxPatch((5.6, 4.5), 4.8, 1.8,
                       boxstyle="round,pad=0.05,rounding_size=0.15",
                       facecolor='#dcfce7', edgecolor='#86efac', lw=1.5, zorder=3)
ax.add_patch(note2)
ax.text(5.85, 6.12, 'Deploy notes:', ha='left', va='center',
        fontsize=11, fontweight='bold', color='#166534', zorder=4)
ax.text(5.85, 5.85, '(1) CNN alone for fast prediction', ha='left', va='center',
        fontsize=10, color='#15803d', zorder=4)
ax.text(5.85, 5.55, '(2) Reuse CNN logits + stats for LR', ha='left', va='center',
        fontsize=10, color='#15803d', zorder=4)
ax.text(5.85, 5.25, '(3) LR explains contribution of each dim', ha='left', va='center',
        fontsize=10, color='#15803d', zorder=4)
ax.text(5.85, 4.97, '    including CNN logit channels', ha='left', va='center',
        fontsize=10, color='#15803d', zorder=4)

legend_box = FancyBboxPatch((0.3, 2.5), 10.1, 1.7,
                            boxstyle="round,pad=0.05,rounding_size=0.15",
                            facecolor='white', edgecolor='#e2e8f0', lw=1.5, zorder=3)
ax.add_patch(legend_box)
ax.text(0.55, 4.02, 'Legend:', ha='left', va='center',
        fontsize=11, fontweight='bold', color='#475569', zorder=4)

legend_items = [
    (0.55, 3.75, COL_TEACHER, 'Soft targets (teacher -> CNN)'),
    (0.55, 3.48, COL_CNN, 'CNN logits (-> LR input)'),
    (0.55, 3.21, COL_FEAT, 'Raw stats (-> LR input)'),
    (4.1, 3.75, COL_LR, 'LR explain path'),
    (4.1, 3.48, COL_LOSS, 'Loss / backprop'),
    (4.1, 3.21, COL_TRAFFIC, 'Traffic data'),
]
for x, y, color, label in legend_items:
    ax.plot([x, x + 0.55], [y, y], color=color, lw=2.5, zorder=4)
    ax.text(x + 0.75, y, label, ha='left', va='center', fontsize=9.5,
            color='#64748b', zorder=4)

ax.annotate('', xy=(7.85, 8.55), xytext=(7.05, 8.08),
            arrowprops=dict(arrowstyle='->', color=COL_CNN, lw=2.5,
                            linestyle=(0, (6, 3))), zorder=5)
ax.text(6.9, 8.25, 'CNN logits', ha='center', va='center', fontsize=9,
        color=COL_CNN, fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.1))

plt.tight_layout(pad=0.5)
out_dir = os.path.join(os.path.dirname(__file__))
plt.savefig(os.path.join(out_dir, 'framework_cnn_lr_explain.png'),
            dpi=150, bbox_inches='tight', facecolor='#fafbfc')
plt.savefig(os.path.join(out_dir, 'framework_cnn_lr_explain.pdf'),
            bbox_inches='tight', facecolor='#fafbfc')
print('Saved:', os.path.join(out_dir, 'framework_cnn_lr_explain.png'))
