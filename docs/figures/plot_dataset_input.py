#!/usr/bin/env python3
"""
Dataset & Input Representation Figure.
Shows packet sequence representation and feature explanation.
"""

import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib_lena')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('#fafbfc')

# Color scheme
COL_PACKET_LEN = '#3b82f6'   # Blue - packet length
COL_DIRECTION = '#22c55e'    # Green - direction
COL_TIME = '#f59e0b'         # Orange - inter-arrival time
COL_BORDER = '#94a3b8'
COL_TEXT = '#334155'
COL_HIGHLIGHT = '#8b5cf6'    # Purple - highlight

# =============================================================================
# Left panel: Sample Representation
# =============================================================================
ax1 = axes[0]
ax1.set_xlim(0, 8)
ax1.set_ylim(0, 8)
ax1.axis('off')
ax1.set_title('Input Representation: Packet Sequence', fontsize=14, fontweight='bold',
              color='#1e293b', pad=20)

# Main matrix box
matrix_box = FancyBboxPatch((0.5, 1.5), 5.5, 4.5,
                              boxstyle="round,pad=0.03,rounding_size=0.15",
                              facecolor='white', edgecolor='#64748b', linewidth=2, zorder=3)
ax1.add_patch(matrix_box)

# Feature labels on the left
features = [
    ('Packet Length', COL_PACKET_LEN, 'l_1, l_2, ..., l_100'),
    ('Direction', COL_DIRECTION, 'd_1, d_2, ..., d_100  (0=in, 1=out)'),
    ('Inter-Time', COL_TIME, 't_1, t_2, ..., t_100  (ms)'),
]

y_positions = [5.0, 3.8, 2.6]
for i, (feat_name, color, desc) in enumerate(features):
    # Feature name box
    feat_box = FancyBboxPatch((0.6, y_positions[i] - 0.35), 1.4, 0.7,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=color+'22', edgecolor=color, linewidth=1.5, zorder=4)
    ax1.add_patch(feat_box)
    ax1.text(1.3, y_positions[i], feat_name, ha='center', va='center',
             fontsize=9, fontweight='bold', color=color, zorder=5)
    # Feature description
    ax1.text(2.1, y_positions[i], desc, ha='left', va='center',
             fontsize=8, color='#64748b', zorder=5)

# Arrow pointing to sequence
ax1.annotate('', xy=(6.2, 3.8), xytext=(5.5, 3.8),
             arrowprops=dict(arrowstyle='->', color=COL_BORDER, lw=2), zorder=4)

# Sequence visualization (simplified)
seq_box = FancyBboxPatch((6.3, 2.0), 1.3, 3.6,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=1.5, zorder=4)
ax1.add_patch(seq_box)

# Draw simplified packets as small rectangles
for j in range(12):
    y_base = 5.2 - j * 0.28
    # Random-ish heights for visual interest
    np.random.seed(j * 7)
    h = 0.12 + np.random.rand() * 0.1
    pkt = patches.Rectangle((6.5, y_base - h/2), 0.9, h,
                              facecolor='#94a3b844', edgecolor='#64748b', linewidth=0.5, zorder=5)
    ax1.add_patch(pkt)

ax1.text(6.95, 1.8, 'pkt_1', ha='center', va='center', fontsize=7, color='#64748b')
ax1.text(6.95, 5.35, 'pkt_n', ha='center', va='center', fontsize=7, color='#64748b')
ax1.text(6.95, 3.55, '...', ha='center', va='center', fontsize=10, color='#64748b')

# Dimension annotation
ax1.annotate('', xy=(2.0, 1.1), xytext=(2.0, 1.4),
             arrowprops=dict(arrowstyle='<->', color=COL_BORDER, lw=1.5))
ax1.text(2.0, 0.85, '3 features', ha='center', va='center', fontsize=8, color='#64748b')

ax1.annotate('', xy=(0.4, 3.8), xytext=(0.9, 3.8),
             arrowprops=dict(arrowstyle='<->', color=COL_BORDER, lw=1.5))
ax1.text(-0.3, 3.8, 'max 100\npackets', ha='center', va='center', fontsize=8, color='#64748b')

# Shape label
shape_box = FancyBboxPatch((5.5, 0.3), 2.1, 0.6,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor='#eff6ff', edgecolor='#93c5fd', linewidth=1.5, zorder=4)
ax1.add_patch(shape_box)
ax1.text(6.55, 0.6, 'Shape: (batch, 100, 3)', ha='center', va='center',
         fontsize=9, fontweight='bold', color='#1e40af', zorder=5)

# =============================================================================
# Right panel: Feature explanation
# =============================================================================
ax2 = axes[1]
ax2.set_xlim(0, 8)
ax2.set_ylim(0, 8)
ax2.axis('off')
ax2.set_title('Feature Explanation', fontsize=14, fontweight='bold',
              color='#1e293b', pad=20)

# Feature cards
feature_cards = [
    {
        'title': 'Packet Length',
        'color': COL_PACKET_LEN,
        'bg': '#3b82f611',
        'icon': '[l1, l2, l3, ...]',
        'desc': 'Data volume pattern',
        'detail': 'Web: small frequent\nVideo: large bursts',
        'y': 6.8
    },
    {
        'title': 'Direction',
        'color': COL_DIRECTION,
        'bg': '#22c55e11',
        'icon': '[0, 1, 1, 0, 1, ...]',
        'desc': 'Bidirectional flow',
        'detail': '0 = client->server\n1 = server->client',
        'y': 5.0
    },
    {
        'title': 'Inter-Arrival Time',
        'color': COL_TIME,
        'bg': '#f59e0b11',
        'icon': '[t1, t2, t3, ...]',
        'desc': 'Timing pattern',
        'detail': 'VoIP: stable & short\nHTTP: irregular',
        'y': 3.2
    }
]

for card in feature_cards:
    # Card background
    card_box = FancyBboxPatch((0.4, card['y'] - 0.7), 7.2, 1.5,
                                boxstyle="round,pad=0.03,rounding_size=0.12",
                                facecolor=card['bg'], edgecolor=card['color'], linewidth=1.5, zorder=3)
    ax2.add_patch(card_box)

    # Color indicator
    ind = patches.Rectangle((0.5, card['y'] + 0.2), 0.08, 0.4,
                              facecolor=card['color'], zorder=4)
    ax2.add_patch(ind)

    # Title
    ax2.text(0.8, card['y'] + 0.45, card['title'], ha='left', va='center',
             fontsize=11, fontweight='bold', color=card['color'], zorder=5)

    # Icon/Format
    ax2.text(2.8, card['y'] + 0.45, card['icon'], ha='center', va='center',
             fontsize=10, color='#1e293b', fontfamily='monospace', zorder=5)

    # Description
    ax2.text(5.5, card['y'] + 0.5, card['desc'], ha='center', va='center',
             fontsize=9, fontweight='bold', color='#475569', zorder=5)

    # Detail
    ax2.text(0.8, card['y'] - 0.15, card['detail'], ha='left', va='center',
             fontsize=8, color='#64748b', zorder=5)

# Key insight box
insight_box = FancyBboxPatch((0.4, 0.4), 7.2, 2.0,
                              boxstyle="round,pad=0.03,rounding_size=0.12",
                              facecolor='#fef3c7', edgecolor='#f59e0b', linewidth=2, zorder=3)
ax2.add_patch(insight_box)

ax2.text(0.6, 2.1, 'Key Insight', ha='left', va='center',
         fontsize=11, fontweight='bold', color='#92400e', zorder=5)

ax2.text(0.6, 1.75, 'vs Traditional Methods (SVM, RF)', ha='left', va='center',
         fontsize=10, fontweight='bold', color='#92400e', zorder=5)

ax2.text(0.6, 1.35, 'Traditional: Extract statistical features first (mean, std, count...)', ha='left', va='center',
         fontsize=9, color='#a16207', zorder=5)
ax2.text(0.6, 1.0, 'Ours: Use raw packet sequences directly', ha='left', va='center',
         fontsize=9, fontweight='bold', color='#92400e', zorder=5)
ax2.text(0.6, 0.65, 'Let the model learn effective temporal patterns automatically', ha='left', va='center',
         fontsize=9, color='#92400e', zorder=5)

# =============================================================================
# Bottom: Dataset info (using fig.text)
# =============================================================================

fig.text(0.5, 0.02, 'Dataset: ISCXVPN2016 / CICFlowMeter CSV  |  Task: Multi-class (VPN/Non-VPN, or App-type)  |  Split: 80% train / 20% test',
         ha='center', va='center', fontsize=9, color='#475569')

plt.tight_layout(pad=1.5)
out_dir = os.path.dirname(__file__)
plt.savefig(os.path.join(out_dir, 'dataset_input_representation.png'),
            dpi=150, bbox_inches='tight', facecolor='#fafbfc')
plt.savefig(os.path.join(out_dir, 'dataset_input_representation.pdf'),
            bbox_inches='tight', facecolor='#fafbfc')
print('Saved:', os.path.join(out_dir, 'dataset_input_representation.png'))
print('Saved:', os.path.join(out_dir, 'dataset_input_representation.pdf'))
