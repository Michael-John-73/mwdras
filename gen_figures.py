"""Regenerate all paper figures (Fig. 2-6) at six data scales (16-512).

Self-contained: the canonical per-scale values below are the re-measured
single-seed numbers used in the manuscript (rotation task, meta-initialized
detector), including the largest scale N=512. Figures are written to
docs/assets/ (referenced by the README and the paper).

Fig. 1 (pipeline diagram) is produced separately by gen_flow.py.

    python gen_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
})

ASSETS = Path(__file__).resolve().parent / 'docs' / 'assets'
ASSETS.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(ASSETS / name)


# ── Canonical six-scale data (rotation task, meta-initialized detector) ──
scales6 = [16, 32, 64, 128, 256, 512]
kstar_meta6 = [2, 1, 0, 0, 0, None]          # None = no recovery within the grid (FAIL)
kstar_b16 = [None, None, None, 2, 1, None]
tpr_k0_6 = [0.125, 0.300, 0.688, 0.875, 0.750, 0.188]
gen_time6 = [12.8, 25.6, 51.2, 102.4, 204.7, 409.4]  # minutes, ~0.8 min/image
FAILY = 5.5

# Rotation TPR trajectory over adaptation steps k, per scale (real single-seed).
ks_traj = [0, 1, 2, 4, 8, 16, 32]
tpr_traj = {
    16:  [0.125, 0.375, 0.750, 0.875, 0.875, 0.875, 0.875],
    32:  [0.300, 0.700, 0.800, 1.000, 1.000, 1.000, 1.000],
    64:  [0.688, 0.938, 0.938, 1.000, 1.000, 0.938, 0.938],
    128: [0.875, 0.562, 0.562, 0.750, 0.875, 0.938, 0.938],
    256: [0.750, 0.438, 0.438, 0.625, 0.812, 0.812, 0.875],
    512: [0.188, 0.188, 0.188, 0.375, 0.625, 0.750, 0.812],
}
kstar_traj = {16: 2, 32: 1, 64: 0, 128: 0, 256: 0, 512: None}
traj_style = {
    16: ('#EF5350', 'o'), 32: ('#FFA726', 's'), 64: ('#66BB6A', '^'),
    128: ('#42A5F5', 'D'), 256: ('#AB47BC', 'v'), 512: ('#00897B', 'P'),
}


# ── Fig 2: Recovery step k* vs scale ──
fig, ax = plt.subplots(figsize=(6.0, 3.9))
x = np.arange(len(scales6))
w = 0.38
meta_plot = [v if v is not None else FAILY for v in kstar_meta6]
b1_plot = [v if v is not None else FAILY for v in kstar_b16]
cmeta = ['#2196F3' if v is not None else '#BBDEFB' for v in kstar_meta6]
cb1 = ['#FF7043' if v is not None else '#FFCCBC' for v in kstar_b16]
ax.bar(x - w / 2, meta_plot, w, color=cmeta, edgecolor='white', label='Meta (FOMAML)')
ax.bar(x + w / 2, b1_plot, w, color=cb1, edgecolor='white', label='B1 (Full Retrain)')
for i, v in enumerate(kstar_meta6):
    ax.text(i - w / 2, (v if v is not None else FAILY) + 0.15, 'FAIL' if v is None else str(v),
            ha='center', va='bottom', fontsize=9, fontweight='bold',
            color=('red' if v is None else 'black'))
for i, v in enumerate(kstar_b16):
    ax.text(i + w / 2, (v if v is not None else FAILY) + 0.15, 'FAIL' if v is None else str(v),
            ha='center', va='bottom', fontsize=9, color=('red' if v is None else 'black'))
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in scales6])
ax.set_xlabel('Training Images (N)')
ax.set_ylabel('Recovery Steps k* (lower is better)')
ax.set_ylim(-0.3, 6.5)
ax.axhline(0, color='green', ls='--', alpha=0.5, lw=1)
ax.text(5.4, 0.2, 'Zero-shot', color='green', fontsize=9, ha='right')
ax.legend(loc='upper center')
ax.set_title('Rotation Recovery: Meta vs. Full Retraining')
save(fig, 'fig2_recovery_kstar.png')
plt.close(fig)


# ── Fig 3: Zero-shot TPR@k=0 vs scale ──
fig, ax = plt.subplots(figsize=(6.0, 3.9))
ax.plot(scales6, tpr_k0_6, 'o-', color='#2196F3', lw=2.5, ms=9, label='Rotation TPR@k=0')
ax.axhline(0.6, color='red', ls='--', alpha=0.7, lw=1.5, label=r'$\beta$ = 0.6 (target)')
ax.fill_between(scales6, 0.6, 1.0, alpha=0.08, color='green')
for s, t in zip(scales6, tpr_k0_6):
    ax.annotate(f'{t:.3f}', (s, t), textcoords='offset points', xytext=(0, 12),
                ha='center', fontsize=9, fontweight='bold')
ax.set_xlabel('Training Images (N)')
ax.set_ylabel('TPR at k = 0')
ax.set_xscale('log', base=2)
ax.set_xticks(scales6)
ax.set_xticklabels([str(s) for s in scales6])
ax.set_ylim(-0.05, 1.05)
ax.legend(loc='lower left')
ax.set_title('Zero-Shot Detection Quality vs. Data Scale')
save(fig, 'fig3_tpr_vs_scale.png')
plt.close(fig)


# ── Fig 4: Rotation TPR adaptation trajectory (circled marker = k*) ──
fig, ax = plt.subplots(figsize=(6.2, 4.2))
xi = np.arange(len(ks_traj))
for N in scales6:
    color, marker = traj_style[N]
    ax.plot(xi, tpr_traj[N], marker=marker, ls='-', color=color, lw=2, ms=8, label=f'N={N}')
    kstar = kstar_traj[N]
    if kstar is not None and kstar in ks_traj:
        j = ks_traj.index(kstar)
        ax.scatter([xi[j]], [tpr_traj[N][j]], s=180, facecolors='none',
                   edgecolors=color, lw=2.2, zorder=5)
ax.axhline(0.6, color='gray', ls='--', alpha=0.7, lw=1.5)
ax.text(len(ks_traj) - 1.4, 0.62, r'$\beta = 0.6$', fontsize=10, color='gray')
ax.fill_between(range(len(ks_traj)), 0.6, 1.05, alpha=0.05, color='green')
ax.set_xticks(range(len(ks_traj)))
ax.set_xticklabels([str(k) for k in ks_traj])
ax.set_xlabel('Test-time adaptation steps $k$')
ax.set_ylabel('TPR (rotation task)')
ax.set_ylim(-0.05, 1.08)
ax.legend(loc='lower right', ncol=2)
ax.set_title('Rotation-task detection-recovery trajectory by scale')
ax.grid(alpha=0.2)
save(fig, 'fig4_adaptation_trajectory.png')
plt.close(fig)


# ── Fig 5: Cost-benefit (generation time vs k*) ──
fig, ax1 = plt.subplots(figsize=(6.0, 4.2))
ax1.bar(range(len(scales6)), gen_time6, color='#FFA726', alpha=0.85, label='ROBIN Generation (min)')
ax1.set_xlabel('Training Images (N)')
ax1.set_ylabel('ROBIN Generation Time (min)', color='#E65100')
ax1.set_xticks(range(len(scales6)))
ax1.set_xticklabels([str(s) for s in scales6])
ax1.tick_params(axis='y', labelcolor='#E65100')
ax1.set_ylim(0, 475)
for i, t in enumerate(gen_time6):
    ax1.text(i, t + 6, f'{t:.0f}m', ha='center', fontsize=9, color='#E65100')
ax2 = ax1.twinx()
meta_line = [v if v is not None else np.nan for v in kstar_meta6]
ax2.plot(range(len(scales6)), meta_line, 's-', color='#1565C0', lw=2.5, ms=9, label='Meta k* (rotation)')
for i, v in enumerate(kstar_meta6):
    if v is None:
        ax2.text(i, 0.25, 'FAIL', ha='center', color='red', fontsize=9, fontweight='bold')
ax2.set_ylabel('k* (Adaptation Steps)', color='#1565C0')
ax2.tick_params(axis='y', labelcolor='#1565C0')
ax2.set_ylim(-0.5, 8)
ax2.set_yticks([0, 1, 2, 3, 4])
for i, k in enumerate(kstar_meta6):
    if k is not None:
        ax2.annotate(f'k*={k}', (i, k), textcoords='offset points', xytext=(10, 6),
                     fontsize=9, color='#1565C0', fontweight='bold')
ax1.annotate('* Sweet Spot', xy=(3, gen_time6[3]), xytext=(1.2, 300),
             fontsize=11, fontweight='bold', color='#2E7D32',
             arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))
l1, la1 = ax1.get_legend_handles_labels()
l2, la2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, la1 + la2, loc='upper left', framealpha=0.9, edgecolor='#CCCCCC')
ax1.set_title('Cost-Benefit: Generation Time vs. Recovery Steps')
plt.tight_layout()
save(fig, 'fig5_cost_benefit.png')
plt.close(fig)


# ── Fig 6: Spearman scatter (all six scales; N=512 single-seed shown) ──
scales5 = [16, 32, 64, 128, 256]
kstar_meta5 = [2, 1, 0, 0, 0]
tpr_k0_5 = [0.125, 0.300, 0.688, 0.875, 0.750]
tpr_k0_512 = 0.188          # real single-seed N=512 TPR@k=0
kstar_fail_y = 3.0          # capped display height for undefined (FAIL) k* at N=512
xticks6 = scales5 + [512]
fig, (axa, axb) = plt.subplots(1, 2, figsize=(9, 3.9))
axa.scatter(scales5, kstar_meta5, s=120, c='#1565C0', zorder=5, edgecolors='white', linewidths=1.5)
xf = np.linspace(16, 256, 100)
za = np.polyfit(np.log2(scales5), kstar_meta5, 1)
axa.plot(xf, np.poly1d(za)(np.log2(xf)), '--', color='#90CAF9', lw=1.5)
for s, k in zip(scales5, kstar_meta5):
    axa.annotate(f'N={s}', (s, k), textcoords='offset points', xytext=(8, 5), fontsize=9)
axa.scatter([512], [kstar_fail_y], s=180, marker='X', c='#D32F2F', zorder=6, edgecolors='white', linewidths=1.5)
axa.annotate('N=512: FAIL', (512, kstar_fail_y), textcoords='offset points', xytext=(-8, 6),
             fontsize=8, color='#D32F2F', fontweight='bold', ha='right')
axa.set_xlabel('Training Images (N)')
axa.set_ylabel('k* (Recovery Steps)')
axa.set_ylim(-0.3, 3.8)
axa.set_title('(a) N -> k*  (rho = -0.894, n=5)')
axa.set_xscale('log', base=2)
axa.set_xticks(xticks6)
axa.set_xticklabels([str(s) for s in xticks6])
axb.scatter(scales5, tpr_k0_5, s=120, c='#2E7D32', zorder=5, edgecolors='white', linewidths=1.5)
zb = np.polyfit(np.log2(scales5), tpr_k0_5, 1)
axb.plot(xf, np.poly1d(zb)(np.log2(xf)), '--', color='#A5D6A7', lw=1.5)
axb.axhline(0.6, color='red', ls=':', alpha=0.6, lw=1)
for s, t in zip(scales5, tpr_k0_5):
    axb.annotate(f'N={s}', (s, t), textcoords='offset points', xytext=(8, 5), fontsize=9)
axb.scatter([512], [tpr_k0_512], s=180, marker='X', c='#D32F2F', zorder=6, edgecolors='white', linewidths=1.5)
axb.annotate('N=512 (outlier)', (512, tpr_k0_512), textcoords='offset points', xytext=(-8, 6),
             fontsize=8, color='#D32F2F', fontweight='bold', ha='right')
axb.set_xlabel('Training Images (N)')
axb.set_ylabel('TPR @ k = 0')
axb.set_title('(b) N -> TPR@k=0  (rho = +0.900, n=5)')
axb.set_xscale('log', base=2)
axb.set_xticks(xticks6)
axb.set_xticklabels([str(s) for s in xticks6])
plt.tight_layout()
save(fig, 'fig6_spearman_scatter.png')
plt.close(fig)

print('Regenerated fig2-fig6 (six scales, 16-512) into', ASSETS)
