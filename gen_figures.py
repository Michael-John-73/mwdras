"""Generate focused individual figures for Paper 4 (IEEE style)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Shared style ──
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
})

# ── Data ──
scales = [16, 32, 64, 128, 256]
kstar_meta = [4, 1, 0, 0, 0]
kstar_b1   = [2, None, None, 2, 1]
tpr_k0_rot = [0.000, 0.300, 0.688, 0.875, 0.750]
auc_meta   = [0.806, 0.880, 0.930, 0.918, 0.941]
gen_time   = [12.8, 25.6, 51.2, 102.4, 204.7]  # minutes
best_iter  = [200, 450, 250, 50, 50]

# ── Fig 2: Recovery step k* vs scale (Q1 core) ──
fig, ax = plt.subplots(figsize=(5.5, 3.8))
x = np.arange(len(scales))
w = 0.35
bars_meta = ax.bar(x - w/2, kstar_meta, w, label='Meta (FOMAML)', color='#2196F3', edgecolor='white')
kstar_b1_plot = [v if v is not None else 5.5 for v in kstar_b1]
colors_b1 = ['#FF7043' if v is not None else '#FFCCBC' for v in kstar_b1]
bars_b1 = ax.bar(x + w/2, kstar_b1_plot, w, label='B1 (Full Retrain)', color=colors_b1, edgecolor='white')
# Mark FAIL
for i, v in enumerate(kstar_b1):
    if v is None:
        ax.text(i + w/2, 5.5 + 0.15, 'FAIL', ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
    else:
        ax.text(i + w/2, v + 0.15, str(v), ha='center', va='bottom', fontsize=9)
for i, v in enumerate(kstar_meta):
    ax.text(i - w/2, v + 0.15, str(v), ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in scales])
ax.set_xlabel('Training Images (N)')
ax.set_ylabel('Recovery Steps k* (lower is better)')
ax.set_ylim(-0.3, 6.5)
ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1)
ax.text(4.4, 0.2, 'Zero-shot', color='green', fontsize=9, ha='right')
ax.legend(loc='upper right')
ax.set_title('Rotation Recovery: Meta vs. Full Retraining')
plt.savefig('d:/RCE/fig2_recovery_kstar.png')
plt.close()

# ── Fig 3: Zero-shot TPR@k=0 vs scale ──
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(scales, tpr_k0_rot, 'o-', color='#2196F3', linewidth=2.5, markersize=9, label='Rotation TPR@k=0')
ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='β = 0.6 (target)')
ax.fill_between(scales, 0.6, 1.0, alpha=0.08, color='green')
ax.text(200, 0.95, 'Success zone\n(TPR ≥ β)', fontsize=9, color='green', ha='center')
for i, (s, t) in enumerate(zip(scales, tpr_k0_rot)):
    ax.annotate(f'{t:.3f}', (s, t), textcoords='offset points', xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')
ax.set_xlabel('Training Images (N)')
ax.set_ylabel('TPR at k = 0')
ax.set_xscale('log', base=2)
ax.set_xticks(scales)
ax.set_xticklabels([str(s) for s in scales])
ax.set_ylim(-0.05, 1.05)
ax.legend(loc='lower right')
ax.set_title('Zero-Shot Detection Quality vs. Data Scale')
plt.savefig('d:/RCE/fig3_tpr_vs_scale.png')
plt.close()

# ── Fig 4 / Fig 5: Cost-Benefit (generation time vs k*) ──
fig, ax1 = plt.subplots(figsize=(5.5, 4.2))
ax1.bar(range(len(scales)), gen_time, color='#FFA726', alpha=0.8, label='ROBIN Generation (min)')
ax1.set_xlabel('Training Images (N)')
ax1.set_ylabel('ROBIN Generation Time (min)', color='#E65100')
ax1.set_xticks(range(len(scales)))
ax1.set_xticklabels([str(s) for s in scales])
ax1.tick_params(axis='y', labelcolor='#E65100')
# y축 상단 50% 여백 확보 → 범례와 겹침 방지
ax1.set_ylim(0, 310)
for i, t in enumerate(gen_time):
    ax1.text(i, t + 4, f'{t:.0f}m', ha='center', fontsize=9, color='#E65100')

ax2 = ax1.twinx()
ax2.plot(range(len(scales)), kstar_meta, 's-', color='#1565C0', linewidth=2.5, markersize=9, label='Meta k* (rotation)')
ax2.set_ylabel('k* (Adaptation Steps)', color='#1565C0')
ax2.tick_params(axis='y', labelcolor='#1565C0')
# 우축도 비례 확대 → k*=4 점이 상단에서 내려옴
ax2.set_ylim(-0.5, 8)
ax2.set_yticks([0, 1, 2, 3, 4])
for i, k in enumerate(kstar_meta):
    offset_y = 8 if k == 0 else 6   # k*=0 라벨은 아래 기준
    va = 'bottom'
    ax2.annotate(f'k*={k}', (i, k), textcoords='offset points',
                 xytext=(10, offset_y), fontsize=9, color='#1565C0',
                 fontweight='bold', va=va)

# Sweet spot annotation (y좌표를 ylim에 맞게 조정)
ax1.annotate('★ Sweet Spot', xy=(3, gen_time[3]), xytext=(1.2, 230),
             fontsize=11, fontweight='bold', color='#2E7D32',
             arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
# 범례를 plot 상단 여백 영역에 배치
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='upper left', bbox_to_anchor=(0.01, 0.99),
           framealpha=0.9, edgecolor='#CCCCCC')
ax1.set_title('Cost–Benefit: Generation Time vs. Recovery Steps')
plt.tight_layout()
plt.savefig('d:/RCE/PAPER4/ACCESS_latex_template_20240429/fig5_cost_benefit.png',
            dpi=200, bbox_inches='tight', pad_inches=0.15)
plt.savefig('d:/RCE/fig4_cost_benefit.png',
            dpi=200, bbox_inches='tight', pad_inches=0.15)
plt.close()

# ── Fig 5: Spearman correlation scatter (N vs k* and N vs TPR) ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

# (a) N vs k*
ax1.scatter(scales, kstar_meta, s=120, c='#1565C0', zorder=5, edgecolors='white', linewidths=1.5)
z = np.polyfit(np.log2(scales), kstar_meta, 1)
x_fit = np.linspace(16, 256, 100)
ax1.plot(x_fit, np.poly1d(z)(np.log2(x_fit)), '--', color='#90CAF9', linewidth=1.5)
for s, k in zip(scales, kstar_meta):
    ax1.annotate(f'N={s}', (s, k), textcoords='offset points', xytext=(8, 5), fontsize=9)
ax1.set_xlabel('Training Images (N)')
ax1.set_ylabel('k* (Recovery Steps)')
ax1.set_title('(a) N → k*  (ρ = −0.894, p = 0.041)')
ax1.set_xscale('log', base=2)
ax1.set_xticks(scales)
ax1.set_xticklabels([str(s) for s in scales])

# (b) N vs TPR@k=0
ax2.scatter(scales, tpr_k0_rot, s=120, c='#2E7D32', zorder=5, edgecolors='white', linewidths=1.5)
z2 = np.polyfit(np.log2(scales), tpr_k0_rot, 1)
ax2.plot(x_fit, np.poly1d(z2)(np.log2(x_fit)), '--', color='#A5D6A7', linewidth=1.5)
ax2.axhline(y=0.6, color='red', linestyle=':', alpha=0.6, linewidth=1)
for s, t in zip(scales, tpr_k0_rot):
    ax2.annotate(f'N={s}', (s, t), textcoords='offset points', xytext=(8, 5), fontsize=9)
ax2.set_xlabel('Training Images (N)')
ax2.set_ylabel('TPR @ k = 0')
ax2.set_title('(b) N → TPR@k=0  (ρ = +0.900, p = 0.037)')
ax2.set_xscale('log', base=2)
ax2.set_xticks(scales)
ax2.set_xticklabels([str(s) for s in scales])

plt.tight_layout()
plt.savefig('d:/RCE/fig5_spearman_scatter.png')
plt.close()

# ── Fig 6: Adaptation trajectory (rotation TPR vs k) ──
ks = [0, 1, 2, 4, 8, 16, 32]
tpr_16  = [0.000, 0.167, 0.333, 0.667, 0.667, 0.833, 0.833]
tpr_32  = [0.300, 0.700, 0.800, 1.000, 1.000, 1.000, 1.000]
tpr_64  = [0.688, 0.938, 0.938, 1.000, 1.000, 0.938, 0.938]
tpr_128 = [0.875, 0.563, 0.563, 0.750, 0.875, 0.938, 0.938]
tpr_256 = [0.750, 0.438, 0.438, 0.625, 0.813, 0.813, 0.875]

fig, ax = plt.subplots(figsize=(6, 4))
for data, label, color, marker in [
    (tpr_16, 'N=16', '#EF5350', 'o'),
    (tpr_32, 'N=32', '#FFA726', 's'),
    (tpr_64, 'N=64', '#66BB6A', '^'),
    (tpr_128, 'N=128', '#42A5F5', 'D'),
    (tpr_256, 'N=256', '#AB47BC', 'v'),
]:
    ax.plot(range(len(ks)), data, f'{marker}-', label=label, color=color, linewidth=2, markersize=8)

ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
ax.text(6.5, 0.62, 'β = 0.6', fontsize=9, color='gray')
ax.fill_between(range(len(ks)), 0.6, 1.05, alpha=0.05, color='green')
ax.set_xticks(range(len(ks)))
ax.set_xticklabels([str(k) for k in ks])
ax.set_xlabel('Adaptation Steps (k)')
ax.set_ylabel('TPR (Rotation Task)')
ax.set_ylim(-0.05, 1.08)
ax.legend(loc='lower right', ncol=2)
ax.set_title('Rotation TPR Adaptation Trajectory by Scale')
plt.savefig('d:/RCE/fig6_adaptation_trajectory.png')
plt.close()

print("All 5 focused figures generated successfully:")
print("  fig2_recovery_kstar.png")
print("  fig3_tpr_vs_scale.png")
print("  fig4_cost_benefit.png")
print("  fig5_spearman_scatter.png")
print("  fig6_adaptation_trajectory.png")
