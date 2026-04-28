"""Generate Fig. 1: Meta-Watermarking Pipeline Flow Diagram (PNG).
v3: compact boxes sized to text, tighter vertical spacing, clear arrows.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.transforms as mtransforms


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def _resolve_access_template_dir() -> Path:
    candidates = sorted(WORKSPACE_ROOT.glob("*/ACCESS_latex_template_20240429"))
    if not candidates:
        raise FileNotFoundError("Could not find ACCESS_latex_template_20240429 under workspace root")
    return candidates[0]


ACCESS_TEMPLATE_DIR = _resolve_access_template_dir()

# ── canvas ──
fig, ax = plt.subplots(figsize=(20, 7.5))
ax.set_xlim(0, 20)
ax.set_ylim(3.8, 10.2)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── helpers ──────────────────────────────────────────────────────────────────
FS = 9.5   # base font size for box text

def auto_box(ax, cx, cy, text, fc, ec, bold=False, lw=2.5,
             pad_x=0.28, pad_y=0.20, tc='#111111'):
    """Draw a box automatically sized to fit `text`, centred at (cx, cy)."""
    fw = 'bold' if bold else 'normal'
    # Render text to measure it
    t = ax.text(cx, cy, text, ha='center', va='center', fontsize=FS,
                fontweight=fw, color=tc, zorder=4, linespacing=1.35,
                visible=False)
    fig.canvas.draw()
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    # Convert pixel bbox → data coordinates
    inv = ax.transData.inverted()
    (x0d, y0d) = inv.transform((bb.x0, bb.y0))
    (x1d, y1d) = inv.transform((bb.x1, bb.y1))
    tw = x1d - x0d
    th = y1d - y0d
    bw = tw + 2 * pad_x
    bh = th + 2 * pad_y
    bx = cx - bw / 2
    by = cy - bh / 2
    box = FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.12",
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(box)
    t.set_visible(True)
    return bx, by, bw, bh   # left, bottom, width, height


def arrow(ax, x1, y1, x2, y2, color, lw=2.2, ms=15, ls='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=ms, linestyle=ls),
                zorder=5)


def elbow(ax, pts, color, lw=2.2, ms=15):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, lw=lw, zorder=5,
            solid_capstyle='round', solid_joinstyle='round')
    ax.annotate('', xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=ms),
                zorder=6)


def section_label(ax, cx, y, text, color):
    ax.text(cx, y, text, ha='center', va='center', fontsize=11,
            fontweight='bold', color=color, zorder=6)

# ── colours ──
C_P1   = '#1B5E20'   # dark green
C_FM   = '#BF360C'   # dark orange
C_P2   = '#0D47A1'   # dark blue
C_OK   = '#2E7D32'   # success green
C_FAIL = '#C62828'   # fail red
C_TH   = '#E65100'   # theta orange

# ─────────────────────────────────────────────────────────────────────────────
# Layout constants — all y values measured from bottom
# We define box centres rather than bottom-left corners so auto_box works.
# ─────────────────────────────────────────────────────────────────────────────

# ── Phase 1 left column centres (x=1.8) ──
P1L_X = 1.80
p1l_ys = [9.0, 7.9, 6.7, 5.6]   # top → bottom (4 boxes)
p1l_texts = [
    'SD 2.1-base\nImage Generation\n(N images)',
    'ROBIN Watermark\nEmbedding (r=5–15)',
    'Attack Application\n(7 attack types)',
    '50-step DDIM\nInversion',
]

# ── Phase 1 right column centres (x=4.6) ──
P1R_X = 4.60
p1r_ys = [9.0, 7.9, 6.7]   # 3 boxes
p1r_texts = [
    'Ring-Feature\nExtraction (10-D)',
    'Z-score\nNormalization',
    'Task Construction\n(support / query)',
]

# ── FOMAML column centres (x=8.0) ──
FM_X = 8.00
fm_ys = [9.0, 7.6, 6.2]
fm_texts = [
    'Outer Update\nθ₀ ← θ₀ − η_meta·∇Σℒᵢ\n(η_meta=0.01, T iter)',
    "Inner Loop (×3 tasks)\nθ'= θ₀ − η·∇ℒ_support\n(η=0.3, 5 steps)",
    'Query-set Loss\nΣℒᵢ(θ\'ᵢ; D_query)',
]

# ── θ₀ box centre ──
TH_X, TH_Y = 8.00, 4.95

# ── Phase 2 left column centres (x=11.8) ──
P2L_X = 11.80
p2l_ys = [9.0, 7.9]
p2l_texts = [
    'New Unseen\nAttack τ',
    'Feature Extraction\n+ Z-score Norm',
]

# ── Diamond centre ──
DM_X, DM_Y = 11.80, 6.8
DM_HW = 0.6    # half-width
DM_HH = 0.45   # half-height

# ── Phase 2 right column centres ──
P2R_X = 16.5
ad_y  = 7.9   # Adapt box
rev_y = 6.8   # Re-eval box
dep_y = 5.7   # Deploy θ_k box

# ── Zero-shot deploy box centre ──
ZS_X, ZS_Y = 11.80, 5.5

# ─────────────────────────────────────────────────────────────────────────────
# Draw phase backgrounds  (after we know approximate extents)
# ─────────────────────────────────────────────────────────────────────────────
from matplotlib.patches import FancyBboxPatch as FBP
bg_kw = dict(boxstyle="round,pad=0.25", linewidth=2.5, zorder=1)
ax.add_patch(FBP((0.3,  4.7), 5.9, 5.1, facecolor='#E8F5E9', edgecolor='#2E7D32', **bg_kw))
ax.add_patch(FBP((6.5,  4.7), 3.0, 5.1, facecolor='#FFF8E1', edgecolor='#F9A825', **bg_kw))
ax.add_patch(FBP((10.1, 4.7), 9.6, 5.1, facecolor='#E3F2FD', edgecolor='#1565C0', **bg_kw))

section_label(ax, 3.25, 9.65, 'Phase 1: Meta-Training (Offline)',        C_P1)
section_label(ax, 8.00, 9.65, 'FOMAML Loop',                             C_FM)
section_label(ax, 15.0, 9.65, 'Phase 2: Test-Time Recovery (Online)',     C_P2)

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 – left column
# ─────────────────────────────────────────────────────────────────────────────
p1l_geom = []
for cy, txt in zip(p1l_ys, p1l_texts):
    g = auto_box(ax, P1L_X, cy, txt, fc='#C8E6C9', ec=C_P1)
    p1l_geom.append(g)   # (bx, by, bw, bh)

# arrows down
for i in range(len(p1l_geom)-1):
    bx,by,bw,bh = p1l_geom[i]
    bx2,by2,bw2,bh2 = p1l_geom[i+1]
    arrow(ax, bx+bw/2, by, bx+bw/2, by2+bh2, C_P1)

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 – right column
# ─────────────────────────────────────────────────────────────────────────────
p1r_geom = []
for cy, txt in zip(p1r_ys, p1r_texts):
    g = auto_box(ax, P1R_X, cy, txt, fc='#A5D6A7', ec=C_P1)
    p1r_geom.append(g)

# arrows down
for i in range(len(p1r_geom)-1):
    bx,by,bw,bh = p1r_geom[i]
    bx2,by2,bw2,bh2 = p1r_geom[i+1]
    arrow(ax, bx+bw/2, by, bx+bw/2, by2+bh2, C_P1)

# cross-column elbow: bottom of left → top of right col (at same y level if aligned)
# Left col bottom → right col top (horizontal then vertical elbow)
lbx,lby,lbw,lbh = p1l_geom[-1]   # bottom left box
rbx,rby,rbw,rbh = p1r_geom[0]    # top right box
mid_x = (lbx + lbw + rbx) / 2
elbow(ax,
    [(lbx+lbw,  lby+lbh/2),
     (mid_x,    lby+lbh/2),
     (mid_x,    rby+rbh/2),
     (rbx,      rby+rbh/2)],
    C_P1)

# ─────────────────────────────────────────────────────────────────────────────
# FOMAML boxes
# ─────────────────────────────────────────────────────────────────────────────
fm_geom = []
for cy, txt in zip(fm_ys, fm_texts):
    g = auto_box(ax, FM_X, cy, txt, fc='#FFE082', ec='#E65100')
    fm_geom.append(g)

# straight arrows down
cx_fm = FM_X
for i in range(len(fm_geom)-1):
    bx,by,bw,bh = fm_geom[i]
    bx2,by2,bw2,bh2 = fm_geom[i+1]
    arrow(ax, bx+bw/2, by, bx+bw/2, by2+bh2, C_FM)

# loop-back elbow on right side
bx0,by0,bw0,bh0 = fm_geom[0]   # Outer Update
bx2,by2,bw2,bh2 = fm_geom[2]   # Query-set Loss
lbx_fm = bx0 + bw0 + 0.3
elbow(ax,
    [(bx2+bw2,       by2+bh2/2),
     (lbx_fm,        by2+bh2/2),
     (lbx_fm,        by0+bh0/2),
     (bx0+bw0,       by0+bh0/2)],
    C_FM)

# arrow: Phase 1 Task Construction → FOMAML Inner Loop
p1r_last = p1r_geom[-1]   # Task Construction
fm_inner = fm_geom[1]      # Inner Loop
arrow(ax,
      p1r_last[0]+p1r_last[2], p1r_last[1]+p1r_last[3]/2,
      fm_inner[0],             fm_inner[1]+fm_inner[3]/2,
      '#33691E', lw=2.2)

# ─────────────────────────────────────────────────────────────────────────────
# θ₀ box
# ─────────────────────────────────────────────────────────────────────────────
th_g = auto_box(ax, TH_X, TH_Y,
                'θ₀ (11 params)\nMeta-Initialization',
                fc='#FFF9C4', ec=C_TH, bold=True, lw=3.0, tc=C_FM)
# arrow from Query-set Loss bottom → θ₀ top
ql = fm_geom[2]
arrow(ax, ql[0]+ql[2]/2, ql[1], th_g[0]+th_g[2]/2, th_g[1]+th_g[3], C_TH, lw=2.5)

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 – left column
# ─────────────────────────────────────────────────────────────────────────────
p2l_geom = []
for cy, txt in zip(p2l_ys, p2l_texts):
    g = auto_box(ax, P2L_X, cy, txt, fc='#BBDEFB', ec='#1565C0')
    p2l_geom.append(g)

p2cx = P2L_X
p2_down = C_P2
arrow(ax, p2l_geom[0][0]+p2l_geom[0][2]/2, p2l_geom[0][1],
          p2l_geom[1][0]+p2l_geom[1][2]/2, p2l_geom[1][1]+p2l_geom[1][3],
      p2_down)

# ─────────────────────────────────────────────────────────────────────────────
# Decision diamond
# ─────────────────────────────────────────────────────────────────────────────
dm = plt.Polygon([
    (DM_X - DM_HW, DM_Y),
    (DM_X,         DM_Y + DM_HH),
    (DM_X + DM_HW, DM_Y),
    (DM_X,         DM_Y - DM_HH),
], closed=True, facecolor='#90CAF9', edgecolor='#0D47A1', linewidth=2.5, zorder=3)
ax.add_patch(dm)
ax.text(DM_X, DM_Y, 'TPR≥β\nFPR≤α ?', ha='center', va='center',
        fontsize=8.5, fontweight='bold', color='#0D47A1', zorder=4)

# Feature Extraction → diamond top
fe = p2l_geom[1]
arrow(ax, fe[0]+fe[2]/2, fe[1], DM_X, DM_Y+DM_HH, p2_down)

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 right column: Adapt, Re-eval, Deploy θ_k
# ─────────────────────────────────────────────────────────────────────────────
P2R_X = 17.0   # right col x (overrides layout constant above)
ad_g  = auto_box(ax, P2R_X, ad_y,  'Adapt: k steps\nθ_k = θ₀ − η∇ℒ\n(k ∈ {1,2,4,8,16,32})',
                 fc='#BBDEFB', ec='#1565C0')
rev_g = auto_box(ax, P2R_X, rev_y, 'Re-evaluate\nTPR, FPR on query',
                 fc='#BBDEFB', ec='#1565C0')
dep_g = auto_box(ax, P2R_X, dep_y, '✓ Deploy θ_k\nat k* steps',
                 fc='#A5D6A7', ec=C_OK, bold=True, lw=3.0)

# diamond right → Adapt left edge (elbow: go right from diamond, then straight to box left)
elbow(ax,
    [(DM_X+DM_HW,  DM_Y),
     (ad_g[0]-0.1, DM_Y),
     (ad_g[0]-0.1, ad_g[1]+ad_g[3]/2),
     (ad_g[0],     ad_g[1]+ad_g[3]/2)],
    C_P2, lw=2.2)
ax.text(ad_g[0]+ad_g[2]/2, ad_g[1]+ad_g[3]+0.12, 'No',
        fontsize=9, color=C_FAIL, fontweight='bold', ha='center')

# Adapt → Re-eval
arrow(ax, ad_g[0]+ad_g[2]/2, ad_g[1], rev_g[0]+rev_g[2]/2, rev_g[1]+rev_g[3], C_P2)

# Re-eval → Deploy
arrow(ax, rev_g[0]+rev_g[2]/2, rev_g[1], dep_g[0]+dep_g[2]/2, dep_g[1]+dep_g[3], C_OK)
ax.text(dep_g[0]+dep_g[2]+0.08, (rev_g[1]+dep_g[1]+dep_g[3])/2,
        'Pass', fontsize=8.5, color=C_OK, fontweight='bold')

# Fail loop-back: Re-eval right → Adapt right (path entirely outside boxes)
fl_x = max(rev_g[0]+rev_g[2], ad_g[0]+ad_g[2]) + 0.55
elbow(ax,
    [(rev_g[0]+rev_g[2], rev_g[1]+rev_g[3]/2),
     (fl_x,              rev_g[1]+rev_g[3]/2),
     (fl_x,              ad_g[1]+ad_g[3]/2),
     (ad_g[0]+ad_g[2],   ad_g[1]+ad_g[3]/2)],
    C_FAIL)
ax.text(fl_x+0.08, (rev_g[1]+rev_g[3]/2+ad_g[1]+ad_g[3]/2)/2,
        'Fail / k++', fontsize=8.5, color=C_FAIL, fontweight='bold')

# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot deploy box
# ─────────────────────────────────────────────────────────────────────────────
zs_g = auto_box(ax, ZS_X, ZS_Y,
                '✓ Zero-Shot Deploy\nθ₀ directly\n(0.079 ms/img)',
                fc='#A5D6A7', ec=C_OK, bold=True, lw=3.0)

# diamond bottom → Zero-Shot
elbow(ax,
    [(DM_X,          DM_Y-DM_HH),
     (DM_X,          zs_g[1]+zs_g[3]+0.05)],
    C_OK)
ax.annotate('', xy=(zs_g[0]+zs_g[2]/2, zs_g[1]+zs_g[3]),
            xytext=(DM_X, zs_g[1]+zs_g[3]+0.05),
            arrowprops=dict(arrowstyle='->', color=C_OK, lw=2.2, mutation_scale=15), zorder=6)
ax.text(DM_X+0.12, DM_Y-DM_HH-0.15, 'Yes (k*=0)',
        fontsize=8.5, color=C_OK, fontweight='bold')

# ─────────────────────────────────────────────────────────────────────────────
# θ₀ transfer dashed arrow: θ₀ box → diamond bottom
# ─────────────────────────────────────────────────────────────────────────────
arrow(ax, th_g[0]+th_g[2], th_g[1]+th_g[3]/2,
          DM_X-DM_HW,      DM_Y,
      C_TH, lw=2.8, ls='--', ms=17)
ax.text((th_g[0]+th_g[2]+DM_X-DM_HW)/2,
        th_g[1]+th_g[3]/2 - 0.22,
        'θ₀ transferred', fontsize=9, color=C_TH,
        fontweight='bold', fontstyle='italic', ha='center')

# ─────────────────────────────────────────────────────────────────────────────
# Caption & Legend
# ─────────────────────────────────────────────────────────────────────────────
# Caption only (no legend in tight layout)
ax.text(10, 4.1,
        'Fig. 1.  Meta-watermarking pipeline overview. '
        'Phase 1 (green): offline meta-training produces shared initialization θ₀. '
        'FOMAML loop (yellow): inner/outer optimization learns task-agnostic weights. '
        'Phase 2 (blue): zero-shot deploy or minimal-step adaptation for unseen attacks.',
        ha='center', va='center', fontsize=8.8, fontstyle='italic', color='#333333',
        wrap=True,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FAFAFA', edgecolor='#BBBBBB', lw=1.5))

ax.text(1.5,  4.6, '■ Offline (one-time cost)',    fontsize=9, color=C_P1, fontweight='bold')
ax.text(7.5,  4.6, '■ Meta-optimization',           fontsize=9, color=C_FM, fontweight='bold')
ax.text(13.5, 4.6, '■ Online (per unseen attack)',  fontsize=9, color=C_P2, fontweight='bold')

# ── Save ──
plt.savefig(ACCESS_TEMPLATE_DIR / 'fig1_pipeline_flow.png',
            dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.2)
plt.savefig(WORKSPACE_ROOT / 'fig1_pipeline_flow.png',
            dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.2)
plt.close()
print("fig1_pipeline_flow.png generated successfully")
