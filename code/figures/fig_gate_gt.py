# -*- coding: utf-8 -*-
# Figure: simulated ground-truth fields for the fit-quality-gate stress test.
# Style matched to fig_sim.py (Figure 4): shared full-height colorbar on the
# right, bold panel labels (a)-(d) placed below each panel, no in-axes titles.
# Regenerates the four scenario fields from gate_robustness_sim.py.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "code/" is importable
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gate_robustness_sim import FIELDS, traj_sample

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "axes.linewidth": 0.8, "mathtext.default": "regular"})

OUT = Path("figs"); OUT.mkdir(exist_ok=True)
g = np.linspace(0, 10, 240); GX, GY = np.meshgrid(g, g); G = np.column_stack([GX.ravel(), GY.ravel()])
panels = [("C", 100), ("MS", 200), ("SC", 300), ("OC", 405)]   # (scenario, seed)
labels = ["(a)", "(b)", "(c)", "(d)"]

def oc_params(seed):
    """Replay make_OC's RNG draws to recover the shadow-wedge centre bearing."""
    rng = np.random.RandomState(seed)
    src = rng.uniform(1, 3, 2); ang = rng.uniform(0, 2 * np.pi)
    rng.uniform(10, 18); shadow_c = ang + rng.uniform(-0.15, 0.15)
    return src, shadow_c

data, allY = [], []
for scen, seed in panels:
    rng = np.random.RandomState(seed); f = FIELDS[scen](rng)
    yG = np.log10(f(G) + 1.0).reshape(GX.shape)
    rng2 = np.random.RandomState(seed); FIELDS[scen](rng2); X = traj_sample(rng2)
    data.append((scen, seed, yG, X)); allY.append(yG)
vmax = float(np.percentile(np.concatenate([y.ravel() for y in allY]), 99.5))

fig = plt.figure(figsize=(7.2, 7.0))
gs = fig.add_gridspec(2, 5, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1, 0.12],
                      wspace=0.55, hspace=0.38)
positions = [(0, slice(0, 2)), (0, slice(2, 4)), (1, slice(0, 2)), (1, slice(2, 4))]
axes = [fig.add_subplot(gs[r, c]) for r, c in positions]
im = None
for ax, (scen, seed, yG, X) in zip(axes, data):
    im = ax.pcolormesh(GX, GY, yG, vmin=0, vmax=vmax, cmap="viridis", shading="auto")
    ax.scatter(X[:, 0], X[:, 1], s=3, c="w", alpha=0.55, linewidths=0)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_xticks([0, 5, 10]); ax.set_yticks([0, 5, 10])
    if scen == "OC":
        src, sc = oc_params(seed); x0, y0 = src[0] + 2.2 * np.cos(sc), src[1] + 2.2 * np.sin(sc)
        ax.annotate("shadow", xy=(x0, y0), xytext=(x0 + 1.8, y0 + 1.9), color="w", fontsize=11,
                    ha="left", arrowprops=dict(arrowstyle="->", color="w", lw=1.2))

cax = fig.add_subplot(gs[:, 4]); pos = cax.get_position()
cax.set_position([pos.x0, pos.y0 + 0.18 * pos.height, pos.width, 0.64 * pos.height])
cb = fig.colorbar(im, cax=cax); cb.set_label("log$_{10}$(count rate + 1)", fontsize=12)
cb.ax.tick_params(labelsize=11)

fig.canvas.draw(); ren = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
row_y = {}
for ax, (r, _) in zip(axes, positions):
    y0 = inv.transform(ax.get_tightbbox(ren))[0][1]; row_y[r] = min(row_y.get(r, 1.0), y0)
for ax, lab, (r, _) in zip(axes, labels, positions):
    bb = ax.get_position()
    fig.text((bb.x0 + bb.x1) / 2, row_y[r] - 0.015, lab, ha="center", va="top",
             fontsize=14, fontweight="bold")

fig.savefig(OUT / "fig_gate_gt.png", dpi=300, bbox_inches="tight")
print("saved", OUT / "fig_gate_gt.png", "| vmax=%.2f" % vmax)
