# -*- coding: utf-8 -*-
# Fig: run-3 source-recovery overlay. Trajectory (colored by log dose rate),
# fitted source, fitted beam direction and nominal-cone, on run-3's own room map.
# Run 3 is the best-behaved run (highest reconstruction R2/F1, correct direction,
# half-angle close to nominal); coordinates are self-consistent (single-run frame).
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from matplotlib.patches import Wedge
from mpl_toolkits.axes_grid1 import make_axes_locatable

DATA = Path(__file__).resolve().parents[2] / "data"
RUN = "3"

# room map (run3 frame), pixel -> metre (fig_exp.py convention)
rgb = Image.open(DATA/RUN/"house.pgm").convert("RGB")
IMW, IMH = rgb.size
P2M = lambda p: p*0.05 - 50.0
extent = (P2M(0), P2M(IMW), P2M(0), P2M(IMH))

# trajectory: metric -> pixel -> P2M (same frame as the map)
df = pd.read_csv(DATA/RUN/"data.txt", sep=r"\s+", comment="#", header=None,
                 names=["t","px","py","dose","d01","date","time","u"])
df = df[(df.px.abs()>1e-20)|(df.py.abs()>1e-20)]
xm = P2M(((df["px"]+50)/0.05).to_numpy(float))
ym = P2M((IMH - (df["py"]+50)/0.05).to_numpy(float))
y = np.log10(df["dose"].to_numpy(float)+1.0)

# fitted source + beam direction (ef_3.json px = metric template params)
px = np.array(json.loads(open(f"ef_{RUN}.json").read())["px"])
sx = P2M((px[0]+50)/0.05); sy = P2M(IMH - (px[1]+50)/0.05)
phi = -px[2]; half = px[3]; deg = np.rad2deg(phi); halfdeg = np.rad2deg(half)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 10,
                     "axes.linewidth": 0.8, "mathtext.default": "regular"})

fig, ax = plt.subplots(figsize=(6.6, 5.6), constrained_layout=False)
ax.imshow(rgb, extent=extent, origin="lower")
sc = ax.scatter(xm, ym, c=y, s=7, cmap="viridis", zorder=3)

L = 4.0
ax.add_patch(Wedge((sx, sy), L, deg-halfdeg, deg+halfdeg, alpha=0.15,
                   color="orange", ec="none", zorder=4))
ax.annotate("", xy=(sx+L*np.cos(phi), sy+L*np.sin(phi)), xytext=(sx, sy),
            arrowprops=dict(arrowstyle="-|>", color="orange", lw=2.6), zorder=5)
ax.scatter([sx], [sy], marker="*", s=340, c="yellow", ec="k", lw=1.2, zorder=6)

ax.plot([], [], color="orange", lw=2.6, label=f"Fitted beam direction ({deg%360:.0f}°)")
ax.fill_between([], [], color="orange", alpha=0.15, label=f"Fitted beam cone ({halfdeg:.1f}° half-angle)")
ax.scatter([], [], marker="*", s=180, c="yellow", ec="k", label=f"Fitted source ({sx:.2f}, {sy:.2f}) m")

pad = 2.5
ax.set_xlim(min(xm.min(), sx)-pad, max(xm.max(), sx)+pad)
ax.set_ylim(min(ym.min(), sy)-pad, max(ym.max(), sy)+pad)
ax.set_aspect("equal")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.legend(loc="upper right", framealpha=0.92, fontsize=9)
div = make_axes_locatable(ax); cax = div.append_axes("right", size="4%", pad=0.08)
cb = fig.colorbar(sc, cax=cax); cb.set_label("log$_{10}$(dose rate + 1)", fontsize=11)
cb.ax.tick_params(labelsize=10)
fig.subplots_adjust(left=0.10, right=0.90, top=0.98, bottom=0.10)
plt.savefig("figs/fig_source_recon.png", dpi=300, bbox_inches="tight")
print("saved figs/fig_source_recon.png")
