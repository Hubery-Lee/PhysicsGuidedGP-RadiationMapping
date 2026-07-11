# -*- coding: utf-8 -*-
# Run 1: measurement trajectory over the house map, with the fitted source
# position and fitted beam direction (to inspect the Fig.7 run-1 anomaly).
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

DATA = Path(__file__).resolve().parents[2] / "data"
RUN = "1"

# ---- house map + pixel->metre extent (resolution 0.05, origin [-50,-50]) ----
# Convention copied from fig_exp.py: everything is plotted in the image PIXEL
# frame, then relabelled to metres via P2M. This is what puts y the right way up.
rgb = Image.open(DATA/RUN/"house.pgm").convert("RGB")
IMW, IMH = rgb.size
P2M = lambda p: p*0.05 - 50.0
extent = (P2M(0), P2M(IMW), P2M(0), P2M(IMH))

# ---- trajectory: metric -> pixel (matching fig_exp.py), then P2M back to metre ----
df = pd.read_csv(DATA/RUN/"data.txt", sep=r"\s+", comment="#", header=None,
                 names=["t","px","py","dose","d01","date","time","u"])
df = df[(df.px.abs()>1e-20)|(df.py.abs()>1e-20)]
xpix = ((df["px"]+50)/0.05).to_numpy(float)
ypix = (IMH - (df["py"]+50)/0.05).to_numpy(float)   # fig_exp.py: 1984 - ...
xm = P2M(xpix); ym = P2M(ypix)
dose = df["dose"].to_numpy(float)
y = np.log10(dose+1.0)

# ---- fitted source + beam direction (ef_1.json px = metric template params) ----
# ef params live in the metric (px,py) frame; convert to the same pixel-then-P2M
# frame as the trajectory so the source sits correctly on the plotted map.
px = np.array(json.loads(open("ef_1.json").read())["px"])
sx = P2M((px[0]+50)/0.05)
sy = P2M((IMH - (px[1]+50)/0.05))
phi = -px[2]          # y-flip negates the angle
half = px[3]

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
                     "font.size":12,"axes.labelsize":13,"xtick.labelsize":11,"ytick.labelsize":11})

fig, ax = plt.subplots(figsize=(7.2,7.0), layout="constrained")
ax.imshow(rgb, extent=extent, origin="lower")
sc = ax.scatter(xm, ym, c=y, s=8, cmap="viridis", zorder=3)
ax.plot(xm, ym, "-", c="k", lw=0.5, alpha=0.4, zorder=2)      # path order
ax.scatter(xm[0], ym[0], marker="^", c="lime", s=120, ec="k", zorder=4, label="start")
ax.scatter(xm[-1], ym[-1], marker="v", c="red", s=120, ec="k", zorder=4, label="end")

# fitted source + fitted beam direction, and its mirror (true direction)
ax.scatter([sx],[sy], marker="*", c="yellow", s=350, ec="k", zorder=5, label="fitted source")
L = 4.0
for ang, col, lab in [(phi, "orange", f"fitted beam ({np.rad2deg(phi)%360:.0f}°)"),
                      (phi+np.pi, "cyan", "mirror direction")]:
    ax.annotate("", xy=(sx+L*np.cos(ang), sy+L*np.sin(ang)), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="->", color=col, lw=2.2), zorder=5)
    ax.plot([], [], "-", c=col, lw=2.2, label=lab)

# tighten to the visited region
pad = 3.0
ax.set_xlim(min(xm.min(),sx)-pad, max(xm.max(),sx)+pad)
ax.set_ylim(min(ym.min(),sy)-pad, max(ym.max(),sy)+pad)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
fig.colorbar(sc, ax=ax, shrink=0.7, label="log$_{10}$(dose rate + 1)")
fig.suptitle(f"Run 1: trajectory, fitted source and beam direction  (half-angle {np.rad2deg(half):.1f}°)",
             fontsize=12)
plt.savefig("figs/fig_run1_traj.png", dpi=200, bbox_inches="tight")
print("saved figs/fig_run1_traj.png")
print(f"source=({sx:.2f},{sy:.2f})  phi={np.rad2deg(phi)%360:.1f}deg  half={np.rad2deg(half):.1f}deg")
print(f"traj x[{xm.min():.1f},{xm.max():.1f}] y[{ym.min():.1f},{ym.max():.1f}]  centroid=({xm.mean():.2f},{ym.mean():.2f})  n={len(xm)}")
