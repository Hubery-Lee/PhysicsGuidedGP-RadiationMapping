r'''
Author: Hubery-Lee hrbeulh@126.com
Description: Generate fig_traj_ate.png (Figure 2) for the Sensors manuscript.
Layout: 2 rows — top: (a) motion-capture photo + (b) APE-colored trajectory;
bottom: (c) APE boxplot spanning full width. Panel labels bottom-centered.
Pipeline reproduces cal_ape.py: evo, TUM format, time association,
SE(3) alignment (no scale), translation-part APE.
'''
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from evo.core import sync, metrics
from evo.tools import file_interface

DATA = str(Path(__file__).resolve().parents[2] / "data_motioncapture")  # NOTE: motion-capture ATE data not bundled; place it here to regenerate Fig. 2
FIGS = str(Path(__file__).resolve().parents[2] / "figs")

errors, rmses, aligned = {}, {}, {}
for g in range(1, 6):
    ref = file_interface.read_tum_trajectory_file(f"{DATA}/ref{g}.txt")
    est = file_interface.read_tum_trajectory_file(f"{DATA}/est{g}.txt")
    ref, est = sync.associate_trajectories(ref, est)
    est.align(ref, correct_scale=False)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((ref, est))
    errors[g] = ape.error * 100.0  # cm
    rmses[g] = ape.get_all_statistics()["rmse"] * 100.0
    aligned[g] = (ref.positions_xyz, est.positions_xyz)

r = np.array([rmses[g] for g in range(1, 6)])
rep = 1  # shown trajectory (longest run, n=110)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "font.size": 10, "axes.labelsize": 11, "axes.linewidth": 0.8})
fig = plt.figure(figsize=(9.5, 8.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.35, 1.0],
                      wspace=0.25, hspace=0.30)
ax0 = fig.add_subplot(gs[0, 0]); ax1 = fig.add_subplot(gs[0, 1]); ax2 = fig.add_subplot(gs[1, :])

# (a) motion-capture experiment photo (downsampled 2x to bound file size)
img = plt.imread(f"{FIGS}/fig_motion_capture.png")
ax0.imshow(img[::2, ::2])
ax0.set_axis_off()

# (b) trajectory overlay: reference dashed gray, estimate APE-colored (jet)
refp, estp = aligned[rep]
err = errors[rep]
ax1.plot(refp[:, 0], refp[:, 1], "--", color="gray", lw=1.2)
pts = estp[:, :2].reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
norm = plt.Normalize(0.0, err.max())
lc = LineCollection(segs, cmap="jet", norm=norm, lw=1.6)
lc.set_array(0.5 * (err[:-1] + err[1:]))
ax1.add_collection(lc)
sc = ax1.scatter(estp[:, 0], estp[:, 1], c=err, cmap="jet", norm=norm, s=8, zorder=4)
cb = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.03)
cb.set_label("APE (cm)", fontsize=10)
allx = np.r_[refp[:, 0], estp[:, 0]]; ally = np.r_[refp[:, 1], estp[:, 1]]
dx, dy = allx.max() - allx.min(), ally.max() - ally.min()
ax1.set_xlim(allx.min() - 0.08 * dx, allx.max() + 0.08 * dx)
ax1.set_ylim(ally.min() - 0.08 * dy, ally.max() + 0.35 * dy)
ax1.set_aspect("equal", adjustable="box")
ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
handles = [Line2D([0], [0], ls="--", color="gray", lw=1.2),
           Line2D([0], [0], color=plt.cm.jet(0.5), lw=1.6)]
ax1.legend(handles, ["Ground truth", "SLAM estimate"],
           loc="upper right", fontsize=9, framealpha=0.9)

# (c) APE distribution per run (full-width bottom row)
data = [errors[g] for g in range(1, 6)]
ax2.boxplot(data, labels=[str(g) for g in range(1, 6)], widths=0.5,
            showfliers=False, patch_artist=True,
            medianprops=dict(color="#1f4e79", lw=1.4),
            boxprops=dict(facecolor="#aec7e8", alpha=0.7, lw=0.8),
            whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
ax2.plot(range(1, 6), [rmses[g] for g in range(1, 6)], "D", color="#d62728",
         ms=6, label="RMSE", zorder=5)
ax2.axhline(r.mean(), color="#d62728", ls=":", lw=1.0,
            label=f"Mean RMSE = {r.mean():.2f} cm")
ax2.set_xlabel("Run"); ax2.set_ylabel("Absolute trajectory error (cm)")
ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax2.set_ylim(bottom=0)

ax0.set_anchor("S"); ax1.set_anchor("S")
fig.tight_layout()
fig.canvas.draw()
# make (a) exactly match (b)'s vertical extent (same top and bottom)
p1 = ax1.get_position()          # (b) frame after aspect is applied
p0 = ax0.get_position()
ar = img.shape[1] / img.shape[0]  # photo w/h
w = p1.height * (fig.get_figheight() / fig.get_figwidth()) * ar
xc = (p0.x0 + p0.x1) / 2
ax0.set_position([xc - w / 2, p1.y0, w, p1.height])
fig.canvas.draw()
ren = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
# (a)/(b) share one label baseline; (c) label below its own decorations
ytop = min(inv.transform(ax.get_tightbbox(ren))[0][1] for ax in (ax0, ax1))
for ax, lab, y0 in [(ax0, "(a)", ytop), (ax1, "(b)", ytop),
                    (ax2, "(c)", inv.transform(ax2.get_tightbbox(ren))[0][1])]:
    bb = ax.get_position()
    fig.text((bb.x0 + bb.x1) / 2, y0 - 0.018, lab, ha="center", va="top", fontsize=12)

fig.savefig(f"{FIGS}/fig_traj_ate.png", dpi=300, bbox_inches="tight")
print("saved")
