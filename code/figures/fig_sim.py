'''
Date: 2026-07-09 18:22:27
LastEditors: Hubery-Lee hrbeulh@126.com
LastEditTime: 2026-07-10 10:10:24
FilePath: code/figures/fig_sim.py
Description: Do not edit
'''
# -*- coding: utf-8 -*-
import warnings, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata
warnings.filterwarnings("ignore")
A,EFF=7.4e8,0.5; DIRN=np.array([10.,10.])/np.hypot(10,10); HALF=np.deg2rad(12.); LEAK=1e-4
def mu(P):
    r=np.clip(np.hypot(P[:,0],P[:,1]),0.3,None); u=P/r[:,None]
    ins=np.arccos(np.clip(u@DIRN,-1,1))<=HALF
    return A*EFF/(4*np.pi*r**2)*np.where(ins,1.0,LEAK)
g=np.linspace(0,10,120); GX,GY=np.meshgrid(g,g); G=np.column_stack([GX.ravel(),GY.ravel()])
YT=np.log10(mu(G)+1.0)
rng=np.random.RandomState(0); X=rng.uniform(0,10,(200,2)); cX=rng.poisson(mu(X)).astype(float); y=np.log10(cX+1.0)
preds={"Ground truth":YT}
p=griddata(X,y,G,method="linear"); nan=np.isnan(p); p[nan]=griddata(X,y,G[nan],method="nearest")
preds["Linear"]=p
gp=GaussianProcessRegressor(ConstantKernel()*RBF(1.0,(1e-2,1e2))+WhiteKernel(1e-2,(1e-6,1e1)),normalize_y=True,n_restarts_optimizer=1,random_state=0).fit(X,y)
preds["GP (RBF)"]=gp.predict(G)
gp2=GaussianProcessRegressor(ConstantKernel()*Matern(1.0,(1e-2,1e2),nu=1.5)+ConstantKernel(0.1,(1e-4,1e2))+WhiteKernel(1e-2,(1e-6,1e1)),normalize_y=True,n_restarts_optimizer=1,random_state=0).fit(X,y)
preds["GP (Matérn 3/2 + bias)"]=gp2.predict(G)
def pk_alpha(c,base=5e-2,lo=1/3,hi=3.0):
    inv=1.0/(c+1.0); rel=np.clip(inv/np.median(inv),lo,hi); return base*rel
gpk=GaussianProcessRegressor(ConstantKernel(1.0,(1e-3,1e3))*Matern(1.0,(1e-2,1e2),nu=1.5)+ConstantKernel(0.1,(1e-4,1e2)),alpha=pk_alpha(cX),normalize_y=True,n_restarts_optimizer=1,random_state=0).fit(X,y)
preds["Poisson kriging"]=gpk.predict(G)
sc=StandardScaler(); Xs=sc.fit_transform(X); Gs=sc.transform(G)
m=MLPRegressor(hidden_layer_sizes=(64,32,16),activation='tanh',solver='lbfgs',alpha=0.01,max_iter=500,random_state=42).fit(Xs,y)
preds["MLP"]=m.predict(Gs)

# ---- Elsevier-style layout: 3 + 2 panels, bottom-aligned labels ----
# Font sizes matched to Fig. 1/2 (font.size 10, axes.labelsize 11) but nudged up,
# and the canvas kept narrow, so that after scaling to \textwidth the on-page type
# is as large as in Figs. 1-2 rather than shrunk down.
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "axes.linewidth": 0.8, "mathtext.default": "regular"})
labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
keys = list(preds.keys())
vmin, vmax = YT.min(), YT.max()

fig = plt.figure(figsize=(9.5, 6.6))
# 6 plotting cols (2 per panel) + 1 narrow col reserved for a shared, full-height colorbar
gs = fig.add_gridspec(2, 7, height_ratios=[1, 1],
                      width_ratios=[1, 1, 1, 1, 1, 1, 0.12],
                      wspace=0.85, hspace=0.40)
# row 1: 3 panels spanning 2 cols each (cols 0-1,2-3,4-5); row 2: 2 panels centered (cols 1-2, 3-4)
positions = [(0, slice(0, 2)), (0, slice(2, 4)), (0, slice(4, 6)),
             (1, slice(0, 2)), (1, slice(2, 4)), (1, slice(4, 6))]
axes = [fig.add_subplot(gs[r, c]) for r, c in positions]

im = None
for ax, k, lab in zip(axes, keys, labels):
    v = preds[k]
    im = ax.pcolormesh(GX, GY, v.reshape(GX.shape), vmin=vmin, vmax=vmax, cmap="viridis", shading="auto")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if k == "Ground truth":
        ax.scatter(X[:, 0], X[:, 1], s=3, c="w", alpha=0.55, linewidths=0)

# shared colorbar: full-height axis in the reserved right column, vertically
# centered over BOTH rows (represents the single global vmin/vmax color scale)
cax = fig.add_subplot(gs[:, 6])
pos = cax.get_position()
cax.set_position([pos.x0, pos.y0 + 0.18 * pos.height, pos.width, 0.64 * pos.height])
cb = fig.colorbar(im, cax=cax)
cb.set_label("log$_{10}$(count rate + 1)", fontsize=12)
cb.ax.tick_params(labelsize=11)

fig.canvas.draw()
ren = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
# bottom-align panel labels: within each row all labels share one baseline
# (only the panel axes enter the calc, never the colorbar)
row_y = {}
for ax, (r, _) in zip(axes, positions):
    y0 = inv.transform(ax.get_tightbbox(ren))[0][1]
    row_y[r] = min(row_y.get(r, 1.0), y0)
for ax, lab, (r, _) in zip(axes, labels, positions):
    bb = ax.get_position()
    fig.text((bb.x0 + bb.x1) / 2, row_y[r] - 0.015, lab, ha="center", va="top",
              fontsize=14, fontweight="bold")

plt.savefig("figs/fig_sim_recon.png", dpi=300, bbox_inches="tight")
print("fig_sim ok")
