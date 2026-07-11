# -*- coding: utf-8 -*-
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ef={i:json.loads(open(f'ef_{i}.json').read()) for i in range(1,8)}
old={i:json.loads(open(f'part_{i}.json').read()) for i in range(1,8)}
qm={i:json.loads(open(f'qm_{i}.json').read()) for i in range(1,8)}
nb={i:json.loads(open(f'nb_{i}.json').read()) for i in range(1,8)}
def arr(g): return np.array([g(i) for i in range(1,8)])
methods=["Linear","Plain GP","IDW","MLP","MLP ens.","RF","MK-GP","Poisson krig.","Physics-GP"]
A=[arr(lambda i: old[i]["A"]["linear"]),arr(lambda i: old[i]["A"]["gp"]),arr(lambda i: qm[i]["A"]["idw"]),
   arr(lambda i: old[i]["A"]["mlp2"]),arr(lambda i: ef[i]["A"]["mlpens"]),arr(lambda i: qm[i]["A"]["rf"]),
   arr(lambda i: nb[i]["A"]["mkgp"]),arr(lambda i: nb[i]["A"]["poissonk"]),
   arr(lambda i: ef[i]["A"]["anisogp"])]
B=[arr(lambda i: old[i]["B"]["linear"]["mean"]),arr(lambda i: old[i]["B"]["gp"]["mean"]),arr(lambda i: qm[i]["B"]["idw"]["mean"]),
   arr(lambda i: old[i]["B"]["mlp2"]["mean"]),arr(lambda i: ef[i]["B"]["mlpens"]["mean"]),arr(lambda i: qm[i]["B"]["rf"]["mean"]),
   arr(lambda i: nb[i]["B"]["mkgp"]["mean"]),arr(lambda i: nb[i]["B"]["poissonk"]["mean"]),
   arr(lambda i: ef[i]["B"]["anisogp"]["mean"])]

# ---- shared style: font sizes matched to Figs. 1/2, narrow canvas so on-page
# type after \textwidth scaling is as large as in Figs. 1-2 ----
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 11,
                     "axes.linewidth": 0.8, "mathtext.default": "regular"})

# ============ Fig. 6: two-protocol MAE (panels (a),(b); no in-axes titles) ============
fig,axes=plt.subplots(1,2,figsize=(9.5,4.0),constrained_layout=False)
xpos=np.arange(len(methods))
# Linear, Plain GP, IDW, MLP, MLP ens., RF, Physics-GP
colors=["#888","#4477aa","#ee8866","#cc6677","#cc6677","#999933","#66ccee","#aa3377","#228833"]
for ax,D in [(axes[0],A),(axes[1],B)]:
    mae=[d[:,0].mean() for d in D]; err=[d[:,0].std() for d in D]
    ax.bar(xpos,mae,yerr=err,capsize=3,color=colors,alpha=0.85)
    ax.set_xticks(xpos); ax.set_xticklabels(methods,rotation=32,ha="right")
    ax.set_ylabel("MAE [log$_{10}$(dose rate + 1)]")
fig.subplots_adjust(wspace=0.28,left=0.09,right=0.98,top=0.97,bottom=0.26)
fig.canvas.draw()
ren=fig.canvas.get_renderer(); inv=fig.transFigure.inverted()
ybase=min(inv.transform(ax.get_tightbbox(ren))[0][1] for ax in axes)
for ax,lab in zip(axes,["(a)","(b)"]):
    bb=ax.get_position()
    fig.text((bb.x0+bb.x1)/2, ybase-0.015, lab, ha="center", va="top",
             fontsize=14, fontweight="bold")
plt.savefig("figs/fig_protocols_mae.png",dpi=300,bbox_inches="tight")

# ============ Fig. 7: source-parameter consistency =============================
# (a) beam-direction rose (polar) + (b) half-angle bars. Direction and half-angle
# are shown on separate panels (they differ in scale and meaning); the polar view
# makes the run-1 mirror-direction degeneracy legible as a single 180deg-flipped arrow.
ds=list(range(1,8))
phi7=np.array([ef[i]["params"]["phi_deg"] for i in ds])   # 0..360
half7=np.array([ef[i]["params"]["half_deg"] for i in ds])
NOMINAL=12.0
fig2=plt.figure(figsize=(9.8,4.6),layout="constrained")
cmap=plt.get_cmap("tab10"); cols=[cmap(k) for k in range(7)]
# (a) beam-direction rose
axp=fig2.add_subplot(1,2,1,projection="polar")
axp.set_theta_zero_location("E"); axp.set_theta_direction(1)
for k,i in enumerate(ds):
    th=np.deg2rad(phi7[k])
    axp.annotate("",xy=(th,1.0),xytext=(0,0),arrowprops=dict(arrowstyle="-|>",color=cols[k],lw=2.2))
    axp.plot([],[],"-",color=cols[k],lw=2.2,label=f"run {i} ({phi7[k]:.0f}°)")
axp.set_rticks([]); axp.set_rlim(0,1.12); axp.set_thetagrids(range(0,360,45),fontsize=10)
axp.legend(loc="upper center",bbox_to_anchor=(0.5,-0.06),ncol=4,columnspacing=0.8,
           handletextpad=0.4,fontsize=8.5,frameon=False)
axp.text(0.5,-0.30,"(a)",transform=axp.transAxes,ha="center",va="top",fontsize=14,fontweight="bold")
# (b) half-angle bars
axb=fig2.add_subplot(1,2,2)
axb.bar(ds,half7,color=cols,alpha=0.85,edgecolor="k",lw=0.5)
axb.axhline(NOMINAL,ls="--",c="gray",lw=1.2,label=f"Nominal half-angle ({NOMINAL:.0f}°)")
for x,h in zip(ds,half7): axb.text(x,h+0.3,f"{h:.1f}",ha="center",va="bottom",fontsize=9)
axb.set_xticks(ds); axb.set_xlabel("Experiment run",labelpad=2)
axb.set_ylabel("Fitted half-angle (°)"); axb.set_ylim(0,half7.max()+3)
axb.legend(loc="upper right",frameon=False)
axb.text(0.5,-0.22,"(b)",transform=axb.transAxes,ha="center",va="top",fontsize=14,fontweight="bold")
plt.savefig("figs/fig_source_params.png",dpi=300,bbox_inches="tight")

# ============ Fig. 8: uncertainty calibration (single axes, no title) ============
cov1=[ef[i]["B"]["calib"]["cov1"] for i in ds]; cov2=[ef[i]["B"]["calib"]["cov2"] for i in ds]
w=0.35
fig3,ax=plt.subplots(figsize=(6.4,4.0),layout="constrained")
ax.bar(np.array(ds)-w/2,cov1,w,label="Empirical coverage |z|<1",color="#4477aa")
ax.bar(np.array(ds)+w/2,cov2,w,label="Empirical coverage |z|<2",color="#66ccee")
ax.axhline(0.683,ls="--",c="#4477aa",lw=1); ax.axhline(0.954,ls="--",c="#66ccee",lw=1)
ax.set_ylim(0,1.05); ax.set_xlabel("Experiment run"); ax.set_ylabel("Coverage")
ax.legend(loc="lower right")
plt.savefig("figs/fig_calibration.png",dpi=300,bbox_inches="tight")
print("stats figs ok")
