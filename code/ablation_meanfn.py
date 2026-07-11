# -*- coding: utf-8 -*-
# ABLATION (referee response, DA fairness challenge): does the physics-guided GP's
# extrapolation advantage come from an UNFAIR embedding of the correct field family,
# or is it a checkable prior that degrades gracefully when misspecified?
#
# Same 7 runs, same Protocol B (spatial block GroupKFold), same residual-GP machinery.
# Only the MEAN function m(x) changes:
#   M0  correct collimated-beam template        (as in the paper)
#   M1  isotropic 1/r^2 mean (no angular cutoff) -> partial prior
#   M2  collimated template with beam direction forced 90deg off -> STRUCTURE ok, PARAM wrong
#   M3  constant mean == plain GP                -> no prior (lower bound)
#
# If M2 collapses toward M3, the advantage is prior-CORRECTNESS-dependent, and correctness
# is judged online by residual fit-cost -> this is the graceful-degradation story, not an
# unfair comparison. Deterministic (fixed seeds).
import json, warnings
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import least_squares
from sklearn.model_selection import GroupKFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data"

def load(i):
    df = pd.read_csv(DATA/str(i)/"data.txt", sep=r"\s+", comment="#", header=None,
                     names=["t","px","py","dose","d01","date","time","u"])
    df = df[(df.px.abs()>1e-20)|(df.py.abs()>1e-20)]
    return df[["px","py"]].to_numpy(float), np.log10(df["dose"].to_numpy(float)+1.0)

def met(y,p):
    sr=np.sum((y-p)**2); st=np.sum((y-np.mean(y))**2)
    return [float(np.mean(np.abs(y-p))), float(np.sqrt(np.mean((y-p)**2))), float(1-sr/st) if st>0 else 0.0]

# ---- generic collimated template; half=pi/leak->1 makes it isotropic ----
def template(p, Xq):
    sx,sy,phi,half,logw,logC,logleak,logbg = p
    dx=Xq[:,0]-sx; dy=Xq[:,1]-sy; r2=dx*dx+dy*dy+0.09
    th=np.abs(np.arctan2(dy,dx)-phi); th=np.minimum(th,2*np.pi-th)
    f=10**logleak+(1-10**logleak)/(1+np.exp((th-half)/max(10**logw,1e-3)))
    return np.log10(10**logC*f/r2+10**logbg+1.0)

def fit_template(X,y,init_px=None,fix_phi=None,isotropic=False):
    i0=np.argmax(y); sx0,sy0=X[i0]
    hot=y>np.percentile(y,85)
    phi0=np.arctan2(np.mean(X[hot,1]-sy0),np.mean(X[hot,0]-sx0)) if hot.sum()>3 else 0.0
    lb=[X[:,0].min()-2,X[:,1].min()-2,-3*np.pi,np.deg2rad(3),-2.5,-2,-6,-2]
    ub=[X[:,0].max()+2,X[:,1].max()+2, 3*np.pi,np.deg2rad(80),0.5,12,0,2]
    if isotropic:  # no angular cutoff: half fixed wide, leak->1 so g(theta)~const
        lb[3]=np.deg2rad(179); ub[3]=np.deg2rad(180); lb[6]=-0.01; ub[6]=0.0
    if init_px is not None:
        inits=[list(init_px), list(init_px[:2])+[init_px[2]+np.pi]+list(init_px[3:])]
    else:
        inits=[[sx0,sy0,phi0+d,np.deg2rad(h),-1.0,np.log10(max(10**y.max()-1,1)*0.09+1),-3.0,0.0]
               for d in [0,np.pi/2,np.pi,-np.pi/2] for h in [10,25]]
    best=None
    for p0 in inits:
        lb2,ub2=list(lb),list(ub)
        p0=list(p0)
        if isotropic:  # start half inside the wide-open bounds, leak near 1
            p0[3]=np.deg2rad(179.5); p0[6]=-0.005
        if fix_phi is not None:  # lock beam direction to a (wrong) value
            p0[2]=fix_phi; lb2[2]=fix_phi-1e-6; ub2[2]=fix_phi+1e-6
        # clip any init into bounds to avoid least_squares raising
        p0=[min(max(v,lo),hi) for v,lo,hi in zip(p0,lb2,ub2)]
        try:
            r=least_squares(lambda p: template(p,X)-y,p0,loss="soft_l1",
                            bounds=(lb2,ub2),max_nfev=500)
            if best is None or r.cost<best.cost: best=r
        except Exception: pass
    if best is None:  # degrade to a flat template (constant) rather than crash
        flat=[X[:,0].mean(),X[:,1].mean(),0.0,np.deg2rad(80),-1.0,-2.0,0.0,float(np.median(y))]
        return np.array(flat), np.inf
    return best.x, best.cost

def gp_resid(Xtr,r,khat=None):
    if khat is None:
        k=ConstantKernel(1.0,(1e-3,1e3))*Matern(1.0,(1e-2,1e2),nu=1.5)+WhiteKernel(1e-2,(1e-6,1e1))
        rng=np.random.RandomState(0); idx=rng.choice(len(r),min(700,len(r)),replace=False)
        g0=GaussianProcessRegressor(kernel=k,normalize_y=True,n_restarts_optimizer=1,random_state=0)
        g0.fit(Xtr[idx],r[idx]); khat=g0.kernel_
    g=GaussianProcessRegressor(kernel=khat,normalize_y=True,optimizer=None); g.fit(Xtr,r)
    return g,khat

def groups(X,nb=4):
    gx=np.digitize(X[:,0],np.linspace(X[:,0].min(),X[:,0].max(),nb+1)[1:-1])
    gy=np.digitize(X[:,1],np.linspace(X[:,1].min(),X[:,1].max(),nb+1)[1:-1])
    return gx*nb+gy

VARIANTS=["M0_correct","M1_isotropic","M2_wrongdir","M3_constant"]
res={v:[] for v in VARIANTS}

for i in range(1,8):
    X,y=load(i)
    # reference correct fit on full data (for wrong-direction offset + khat seed)
    px_full,_=fit_template(X,y)
    khat=None
    g4=groups(X); splits=list(GroupKFold(n_splits=min(5,len(np.unique(g4)))).split(X,y,groups=g4))
    per={v:[] for v in VARIANTS}
    for tr,te in splits:
        Xt,yt,Xv,yv=X[tr],y[tr],X[te],y[te]
        # M0 correct template
        p0,_=fit_template(Xt,yt,init_px=px_full)
        g,khat=gp_resid(Xt,yt-template(p0,Xt),khat=khat)
        pred=template(p0,Xv)+g.predict(Xv); per["M0_correct"].append(met(yv,pred))
        # M1 isotropic mean (1/r^2, no angular cutoff)
        p1,_=fit_template(Xt,yt,isotropic=True)
        g1,_=gp_resid(Xt,yt-template(p1,Xt),khat=khat)
        pred=template(p1,Xv)+g1.predict(Xv); per["M1_isotropic"].append(met(yv,pred))
        # M2 wrong direction (structure ok, beam 90deg off)
        p2,_=fit_template(Xt,yt,fix_phi=px_full[2]+np.pi/2)
        g2,_=gp_resid(Xt,yt-template(p2,Xt),khat=khat)
        pred=template(p2,Xv)+g2.predict(Xv); per["M2_wrongdir"].append(met(yv,pred))
        # M3 constant mean == plain GP
        c=float(np.mean(yt))
        g3,_=gp_resid(Xt,yt-c,khat=khat)
        pred=c+g3.predict(Xv); per["M3_constant"].append(met(yv,pred))
    for v in VARIANTS: res[v].append(np.array(per[v]).mean(0).tolist())
    print(f"run{i} done: "+"  ".join(f"{v.split('_')[0]} R2={np.array(per[v]).mean(0)[2]:+.2f}" for v in VARIANTS),flush=True)

out={}
print("\n=== ablation summary (Protocol B, mean over 7 runs) ===")
print(f"{'variant':14s} {'MAE':>14s} {'RMSE':>14s} {'R2':>16s}")
for v in VARIANTS:
    a=np.array(res[v])
    out[v]={"mean":a.mean(0).tolist(),"std":a.std(0).tolist(),"runs":res[v]}
    print(f"{v:14s} {a[:,0].mean():.3f}±{a[:,0].std():.3f}  {a[:,1].mean():.3f}±{a[:,1].std():.3f}  {a[:,2].mean():+.3f}±{a[:,2].std():.3f}")
open("ablation_results.json","w").write(json.dumps(out,ensure_ascii=False,indent=1))
print("\nsaved ablation_results.json")
