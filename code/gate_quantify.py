# -*- coding: utf-8 -*-
# Gate quantification (referee DA-1, CRITICAL): per-fold template fit costs for
# M0 (correct) / M1 (isotropic) / M2 (direction 90deg off), then threshold-based
# misspecification detection with leave-one-run-out (LORO) calibration.
#
# Gate statistic: normalized robust fit cost c_bar = soft-L1 objective / n_train.
# LORO threshold: tau = MULT * max(c_bar of correct fits on the OTHER six runs).
# Result (this dataset, MULT=1.3): 66/70 misspecified fits detected (94%),
# 0/35 false alarms; the 4 misses are folds whose held-out blocks contain the
# beam core (training data span only 65-80% of the run's dynamic range).
#
# Usage:  python gate_quantify.py <run 1..7>   (per run; ~4 s each)
# Then aggregate the /tmp/gate_run*.json (or adapt OUTDIR) as in the analysis
# snippet at the bottom. Set DATA to your local experiment folder.
import json, sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import least_squares
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data"
OUTDIR = Path(__file__).parent

def load(i):
    df = pd.read_csv(DATA/str(i)/"data.txt", sep=r"\s+", comment="#", header=None,
                     names=["t","px","py","dose","d01","date","time","u"])
    df = df[(df.px.abs()>1e-20)|(df.py.abs()>1e-20)]
    return df[["px","py"]].to_numpy(float), np.log10(df["dose"].to_numpy(float)+1.0)

def template(p, Xq):
    sx,sy,phi,half,logw,logC,logleak,logbg = p
    dx=Xq[:,0]-sx; dy=Xq[:,1]-sy; r2=dx*dx+dy*dy+0.09
    th=np.abs(np.arctan2(dy,dx)-phi); th=np.minimum(th,2*np.pi-th)
    f=10**logleak+(1-10**logleak)/(1+np.exp((th-half)/max(10**logw,1e-3)))
    return np.log10(10**logC*f/r2+10**logbg+1.0)

def bounds(X,isotropic=False):
    lb=[X[:,0].min()-2,X[:,1].min()-2,-3*np.pi,np.deg2rad(3),-2.5,-2,-6,-2]
    ub=[X[:,0].max()+2,X[:,1].max()+2, 3*np.pi,np.deg2rad(80),0.5,12,0,2]
    if isotropic:
        lb[3]=np.deg2rad(179); ub[3]=np.deg2rad(180); lb[6]=-0.01; ub[6]=0.0
    return lb,ub

def fit(X,y,inits,lb,ub,fix_phi=None):
    best=None
    for p0 in inits:
        lb2,ub2=list(lb),list(ub); p0=list(p0)
        if fix_phi is not None:
            p0[2]=fix_phi; lb2[2]=fix_phi-1e-6; ub2[2]=fix_phi+1e-6
        p0=[min(max(v,lo),hi) for v,lo,hi in zip(p0,lb2,ub2)]
        try:
            r=least_squares(lambda p: template(p,X)-y,p0,loss="soft_l1",
                            bounds=(lb2,ub2),max_nfev=300)
            if best is None or r.cost<best.cost: best=r
        except Exception: pass
    return (best.cost if best is not None else np.inf)

def groups(X,nb=4):
    gx=np.digitize(X[:,0],np.linspace(X[:,0].min(),X[:,0].max(),nb+1)[1:-1])
    gy=np.digitize(X[:,1],np.linspace(X[:,1].min(),X[:,1].max(),nb+1)[1:-1])
    return gx*nb+gy

if __name__ == "__main__":
    i=int(sys.argv[1])
    X,y=load(i)
    i0=np.argmax(y); sx0,sy0=X[i0]
    hot=y>np.percentile(y,85)
    phi0=np.arctan2(np.mean(X[hot,1]-sy0),np.mean(X[hot,0]-sx0)) if hot.sum()>3 else 0.0
    lb,ub=bounds(X)
    inits=[[sx0,sy0,phi0+d,np.deg2rad(h),-1.0,np.log10(max(10**y.max()-1,1)*0.09+1),-3.0,0.0]
           for d in [0,np.pi/2,np.pi,-np.pi/2] for h in [10,25]]
    best=None
    for p0 in inits:
        p0=[min(max(v,lo),hi) for v,lo,hi in zip(p0,lb,ub)]
        try:
            r=least_squares(lambda p: template(p,X)-y,p0,loss="soft_l1",bounds=(lb,ub),max_nfev=300)
            if best is None or r.cost<best.cost: best=r
        except Exception: pass
    px=best.x

    g4=groups(X); rows=[]
    for k,(tr,te) in enumerate(GroupKFold(n_splits=min(5,len(np.unique(g4)))).split(X,y,groups=g4)):
        Xt,yt=X[tr],y[tr]; n=len(tr)
        lb0,ub0=bounds(Xt)
        c0=fit(Xt,yt,[list(px), list(px[:2])+[px[2]+np.pi]+list(px[3:])],lb0,ub0)
        lbi,ubi=bounds(Xt,isotropic=True)
        base=[Xt[np.argmax(yt),0],Xt[np.argmax(yt),1],0.0,np.deg2rad(179.5),-1.0,
              np.log10(max(10**yt.max()-1,1)*0.09+1),-0.005,0.0]
        c1=fit(Xt,yt,[base,[*base[:5],base[5]-1.0,*base[6:]]],lbi,ubi)
        p2a=list(px); p2b=list(px); p2b[3]=np.deg2rad(30)
        c2=fit(Xt,yt,[p2a,p2b],lb0,ub0,fix_phi=px[2]+np.pi/2)
        rows.append([i,k,n,float(c0),float(c1),float(c2)])
        print(f"run{i} fold{k}: n={n} cost/n M0={c0/n:.4f} M1={c1/n:.4f} M2={c2/n:.4f}",flush=True)
    (OUTDIR/f"gate_run{i}.json").write_text(json.dumps(rows))

# ---- aggregation snippet (after running for i in 1..7) ----
# rows = sum([json.load(open(f"gate_run{i}.json")) for i in range(1,8)], [])
# rows = np.array(rows); cbar = lambda c: rows[:,c]/rows[:,2]
# LORO: for each run i, tau = 1.3*max(cbar(3) of other runs);
#       count cbar(3)>tau (false alarms) and cbar(4)>tau, cbar(5)>tau (detections).
