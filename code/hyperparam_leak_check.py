# -*- coding: utf-8 -*-
# R1-W2 sensitivity: kernel hyperparams (a) estimated once on Protocol-A train, reused
# across Protocol-B folds (paper protocol) vs (b) refit per fold from fold-train only.
import json, sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import least_squares
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
warnings.filterwarnings("ignore")
DATA = Path(__file__).resolve().parents[1] / "data"

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

def fit_template(X,y,init_px=None):
    i0=np.argmax(y); sx0,sy0=X[i0]
    hot=y>np.percentile(y,85)
    phi0=np.arctan2(np.mean(X[hot,1]-sy0),np.mean(X[hot,0]-sx0)) if hot.sum()>3 else 0.0
    lb=[X[:,0].min()-2,X[:,1].min()-2,-3*np.pi,np.deg2rad(3),-2.5,-2,-6,-2]
    ub=[X[:,0].max()+2,X[:,1].max()+2, 3*np.pi,np.deg2rad(80),0.5,12,0,2]
    inits=([list(init_px), list(init_px[:2])+[init_px[2]+np.pi]+list(init_px[3:])] if init_px is not None
           else [[sx0,sy0,phi0+d,np.deg2rad(h),-1.0,np.log10(max(10**y.max()-1,1)*0.09+1),-3.0,0.0]
                 for d in [0,np.pi/2,np.pi,-np.pi/2] for h in [10,25]])
    best=None
    for p0 in inits:
        p0=[min(max(v,lo),hi) for v,lo,hi in zip(p0,lb,ub)]
        try:
            r=least_squares(lambda p: template(p,X)-y,p0,loss="soft_l1",bounds=(lb,ub),max_nfev=300)
            if best is None or r.cost<best.cost: best=r
        except Exception: pass
    return best.x

def learn_khat(X,r):
    k=ConstantKernel(1.0,(1e-3,1e3))*Matern(1.0,(1e-2,1e2),nu=1.5)+WhiteKernel(1e-2,(1e-6,1e1))
    rng=np.random.RandomState(0); idx=rng.choice(len(r),min(700,len(r)),replace=False)
    g=GaussianProcessRegressor(kernel=k,normalize_y=True,n_restarts_optimizer=1,random_state=0)
    g.fit(X[idx],r[idx]); return g.kernel_

def condition(X,r,khat):
    g=GaussianProcessRegressor(kernel=khat,normalize_y=True,optimizer=None); g.fit(X,r); return g

def met(y,p):
    sr=np.sum((y-p)**2); st=np.sum((y-np.mean(y))**2)
    return np.array([np.mean(np.abs(y-p)), np.sqrt(np.mean((y-p)**2)), 1-sr/st if st>0 else 0.0])

def groups(X,nb=4):
    gx=np.digitize(X[:,0],np.linspace(X[:,0].min(),X[:,0].max(),nb+1)[1:-1])
    gy=np.digitize(X[:,1],np.linspace(X[:,1].min(),X[:,1].max(),nb+1)[1:-1])
    return gx*nb+gy

i=int(sys.argv[1])
X,y=load(i)
px_full=fit_template(X,y)
# (a) paper protocol: khat from Protocol-A training set (80% random, seed 42)
itr,_=train_test_split(np.arange(len(y)),test_size=0.2,random_state=42)
pA=fit_template(X[itr],y[itr],init_px=px_full)
khat_A=learn_khat(X[itr],y[itr]-template(pA,X[itr]))
g4=groups(X)
res={"reuse":[],"refit":[]}
for tr,te in GroupKFold(n_splits=min(5,len(np.unique(g4)))).split(X,y,groups=g4):
    Xt,yt,Xv,yv=X[tr],y[tr],X[te],y[te]
    pf=fit_template(Xt,yt,init_px=px_full)
    resid=yt-template(pf,Xt)
    # (a) reuse khat_A
    g=condition(Xt,resid,khat_A)
    res["reuse"].append(met(yv,template(pf,Xv)+g.predict(Xv)))
    # (b) refit khat within fold
    khat_f=learn_khat(Xt,resid)
    g=condition(Xt,resid,khat_f)
    res["refit"].append(met(yv,template(pf,Xv)+g.predict(Xv)))
for k in res:
    a=np.array(res[k]); print(f"run{i} {k}: MAE={a[:,0].mean():.4f} RMSE={a[:,1].mean():.4f} R2={a[:,2].mean():+.4f}")
Path(f"/tmp/leak_run{i}.json").write_text(json.dumps({k:np.array(v).tolist() for k,v in res.items()}))
