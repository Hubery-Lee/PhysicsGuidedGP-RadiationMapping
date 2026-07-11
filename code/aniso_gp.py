# -*- coding: utf-8 -*-
# 各向异性准直模板GP原型: mean = log10( C/(r^2+eps)*f(theta) + bg + 1 ), f=扇形软过渡
import json, sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import least_squares
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
warnings.filterwarnings("ignore")
DATA=Path(__file__).resolve().parents[1] / "data"
PART=Path(__file__).resolve().parents[1] / "results"

def load(i):
    df=pd.read_csv(DATA/str(i)/"data.txt",sep=r"\s+",comment="#",header=None,
        names=["t","px","py","dose","d01","date","time","u"])
    df=df[(df.px.abs()>1e-20)|(df.py.abs()>1e-20)]
    return df[["px","py"]].to_numpy(float), np.log10(df["dose"].to_numpy(float)+1.0)

def met(y,p):
    sr=np.sum((y-p)**2); st=np.sum((y-np.mean(y))**2)
    return [float(np.mean(np.abs(y-p))), float(np.sqrt(np.mean((y-p)**2))), float(1-sr/st) if st>0 else 0.0]

def template(p,Xq):
    sx,sy,phi,half,logw,logC,logleak,logbg=p
    dx=Xq[:,0]-sx; dy=Xq[:,1]-sy
    r2=dx*dx+dy*dy+0.09
    theta=np.abs(np.arctan2(dy,dx)-phi)
    theta=np.minimum(theta,2*np.pi-theta)
    w=10**logw
    leak=10**logleak
    f=leak+(1-leak)/(1+np.exp((theta-half)/max(w,1e-3)))
    return np.log10(10**logC*f/r2+10**logbg+1.0)

def fit_template(X,y):
    i0=np.argmax(y); sx0,sy0=X[i0]
    hot=y>np.percentile(y,85)
    phi0=np.arctan2(np.mean(X[hot,1]-sy0),np.mean(X[hot,0]-sx0)) if hot.sum()>3 else 0.0
    best=None
    for half0 in [np.deg2rad(10),np.deg2rad(20),np.deg2rad(35)]:
        p0=[sx0,sy0,phi0,half0,-1.0,np.log10(max(10**y.max()-1,1)*0.09+1),-3.0,0.0]
        try:
            r=least_squares(lambda p: template(p,X)-y, p0, loss="soft_l1",
                bounds=([X[:,0].min()-2,X[:,1].min()-2,-2*np.pi,np.deg2rad(3),-2.5,-2,-6,-2],
                        [X[:,0].max()+2,X[:,1].max()+2, 2*np.pi,np.deg2rad(80), 0.5,12, 0, 2]),
                max_nfev=3000)
            if best is None or r.cost<best.cost: best=r
        except Exception: pass
    return best

def aniso_gp(Xtr,ytr,Xte,khat=None,px=None):
    if px is None:
        b=fit_template(Xtr,ytr); px=b.x
    r=ytr-template(px,Xtr)
    if khat is None:
        k=ConstantKernel(1.0,(1e-3,1e3))*Matern(1.0,(1e-2,1e2),nu=1.5)+WhiteKernel(1e-2,(1e-6,1e1))
        rng=np.random.RandomState(0); idx=rng.choice(len(r),min(800,len(r)),replace=False)
        g0=GaussianProcessRegressor(kernel=k,normalize_y=True,n_restarts_optimizer=1,random_state=0)
        g0.fit(Xtr[idx],r[idx]); khat=g0.kernel_
    g=GaussianProcessRegressor(kernel=khat,normalize_y=True,optimizer=None); g.fit(Xtr,r)
    return template(px,Xte)+g.predict(Xte), khat, px

def groups(X,nb=4):
    gx=np.digitize(X[:,0],np.linspace(X[:,0].min(),X[:,0].max(),nb+1)[1:-1])
    gy=np.digitize(X[:,1],np.linspace(X[:,1].min(),X[:,1].max(),nb+1)[1:-1])
    return gx*nb+gy

for a in sys.argv[1:]:
    i=int(a); X,y=load(i); out={}
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
    pA,khat,px=aniso_gp(Xtr,ytr,Xte)
    out["A"]={"anisogp":met(yte,pA),"template_only":met(yte,template(px,Xte))}
    out["params"]={"src":[round(px[0],2),round(px[1],2)],"phi_deg":round(np.rad2deg(px[2]),1),
                   "half_deg":round(np.rad2deg(px[3]),1),"kernel":str(khat)}
    g=groups(X); fold=[]; foldT=[]
    for tr,te in GroupKFold(n_splits=min(5,len(np.unique(g)))).split(X,y,groups=g):
        p,_,pxf=aniso_gp(X[tr],y[tr],X[te],khat=khat)
        fold.append(met(y[te],p)); foldT.append(met(y[te],template(pxf,X[te])))
    out["B"]={"anisogp":{"mean":np.array(fold).mean(0).tolist()},
              "template_only":{"mean":np.array(foldT).mean(0).tolist()}}
    (PART/f"ag_{i}.json").write_text(json.dumps(out,ensure_ascii=False))
    print(i,"done",out["params"]["src"],out["params"]["phi_deg"],out["params"]["half_deg"],flush=True)
