# -*- coding: utf-8 -*-
# 终版实验: 多起点aniso模板GP + 边界F1 + 校准 + MLP集成, 两协议
import json, sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import least_squares
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
DATA=Path(__file__).resolve().parents[1] / "data"
PART=Path(__file__).resolve().parents[1] / "results"
THR=np.log10(11.0)  # 边界阈值: 10 uSv/h 高于本底一个量级 -> log10(10+1)

def load(i):
    df=pd.read_csv(DATA/str(i)/"data.txt",sep=r"\s+",comment="#",header=None,
        names=["t","px","py","dose","d01","date","time","u"])
    df=df[(df.px.abs()>1e-20)|(df.py.abs()>1e-20)]
    return df[["px","py"]].to_numpy(float), np.log10(df["dose"].to_numpy(float)+1.0)

def met(y,p):
    sr=np.sum((y-p)**2); st=np.sum((y-np.mean(y))**2)
    return [float(np.mean(np.abs(y-p))), float(np.sqrt(np.mean((y-p)**2))), float(1-sr/st) if st>0 else 0.0]

def f1(y,p,thr=THR):
    a=y>thr; b=p>thr
    tp=np.sum(a&b); fp=np.sum(~a&b); fn=np.sum(a&~b)
    if tp+fp==0 or tp+fn==0: return float("nan")
    pr=tp/(tp+fp); rc=tp/(tp+fn)
    return float(2*pr*rc/(pr+rc)) if pr+rc>0 else 0.0

def template(p,Xq):
    sx,sy,phi,half,logw,logC,logleak,logbg=p
    dx=Xq[:,0]-sx; dy=Xq[:,1]-sy; r2=dx*dx+dy*dy+0.09
    th=np.abs(np.arctan2(dy,dx)-phi); th=np.minimum(th,2*np.pi-th)
    f=10**logleak+(1-10**logleak)/(1+np.exp((th-half)/max(10**logw,1e-3)))
    return np.log10(10**logC*f/r2+10**logbg+1.0)

def fit_template(X,y,init_px=None):
    i0=np.argmax(y); sx0,sy0=X[i0]
    hot=y>np.percentile(y,85)
    phi0=np.arctan2(np.mean(X[hot,1]-sy0),np.mean(X[hot,0]-sx0)) if hot.sum()>3 else 0.0
    best=None
    inits=[]
    if init_px is not None:
        inits=[list(init_px), list(init_px[:2])+[init_px[2]+np.pi]+list(init_px[3:])]
    else:
        inits=[[sx0,sy0,phi0+d,np.deg2rad(h),-1.0,np.log10(max(10**y.max()-1,1)*0.09+1),-3.0,0.0]
               for d in [0,np.pi/2,np.pi,-np.pi/2] for h in [10,25]]
    for p0 in inits:
        if True:
            try:
                r=least_squares(lambda p: template(p,X)-y,p0,loss="soft_l1",
                  bounds=([X[:,0].min()-2,X[:,1].min()-2,-3*np.pi,np.deg2rad(3),-2.5,-2,-6,-2],
                          [X[:,0].max()+2,X[:,1].max()+2, 3*np.pi,np.deg2rad(80),0.5,12,0,2]),max_nfev=500)
                if best is None or r.cost<best.cost: best=r
            except Exception: pass
    return best.x

def gp_resid(Xtr,r,khat=None):
    if khat is None:
        k=ConstantKernel(1.0,(1e-3,1e3))*Matern(1.0,(1e-2,1e2),nu=1.5)+WhiteKernel(1e-2,(1e-6,1e1))
        rng=np.random.RandomState(0); idx=rng.choice(len(r),min(700,len(r)),replace=False)
        g0=GaussianProcessRegressor(kernel=k,normalize_y=True,n_restarts_optimizer=1,random_state=0)
        g0.fit(Xtr[idx],r[idx]); khat=g0.kernel_
    g=GaussianProcessRegressor(kernel=khat,normalize_y=True,optimizer=None); g.fit(Xtr,r)
    return g,khat

def mlp_ens(Xtr,ytr,Xte,seeds=(0,1,42)):
    sc=StandardScaler(); Xs=sc.fit_transform(Xtr); Xe=sc.transform(Xte)
    ps=[]
    for s in seeds:
        m=MLPRegressor(hidden_layer_sizes=(64,32,16),activation='tanh',solver='lbfgs',
                       alpha=0.01,max_iter=300,random_state=s)
        m.fit(Xs,ytr); ps.append(m.predict(Xe))
    return np.mean(ps,axis=0)

def groups(X,nb=4):
    gx=np.digitize(X[:,0],np.linspace(X[:,0].min(),X[:,0].max(),nb+1)[1:-1])
    gy=np.digitize(X[:,1],np.linspace(X[:,1].min(),X[:,1].max(),nb+1)[1:-1])
    return gx*nb+gy

for a in sys.argv[1:]:
    proto="AB"
    if ":" in a: a,proto=a.split(":")
    i=int(a); X,y=load(i)
    fp=PART/f"ef_{i}.json"
    out=json.loads(fp.read_text()) if fp.exists() else {}
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
    px=np.array(out["px"]) if "px" in out else fit_template(Xtr,ytr)
    g,khat=gp_resid(Xtr,ytr-template(px,Xtr))
    out["px"]=list(map(float,px))
    if "A" in proto:
        mu,sd=g.predict(Xte,return_std=True); predA=template(px,Xte)+mu
        out["A"]={"anisogp":met(yte,predA),"anisogp_f1":f1(yte,predA)}
        pe=mlp_ens(Xtr,ytr,Xte); out["A"]["mlpens"]=met(yte,pe); out["A"]["mlpens_f1"]=f1(yte,pe)
        out["params"]={"src":[round(float(px[0]),2),round(float(px[1]),2)],"phi_deg":round(float(np.rad2deg(px[2]))%360,1),
                       "half_deg":round(float(np.rad2deg(px[3])),1)}
    if ("B" not in proto) and ("E" not in proto):
        fp.write_text(json.dumps(out,ensure_ascii=False)); print(i,proto,"done",flush=True); continue
    g4=groups(X); splits=list(GroupKFold(n_splits=min(5,len(np.unique(g4)))).split(X,y,groups=g4))
    if "B" in proto:
        foldM=[]; foldF=[]; zs=[]; sds=[]; errs=[]
        for tr,te in splits:
            pxf=fit_template(X[tr],y[tr],init_px=px); gf,_=gp_resid(X[tr],y[tr]-template(pxf,X[tr]),khat=khat)
            mu,sd=gf.predict(X[te],return_std=True); pb=template(pxf,X[te])+mu
            foldM.append(met(y[te],pb)); foldF.append(f1(y[te],pb))
            err=y[te]-pb; zs.extend((err/np.maximum(sd,1e-6)).tolist())
            sds.extend(sd.tolist()); errs.extend(np.abs(err).tolist())
        z=np.array(zs); rho=spearmanr(np.array(sds),np.array(errs)).statistic
        out.setdefault("B",{}); out["B"]["anisogp"]={"mean":np.array(foldM).mean(0).tolist()}
        out["B"]["anisogp_f1"]=float(np.nanmean(foldF))
        out["B"]["calib"]={"cov1":float(np.mean(np.abs(z)<1)),"cov2":float(np.mean(np.abs(z)<2)),
                           "spearman_sd_err":float(rho)}
    if "E" in proto:
        foldE=[]; foldEF=[]
        for tr,te in splits:
            pef=mlp_ens(X[tr],y[tr],X[te]); foldE.append(met(y[te],pef)); foldEF.append(f1(y[te],pef))
        out.setdefault("B",{}); out["B"]["mlpens"]={"mean":np.array(foldE).mean(0).tolist()}
        out["B"]["mlpens_f1"]=float(np.nanmean(foldEF))
    fp.write_text(json.dumps(out,ensure_ascii=False))
    print(i,proto,"done",flush=True)
