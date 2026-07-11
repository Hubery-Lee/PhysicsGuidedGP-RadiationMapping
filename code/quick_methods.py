# -*- coding: utf-8 -*-
# 候选方法实测: physGP(逆平方均值+Matern残差GP) / RF / IDW + MLP种子稳定性
import json, sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import least_squares
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore")
DATA=Path(__file__).resolve().parents[1] / "data"
PART=Path(__file__).resolve().parents[1] / "results"

def load(i):
    df=pd.read_csv(DATA/str(i)/"data.txt",sep=r"\s+",comment="#",header=None,
        names=["t","px","py","dose","d01","date","time","u"])
    df=df[(df.px.abs()>1e-20)|(df.py.abs()>1e-20)]
    return df[["px","py"]].to_numpy(float), np.log10(df["dose"].to_numpy(float)+1.0)

def met(y,p):
    mae=float(np.mean(np.abs(y-p))); rmse=float(np.sqrt(np.mean((y-p)**2)))
    sr=np.sum((y-p)**2); st=np.sum((y-np.mean(y))**2)
    return [mae,rmse,float(1-sr/st) if st>0 else 0.0]

# --- 物理趋势: log10(C/((r2+eps)) + bg + 1) ---
def phys_fit(X,y):
    def model(p,Xq):
        sx,sy,logC,logbg=p
        r2=(Xq[:,0]-sx)**2+(Xq[:,1]-sy)**2+0.09
        return np.log10(10**logC/r2+10**logbg+1.0)
    i0=np.argmax(y); p0=[X[i0,0],X[i0,1],np.log10(max(10**y.max()-1,1)*0.09+1),0.0]
    try:
        res=least_squares(lambda p: model(p,X)-y, p0, max_nfev=2000)
        return lambda Xq: model(res.x,Xq), res.x
    except Exception:
        return lambda Xq: np.full(len(Xq),y.mean()), p0

def phys_gp(Xtr,ytr,Xte,seed=0,khat=None):
    f,px=phys_fit(Xtr,ytr)
    r=ytr-f(Xtr)
    if khat is None:
        k=ConstantKernel(1.0,(1e-3,1e3))*Matern(1.0,(1e-2,1e2),nu=1.5)+WhiteKernel(1e-2,(1e-6,1e1))
        rng=np.random.RandomState(seed); idx=rng.choice(len(r),min(800,len(r)),replace=False)
        g0=GaussianProcessRegressor(kernel=k,normalize_y=True,n_restarts_optimizer=1,random_state=seed)
        g0.fit(Xtr[idx],r[idx]); khat=g0.kernel_
    g=GaussianProcessRegressor(kernel=khat,normalize_y=True,optimizer=None); g.fit(Xtr,r)
    return f(Xte)+g.predict(Xte), khat

def idw(Xtr,ytr,Xte,k=8,p=2):
    nn=NearestNeighbors(n_neighbors=min(k,len(ytr))).fit(Xtr)
    d,ix=nn.kneighbors(Xte); d=np.maximum(d,1e-6); w=1/d**p
    return np.sum(w*ytr[ix],axis=1)/np.sum(w,axis=1)

def rf(Xtr,ytr,Xte,seed=0):
    m=RandomForestRegressor(200,random_state=seed,n_jobs=2); m.fit(Xtr,ytr); return m.predict(Xte)

def mlp(Xtr,ytr,Xte,seed=42):
    sc=StandardScaler(); Xs=sc.fit_transform(Xtr); Xe=sc.transform(Xte)
    m=MLPRegressor(hidden_layer_sizes=(64,32,16),activation='tanh',solver='lbfgs',alpha=0.01,max_iter=500,random_state=seed)
    m.fit(Xs,ytr); return m.predict(Xe)

def groups(X,nb=4):
    gx=np.digitize(X[:,0],np.linspace(X[:,0].min(),X[:,0].max(),nb+1)[1:-1])
    gy=np.digitize(X[:,1],np.linspace(X[:,1].min(),X[:,1].max(),nb+1)[1:-1])
    return gx*nb+gy

for a in sys.argv[1:]:
    i=int(a); X,y=load(i); out={}
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
    pA,khat=phys_gp(Xtr,ytr,Xte)
    out["A"]={"physgp":met(yte,pA),"rf":met(yte,rf(Xtr,ytr,Xte)),"idw":met(yte,idw(Xtr,ytr,Xte))}
    out["phys_kernel"]=str(khat)
    # MLP种子稳定性 (协议A, 6种子)
    seeds=[met(yte,mlp(Xtr,ytr,Xte,seed=s)) for s in [0,1,2,7,42,123]]
    sa=np.array(seeds); out["mlp_seed_A"]={"R2_per_seed":sa[:,2].tolist(),"R2_mean":float(sa[:,2].mean()),"R2_std":float(sa[:,2].std())}
    # 协议B
    g=groups(X); fold={m:[] for m in ["physgp","rf","idw"]}
    for tr,te in GroupKFold(n_splits=min(5,len(np.unique(g)))).split(X,y,groups=g):
        p,_=phys_gp(X[tr],y[tr],X[te],khat=khat)
        fold["physgp"].append(met(y[te],p))
        fold["rf"].append(met(y[te],rf(X[tr],y[tr],X[te])))
        fold["idw"].append(met(y[te],idw(X[tr],y[tr],X[te])))
    out["B"]={m:{"mean":np.array(v).mean(0).tolist()} for m,v in fold.items()}
    (PART/f"qm_{i}.json").write_text(json.dumps(out,ensure_ascii=False))
    print(i,"done",flush=True)
