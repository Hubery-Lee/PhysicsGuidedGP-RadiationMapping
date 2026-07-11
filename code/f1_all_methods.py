# -*- coding: utf-8 -*-
# P1-4: boundary F1 for linear / plain GP / IDW / RF / single MLP, protocols A+B.
# Configs mirror r1_run.py / quick_methods.py exactly; F1 aggregation mirrors exp_final.py.
import json, sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore")
DATA = Path(__file__).resolve().parents[1] / "data"
THR = np.log10(11.0)

def load(i):
    df = pd.read_csv(DATA/str(i)/"data.txt", sep=r"\s+", comment="#", header=None,
                     names=["t","px","py","dose","d01","date","time","u"])
    df = df[(df.px.abs()>1e-20)|(df.py.abs()>1e-20)]
    return df[["px","py"]].to_numpy(float), np.log10(df["dose"].to_numpy(float)+1.0)

def f1(y,p,thr=THR):
    a=y>thr; b=p>thr
    tp=np.sum(a&b); fp=np.sum(~a&b); fn=np.sum(a&~b)
    return float(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn)>0 else float("nan")

def lin(Xtr,ytr,Xte):
    p=griddata(Xtr,ytr,Xte,method="linear"); nan=np.isnan(p)
    if nan.any(): p[nan]=griddata(Xtr,ytr,Xte[nan],method="nearest")
    return p

def gp_learn(Xtr,ytr,seed=0):
    k=ConstantKernel(1.0,(1e-3,1e3))*Matern(1.0,(1e-2,1e2),nu=1.5)+ConstantKernel(0.1,(1e-4,1e2))+WhiteKernel(1e-2,(1e-6,1e1))
    rng=np.random.RandomState(seed); idx=rng.choice(len(ytr),min(800,len(ytr)),replace=False)
    g0=GaussianProcessRegressor(kernel=k,normalize_y=True,n_restarts_optimizer=1,random_state=seed)
    g0.fit(Xtr[idx],ytr[idx]); return g0.kernel_

def gp_pred(Xtr,ytr,Xte,khat):
    g=GaussianProcessRegressor(kernel=khat,normalize_y=True,optimizer=None); g.fit(Xtr,ytr); return g.predict(Xte)

def idw(Xtr,ytr,Xte,k=8,p=2):
    nn=NearestNeighbors(n_neighbors=min(k,len(ytr))).fit(Xtr)
    d,ix=nn.kneighbors(Xte); d=np.maximum(d,1e-6); w=1/d**p
    return np.sum(w*ytr[ix],axis=1)/np.sum(w,axis=1)

def rf(Xtr,ytr,Xte,seed=0):
    m=RandomForestRegressor(200,random_state=seed,n_jobs=2); m.fit(Xtr,ytr); return m.predict(Xte)

def mlp(Xtr,ytr,Xte,seed=42):
    sc=StandardScaler(); Xs=sc.fit_transform(Xtr); Xq=sc.transform(Xte)
    m=MLPRegressor(hidden_layer_sizes=(64,32,16),activation='tanh',solver='lbfgs',
                   alpha=0.01,max_iter=500,random_state=seed)
    m.fit(Xs,ytr); return m.predict(Xq)

def groups(X,nb=4):
    gx=np.digitize(X[:,0],np.linspace(X[:,0].min(),X[:,0].max(),nb+1)[1:-1])
    gy=np.digitize(X[:,1],np.linspace(X[:,1].min(),X[:,1].max(),nb+1)[1:-1])
    return gx*nb+gy

i=int(sys.argv[1])
X,y=load(i)
itr,ite=train_test_split(np.arange(len(y)),test_size=0.2,random_state=42)
Xtr,ytr,Xte,yte=X[itr],y[itr],X[ite],y[ite]
khat=gp_learn(Xtr,ytr)
METH={"linear":lin,"gp":lambda a,b,c: gp_pred(a,b,c,khat),"idw":idw,"rf":rf,"mlp":mlp}
out={"A":{},"B":{}}
for m,fn in METH.items():
    out["A"][m]=f1(yte,fn(Xtr,ytr,Xte))
g4=groups(X)
splits=list(GroupKFold(n_splits=min(5,len(np.unique(g4)))).split(X,y,groups=g4))
for m,fn in METH.items():
    fs=[f1(y[te],fn(X[tr],y[tr],X[te])) for tr,te in splits]
    out["B"][m]=float(np.nanmean(fs))
print(i, json.dumps(out))
Path(f"/tmp/f1_run{i}.json").write_text(json.dumps(out))
