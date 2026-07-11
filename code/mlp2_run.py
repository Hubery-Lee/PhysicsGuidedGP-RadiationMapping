# -*- coding: utf-8 -*-
# 用作者精确MLP配置(alpha=0.01, 仅X标准化, seed=42, (64,32,16), tanh, lbfgs, 500iter)
# 在与linear/GP相同的外层测试集/CV折上补算 -> part_i.json 增加 mlp2
import json, sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
DATA = Path(__file__).resolve().parents[1] / "data"
PART = Path(__file__).resolve().parents[1] / "results"

def load(i):
    df = pd.read_csv(DATA/str(i)/"data.txt", sep=r"\s+", comment="#", header=None,
        names=["t","px","py","dose","d01","date","time","u"])
    df = df[(df.px.abs()>1e-20)|(df.py.abs()>1e-20)]
    return df[["px","py"]].to_numpy(float), np.log10(df["dose"].to_numpy(float)+1.0)

def met(y,p):
    mae=float(np.mean(np.abs(y-p))); rmse=float(np.sqrt(np.mean((y-p)**2)))
    sr=np.sum((y-p)**2); st=np.sum((y-np.mean(y))**2)
    return [mae, rmse, float(1-sr/st) if st>0 else 0.0]

def mlp_author(Xtr,ytr,Xte):
    sc=StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
    m=MLPRegressor(hidden_layer_sizes=(64,32,16),activation='tanh',solver='lbfgs',
                   alpha=0.01,max_iter=500,random_state=42)
    m.fit(Xtr_s,ytr); return m.predict(Xte_s)

def groups(X,nb=4):
    gx=np.digitize(X[:,0],np.linspace(X[:,0].min(),X[:,0].max(),nb+1)[1:-1])
    gy=np.digitize(X[:,1],np.linspace(X[:,1].min(),X[:,1].max(),nb+1)[1:-1])
    return gx*nb+gy

for a in sys.argv[1:]:
    i=int(a); X,y=load(i)
    fp=PART/f"part_{i}.json"; out=json.loads(fp.read_text())
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
    out["A"]["mlp2"]=met(yte,mlp_author(Xtr,ytr,Xte))
    g=groups(X); folds=[]
    for tr,te in GroupKFold(n_splits=min(5,len(np.unique(g)))).split(X,y,groups=g):
        folds.append(met(y[te],mlp_author(X[tr],y[tr],X[te])))
    fa=np.array(folds)
    out["B"]["mlp2"]={"mean":fa.mean(0).tolist(),"std":fa.std(0).tolist(),"folds":fa.tolist()}
    fp.write_text(json.dumps(out,ensure_ascii=False))
    print(i,"done",flush=True)
