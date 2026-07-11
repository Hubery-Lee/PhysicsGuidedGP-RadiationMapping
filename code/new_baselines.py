# -*- coding: utf-8 -*-
# 新增审稿基线: 多核加权GP (MKGP, Zhang2025风格) + 泊松克里金 (PoissonK)
# 完全对齐 r1_run.py / quick_methods.py 的数据加载、协议A/B划分与指标口径。
import json, sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel, RBF
warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parents[1] / "results"
THR = np.log10(11.0)  # 边界阈值: 高于本底一个量级 (10 uSv/h)

def load(i):
    df = pd.read_csv(DATA/str(i)/"data.txt", sep=r"\s+", comment="#", header=None,
        names=["t","px","py","dose","d01","date","time","u"])
    df = df[(df.px.abs()>1e-20)|(df.py.abs()>1e-20)]
    X = df[["px","py"]].to_numpy(float)
    dose = df["dose"].to_numpy(float)
    y = np.log10(dose+1.0)
    return X, y, dose

def met(y,p):
    sr=np.sum((y-p)**2); st=np.sum((y-np.mean(y))**2)
    return [float(np.mean(np.abs(y-p))), float(np.sqrt(np.mean((y-p)**2))),
            float(1-sr/st) if st>0 else 0.0]

def f1(y,p,thr=THR):
    a=y>thr; b=p>thr
    tp=np.sum(a&b); fp=np.sum(~a&b); fn=np.sum(a&~b)
    if tp+fp==0 or tp+fn==0: return float("nan")
    pr=tp/(tp+fp); rc=tp/(tp+fn)
    return float(2*pr*rc/(pr+rc)) if pr+rc>0 else 0.0

def groups(X,nb=4):
    gx=np.digitize(X[:,0],np.linspace(X[:,0].min(),X[:,0].max(),nb+1)[1:-1])
    gy=np.digitize(X[:,1],np.linspace(X[:,1].min(),X[:,1].max(),nb+1)[1:-1])
    return gx*nb+gy

# ---------- 多核加权GP (Zhang 2025 风格) ----------
# 多个不同长度尺度的核加权和 + 白噪声; ConstantKernel 因子即"学习到的权重",
# 由边际似然优化。为速度/确定性: 800点子样拟合超参 -> 固定 -> 全量条件化。
def mkgp_kernel():
    return (ConstantKernel(1.0,(1e-3,1e3))*Matern(0.3,(1e-2,1e1),nu=1.5)
            + ConstantKernel(1.0,(1e-3,1e3))*Matern(1.5,(1e-1,1e2),nu=2.5)
            + ConstantKernel(0.5,(1e-4,1e2))*RBF(3.0,(3e-1,3e2))
            + ConstantKernel(0.1,(1e-4,1e2))
            + WhiteKernel(1e-2,(1e-6,1e1)))

def mkgp_fit(Xtr,ytr,seed=0):
    rng=np.random.RandomState(seed); idx=rng.choice(len(ytr),min(800,len(ytr)),replace=False)
    g0=GaussianProcessRegressor(kernel=mkgp_kernel(),normalize_y=True,
                                n_restarts_optimizer=1,random_state=seed)
    g0.fit(Xtr[idx],ytr[idx])
    return g0.kernel_

def mkgp_predict(Xtr,ytr,Xte,khat):
    g=GaussianProcessRegressor(kernel=khat,normalize_y=True,optimizer=None)
    g.fit(Xtr,ytr); return g.predict(Xte)

# ---------- 泊松克里金 (point-support) ----------
# 常均值 Matern3/2 kriging, 每点异方差噪声来自泊松计数统计:
# 计数 N∝dose, log10 域 Var(log10 N) ≈ (1/ln10)^2 / N ∝ 1/dose (delta 法)。
# 实验探测器输出剂量率而非原始计数, 故用相对权重: alpha_i ∝ 1/(dose_i+1),
# 归一化到与 plain GP 相当的名义 nugget 量级, 使高剂量(高计数)点权重更大。
def pk_alpha(dose, base=5e-2, lo=1.0/3, hi=3.0):
    # 每点相对方差 ∝ 1/计数 ∝ 1/dose; 按中位数归一后裁剪到 [1/3,3] 倍,
    # 防止高计数点 nugget 趋零导致过拟合; base 校准到 plain GP 噪声量级。
    inv = 1.0/(dose+1.0)
    rel = np.clip(inv/np.median(inv), lo, hi)
    return base * rel

def pk_fit(Xtr,ytr,dtr,seed=0):
    k=ConstantKernel(1.0,(1e-3,1e3))*Matern(1.0,(1e-2,1e2),nu=1.5)+ConstantKernel(0.1,(1e-4,1e2))
    rng=np.random.RandomState(seed); idx=rng.choice(len(ytr),min(800,len(ytr)),replace=False)
    g0=GaussianProcessRegressor(kernel=k,alpha=pk_alpha(dtr)[idx],normalize_y=True,
                                n_restarts_optimizer=1,random_state=seed)
    g0.fit(Xtr[idx],ytr[idx])
    return g0.kernel_

def pk_predict(Xtr,ytr,dtr,Xte,khat):
    g=GaussianProcessRegressor(kernel=khat,alpha=pk_alpha(dtr),normalize_y=True,optimizer=None)
    g.fit(Xtr,ytr); return g.predict(Xte)

def run(i):
    X,y,dose = load(i)
    out={"n":int(len(y))}
    # ---- 协议A: 随机 80/20, random_state=42 ----
    Xtr,Xte,ytr,yte,dtr,dte = train_test_split(X,y,dose,test_size=0.2,random_state=42)
    kh_mk = mkgp_fit(Xtr,ytr); kh_pk = pk_fit(Xtr,ytr,dtr)
    pA_mk = mkgp_predict(Xtr,ytr,Xte,kh_mk)
    pA_pk = pk_predict(Xtr,ytr,dtr,Xte,kh_pk)
    out["A"]={"mkgp":met(yte,pA_mk),"mkgp_f1":f1(yte,pA_mk),
              "poissonk":met(yte,pA_pk),"poissonk_f1":f1(yte,pA_pk)}
    out["mk_kernel"]=str(kh_mk); out["pk_kernel"]=str(kh_pk)
    # ---- 协议B: 4x4 分块 GroupKFold (核超参固定复用, 与 plain GP 口径一致) ----
    g=groups(X); splits=list(GroupKFold(n_splits=min(5,len(np.unique(g)))).split(X,y,groups=g))
    fold={"mkgp":[],"poissonk":[]}; foldF={"mkgp":[],"poissonk":[]}
    for tr,te in splits:
        pmk=mkgp_predict(X[tr],y[tr],X[te],kh_mk)
        ppk=pk_predict(X[tr],y[tr],dose[tr],X[te],kh_pk)
        fold["mkgp"].append(met(y[te],pmk)); foldF["mkgp"].append(f1(y[te],pmk))
        fold["poissonk"].append(met(y[te],ppk)); foldF["poissonk"].append(f1(y[te],ppk))
    out["B"]={m:{"mean":np.array(v).mean(0).tolist(),"folds":np.array(v).tolist()} for m,v in fold.items()}
    for m in foldF: out["B"][m]["f1"]=float(np.nanmean(foldF[m]))
    (OUT/f"nb_{i}.json").write_text(json.dumps(out,ensure_ascii=False))
    print("run %d done n=%d | A mkgp R2=%.3f pk R2=%.3f | B mkgp MAE=%.3f pk MAE=%.3f"
          % (i,out["n"],out["A"]["mkgp"][2],out["A"]["poissonk"][2],
             out["B"]["mkgp"]["mean"][0],out["B"]["poissonk"]["mean"][0]),flush=True)
    return out

if __name__=="__main__":
    args=sys.argv[1:] or [str(i) for i in range(1,8)]
    for a in args: run(int(a))
