# -*- coding: utf-8 -*-
# R4: 准直Cs-137源模拟, 计数直接泊松抽样(替代原10%-30%相对噪声)
# 5方法对比, 评价基准=理论真值场(60x60网格), 5个随机种子
import json, warnings
import numpy as np
from scipy.interpolate import griddata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

A, EFF = 7.4e8, 0.5
DIRN = np.array([10.,10.])/np.hypot(10,10); HALF = np.deg2rad(12.)  # 准直半角12°(证书)
LEAK = 1e-4

def mu(P, t=1.0):
    r = np.clip(np.hypot(P[:,0],P[:,1]), 0.3, None)
    u = P/ r[:,None]
    inside = np.arccos(np.clip(u@DIRN, -1, 1)) <= HALF
    rate = A*EFF/(4*np.pi*r**2)*np.where(inside, 1.0, LEAK)
    return rate*t

def truth_log(P): return np.log10(mu(P)+1.0)

def met(y,p):
    mae=float(np.mean(np.abs(y-p))); rmse=float(np.sqrt(np.mean((y-p)**2)))
    sr=np.sum((y-p)**2); st=np.sum((y-np.mean(y))**2)
    return [mae, rmse, float(1-sr/st)]

g = np.linspace(0,10,60); GX,GY = np.meshgrid(g,g)
G = np.column_stack([GX.ravel(),GY.ravel()]); YG = truth_log(G)

def gp_fit(X,y,Xte,kernel):
    m=GaussianProcessRegressor(kernel=kernel,normalize_y=True,n_restarts_optimizer=1,random_state=0)
    m.fit(X,y); return m.predict(Xte), str(m.kernel_)

res={m:[] for m in ["linear","gp_rbf","gp_matern_bias","kriging","mlp"]}; kern_log={}
for seed in range(5):
    rng=np.random.RandomState(seed)
    X = rng.uniform(0,10,(200,2))
    y = np.log10(rng.poisson(mu(X))+1.0)
    p = griddata(X,y,G,method="linear"); nan=np.isnan(p)
    p[nan]=griddata(X,y,G[nan],method="nearest")
    res["linear"].append(met(YG,p))
    p,k1 = gp_fit(X,y,G, ConstantKernel()*RBF(1.0,(1e-2,1e2))+WhiteKernel(1e-2,(1e-6,1e1)))
    res["gp_rbf"].append(met(YG,p))
    p,k2 = gp_fit(X,y,G, ConstantKernel()*Matern(1.0,(1e-2,1e2),nu=1.5)+ConstantKernel(0.1,(1e-4,1e2))+WhiteKernel(1e-2,(1e-6,1e1)))
    res["gp_matern_bias"].append(met(YG,p))
    p,k3 = gp_fit(X,y,G, ConstantKernel()*Matern(1.0,(1e-2,1e2),nu=1.5)+WhiteKernel(1e-2,(1e-6,1e1)))
    res["kriging"].append(met(YG,p))
    sc=StandardScaler(); Xs=sc.fit_transform(X); Gs=sc.transform(G)
    m=MLPRegressor(hidden_layer_sizes=(64,32,16),activation='tanh',solver='lbfgs',alpha=0.01,max_iter=500,random_state=42)
    m.fit(Xs,y); res["mlp"].append(met(YG,m.predict(Gs)))
    if seed==0: kern_log={"rbf":k1,"matern_bias":k2,"kriging":k3}
    print("seed",seed,"done",flush=True)

out={"metrics":{m:{"mean":np.array(v).mean(0).tolist(),"std":np.array(v).std(0).tolist(),"runs":v} for m,v in res.items()},
     "kernels_seed0":kern_log,
     "poisson_stats":{"note":"counts~Poisson(mu*1s); 准直内r=1m时mu~2.9e7 (rel err 0.02%), 准直外r=14m时mu~15 (rel err ~26%)"}}
open("r4_results.json","w").write(json.dumps(out,ensure_ascii=False,indent=1))
for m,v in res.items():
    a=np.array(v); print(f"{m:15s} MAE {a[:,0].mean():.4f}±{a[:,0].std():.4f} RMSE {a[:,1].mean():.4f}±{a[:,1].std():.4f} R2 {a[:,2].mean():.4f}±{a[:,2].std():.4f}")
