# -*- coding: utf-8 -*-
# 在 r4 仿真协议(准直源,200均匀采样,泊松计数,5种子,60x60真值网格)上跑
# 多核加权GP + Poisson kriging, 口径与 r4_poisson_sim.py / new_baselines.py 一致.
import json, numpy as np, warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel, WhiteKernel
warnings.filterwarnings("ignore")
A,EFF=7.4e8,0.5; DIRN=np.array([10.,10.])/np.hypot(10,10); HALF=np.deg2rad(12.); LEAK=1e-4
def mu(P):
    r=np.clip(np.hypot(P[:,0],P[:,1]),0.3,None); u=P/r[:,None]
    ins=np.arccos(np.clip(u@DIRN,-1,1))<=HALF
    return A*EFF/(4*np.pi*r**2)*np.where(ins,1.0,LEAK)
def met(y,p):
    sr=np.sum((y-p)**2); st=np.sum((y-np.mean(y))**2)
    return [float(np.mean(np.abs(y-p))),float(np.sqrt(np.mean((y-p)**2))),float(1-sr/st)]
g=np.linspace(0,10,60); GX,GY=np.meshgrid(g,g); G=np.column_stack([GX.ravel(),GY.ravel()]); YG=np.log10(mu(G)+1.0)

def mkgp(X,y,G):
    k=(ConstantKernel(1.0,(1e-3,1e3))*Matern(0.3,(1e-2,1e1),nu=1.5)
       +ConstantKernel(1.0,(1e-3,1e3))*Matern(1.5,(1e-1,1e2),nu=2.5)
       +ConstantKernel(0.5,(1e-4,1e2))*RBF(3.0,(3e-1,3e2))
       +ConstantKernel(0.1,(1e-4,1e2))+WhiteKernel(1e-2,(1e-6,1e1)))
    m=GaussianProcessRegressor(kernel=k,normalize_y=True,n_restarts_optimizer=1,random_state=0).fit(X,y)
    return m.predict(G)
def pk_alpha(c,base=5e-2,lo=1/3,hi=3.0):
    inv=1.0/(c+1.0); rel=np.clip(inv/np.median(inv),lo,hi); return base*rel
def poissonk(X,y,c,G):
    k=ConstantKernel(1.0,(1e-3,1e3))*Matern(1.0,(1e-2,1e2),nu=1.5)+ConstantKernel(0.1,(1e-4,1e2))
    m=GaussianProcessRegressor(kernel=k,alpha=pk_alpha(c),normalize_y=True,n_restarts_optimizer=1,random_state=0).fit(X,y)
    return m.predict(G)

res={"mkgp":[],"poissonk":[]}
for seed in range(5):
    rng=np.random.RandomState(seed); X=rng.uniform(0,10,(200,2)); c=rng.poisson(mu(X)).astype(float); y=np.log10(c+1.0)
    res["mkgp"].append(met(YG,mkgp(X,y,G)))
    res["poissonk"].append(met(YG,poissonk(X,y,c,G)))
open("sim_newbaselines.json","w").write(json.dumps(res))
for m,v in res.items():
    a=np.array(v); print("%-10s MAE %.3f±%.3f RMSE %.3f±%.3f R2 %.3f±%.3f"%(m,a[:,0].mean(),a[:,0].std(),a[:,1].mean(),a[:,1].std(),a[:,2].mean(),a[:,2].std()),flush=True)
