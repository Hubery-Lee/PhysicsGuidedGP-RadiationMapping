# -*- coding: utf-8 -*-
# 检验: 泊松模拟突变场中 MLP优势是否对"网络初始化种子"稳健
import warnings, json
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
A,EFF=7.4e8,0.5; DIRN=np.array([10.,10.])/np.hypot(10,10); HALF=np.deg2rad(12.); LEAK=1e-4
def mu(P):
    r=np.clip(np.hypot(P[:,0],P[:,1]),0.3,None); u=P/r[:,None]
    ins=np.arccos(np.clip(u@DIRN,-1,1))<=HALF
    return A*EFF/(4*np.pi*r**2)*np.where(ins,1.0,LEAK)
def met(y,p):
    sr=np.sum((y-p)**2); st=np.sum((y-np.mean(y))**2)
    return float(np.mean(np.abs(y-p))), float(1-sr/st)
g=np.linspace(0,10,60); GX,GY=np.meshgrid(g,g)
G=np.column_stack([GX.ravel(),GY.ravel()]); YG=np.log10(mu(G)+1.0)
res={"mlp":[], "gp":[]}
for ds in range(3):  # 数据种子
    rng=np.random.RandomState(ds)
    X=rng.uniform(0,10,(200,2)); y=np.log10(rng.poisson(mu(X))+1.0)
    sc=StandardScaler(); Xs=sc.fit_transform(X); Gs=sc.transform(G)
    for ms in [0,1,2,7,42,123]:  # 网络初始化种子
        m=MLPRegressor(hidden_layer_sizes=(64,32,16),activation='tanh',solver='lbfgs',
                       alpha=0.01,max_iter=500,random_state=ms)
        m.fit(Xs,y); mae,r2=met(YG,m.predict(Gs))
        res["mlp"].append({"data_seed":ds,"net_seed":ms,"MAE":mae,"R2":r2})
    k=ConstantKernel()*Matern(1.0,(1e-2,1e2),nu=1.5)+ConstantKernel(0.1,(1e-4,1e2))+WhiteKernel(1e-2,(1e-6,1e1))
    gp=GaussianProcessRegressor(kernel=k,normalize_y=True,n_restarts_optimizer=1,random_state=0)
    gp.fit(X,y); mae,r2=met(YG,gp.predict(G))
    res["gp"].append({"data_seed":ds,"MAE":mae,"R2":r2})
r2s=np.array([d["R2"] for d in res["mlp"]]); maes=np.array([d["MAE"] for d in res["mlp"]])
print("MLP 18次(3数据种子x6网络种子): R2 mean %.3f std %.3f min %.3f max %.3f"%(r2s.mean(),r2s.std(),r2s.min(),r2s.max()))
print("MLP MAE: mean %.3f std %.3f min %.3f max %.3f"%(maes.mean(),maes.std(),maes.min(),maes.max()))
for d in res["gp"]: print("GP data_seed %d: MAE %.3f R2 %.3f"%(d["data_seed"],d["MAE"],d["R2"]))
per=np.array([[d["R2"] for d in res["mlp"] if d["data_seed"]==s] for s in range(3)])
for s in range(3): print("数据种子%d 各网络种子R2:"%s, np.round(per[s],3).tolist())
open("sim_stability.json","w").write(json.dumps(res))
