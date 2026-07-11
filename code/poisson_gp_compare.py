# -*- coding: utf-8 -*-
# 意见2: 泊松似然GP (Laplace近似, log链接) vs log-高斯GP, 纯仿真.
# 目的: (1) log10(count+1)是否已稳定泊松异方差; (2) 泊松似然能否改善单点σ区分.
import json, sys, warnings
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.stats import spearmanr, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
warnings.filterwarnings("ignore")

A, EFF, LEAK = 7.4e8, 0.5, 1e-4
DIRN = np.array([1.,1.])/np.sqrt(2); HALF = np.deg2rad(12.)
SCALE = 1e-6   # 计数缩放: 把探测率(~1e7)降到可数值处理的期望计数量级, 保持泊松结构

def mu_count(P, src=np.array([2.,2.])):
    d=P-src; r=np.clip(np.hypot(d[:,0],d[:,1]),0.3,None); u=d/r[:,None]
    ins=np.arccos(np.clip(u@DIRN,-1,1))<=HALF
    return A*EFF/(4*np.pi*r**2)*np.where(ins,1.0,LEAK)*SCALE   # 期望计数 λ(x)

def traj_sample(rng,n=252,nlines=6):
    ys=np.sort(rng.uniform(0.5,9.5,nlines)); per=n//nlines; pts=[]
    for yy in ys:
        xs=np.sort(rng.uniform(0,10,per)); jit=rng.normal(0,0.15,per)
        pts.append(np.column_stack([xs,np.clip(yy+jit,0,10)]))
    return np.vstack(pts)

# ---------- Laplace 泊松GP: y~Poisson(exp(f)), f~GP(0,K) ----------
def laplace_poisson_gp(Xtr,ytr,Xte,kfun,jitter=1e-6,tol=1e-6,maxit=100):
    # 常数先验均值 m0=log(平均计数): 令潜变量在无数据处回归到平均率而非 rate=1,
    # 与 log-高斯GP 的 normalize_y 对齐, 保证公平.
    m0=np.log(max(np.mean(ytr),0.1))
    K=kfun(Xtr,Xtr)+jitter*np.eye(len(Xtr))
    f=np.full(len(Xtr), m0)                 # 初值=先验均值
    for _ in range(maxit):
        f=np.clip(f,-20,20)
        ef=np.exp(f); W=np.clip(ef,1e-8,1e10); sW=np.sqrt(W)
        B=np.eye(len(Xtr))+ (sW[:,None]*K*sW[None,:]); L=cholesky(B,lower=True)
        grad=ytr-ef
        b=W*(f-m0)+grad
        tmp=solve_triangular(L, sW*(K@b), lower=True)
        tmp=solve_triangular(L.T, tmp, lower=False)
        a=b - sW*tmp
        f_target=m0+K@a
        f_new=f+0.7*(f_target-f)             # 阻尼牛顿步, 防止过冲发散
        if np.max(np.abs(f_new-f))<tol: f=f_new; break
        f=f_new
    f=np.clip(f,-20,20)
    ef=np.exp(f); W=ef; sW=np.sqrt(W)
    B=np.eye(len(Xtr))+(sW[:,None]*K*sW[None,:]); L=cholesky(B,lower=True)
    grad=ytr-ef
    Ks=kfun(Xtr,Xte); Kss=kfun(Xte,Xte)
    fmean=m0+Ks.T@grad                       # 预测潜变量均值(含先验均值)
    v=solve_triangular(L, sW[:,None]*Ks, lower=True)
    fvar=np.diag(Kss)-np.sum(v*v,axis=0)
    fvar=np.maximum(fvar,1e-9)
    return fmean, fvar

def matern_kfun(ls, var=1.0):
    k=ConstantKernel(var)*Matern(ls,nu=1.5)
    return lambda Xa,Xb: k(Xa,Xb)

def met(y,p):
    sr=np.sum((y-p)**2); st=np.sum((y-np.mean(y))**2)
    return float(np.mean(np.abs(y-p))), float(np.sqrt(np.mean((y-p)**2))), float(1-sr/st) if st>0 else 0.0

def one(seed):
    rng=np.random.RandomState(seed)
    G=np.column_stack([v.ravel() for v in np.meshgrid(np.linspace(0,10,50),np.linspace(0,10,50))])
    lamG=mu_count(G)                          # 真值期望计数
    tgtG=np.log10(lamG+1.0)                   # 评价域: log10(rate+1)
    X=traj_sample(rng); lamX=mu_count(X)
    cX=rng.poisson(lamX).astype(float)        # 观测计数(泊松)
    yX=np.log10(cX+1.0)

    # --- log-高斯GP (现方法) ---
    k=ConstantKernel(1.0,(1e-3,1e3))*Matern(1.0,(1e-2,1e2),nu=1.5)+ConstantKernel(0.1,(1e-4,1e2))+WhiteKernel(1e-2,(1e-6,1e1))
    gg=GaussianProcessRegressor(kernel=k,normalize_y=True,n_restarts_optimizer=1,random_state=0).fit(X,yX)
    mg,sg=gg.predict(G,return_std=True)       # 已在 log10(count+1) 域
    ls=[v for k,v in gg.kernel_.get_params().items() if k.endswith('length_scale') and np.isscalar(v)][0]  # 复用长度尺度

    # --- 泊松GP (Laplace) ---
    kf=matern_kfun(ls, var=1.0)
    fm,fv=laplace_poisson_gp(X,cX,G,kf)
    # 点估计用后验中位数 rate=exp(fm) (对 MAE 最优, 避免对数正态均值在空白区膨胀)
    lam_hat=np.exp(fm)
    mp=np.log10(lam_hat+1.0)
    # 评价域σ: delta法 g(f)=log10(exp(f)+1), g'(fm)=exp(fm)/((exp(fm)+1)ln10)
    gp_=lam_hat/((lam_hat+1.0)*np.log(10))
    sp=gp_*np.sqrt(fv)

    err_g=np.abs(tgtG-mg); err_p=np.abs(tgtG-mp)
    out={"seed":int(seed),
         "logGP":{"met":met(tgtG,mg),
                  "spearman":float(spearmanr(sg,err_g).statistic),
                  "cov1":float(np.mean(err_g<=sg)),"cov2":float(np.mean(err_g<=2*sg))},
         "poisGP":{"met":met(tgtG,mp),
                  "spearman":float(spearmanr(sp,err_p).statistic),
                  "cov1":float(np.mean(err_p<=sp)),"cov2":float(np.mean(err_p<=2*sp))}}
    return out

if __name__=="__main__":
    seeds=[int(x) for x in sys.argv[1:]] or list(range(10))
    res=[one(s) for s in seeds]
    import pathlib; pathlib.Path("poisson_cmp.json").write_text(json.dumps(res,ensure_ascii=False))
    for r in res:
        g=r["logGP"]; p=r["poisGP"]
        print("seed%d | logGP MAE=%.3f R2=%.2f rho=%.2f cov2=%.2f || poisGP MAE=%.3f R2=%.2f rho=%.2f cov2=%.2f"
              %(r["seed"],g["met"][0],g["met"][2],g["spearman"],g["cov2"],p["met"][0],p["met"][2],p["spearman"],p["cov2"]),flush=True)
