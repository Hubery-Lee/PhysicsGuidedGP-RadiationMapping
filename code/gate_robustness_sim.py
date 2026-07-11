# -*- coding: utf-8 -*-
# 意见3: 拟合门控在"新失配类型"上的稳健性压测 (纯仿真, 自洽标定)
# 场景: C=正确单源(准直), MS=多源叠加, SC=散射主导, OC=屏蔽遮挡
# 每realization: 轨迹式采样+泊松计数 -> log10(count+1) -> 拟合单源模板
#   门控统计量 c_bar = soft_L1 代价 / n_train (与实验管线同口径)
# 并对比 模板GP(误用) vs plain GP(回退) 相对真值的重建MAE, 证明门控触发后回退更安全。
import json, sys, warnings
import numpy as np
from scipy.optimize import least_squares
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
warnings.filterwarnings("ignore")

A, EFF, LEAK = 7.4e8, 0.5, 1e-4
from pathlib import Path
OUT = Path(__file__).parent

def collimated(P, src, dirn, half):
    d = P - src
    r = np.clip(np.hypot(d[:,0], d[:,1]), 0.3, None)
    u = d / r[:,None]
    ins = np.arccos(np.clip(u@dirn, -1, 1)) <= half
    return A*EFF/(4*np.pi*r**2)*np.where(ins, 1.0, LEAK)

def make_C(rng):
    src=rng.uniform(1,3,2); ang=rng.uniform(0,2*np.pi); dirn=np.array([np.cos(ang),np.sin(ang)])
    half=np.deg2rad(rng.uniform(10,16))
    return lambda P: collimated(P, src, dirn, half)

def make_MS(rng):
    k=int(rng.choice([2,3])); ps=[]
    for _ in range(k):
        src=rng.uniform(0.5,9.5,2); ang=rng.uniform(0,2*np.pi); dirn=np.array([np.cos(ang),np.sin(ang)])
        ps.append((src,dirn,np.deg2rad(rng.uniform(10,18))))
    return lambda P: sum(collimated(P,s,d,h) for s,d,h in ps)

def make_SC(rng):
    src=rng.uniform(1,3,2); ang=rng.uniform(0,2*np.pi); dirn=np.array([np.cos(ang),np.sin(ang)])
    half=np.deg2rad(rng.uniform(10,16)); frac=rng.uniform(0.3,0.6)
    def f(P):
        beam=collimated(P, src, dirn, half)
        r=np.clip(np.hypot((P-src)[:,0],(P-src)[:,1]),0.3,None)
        return beam + frac*A*EFF/(4*np.pi*r**2)
    return f

def make_OC(rng):
    src=rng.uniform(1,3,2); ang=rng.uniform(0,2*np.pi); dirn=np.array([np.cos(ang),np.sin(ang)])
    half=np.deg2rad(rng.uniform(12,18)); shadow_c=ang+rng.uniform(-0.15,0.15); shadow_w=rng.uniform(0.10,0.20)
    def f(P):
        rate=collimated(P, src, dirn, half)
        d=P-src; bearing=np.arctan2(d[:,1],d[:,0])
        dth=np.abs(np.angle(np.exp(1j*(bearing-shadow_c))))
        return np.where(dth<shadow_w, rate*LEAK, rate)
    return f

FIELDS={"C":make_C,"MS":make_MS,"SC":make_SC,"OC":make_OC}
BASE={"C":100,"MS":200,"SC":300,"OC":400}

def traj_sample(rng, n=252, nlines=6):
    ys=np.sort(rng.uniform(0.5,9.5,nlines)); per=n//nlines; pts=[]
    for yy in ys:
        xs=np.sort(rng.uniform(0,10,per)); jit=rng.normal(0,0.15,per)
        pts.append(np.column_stack([xs, np.clip(yy+jit,0,10)]))
    return np.vstack(pts)

def template(p, Xq):
    sx,sy,phi,half,logw,logC,logleak,logbg = p
    dx=Xq[:,0]-sx; dy=Xq[:,1]-sy; r2=dx*dx+dy*dy+0.09
    th=np.abs(np.arctan2(dy,dx)-phi); th=np.minimum(th,2*np.pi-th)
    f=10**logleak+(1-10**logleak)/(1+np.exp((th-half)/max(10**logw,1e-3)))
    return np.log10(10**logC*f/r2+10**logbg+1.0)

def fit_template(X,y):
    i0=np.argmax(y); sx0,sy0=X[i0]
    hot=y>np.percentile(y,85)
    phi0=np.arctan2(np.mean(X[hot,1]-sy0),np.mean(X[hot,0]-sx0)) if hot.sum()>3 else 0.0
    lb=[X[:,0].min()-2,X[:,1].min()-2,-3*np.pi,np.deg2rad(3),-2.5,-2,-6,-2]
    ub=[X[:,0].max()+2,X[:,1].max()+2, 3*np.pi,np.deg2rad(80),0.5,12,0,2]
    inits=[[sx0,sy0,phi0+d,np.deg2rad(h),-1.0,np.log10(max(10**y.max()-1,1)*0.09+1),-3.0,0.0]
           for d in [0,np.pi/2,np.pi,-np.pi/2] for h in [10,25]]
    best=None
    for p0 in inits:
        p0=[min(max(v,lo),hi) for v,lo,hi in zip(p0,lb,ub)]
        try:
            r=least_squares(lambda p: template(p,X)-y,p0,loss="soft_l1",bounds=(lb,ub),max_nfev=500)
            if best is None or r.cost<best.cost: best=r
        except Exception: pass
    return best

def met(y,p): return float(np.mean(np.abs(y-p)))

def resid_gp(Xtr,r):
    k=ConstantKernel(1.0,(1e-3,1e3))*Matern(1.0,(1e-2,1e2),nu=1.5)+WhiteKernel(1e-2,(1e-6,1e1))
    g=GaussianProcessRegressor(kernel=k,normalize_y=True,n_restarts_optimizer=1,random_state=0); g.fit(Xtr,r); return g

def plain_gp(Xtr,ytr):
    k=ConstantKernel(1.0,(1e-3,1e3))*Matern(1.0,(1e-2,1e2),nu=1.5)+ConstantKernel(0.1,(1e-4,1e2))+WhiteKernel(1e-2,(1e-6,1e1))
    g=GaussianProcessRegressor(kernel=k,normalize_y=True,n_restarts_optimizer=1,random_state=0); g.fit(Xtr,ytr); return g

def one(scenario, seed):
    rng=np.random.RandomState(seed)
    f=FIELDS[scenario](rng)
    G=np.column_stack([v.ravel() for v in np.meshgrid(np.linspace(0,10,60),np.linspace(0,10,60))])
    yG=np.log10(f(G)+1.0)
    X=traj_sample(rng)
    y=np.log10(rng.poisson(f(X))+1.0)
    best=fit_template(X,y); cbar=float(best.cost/len(X)); px=best.x
    gt=resid_gp(X, y-template(px,X)); pred_t=template(px,G)+gt.predict(G)
    gp=plain_gp(X,y); pred_p=gp.predict(G)
    return {"scenario":scenario,"seed":int(seed),"cbar":cbar,
            "mae_template":met(yG,pred_t),"mae_plaingp":met(yG,pred_p),
            "yrange":float(yG.max()-yG.min())}

if __name__=="__main__":
    scen=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 25
    out=[one(scen,s) for s in range(BASE[scen],BASE[scen]+n)]
    (OUT/("gaterob_%s.json"%scen)).write_text(json.dumps(out,ensure_ascii=False))
    cb=np.array([o["cbar"] for o in out])
    print("%s: n=%d c_bar med=%.4f [%.4f,%.4f] mae_tmpl=%.3f mae_plain=%.3f"
          % (scen,len(out),np.median(cb),np.percentile(cb,10),np.percentile(cb,90),
             np.mean([o["mae_template"] for o in out]),np.mean([o["mae_plaingp"] for o in out])),flush=True)
