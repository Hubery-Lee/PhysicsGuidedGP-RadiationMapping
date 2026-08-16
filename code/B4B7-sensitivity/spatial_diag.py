# -*- coding: utf-8 -*-
# B4: 空间残差诊断 (留块GP标准化残差) — 复用 gate_robustness_sim 原版生成器/模板/门控口径
import json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from gate_robustness_sim import FIELDS, BASE, traj_sample, template, fit_template, resid_gp
T0=time.time(); BUDGET=140
CK='/tmp/ana/sd.jsonl'

def gp_block_z(X, y, nb=4, seed=0):
    xe=np.linspace(0,10,nb+1); ye=np.linspace(0,10,nb+1)
    bid=np.clip(np.searchsorted(xe,X[:,0])-1,0,nb-1)*nb+np.clip(np.searchsorted(ye,X[:,1])-1,0,nb-1)
    blocks=[b for b in np.unique(bid) if (bid==b).sum()>=10]
    rs=np.random.RandomState(seed)
    order=rs.permutation(blocks)
    folds=[order[i::5] for i in range(5)]
    zmap={}
    for fold in folds:
        te=np.isin(bid,fold); tr=~te
        if te.sum()==0 or tr.sum()<50: continue
        best=fit_template(X[tr],y[tr]); px=best.x
        r_tr=y[tr]-template(px,X[tr])
        g=resid_gp(X[tr],r_tr)
        mu,sd=g.predict(X[te],return_std=True)
        pred=template(px,X[te])+mu
        z=(y[te]-pred)/np.maximum(sd,1e-6)
        bt=bid[te]
        for b in np.unique(bt):
            m=bt==b
            if m.sum()>=10: zmap[int(b)]=float(np.mean(z[m])*np.sqrt(m.sum()))
    return (max(abs(v) for v in zmap.values()) if zmap else 0.0), zmap

def one(scen, seed):
    rng=np.random.RandomState(seed)
    f=FIELDS[scen](rng)
    X=traj_sample(rng)
    y=np.log10(rng.poisson(f(X))+1.0)
    best=fit_template(X,y); cbar=float(best.cost/len(X))
    Tz,zmap=gp_block_z(X,y,seed=seed)
    return dict(scen=scen,seed=int(seed),cbar=cbar,Tz=float(Tz),zmap=zmap)

JOBS=[('C',100+i) for i in range(22)]+[('OC',400+i) for i in range(20)]
done=set()
if os.path.exists(CK):
    for ln in open(CK): d=json.loads(ln); done.add((d['scen'],d['seed']))
fo=open(CK,'a')
for scen,seed in JOBS:
    if (scen,seed) in done: continue
    if time.time()-T0>BUDGET: print('PAUSE', scen, seed, len(done)); sys.exit(0)
    r=one(scen,seed); fo.write(json.dumps(r)+'\n'); fo.flush(); done.add((scen,seed))
print('ALL DONE', len(done))
