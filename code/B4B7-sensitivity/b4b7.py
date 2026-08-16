import numpy as np, json, os, sys, time
from scipy.optimize import least_squares
T0 = time.time(); BUDGET = 140

A, EFF, LEAK, HALF = 7.4e8, 0.5, 1e-4, np.deg2rad(12.0)
W_PEN = np.deg2rad(1.5)
def g_sector(theta, half=HALF, w=W_PEN, leak=LEAK):
    z = np.clip((half-theta)/w, -50, 50)
    return leak + (1-leak)/(1+np.exp(-z))
def ang_off(x, y, sx, sy, phi):
    a = np.arctan2(y-sy, x-sx) - phi
    return np.abs((a+np.pi) % (2*np.pi) - np.pi)
def rate(x, y, sx, sy, phi, half=HALF, occl=None):
    r2 = (x-sx)**2 + (y-sy)**2 + 0.09
    th = ang_off(x, y, sx, sy, phi)
    lam = A*EFF*g_sector(th, half)/(4*np.pi*r2)
    if occl is not None:
        phc, wid, r0, att = occl
        a = np.arctan2(y-sy, x-sx) - phc
        a = np.abs((a+np.pi) % (2*np.pi) - np.pi)
        r = np.hypot(x-sx, y-sy)
        lam = np.where((a < wid/2) & (r > r0), lam*att, lam)
    return lam
def sample_traj(dx=0.15):
    xs, ys = [], []
    for i, yy in enumerate(np.linspace(0.6, 7.4, 6)):
        xx = np.arange(0.5, 9.5, dx)
        if i % 2: xx = xx[::-1]
        xs.append(xx); ys.append(np.full_like(xx, yy))
    return np.concatenate(xs), np.concatenate(ys)
def template(p, x, y):
    logC, sx, sy, phi, half, logw, logit_l, b = p
    th = ang_off(x, y, sx, sy, phi)
    w = np.exp(logw); l = 1e-6 + (0.1-1e-6)/(1+np.exp(-logit_l))
    z = np.clip((half-th)/w, -50, 50)
    gg = l + (1-l)/(1+np.exp(-z))
    r2 = (x-sx)**2 + (y-sy)**2 + 0.09
    return np.log10(np.maximum(10.0**np.clip(logC, -30, 30) * gg / r2 + b + 1.0, 1e-12))
XSC = [1., 1., 1., 0.5, 0.2, 1., 1., 1.]
def fit_template(x, y, yy):
    i0 = np.argmax(yy)
    hot = yy >= np.quantile(yy, 0.85)
    cx, cy = np.average(x[hot], weights=yy[hot]), np.average(y[hot], weights=yy[hot])
    best = None
    for dphi in [0, np.pi/2, -np.pi/2, np.pi]:
      for h0 in [np.deg2rad(10), np.deg2rad(20)]:
        phi0 = np.arctan2(cy-y[i0]+1e-9, cx-x[i0]+1e-9) + dphi
        p0 = [np.log10(A*EFF/(4*np.pi)), x[i0], y[i0], phi0, h0, np.log(np.deg2rad(2)), 0.0, 0.0]
        try:
            res = least_squares(lambda p: template(p, x, y)-yy, p0, loss='soft_l1', f_scale=0.3,
                                max_nfev=3000, x_scale=XSC)
        except Exception: continue
        if best is None or res.cost < best.cost: best = res
    return best.x, best.cost/len(yy)
def block_stat(x, y, res, nb=4):
    xe = np.linspace(x.min()-1e-9, x.max()+1e-9, nb+1); ye = np.linspace(y.min()-1e-9, y.max()+1e-9, nb+1)
    means = []
    mp = np.full((nb, nb), np.nan)
    for i in range(nb):
        for j in range(nb):
            m = (x >= xe[i]) & (x < xe[i+1]) & (y >= ye[j]) & (y < ye[j+1])
            if m.sum() >= 10:
                mp[j, i] = np.mean(res[m]); means.append((j, i, mp[j, i]))
    v = np.array([t[2] for t in means])
    med = np.median(v); mad = 1.4826*np.median(np.abs(v-med)) + 1e-12
    T = float(np.max(np.abs(v-med))/mad)
    mp = (mp-med)/mad
    return T, mp

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

def gp_block_stat(x, y, yy, nb=4, seed=0):
    xe = np.linspace(x.min()-1e-9, x.max()+1e-9, nb+1); ye = np.linspace(y.min()-1e-9, y.max()+1e-9, nb+1)
    bid = np.clip(np.searchsorted(xe, x)-1, 0, nb-1)*nb + np.clip(np.searchsorted(ye, y)-1, 0, nb-1)
    blocks = [b for b in np.unique(bid) if (bid == b).sum() >= 10]
    rs = np.random.default_rng(seed)
    order = rs.permutation(blocks)
    folds = [order[i::5] for i in range(5)]
    zmap = {}
    for fold in folds:
        te = np.isin(bid, fold); tr = ~te
        if te.sum() == 0 or tr.sum() < 50: continue
        p, _ = fit_template(x[tr], y[tr], yy[tr])
        res_tr = yy[tr] - template(p, x[tr], y[tr])
        ker = ConstantKernel(1.0, (1e-3, 1e3))*Matern(length_scale=1.0, length_scale_bounds=(0.1, 10), nu=1.5) \
              + WhiteKernel(1e-2, (1e-8, 1e1))
    # subsample for GP speed
        ii = np.arange(tr.sum())
        if ii.size > 350: ii = rs.choice(ii, 350, replace=False)
        gp = GaussianProcessRegressor(kernel=ker, normalize_y=True, n_restarts_optimizer=1, random_state=0)
        gp.fit(np.c_[x[tr][ii], y[tr][ii]], res_tr[ii])
        mu, sd = gp.predict(np.c_[x[te], y[te]], return_std=True)
        pred = template(p, x[te], y[te]) + mu
        z = (yy[te] - pred)/np.maximum(sd, 1e-6)
        bt = bid[te]
        for b in np.unique(bt):
            m = bt == b
            if m.sum() >= 10:
                zmap[int(b)] = float(np.mean(z[m])*np.sqrt(m.sum()))
    T = max(abs(v) for v in zmap.values())
    return T, zmap

def job(kind, idx):
    gpmode = kind.endswith('g')
    base = kind[:-1] if gpmode else kind
    kind_seed = base
    rng = np.random.default_rng(100000 + hash(kind_seed) % 1000 * 100 + idx)
    sx, sy = rng.uniform(0.8, 2.0), rng.uniform(2.5, 5.5)
    phi = rng.uniform(-0.3, 0.3)
    xt, yt = sample_traj()
    if base == 'corr' or base == 'occl':
        occ = None
        if base == 'occl':
            phc = phi + rng.uniform(-0.6, 0.6)*HALF
            occ = (phc, np.deg2rad(rng.uniform(3, 6)), rng.uniform(3.0, 5.0), 0.1)
        cnt = rng.poisson(rate(xt, yt, sx, sy, phi, occl=occ))
        yy = np.log10(cnt + 1.0)
        if gpmode:
            Tg, zmap = gp_block_stat(xt, yt, yy, seed=idx)
            out = dict(kind=kind, idx=idx, Tg=float(Tg))
            if kind == 'occlg' and idx == 0:
                json.dump(dict(zmap=zmap, src=[sx, sy, phi], occ=list(occ)), open('/tmp/ana/occg_example.json','w'))
            return out
        p, c = fit_template(xt, yt, yy)
        r = yy - template(p, xt, yt)
        T, mp = block_stat(xt, yt, r)
        out = dict(kind=kind, idx=idx, cbar=float(c), T=float(T))
        if kind == 'occl' and idx == 0:
            np.savez('/tmp/ana/occ_example.npz', x=xt, y=yt, res=r, mp=mp, src=[sx, sy, phi], occ=list(occ))
        return out
    # b7: kind in ('clean','pert')
    kind = base
    lam = rate(xt, yt, sx, sy, phi)
    if base == 'pert':
        npts = max(1, int(round(0.4/0.15)))
        lam = np.convolve(lam, np.ones(npts)/npts, mode='same')
        xs = xt + rng.normal(0, 0.0272/np.sqrt(2), xt.size)
        ys = yt + rng.normal(0, 0.0272/np.sqrt(2), yt.size)
    else:
        xs, ys = xt, yt
    cnt = rng.poisson(lam); yy = np.log10(cnt + 1.0)
    p, c = fit_template(xs, ys, yy)
    dpos = float(np.hypot(p[1]-sx, p[2]-sy))
    ddir = float(np.rad2deg(np.abs((p[3]-phi+np.pi) % (2*np.pi) - np.pi)))
    dhalf = float(np.rad2deg(p[4]) - 12.0)
    berr = []
    for r0 in (3., 4., 5.):
        ths = np.linspace(-np.pi/3, np.pi/3, 2001)
        gx, gy = sx + r0*np.cos(phi+ths), sy + r0*np.sin(phi+ths)
        ytrue = np.log10(rate(gx, gy, sx, sy, phi) + 1)
        ypred = template(p, gx, gy)
        lvl = 0.5*(ytrue.max() + np.median(ytrue[np.abs(ths) > np.deg2rad(25)]))
        for sgn in (1, -1):
            m = sgn*ths > 0
            tc = ths[m][np.argmin(np.abs(ytrue[m]-lvl))]
            pc = ths[m][np.argmin(np.abs(ypred[m]-lvl))]
            berr.append(abs(r0*(tc-pc)))
    return dict(kind=kind, idx=idx, dpos=dpos, ddir=ddir, dhalf=dhalf, berr=float(np.mean(berr)), cbar=float(c))

JOBS = [('corr', i) for i in range(22)] + [('occl', i) for i in range(20)] + \
       [('clean', i) for i in range(20)] + [('pert', i) for i in range(20)] + \
       [('corrg', i) for i in range(22)] + [('occlg', i) for i in range(20)]
CK = '/tmp/ana/ck.jsonl'
done = set()
if os.path.exists(CK):
    for ln in open(CK):
        d = json.loads(ln); done.add((d['kind'], d['idx']))
f = open(CK, 'a')
n_done_now = 0
for kind, idx in JOBS:
    if (kind, idx) in done: continue
    if time.time() - T0 > BUDGET:
        print('PAUSE at', kind, idx, 'done_total=', len(done) + n_done_now); sys.exit(0)
    r = job(kind, idx)
    f.write(json.dumps(r) + '\n'); f.flush(); n_done_now += 1
print('ALL DONE', len(done) + n_done_now)
