#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R1/R2/R5 重分析：7组实验数据
- 协议A：随机80/20划分（复现原文做法, random_state=42）
- 协议B：空间分块交叉验证（4x4分块, GroupKFold 5折）
- 方法：线性插值 / Matérn3/2+偏差GP（边际似然优化, 披露超参数） / MLP(64,32,16, tanh, LBFGS)
- 统计：7组配对 Wilcoxon 符号秩检验 (MLP vs GP, MLP vs Linear)
输出: reanalysis_results.json
"""
import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT = Path("reanalysis_results.json")

def load_dataset(i):
    df = pd.read_csv(DATA_DIR / str(i) / "data.txt", sep=r"\s+", comment="#", header=None,
                     names=["timeStamp","px","py","doseRate","doseRate01","date","time","unknown"])
    m = (df.px.abs() > 1e-20) | (df.py.abs() > 1e-20)   # 剔除SLAM未收敛的初始零位姿
    df = df[m]
    X = df[["px","py"]].to_numpy(float)
    y_raw = df["doseRate"].to_numpy(float)
    y = np.log10(y_raw + 1.0)                            # 与作者代码一致: log10(dose+1)
    return X, y

def metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2); ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return mae, rmse, r2

def fit_predict_linear(Xtr, ytr, Xte):
    p = griddata(Xtr, ytr, Xte, method="linear")
    nan = np.isnan(p)
    if nan.any():
        p[nan] = griddata(Xtr, ytr, Xte[nan], method="nearest")
    return p

def fit_predict_gp(Xtr, ytr, Xte, seed=0, fixed_kernel=None):
    if fixed_kernel is not None:
        gp = GaussianProcessRegressor(kernel=fixed_kernel, normalize_y=True, optimizer=None)
        gp.fit(Xtr, ytr)
        return gp.predict(Xte), None, None
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) \
             + ConstantKernel(0.1, (1e-4, 1e2)) + WhiteKernel(1e-2, (1e-6, 1e1))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ytr), min(800, len(ytr)), replace=False)
    gp0 = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=1, random_state=seed)
    gp0.fit(Xtr[idx], ytr[idx])
    gp = GaussianProcessRegressor(kernel=gp0.kernel_, normalize_y=True, optimizer=None)
    gp.fit(Xtr, ytr)
    return gp.predict(Xte), gp0.kernel_, float(gp0.log_marginal_likelihood_value_)

def fit_predict_mlp(Xtr, ytr, Xte, seed=0):
    sx, sy = StandardScaler(), StandardScaler()
    Xtr_s = sx.fit_transform(Xtr); Xte_s = sx.transform(Xte)
    ytr_s = sy.fit_transform(ytr.reshape(-1, 1)).ravel()
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation="tanh", solver="lbfgs",
                       max_iter=500, random_state=seed)
    mlp.fit(Xtr_s, ytr_s)
    return sy.inverse_transform(mlp.predict(Xte_s).reshape(-1, 1)).ravel()

def block_groups(X, nb=4):
    gx = np.digitize(X[:, 0], np.linspace(X[:, 0].min(), X[:, 0].max(), nb + 1)[1:-1])
    gy = np.digitize(X[:, 1], np.linspace(X[:, 1].min(), X[:, 1].max(), nb + 1)[1:-1])
    return gx * nb + gy

import sys
PART = Path(__file__).resolve().parents[1] / "results"

def run_dataset(i, proto="AB"):
    X, y = load_dataset(i)
    fp = PART / f"part_{i}.json"
    out = json.loads(fp.read_text()) if fp.exists() else {}
    out["n"] = int(len(y))
    # ---------- 协议A：随机80/20 ----------
    if "A" not in proto:
        khat = None
    else:
        pass
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    if "A" in proto:
        rowA = {}
        rowA["linear"] = metrics(yte, fit_predict_linear(Xtr, ytr, Xte))
        gp_pred, khat, lml = fit_predict_gp(Xtr, ytr, Xte)
        rowA["gp"] = metrics(yte, gp_pred)
        rowA["mlp"] = metrics(yte, fit_predict_mlp(Xtr, ytr, Xte))
        out["A"] = rowA
        out["gp_kernel"] = repr(khat); out["lml"] = lml
    else:
        _, khat, _ = fit_predict_gp(Xtr[:0+len(ytr)], ytr, Xtr[:1]) if False else (None, None, None)
    if "B" not in proto:
        (PART / f"part_{i}.json").write_text(json.dumps(out, ensure_ascii=False))
        print(f"dataset {i} proto {proto} done n={out['n']}", flush=True)
        return
    if out.get("gp_kernel"):
        import sklearn.gaussian_process.kernels as K
        khat = eval(out["gp_kernel"], {**K.__dict__})
    # ---------- 协议B：空间分块 GroupKFold（GP超参数固定复用） ----------
    g = block_groups(X, nb=4)
    n_folds = min(5, len(np.unique(g)))
    gkf = GroupKFold(n_splits=n_folds)
    fold_m = {m: [] for m in ["linear", "gp", "mlp"]}
    for tr, te in gkf.split(X, y, groups=g):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        fold_m["linear"].append(metrics(yte, fit_predict_linear(Xtr, ytr, Xte)))
        p, _, _ = fit_predict_gp(Xtr, ytr, Xte, fixed_kernel=khat)
        fold_m["gp"].append(metrics(yte, p))
        fold_m["mlp"].append(metrics(yte, fit_predict_mlp(Xtr, ytr, Xte)))
    rowB = {}
    for m, v in fold_m.items():
        a = np.array(v)
        rowB[m] = {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "folds": a.tolist()}
    out["B"] = rowB
    (PART / f"part_{i}.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"dataset {i} done n={out['n']}", flush=True)

if len(sys.argv) > 1 and sys.argv[1] != "merge":
    for a in sys.argv[1:]:
        if ":" in a:
            i, proto = a.split(":")
            run_dataset(int(i), proto)
        else:
            run_dataset(int(a))
    sys.exit(0)

# ---------- merge: 汇总 + Wilcoxon ----------
results = {"protocol_A_random": {}, "protocol_B_blockcv": {}, "gp_hyperparams": {}, "n_points": {}}
for i in range(1, 8):
    part = json.loads((PART / f"part_{i}.json").read_text())
    results["n_points"][i] = part["n"]
    results["protocol_A_random"][i] = part["A"]
    results["protocol_B_blockcv"][i] = part["B"]
    results["gp_hyperparams"][i] = {"kernel": part["gp_kernel"], "log_marginal_likelihood": part["lml"]}

def summarize(proto_key, get):
    per = {m: np.array([get(results[proto_key][i], m) for i in range(1, 8)]) for m in ["linear", "gp", "mlp"]}
    summ = {m: {"mean": per[m].mean(0).tolist(), "std": per[m].std(0).tolist(),
                "per_dataset": per[m].tolist()} for m in per}
    tests = {}
    for a, b in [("mlp", "gp"), ("mlp", "linear")]:
        tests[f"{a}_vs_{b}"] = {}
        for j, name in enumerate(["MAE", "RMSE", "R2"]):
            stat, p = wilcoxon(per[a][:, j], per[b][:, j])
            tests[f"{a}_vs_{b}"][name] = {"W": float(stat), "p": float(p)}
    return {"summary": summ, "wilcoxon": tests}

results["summary_A"] = summarize("protocol_A_random", lambda row, m: row[m])
results["summary_B"] = summarize("protocol_B_blockcv", lambda row, m: row[m]["mean"])

OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1))
print("ALL DONE")
