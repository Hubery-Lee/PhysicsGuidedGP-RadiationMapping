# Manifest: files -> paper artifacts

## Code -> results -> paper location

| Script (`code/`) | Produces (`results/`) | Paper artifact |
|---|---|---|
| `r1_run.py` | `part_*.json`, `reanalysis_results.json` | Tables 2, 3 (linear, plain GP, MLP) |
| `mlp2_run.py` | `part_*.json` (mlp2 field) | MLP single-network entries |
| `quick_methods.py` | `qm_*.json` | Tables 2, 3 (RF, IDW); MLP seed stability (Sec. 3.2) |
| `exp_final.py` | `ef_*.json` | Tables 2, 3 (physics-guided GP, MLP ensemble); calibration (Fig. calibration) |
| `aniso_gp.py` | `ag_*.json` | anisotropic-template component |
| `new_baselines.py` | `nb_*.json` | Tables 2, 3 (multi-kernel weighted GP, Poisson kriging) |
| `sim_newbaselines.py` | `sim_newbaselines.json` | Table 1 (MKGP, Poisson kriging on simulation) |
| `r4_poisson_sim.py` | `r4_results.json` | Table 1 (5-method simulation) |
| `sim_stability.py` | `sim_stability.json` | Sec. 3.1 (MLP R^2 across 18 seed combinations) |
| `ablation_meanfn.py` | `ablation_results.json` | Table 4 (mean-function ablation M0-M3) |
| `gate_quantify.py` | `gate_costs.json` | Sec. 4.2 (gate 94% detection, 0 false alarm) |
| `gate_robustness_sim.py` | `gaterob_{C,MS,SC,OC}.json` | Table 5 + Fig. (gate on unseen misspecification) |
| `poisson_gp_compare.py` | `poisson_cmp.json` | Table 6 (Poisson GP vs log-Gaussian GP) |
| `f1_all_methods.py` | `f1_all_results.json` | Tables 2, 3 (boundary F1 column) |
| `hyperparam_leak_check.py` | `leak_check_results.json` | Sec. 2.4 (kernel-hyperparameter reuse control) |

## Figures (`code/figures/`)

| Script | Paper figure |
|---|---|
| `fig_sim.py` | Fig. 4 — simulation reconstructions (incl. Poisson kriging panel) |
| `fig_exp.py` | Fig. 5 — experimental run-3 reconstructions |
| `fig_stats.py` | Fig. 6 (protocol MAE, 9 methods), Fig. 7 (source parameters), Fig. 9 (calibration) |
| `fig_source_recon.py` | Fig. 8 — source recovery overlay |
| `fig_run1_traj.py` | run-1 trajectory / mirror-direction degeneracy |
| `make_fig_traj_ate.py` | Fig. 2 — localization ATE (requires motion-capture data, not bundled) |

## Data
`data/1/` … `data/7/` — seven independent robotic survey runs
(521–2325 georeferenced dose-rate measurements each).
