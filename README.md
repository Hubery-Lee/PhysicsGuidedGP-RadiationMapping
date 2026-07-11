# Physics-Guided Gaussian Process Mapping of Strong-Gradient Radiation Fields

Data and analysis code accompanying the paper
*"Physics-Guided Gaussian Process Mapping of Strong-Gradient Radiation Fields
from Mobile Robot Surveys: The Role of Sampling Geometry."*

A tracked mobile robot surveyed a collimated Cs-137 field in seven independent
runs. This repository contains the georeferenced dose-rate measurements, the
reconstruction and evaluation code (all baselines + the proposed physics-guided
GP), the fit-quality-gate stress tests, the Poisson-likelihood comparison, and
every per-run / per-realization result reported in the paper.

## Repository layout

```
data/            7 runs of georeferenced dose-rate measurements (run i -> data/i/data.txt)
code/            analysis pipeline (see mapping below)
code/figures/    figure-generation scripts
results/         precomputed per-run / per-realization outputs (JSON) for every table/figure
requirements.txt Python dependencies
MANIFEST.md      file -> paper table/figure mapping
```

### Data format
Each `data/i/data.txt` is whitespace-delimited with a `#` header line:
```
#timeStamp  px(m)  py(m)  doseRate(uSv/h)
```
`px, py` are SLAM poses (metres); `doseRate` is the ambient dose-equivalent rate
H*(10) in uSv/h. Rows with near-zero pose (pre-SLAM-convergence) are filtered in
the loaders. All methods transform the target as `y = log10(doseRate + 1)`.

## Reproducing the results

```bash
pip install -r requirements.txt

# Core baselines (linear / plain GP / MLP), Protocols A & B  -> results/part_*.json
python code/r1_run.py 1 2 3 4 5 6 7

# Physics-guided GP + MLP ensemble + calibration            -> results/ef_*.json
python code/exp_final.py 1:AB 2:AB 3:AB 4:AB 5:AB 6:AB 7:AB

# RF / IDW / physGP-trend + MLP seed stability              -> results/qm_*.json
python code/quick_methods.py 1 2 3 4 5 6 7

# New advanced baselines: multi-kernel weighted GP + Poisson kriging (experiments)
python code/new_baselines.py                                 # -> results/nb_*.json
# ... and on the Poisson simulation (Table 1)
python code/sim_newbaselines.py                              # -> results/sim_newbaselines.json

# Simulation 5-method comparison (Table 1)                  -> results/r4_results.json
python code/r4_poisson_sim.py

# Mean-function ablation M0-M3 (Table 4)                     -> results/ablation_results.json
python code/ablation_meanfn.py 1 2 3 4 5 6 7

# Gate detection on experimental misspecification (Sec. 4.2, 94%) -> results/gate_costs.json
python code/gate_quantify.py 1 2 3 4 5 6 7

# Gate stress test on unseen misspecification types (Table 5, Fig.) -> results/gaterob_*.json
python code/gate_robustness_sim.py C 22
python code/gate_robustness_sim.py MS 20
python code/gate_robustness_sim.py SC 20
python code/gate_robustness_sim.py OC 20

# Poisson-likelihood GP vs log-Gaussian GP (Table 6)        -> results/poisson_cmp.json
python code/poisson_gp_compare.py 0 1 2 3 4 5 6 7 8 9 10 11

# Boundary F1 for all methods (Tables 2/3 F1 column)        -> results/f1_all_results.json
python code/f1_all_methods.py 1 2 3 4 5 6 7
```

`results/` already contains all of these outputs, so tables can be regenerated
without rerunning the (GP-heavy) pipeline.

### Figures
Figure scripts in `code/figures/` read the JSON outputs from `results/`. Run them
with `results/` as the working directory, e.g.:
```bash
cd results && python ../code/figures/fig_stats.py     # Fig. 6 (protocol MAE), 7, 9
cd results && python ../code/figures/fig_sim.py        # Fig. 4 (simulation reconstructions)
cd results && python ../code/figures/fig_exp.py        # Fig. 5 (experimental reconstructions)
```
`make_fig_traj_ate.py` (Fig. 2, localization/ATE) additionally requires the
eight-camera motion-capture dataset, which is **not bundled** here; place it in
`data_motioncapture/` to regenerate that figure.

## Notes
- Paths are resolved relative to the repository, so no local paths need editing.
- The raw survey data were acquired at a national secondary calibration facility;
  confirm you have permission to release them publicly before pushing `data/`.
  Remove `data/` if release is restricted — all `results/` JSONs remain usable.
- Implemented with scikit-learn and SciPy; the physics template is fitted by
  robust (soft-L1) nonlinear least squares with multi-start initialization.

## Citation
Please cite the paper (details to be added upon publication). A `CITATION.cff`
can be added at that point.

## License
Code released under the MIT License (see `LICENSE`); update the copyright holder.
Data, if released, are provided for academic reuse — specify a data license
(e.g., CC-BY-4.0) as appropriate.
