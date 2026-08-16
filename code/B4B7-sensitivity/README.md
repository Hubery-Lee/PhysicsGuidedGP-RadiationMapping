# B4/B7 — Revision analyses for sensors-4463439

Analyses added during the major revision of the Sensors manuscript
(sensors-4463439), responding to Reviewer 2.

## Contents

| File | Purpose |
|---|---|
| `b4b7.py` | Checkpoint-resumable driver. **B7** (`clean`/`pert` jobs): paired simulation of template-parameter recovery under ideal sampling vs. injected perturbations (Gaussian pose noise, 2.72 cm RMSE; 0.4 m along-track boxcar smearing before Poisson sampling); 20 realizations. Produces the numbers behind the "Sensitivity to Localization Error and Detector Smearing" subsection (Table `tab:smearing`): median [IQR] of source-position error, beam-direction error, half-angle bias, boundary displacement; 3/20 gross misfits flagged by the fit-quality gate at 35–48× the ensemble-median cost. Also contains **B4** (`corr`/`occl`/`corrg`/`occlg` jobs): spatial residual diagnostic evaluation on a sweep-geometry occlusion ensemble (20 occluded + 22 correctly-specified fields): global gate 15/20, block-level leave-block-out GP z-score 14/20, union 18/20, zero false alarms under leave-one-out calibration. |
| `spatial_diag.py` | Independent B4 variant that reuses `../gate_robustness_sim.py` (original field generators / template fit / gate statistic) verbatim; block-level standardized residual diagnostic on the original random-trajectory ensemble. Included for cross-checking; the sparse random trajectories give shadows that often fall between survey lines, which is the regime where the block diagnostic adds little (as stated in the manuscript). |
| `ck.jsonl` | Per-realization results for all `b4b7.py` jobs (checkpoint format, JSON Lines). |
| `sd.jsonl` | Per-realization results for `spatial_diag.py`. |

## Reproducing

```bash
pip install numpy scipy scikit-learn
python b4b7.py          # checkpoint-resumable; safe to rerun until "ALL DONE"
python spatial_diag.py  # requires ../gate_robustness_sim.py in the same directory tree
```

`b4b7.py` uses a ~140 s time budget per invocation and resumes from
`ck.jsonl`; run it repeatedly until it prints `ALL DONE`.

## Mapping to the manuscript

- Spatial residual diagnostic paragraph (end of "Gate Robustness to Unseen
  Misspecification Types"): 15/20, 14/20, 18/20, zero false alarms — from
  `ck.jsonl` kinds `corr`/`occl`/`corrg`/`occlg`.
- "Sensitivity to Localization Error and Detector Smearing" (Table
  `tab:smearing`): medians/IQRs from `ck.jsonl` kinds `clean`/`pert`
  (outliers: pert indices 10, 11, 18).
