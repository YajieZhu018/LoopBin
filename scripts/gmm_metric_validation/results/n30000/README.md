# N_SAMPLES=30000 run (2026-08-26)

Archived output of `01_generate_toy_data.py` + `02_evaluate_metric_weights.py` +
`03_plot_metric_selection.py` with `N_SAMPLES` bumped from 10000 to 30000,
submitted as SLURM job `15519814` via `../run_bump_samples.sbatch`. Pre-bump
`toy_data/`/`pretrained/` backed up to `../../n10000_backup/` (which also has a
copy of that run's `results/` under `results_n10000/`). See the "2026-08-26:
N_SAMPLES bump to 30000" entry in `../../run_report.md` for the full writeup —
this file is a short index for what's in this folder.

## This is also the first real GPU-accelerated run

`run_bump_samples.sbatch` now extracts the CUDA 11 runtime libraries (see the
CUDA/GPU fix entries in `run_report.md`) before running Python. Job log
(`../run_bump_samples_15519814.err`) confirms `libcudart.so.11.0` loaded and the
job completed in **22 minutes** — despite 3x the data of the N=10000 run, which
took 24 minutes on CPU (no working GPU libs at the time). Real GPU speedup,
confirmed, not just theoretical.

## Dead-unit fraction: still improving, but slowly (diminishing returns)

Spot-checked `mild_k4`/`severe_k10`: 8/10 dead (N=3000) → 6/10 dead (N=10000) →
6/10 and 5/10 dead respectively (N=30000). Tripling `N_SAMPLES` again bought much
less improvement than the first bump did — `N_SAMPLES` alone is not going to fully
fix the dead-latent-unit problem; diminishing returns have set in.

**`cov_health` is finally partially testable**: 4/9 datasets now have non-constant
`cov_health` across their 10 candidates (was 0/9 at both N=3000 and N=10000). Its
correlation with ARI in those 4 is negative (mean within-dataset rho -0.296,
n=4) — a genuinely new data point, but on a small, non-random subset of datasets
(only datasets whose latent space happened to have few enough dead dims), so
treat this as suggestive, not conclusive.

**More datasets hit total survivor-filter wipeout** (`n_survivors=0/10`) at
N=30000 than at N=10000: `mild_k10`, `moderate_k6`, `moderate_k10`, `severe_k10`
all have zero candidates passing `MIN_WEIGHT_THRESHOLD`/`COLLAPSE_RATIO_THRESHOLD`
now (vs. only reduced — not zeroed — survivor counts for the k10 datasets at
N=10000). Not fully explained; noted here as something to look into if this
matters for real pipeline runs, not chased down further.

## The big story: metric rankings are NOT stable across sample sizes

Comparing mean within-dataset Spearman rho across all three runs so far:

| metric | N=3000 | N=10000 | N=30000 |
|---|---|---|---|
| min_weight | (not computed*) | **+0.280** | **+0.333** |
| separation | (not computed*) | +0.232 | +0.015 |
| assignment_confidence | (not computed*) | -0.136 | -0.122 |
| log_likelihood | (not computed*) | +0.016 | -0.133 |
| cov_health | NaN (all constant) | NaN (all constant) | -0.296 (4/9 datasets only) |

\* within-dataset correlation wasn't computed for the N=3000 run — it predates
that instrumentation; only pooled numbers exist for it (see `run_report.md`).

**`min_weight` is the only metric that stays consistently positive across both
runs where it's measurable, and gets *stronger*, not weaker, as `N_SAMPLES`
grows.** Every other metric either flips sign (`separation`: strong positive at
N=10000, ~zero at N=30000) or stays weakly negative/noisy (`assignment_confidence`,
`log_likelihood`). Given `separation` was the standout "winner" in the N=10000
report and has now evaporated, **don't trust a single run's within-dataset ranking
either** — the per-dataset rhos are individually noisy (e.g. `separation` ranges
from -0.83 to +0.96 across the 9 datasets at N=30000) and the *pooled* correlation
for `separation` actually flipped from strongly positive (0.826) to weakly
negative (-0.189) between these two runs, which is exactly the kind of
between-dataset-scale instability the within-dataset analysis was introduced to
catch — it just turns out `separation` itself isn't reliable either, not only the
pooling method.

## Files

Same set as `../n10000/`, plus `training_curves.json` (this run is the first with
AE loss-vs-epoch tracking, so `metric_selection_report.pdf` also has the new
per-dataset loss-curve page).
