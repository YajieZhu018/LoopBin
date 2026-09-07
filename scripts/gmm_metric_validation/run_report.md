# GMM metric-selection validation on the real AE latent space — run report

**Date:** 2026-08-25
**Scope:** rewrote `01_generate_toy_data.py` and `02_evaluate_metric_weights.py` in place
so that `select_balanced_gmm`'s five candidate-selection metrics are validated against
ARI on a real AE-pretrained latent space (with known ground truth), instead of on
synthetic data generated directly in latent space with no pretraining step.
`03_plot_metric_selection.py` and `04_test_real_pretrain_gmm.py` were left untouched.

## What changed

- **`01_generate_toy_data.py`**: now generates raw, pre-AE feature vectors
  `(3000, 384)` per dataset — not latent-space blobs. Each dataset samples cluster
  structure in a 10-D "true biological state" space (elliptical, correlated
  covariances, 3 blur levels: mild/moderate/severe separation), then nonlinearly maps
  each point to a per-feature Poisson rate and samples counts, then applies
  `log1p` + per-feature min-max normalization — mirroring the real pipeline's
  `log_epi`/`log_microc` + normalize steps. 9 datasets total: 3 blur levels x
  n_centroid in {4, 6, 10}.
- **`02_evaluate_metric_weights.py`**: for each dataset, pretrains a real AE
  (`src/model/ae.py`'s `AE` class, same architecture/hyperparameters as
  `main.py:pretrain_ae` — 500-500-2000-10, Adam lr=0.002, batch 256, seed=0, 100
  epochs), saves the pretrained model + latent embedding for reuse, then fits GMM
  candidates on the **latent space** (`_fit_gmm_candidates` from
  `src/model/vade_model.py`) and computes ARI/NMI against the known ground truth
  there. Everything downstream (survivor filtering, oracle/naive/random baselines,
  the 7-way `metric_weights` sweep) is the original logic, unchanged, just operating
  on latent candidates instead of raw-data candidates.

## A real bug found and fixed along the way

The first version of `01`'s generator (smooth tanh-projection + additive Gaussian
noise + min-max normalize, producing dense, ~0.5-mean, symmetric-per-feature data)
caused the AE to **completely collapse**: all 10 latent bottleneck units stuck at
exactly zero, reconstruction loss frozen at `log(2)` (i.e., no learning at all), for
every one of the 9 datasets, regardless of AE pretrain seed (tested 0, 1, 2, 3, 42)
or epoch count (5 vs 100 — collapse happens within the first few epochs and never
recovers, since a fully dead ReLU unit has zero gradient forever).

I compared this against the real pipeline's actual pretraining input
(`data/02_process/merged_log_data.npy`, N=44828, D=384) and found it has a very
different profile: mean **0.05** (not 0.5), 68% of values below 0.05, ~4.75% exact
zeros — i.e. real biological signal is sparse and right-skewed, not dense and
symmetric. Reproducing that shape (via a Poisson-count generative process instead of
Gaussian-noise-on-a-smooth-manifold) was what fixed the collapse. I also ruled out
weight initialization as the primary cause — swapping in He-normal init on top of the
old symmetric data did **not** fix it, confirming the data distribution itself was the
driver, not just an unlucky seed.

The fix is not complete, though — see limitations below.

## Final results (9 datasets, 10 GMM candidates each = 90 candidates pooled)

**Spearman correlation of each raw metric with ARI:**

| metric | rho | p-value |
|---|---|---|
| assignment_confidence | **0.446** | 1.1e-05 |
| separation | 0.340 | 1.0e-03 |
| min_weight | 0.262 | 1.3e-02 |
| log_likelihood | -0.154 | 0.147 (not significant) |
| cov_health | **NaN** | NaN — see limitation below |

**Mean ARI by selection strategy (averaged over the 9 datasets):**

| rank | strategy | mean ARI |
|---|---|---|
| 1 | oracle (upper bound) | 0.219 |
| 2 | confidence_only | 0.214 |
| 3 | confidence_loglik | 0.214 |
| 4 | naive_loglik (sklearn default behavior) | 0.212 |
| 5 | loglik_only | 0.212 |
| 6 | random_pick (lower bound) | 0.210 |
| 7 | equal_current_default (select_balanced_gmm's actual default) | 0.207 |
| 8 | separation_confidence | 0.206 |
| 9 | separation_loglik | 0.205 |
| 10 | separation_only | 0.205 |

Full report/plots: `results/metric_selection_report.pdf`,
`results/metric_selection_summary.csv`. Raw JSON: `results/*.json`.

## Limitation to know about before reading too much into the ranking above

Even with the Poisson-count fix, the AE's 10-D latent space is still typically
**70-90% dead** (dims collapsed to ~zero variance) on this toy-scale data (N=3000,
vs. ~45k in the real pipeline — more samples per epoch means more stable gradients
and less risk of a unit dying early). Two visible symptoms:

1. **`cov_health` is degenerate** — pinned at `0.9999990000010001` for essentially
   all 90 candidates (std ~1e-16), because with most dims dead, nearly every
   component's per-dim variance sits at the `reg_covar=1e-4` floor everywhere, so
   `min/median` of that floor value is trivially ~1 regardless of the fit. Spearman
   correlation against a constant is undefined (`NaN`) — this is a real property of
   this run's degenerate latent space, not a bug in `02`'s computation. (`03`'s plots
   still render correctly; matplotlib just prints a harmless "posx and posy should be
   finite values" warning when placing the label for that one bar.)
2. **The 10 GMM candidates within a dataset are unusually similar to each other** —
   oracle (0.219) and random_pick (0.210) are barely 0.01 apart, and every
   weight-combo strategy clusters within that same narrow band. With so few "alive"
   informative dimensions, different random seeds' k-means inits tend to converge to
   similar solutions, leaving little for the composite-score weighting to actually
   discriminate between.

So: the **relative ranking** of `assignment_confidence` and `separation` as the
strongest ARI predictors, and `log_likelihood` alone as a poor one, is a real and
plausible signal (consistent with why `select_balanced_gmm` was written in the first
place) — but the **absolute mean-ARI gaps** between strategies are compressed by the
limited candidate diversity, and `cov_health` didn't get a fair test here at all.
If you want a tighter read on `cov_health` specifically, or larger spreads between
strategies, the most direct lever is increasing `N_SAMPLES` in `01` (more data per
AE pretrain epoch → fewer dead units) — I didn't push on this further since it starts
trading off toy-script runtime, and wanted to leave that call to you rather than
keep tuning unilaterally.

## Files produced

- `toy_data/*.npz` + `manifest.json` — 9 raw biological-like datasets (regenerated).
- `pretrained/<dataset>/` — 9 pretrained AE SavedModel directories (reusable).
- `pretrained/<dataset>_latent.npz` — 9 latent embeddings (`z`, `y_true`) — reload
  these directly if you want to re-run GMM experiments without repeating AE pretraining.
- `results/per_dataset_summary.json`, `results/per_candidate_metrics.json`,
  `results/correlations.json`, `results/strategy_mean_ari.json` — same schema as
  before, so `03_plot_metric_selection.py` needed no changes.
- `results/metric_selection_report.pdf`, `results/metric_selection_summary.csv`.

## Verification performed

- Reran `01` → 9/9 datasets generated, correct shapes `(3000, 384)`, balanced labels.
- Reran `02` end-to-end → all 9 AE pretrains completed, all pretrained
  models/latents/results written.
- Spot-checked one dataset (`mild_k4`): reloaded `pretrained/mild_k4_latent.npz`,
  refit the `random_state=0` GMM candidate independently, got ARI matching the
  recorded value exactly (`0.502781992082551` both times).
- Reran `03` unmodified against the new JSON → completed without error, confirming
  schema compatibility (one harmless matplotlib warning, explained above).

## Suggested next step (not done — flagging for your call)

If you want `cov_health` to actually get exercised and the strategy gaps to widen,
try bumping `N_SAMPLES` in `01_generate_toy_data.py` (e.g. 3000 → 8000-10000) and
rerunning `01` + `02`. That will roughly proportionally increase `02`'s pretraining
time (currently ~10-15 min for the full 9-dataset sweep). I didn't do this myself
since it's a real runtime/thoroughness tradeoff you might want to weigh in on.

## 2026-08-26: N_SAMPLES bump to 10000, and a bigger problem found along the way

Followed up on the "suggested next step" above: bumped `N_SAMPLES` 3000 → 10000 in
`01_generate_toy_data.py`, backed up the original N=3000 `toy_data/`, `pretrained/`,
`results/` to `n3000_backup/`, and reran `01` → `02` → `03` as SLURM job `15516809`
(`run_bump_samples.sbatch`, `scc-gpu` partition, A100). Job completed in ~24 min.
Results archived to `results/n10000/` (see that folder's `README.md` for the
detailed file-by-file breakdown) — `results/` no longer holds a bare top-level run;
each run now gets its own named subfolder.

### Dead-unit hypothesis: partially confirmed

Spot-checked datasets (`mild_k4`, `severe_k10`) went from **8/10 dead latent dims at
N=3000 to 6/10 dead at N=10000**. More samples per pretrain epoch does reduce the
dead-unit problem, as hypothesized — but the latent space is still majority-dead,
not fixed. `cov_health` is still completely degenerate (pinned constant, `NaN`
correlation) as a direct consequence — still hasn't gotten a fair test.

### A bigger, separate problem: pooled correlations are confounded across datasets

While digging into *why* `log_likelihood` correlated so poorly with ARI (pooled
rho went from -0.154 at N=3000 to -0.244 at N=10000), I checked the Spearman
correlation **within each dataset separately** (i.e. only comparing the 10 GMM
candidates fit on the *same* dataset against each other — the actual comparison
`select_balanced_gmm` needs to make) instead of pooling all 90 candidates from 9
datasets into one correlation, which is what `02_evaluate_metric_weights.py`
had been doing. The two numbers disagree sharply:

| metric | pooled rho (N=10000) | mean within-dataset rho |
|---|---|---|
| min_weight | 0.102 (n.s.) | **+0.280** |
| separation | 0.826 | **+0.232** |
| assignment_confidence | 0.466 | **-0.136** |
| log_likelihood | -0.244 | **+0.016** |
| cov_health | NaN | NaN (untested) |

Two things follow from this:

1. **`log_likelihood`'s pooled -0.244 is not "higher likelihood predicts worse
   clustering."** Within a dataset, log-likelihood barely varies across the 10
   candidates at all (near-flat, e.g. `mild_k4`: 18.5 for every candidate) — mean
   within-dataset rho is ≈0.016, i.e. no real signal, positive or negative. The
   negative pooled number is an artifact of pooling: datasets with higher raw
   log-likelihood scale (driven by `n_centroid`/blur level) happen to have lower
   achievable ARI ceilings, and vice versa — a cross-dataset difficulty confound
   masquerading as an anti-correlation.
2. **`assignment_confidence` — the metric that looked like the strongest predictor
   in the original N=3000 report (rho=0.446, mean-ARI essentially tied with
   oracle) — has a *slightly negative* mean within-dataset rho (-0.136).** Within 5
   of 9 datasets, picking the higher-confidence candidate is on average very
   slightly *worse* for ARI, not better (`mild_k6`: -0.900; `severe_k6`: -0.960).
   The pooled positive correlation comes almost entirely from easy (mild-blur)
   datasets having both higher confidence values and higher ARI ceilings than hard
   (severe-blur) datasets — not from confidence actually discriminating good
   candidates from bad ones within a dataset.
3. `separation` is the only metric whose pooled correlation direction survives
   this check, though far weaker than the pooled 0.826 suggests (mean +0.232, and
   noisy — individual per-dataset rhos range from -0.64 to +0.96). `min_weight`
   was actually *understated* by pooling (+0.280 within-dataset vs. only 0.102
   pooled).

**This reframes both runs' headline conclusions.** `assignment_confidence`'s
apparent strength (in both the N=3000 and N=10000 reports) looks like it was
mostly/entirely a between-dataset difficulty confound, not real predictive power.
`separation` and possibly `min_weight` are the more credible signals, though both
are weak and dataset-dependent, and `cov_health` remains completely untested. This
is a more fundamental issue than the dead-unit/sample-size one — bumping
`N_SAMPLES` further won't fix a pooling artifact.

### Code changes made in response

- **`02_evaluate_metric_weights.py`**: added `compute_within_dataset_correlations`,
  which computes Spearman rho per dataset (only among that dataset's own
  candidates) and reports the mean/median across datasets alongside the existing
  pooled correlation. Saves `results/within_dataset_correlations.json` in addition
  to `correlations.json`; both are now printed at the end of a run, with the
  pooled one explicitly labeled as confounded. Also now captures each dataset's
  full per-epoch AE loss curve (`history.history['loss']`, previously only the
  final value was kept) and saves it to `results/training_curves.json`.
- **`03_plot_metric_selection.py`**: added a new report page
  (`plot_within_vs_pooled_page`) putting pooled and within-dataset rho side by
  side per metric, and a new page (`plot_training_curves_page`) plotting AE
  pretraining loss vs. epoch per dataset (rendered only if `training_curves.json`
  is present, so it degrades gracefully for older runs). `plot_summary_page`'s
  text page now lists both correlation rankings. Fixed two pre-existing hardcoded,
  stale counts in plot titles/labels (`"240 candidates"`, `"24 toy datasets"`
  — actual counts are 90 candidates / 9 datasets) to compute the real count instead.
- `results/n10000/within_dataset_correlations.json` and its
  `metric_selection_report.pdf` were regenerated using this new code, applied
  post hoc to the already-saved `per_candidate_metrics.json` — the expensive AE
  pretraining step was not rerun for this.
- No code changes needed to `01_generate_toy_data.py` beyond the `N_SAMPLES`
  constant itself.

### Aside: the SLURM job likely ran on CPU, not the requested GPU -- now fixed

While investigating this, found that TensorFlow 2.5.0 (pinned in the `loopbin`
venv) can't find CUDA 11.x runtime libraries on this cluster's GPU nodes (only
CUDA 12.8 is installed system-wide; there is no `cuda` or `nvhpc` environment
module on this cluster at all, unlike GWDG's separate "Grete" GPU cluster, which
does have one). So `job 15516809` almost certainly pretrained all 9 AEs on CPU
despite requesting an A100 — this doesn't affect correctness of the results above,
only means the ~24 min runtime doesn't reflect real GPU speed.

**Fix, now applied and verified end-to-end**: pip-installed CUDA 11 runtime
libraries (`nvidia-cuda-runtime-cu11`, `nvidia-cudnn-cu11==8.6.0.163`,
`nvidia-cublas-cu11`, `nvidia-cufft-cu11`, `nvidia-curand-cu11`,
`nvidia-cusolver-cu11`, `nvidia-cusparse-cu11` — the cuDNN version is pinned to an
8.x release since TF 2.5.0 needs cuDNN 8.1 and PyPI's earliest available is
8.5.0.96; 8.6.0.163 works) packaged as a single tarball,
`src/jobs/cuda11_runtime_libs.tar` (2.8GB). Referencing the many small `.so` files
directly on `ceph-hdd` doesn't work on this cluster — both a `pip install
--target=<ceph-hdd path>` and a plain `cp -r` of the extracted directory hung for
10+ minutes with negligible progress (this mount's throughput for many-small-file
payloads is apparently very poor, independent of which tool does the writing);
packing the same ~2.8GB into one tar file and copying that single file took
**2.2 seconds**. `run_bump_samples.sbatch` now extracts this tarball to the compute
node's local `/tmp` (fast, since it's local disk, not `ceph-hdd`) at job start and
points `LD_LIBRARY_PATH` at the extracted `nvidia/*/lib` directories before
running Python.

Verified via `srun` with the exact same extract+`LD_LIBRARY_PATH` logic added to
the sbatch script: `tf.config.list_physical_devices('GPU')` returns the A100 (79GB
visible), all CUDA libraries load cleanly (`libcudart`, `libcublas`, `libcublasLt`,
`libcufft`, `libcurand`, `libcusolver`, `libcusparse`, `libcudnn.so.8`), and a test
matmul actually runs on `/device:GPU:0`. Tarball extraction takes ~30s.

This fix is now wired into `run_bump_samples.sbatch` and
`src/jobs/test_balanced_gmm.sbatch` (the only other currently-functional,
GPU-requesting job script — it runs real AE pretraining via
`04_test_real_pretrain_gmm.py` on the production `merged_log_data.npy`).

## 2026-08-26: N_SAMPLES bump to 30000 (first real GPU-accelerated run)

Bumped `N_SAMPLES` 10000 → 30000 in `01_generate_toy_data.py`, backed up the
N=10000 `toy_data/`/`pretrained/`/`results/` to `n10000_backup/`, and reran
`run_bump_samples.sbatch` (now with the CUDA fix above wired in) as SLURM job
`15519814`. Results archived to `results/n30000/` (see its `README.md` for the
detailed breakdown).

**GPU fix confirmed working in a real production-shaped job, not just the
`srun` smoke test**: `libcudart.so.11.0` loaded successfully per the job's `.err`
log, and the job completed in **22 minutes despite 3x the data of the N=10000
run**, which took 24 minutes on CPU. That's a real, substantial speedup, not just
a passed check.

### Dead-unit fraction: diminishing returns

Spot-checked datasets went from 6/10 dead (N=10000) to 6/10 and 5/10 dead
(N=30000, `mild_k4`/`severe_k10` respectively) — tripling `N_SAMPLES` again
bought much less improvement than the first bump did. `cov_health` is partially
testable for the first time (4/9 datasets have non-constant values now, vs. 0/9
at both N=3000 and N=10000), with a negative mean within-dataset rho (-0.296,
n=4) — a new data point, but on a small non-random subset, so treat as
suggestive only. Also newly noted: 4 of 9 datasets (all the `k10`/`k6`-heavy
configs except `mild_k6`/`severe_k6`) now hit total survivor-filter wipeout
(`n_survivors=0/10`), more than at N=10000 — not chased down further here.

### The bigger finding: metric rankings don't replicate across sample sizes

| metric | mean within-dataset rho, N=10000 | mean within-dataset rho, N=30000 |
|---|---|---|
| min_weight | +0.280 | **+0.333** |
| separation | +0.232 | **+0.015** |
| assignment_confidence | -0.136 | -0.122 |
| log_likelihood | +0.016 | -0.133 |
| cov_health | NaN | -0.296 (4/9 datasets only) |

`min_weight` is the only metric that stays consistently positive and gets
*stronger* as `N_SAMPLES` increases — across both runs where it's measurable.
Every other metric is unstable: most strikingly, `separation` — the standout
"strongest predictor" in the N=10000 report, both pooled (0.826) and
within-dataset (+0.232) — collapses to essentially zero within-dataset signal
(+0.015) at N=30000, and its *pooled* correlation actually flips sign entirely
(0.826 → -0.189). That pooled flip is exactly the kind of between-dataset-scale
instability the within-dataset analysis (added after the N=10000 run) was meant
to catch — except here it shows the within-dataset number isn't fully stable
either, just less wildly so. `assignment_confidence` and `log_likelihood` stay
uninformative-to-weakly-negative in both runs, consistent with the N=10000
finding.

**Practical takeaway so far**: `min_weight` is the one metric with a
reproducible, strengthening signal across two independent runs. Nothing else has
earned trust yet — `separation` looked good once and stopped looking good;
`cov_health` has barely been tested at all across three runs combined. If
`select_balanced_gmm`'s default weighting is revisited before this stabilizes
further, weighting toward `min_weight` (not `assignment_confidence`, the
original default's other main input) is the best-supported change from the data
collected so far — but "three runs, one of five metrics behaving consistently"
is not yet a strong enough basis to commit to a specific weighting scheme.

### Aside: applying the same GPU fix elsewhere

The older GPU job scripts under `src/jobs/` (`03_train_ae.sbatch`,
`041_save_model_each_200_epochs.sbatch`, `04_temp.sbatch`, `04_train_vade.sbatch`,
`05_predict_clusters.sbatch`, `05_predict_each_200_epochs.sbatch`, `06_test.sbatch`,
`06_train_vade_with_test.sbatch`, `07_calculate_generaliability.sbatch`,
`07_temp.sbatch`, `07_temp2.sbatch`) were **not** touched. They're already
non-functional for reasons unrelated to CUDA: they request partition `gpu`, which
no longer exists on this cluster (`sinfo -p gpu` returns no nodes), point at a
stale path (`/usr/users/yzhu1/LoopBin/...` instead of the current
`/mnt/ceph-hdd/.../Yajie/LoopBin`), and never activate the `loopbin` venv or load
any module — leftovers from before some earlier cluster/environment migration.
Applying the CUDA-libs fix to them wouldn't make them runnable; fixing the
partition/path/venv issues first is a separate, bigger task not attempted here.
`test_gpu_gmm.sbatch` is an empty stub (no actual work) and `test_gmm.sbatch`
doesn't request a GPU at all, so neither needed this fix.

## 2026-08-27: the dead latent dims were dying ReLU, not too little data

Three runs of `N_SAMPLES` scaling (3000 → 10000 → 30000) moved the dead-dim count
from 8/10 to ~5.6/10 and were clearly hitting diminishing returns. The cause turned
out not to be sample size at all.

`src/model/ae.py`'s bottleneck was `Dense(10, activation='relu')`. A ReLU unit whose
pre-activation is negative for every sample outputs exactly 0 and receives exactly 0
gradient — permanently, with no path back. In a 500- or 2000-unit hidden layer that's
survivable; in a 10-unit bottleneck it's the whole problem. (This also explains the
N=3000 total collapse documented above: same mechanism, all 10 units at once.)

Two further costs, beyond the dead units:

- **It fights the GMM fitted on it.** ReLU confines the latent to the non-negative
  orthant with a mass spike at exactly 0 per axis. Dims at that spike have near-zero
  variance, so every component's variance sits at the `reg_covar=1e-4` floor, so
  `cov_health` = `min/median` of a constant ≈ 1 — which is precisely why `cov_health`
  was degenerate and untestable for three runs. It was never a toy-data artifact.
- **It's inconsistent with VaDE's own encoder.** `vade_model.py:build_encoder` uses a
  *linear* `z_mean`, and `load_pretrained_weights` copies the AE's ReLU bottleneck
  weights straight into it (`vade_model.py:267`). The GMM centers are fit on a
  non-negative latent, then initialized into a space where `z_mean` can go negative.

### Change made

`AE.__init__` now takes `latent_activation`, **defaulting to `'relu'`** — production
`main.py:pretrain_ae` and `04_test_real_pretrain_gmm.py` are untouched, and every
model under `trials/` stays reproducible. `02_evaluate_metric_weights.py` opts in with
`LATENT_ACTIVATION = None`. This was kept as an opt-in A/B rather than a production
flip on purpose; flipping the default is a separate, one-line decision.

Also added, per the "should we monitor latent correlation too?" question: `02` now
computes `compute_latent_diagnostics` → `results/latent_diagnostics.json`, and `03`
renders it as the report's first page. It tracks alive dims, **effective rank**
(participation ratio of `cov(z)`'s eigenvalues — a continuous "how many dimensions is
this really" that, unlike the dead-dim count, also penalizes alive-but-redundant
dims), and **within-cluster** vs. global dimension correlation. Within-cluster is the
one that matters: `covariance_type='diag'` and VaDE's `GMM` layer (`lambda_p` is
per-dim, not a matrix) both assume dims are uncorrelated *within* a component. Global
correlation is expected to be high whenever clusters separate along a diagonal, so
testing against it would flag healthy latent spaces as broken.

### Result: SLURM job `15546947`, 13 min, results in `results/n30000_linear/`

Controlled A/B — `01` was not rerun, so both runs see byte-identical `toy_data/`; only
the activation differs. The ReLU AEs/latents were moved to `pretrained_n30000_relu/`,
and `results/n30000/latent_diagnostics.json` was backfilled post hoc from them (no
retraining) so both sides are measured identically.

| | ReLU | linear |
|---|---|---|
| alive latent dims | 4.4 / 10 | **10.0 / 10** (9/9 datasets) |
| effective rank | 2.29 | **9.39** |
| within-cluster mean \|r\| | 0.356 | **0.086** |
| global mean \|r\| | 0.365 | 0.043 |
| AE reconstruction loss | 0.2952 | **0.2813** (better on 9/9) |
| datasets with 0/10 GMM survivors | 4 / 9 | **0 / 9** |
| datasets where `cov_health` is measurable | 4 / 9 | **9 / 9** |
| oracle mean ARI | 0.1237 | **0.1414** |

One activation did what tripling the data twice could not. Worth noting the
correlation result specifically: under ReLU, within-cluster |r| (0.356) and global
|r| (0.365) were *equal*, meaning the correlation was intrinsic to each component —
the exact case a diagonal covariance cannot represent, and a sign the GMM had been
splitting tilted clusters into multiple axis-aligned ones. Under linear both collapse
toward zero, so `covariance_type='diag'` is now a well-specified choice rather than a
silently violated assumption. Effective rank also shows the dead-dim count had been
*overstating* latent capacity roughly 2x under ReLU (4.4 alive dims but only 2.29
effective) — the surviving dims were about half redundant with each other.

### The metric-selection finding this exposes

Every strategy's ARI improved, but `random_pick` (0.1301) still beats every actual
selection strategy, the best being `loglik_only`/`naive_loglik` at 0.1274. Since all
9 datasets now have 10/10 survivors, this is the first run where the composite scoring
is exercised on its intended path rather than the no-survivor fallback — and it loses
to picking at random.

Within-dataset mean rho vs. ARI across all three measurable runs:

| metric | N=10000 | N=30000 ReLU | N=30000 linear |
|---|---|---|---|
| min_weight | +0.280 | +0.333 | **+0.289** |
| log_likelihood | +0.016 | -0.133 | +0.036 |
| cov_health | NaN | -0.296 (4/9) | -0.068 (9/9), median -0.370 |
| separation | +0.232 | +0.015 | -0.179 |
| assignment_confidence | -0.136 | -0.122 | -0.223 |

**`min_weight` is the only metric positive in all three runs, and it is the one metric
`metric_weights` cannot express.** `select_balanced_gmm`'s `metric_weights` tuple is
`(separation, assignment_confidence, log_likelihood)` (`vade_model.py:163`) —
`min_weight` appears only as a hard survivor filter (line 150) and a fallback tiebreak
(line 172), never as a ranking term. So the composite ranks candidates using the three
metrics with no reliable signal, while the one metric that does predict ARI is
structurally excluded from ranking. That is a plausible mechanical explanation for why
every weight combo loses to random, and it is testable without touching production:
add `min_weight` as a fourth weighted term and rerun the sweep.

`cov_health` finally got its fair test (9/9 datasets, three runs after first being
flagged as untestable) and does not earn its place: mean -0.068, median -0.370.
`separation` has now gone +0.232 → +0.015 → -0.179 across three runs — no stable
signal, and its apparent strength in the N=10000 report should be treated as dead.

### Not done — flagging for your call

1. **Flipping `ae.py`'s default to linear.** The toy evidence is strong and it also
   makes AE pretraining consistent with VaDE's linear `z_mean`, but it changes the
   production model and would shift published cluster assignments. Worth confirming
   on the real `merged_log_data.npy` latent space first — `04_test_real_pretrain_gmm.py`
   already pretrains a real AE and would show whether production is majority-dead too.
   That has never been checked, and it is the more consequential question: if it is,
   the published clusters were computed in an effectively ~2-D space.
2. **Adding `min_weight` to `metric_weights`.** Cheap to test in this harness, and the
   best-supported change from three runs of data.
3. **`covariance_type='full'`** is no longer obviously needed — the linear bottleneck
   fixed the within-cluster correlation that motivated it. Reconsider only if the real
   latent space shows high within-cluster |r| after the activation change.

## 2026-08-27: confirmed on the REAL data — production's latent space is majority-dead

Ran item 1 above. SLURM job `15548074`, 41 min, results in `results/real_latent_relu/`
and `results/real_latent_linear/`.

Both arms replicate `main.py:pretrain_ae` **exactly** — full `merged_log_data.npy`
(44828 x 384, no subsample), 200 epochs, batch 256, Adam lr=0.002, seed 0,
`num_clusters=10` to match `trials/vade_10clusters_merged_control_degron_rep1_cov_dia_run1`.
The only difference between arms is the bottleneck activation, so the `relu` arm is a
faithful reproduction of the model the published clusters came from, not a proxy.

To make this runnable, `04_test_real_pretrain_gmm.py` gained `--latent-activation` and
a `--n-samples 0` = "use everything" mode (it previously *always* subsampled, so it
could not replicate production), and `compute_latent_diagnostics` moved into a shared
`latent_diagnostics.py` so `02` and `04` measure identically rather than via a copy.

| | relu (= production) | linear |
|---|---|---|
| alive latent dims | **5 / 10** | **10 / 10** |
| effective rank | **2.93** | **9.42** |
| per-dim variance | `.77 .67 .67 .65 .60 0 0 0 0 0` | `.20 .16 .15 .14 .14 .14 .13 .13 .13 .11` |
| AE reconstruction loss | 0.1811 | **0.1779** |
| GMM candidates below the 0.02 `min_weight` floor | **4 / 10** | **0 / 10** |
| `min_weight` range across candidates | 0.012 - 0.032 | 0.036 - 0.057 |
| within-cluster mean \|r\| | 0.078 | 0.070 |
| global mean \|r\| | 0.406 | 0.049 |

**Answer to the question: yes.** Five of the ten latent dimensions are dead — variance
exactly 0.0000, not merely small — in the real production pretrain. Effective rank is
2.93. Every clustering result under `trials/` was produced by fitting a 10-component
GMM in what is effectively a **~3-dimensional** space. This was never measured before;
the toy harness had only ever been a proxy for it.

The linear bottleneck fixes it on real data exactly as it did on toy data, and again
with *better* reconstruction loss, so this is not a capacity/quality tradeoff.

### Correction: the within-cluster correlation problem does NOT transfer to real data

This is the one place the toy harness misled. On toy data, within-cluster |r| (0.356)
and global |r| (0.365) were equal, which said the correlation was intrinsic to each
component and `covariance_type='diag'` was misspecified. **On real data that is not
the case**: within |r| is 0.078 against a global |r| of 0.406 — global sits far above
within, which is the healthy pattern (clusters separating along diagonal directions).

So `covariance_type='diag'` is a sound choice for the real pipeline, and item 3 above
(`covariance_type='full'`) should be considered closed, not deferred. The toy
correlation finding was an artifact of `01_generate_toy_data.py` deliberately sampling
"elliptical, correlated covariances" — a property of the generator, not of chromatin
loop data. Worth remembering when reading anything else the toy harness produces.

(Caveat as designed: real data has no ground truth, so the within-cluster statistic is
grouped by the selected GMM's own hard assignments, which makes it optimistic. But
0.078 is low enough, and the global/within gap wide enough, that the conclusion holds.)

### The dead dims were causing the cluster imbalance select_balanced_gmm exists to fight

Under relu, 4 of 10 GMM candidates fall below the `min_weight >= 0.02` survivor floor,
and the best candidate reaches only 0.032 — against 0.1 for a perfectly balanced
10-cluster split. Under linear, **0 of 10** fall below the floor and the range rises to
0.036-0.057.

That is a plausible causal chain worth stating plainly: `select_balanced_gmm` was
written to work around GMM fits that produce one tight component plus several
near-empty ones. A latent space with ~3 effective dimensions being asked to support 10
Gaussian components is a direct cause of exactly that degeneracy. Fixing the activation
attacks the root cause; `select_balanced_gmm` treats the symptom. This also explains
why `min_weight` kept coming out as the only ARI-predictive metric across three toy
runs — it was the one metric measuring the actual pathology.

Also of note (weak evidence, no ground truth): the balanced pick agreed with the other
candidates more than the naive best-log-likelihood pick did, in both arms
(0.711 vs 0.613 relu; 0.676 vs 0.552 linear). That is candidate *agreement*, not
correctness — a more consensus-like partition is not necessarily a better one — so it
should not be read as validating the composite score.

### Recommended next step

Flipping `src/model/ae.py`'s default to linear is now supported on the real data, not
just on toy data: more usable dimensions, better reconstruction, better-balanced GMM
candidates, and consistency with VaDE's already-linear `z_mean`. It remains **not
done** — it changes the production model and would shift every published cluster
assignment, so it needs an explicit decision plus a rerun of the affected `trials/`.

The honest framing for that decision: this is not a tuning tweak, it is a bug fix. The
current model discards half its latent capacity before the GMM ever sees it.

## 2026-09-03: pi_mode (theta_p handling) comparison — settling on `uniform_fixed`

**Question:** `main.py -f train` supports four ways of handling the GMM prior's mixture
weights `theta_p` during VaDE training (`-pi_mode`): `uniform_fixed` (frozen at `1/k`,
the original default), `gmm_fixed` (frozen at the pretrained GMM's own fitted weights),
`em` (frozen w.r.t. gradients, updated by a periodic EM M-step), and `gradient`
(trained by backprop). Which one produces stable, well-populated clusters?

**`gradient` was already known to fail** (`src/jobs/04_train_vade.sbatch` baseline,
seed 48, job `15615554` arm 2): `theta_p` trained directly by backprop off the ELBO
collapses to 2 of 10 clusters (`vade_..._pi_trained_run1`, 62%/38% split) — component
starvation, a known failure mode of letting mixture weights receive gradient directly
in a joint VAE+GMM objective. `uniform_fixed` on the *same* pretrained AE and the *same
selected prior* (job `15615554` arm 1) stayed at a healthy 9 clusters. This is why
`gradient` was never seriously considered further; the open question was whether
`gmm_fixed`/`em` — which don't backprop through `theta_p` — would do better.

### First attempt (job `15665592`, 2026-08-31) was confounded, not a real comparison

`gmm_fixed` and `em` were run against a GMM prior re-selected on that job's own
hardware (`-mw 1,0,0`, separation-only — **not** `1,1,0`; despite discussion of adding
a confidence term, no run ever actually used it, confirmed by grepping every `.out`
log's `GMM metric_weights` line). Because `encoder.predict` differs by more than fp32
rounding across machines, this job's candidate search picked `random_state=13`
(separation 1.3466) where the earlier CPU-fallback job `15615554` had picked
`random_state=73` (separation 1.3651) from nominally the same setup. Both `gmm_fixed`
(3 seeds) and `em` (3 seeds) collapsed badly on the rs=13 prior — `gmm_fixed` to 2-ish
dominant clusters (84.6%/86.5%/87.4% in the largest), `em` to a single cluster
(100%) in all 3 seeds — but since the prior itself also differed from the healthy
baseline, this didn't cleanly separate "bad prior" from "bad pi-handling."

**Checked whether it actually was a bad prior — it wasn't.** Loaded both pickled GMMs
(`.../pi_frozen_run1/gmm_models/*.pkl` = rs73, `.../pi_gmminit_fixed_run1/gmm_models/*.pkl`
= rs13) and did an optimal 1:1 component match (Hungarian algorithm on center distance).
Every matched pair sits 0.005-0.08 pooled-stds apart, against ~1.3-4.6 stds between
genuinely distinct components in either GMM, and matched weights agree to within 0.002.
**rs=13 and rs=73 are the same 10-way partition of the data, just permuted** — the 1.4%
separation-score gap is CPU/GPU forward-pass noise, not a real quality difference. Both
were already the argmax of their own 10-candidate pool. So "search more seeds and pick
the best separation" was ruled out as a fix before it was tried: two independent
10-seed searches on two machines already converged on the same structure.

### Clean test (job `15729713`, 2026-09-03): one pinned prior, three pi_modes, three seeds

To isolate `theta_p` handling from prior selection, `04_train_vade_pinned_prior.sbatch`
pins **one** GMM fit (`data/03_prior/relu_rs73/gmm_prior.pkl`, the rs=73 candidate
copied out of job `15615554`'s `gmm_models/`, not refit) across all 9 array tasks via
the new `-gmm_prior <path>` flag (`main.py`, `VADE.load_pretrained_weights`), so
`u_p`/`lambda_p` are identical in every arm and only `theta_p` handling varies.

| pi_mode | run1 (seed 48) | run2 (seed 73) | run3 (seed 101) |
|---|---|---|---|
| `uniform_fixed` | 9 clusters, 0.2-20.7% each | 9 clusters, 0.0-22.4% each | 9 clusters, 0.5-19.7% each |
| `gmm_fixed` | 5 clusters, 71.8% dominant | 6 clusters, 64.2% dominant | 2 clusters, 88.5% dominant |
| `em` | 1 cluster, 100% | 1 cluster, 100% | 1 cluster, 100% |

`uniform_fixed` reproduces the original healthy baseline shape in all 3 seeds.
`gmm_fixed` and `em` collapse in all 3 seeds, on a prior that is structurally identical
to the one `uniform_fixed` handles fine — **conclusively ruling out prior quality as the
cause.** The fix belongs in `pi_mode`, not in GMM candidate selection.

**`em`'s collapse is deterministic and fast**, confirmed from `theta_history.npy`
(logged every 10 epochs post-warmup) — all 3 seeds cross the same thresholds within a
few epochs of each other, and all 3 converge onto the *same* winning component (index
3, the one with the largest initial weight, 0.1765, in the pinned prior):

| threshold | epoch |
|---|---|
| dominant component > 50% | 70 (2 EM steps after the 50-epoch warmup) |
| dominant component > 90% | 120-130 |
| only 1 component alive (rest < 1%) | 140 |
| dominant component > 99% | 290 |
| dominant component > 99.9% (≈ final 100%) | 360 (floor schedule fully ramped off by 300) |

Mechanism: the periodic M-step (`pi <- 0.7*pi + 0.3*mean(gamma)`) is a positive-feedback
loop — whichever component starts with the largest prior weight gets more average
responsibility, which raises its EM-updated weight further, compounding every 10
epochs. It is a function of the *initial* weight vector, independent of training seed
(all 3 seeds picked the same winner) and independent of which structurally-equivalent
prior you hand it (rs=13's run picked its own largest-initial-weight component too).

**`gmm_fixed`'s collapse epoch could not be determined** — `theta_p` never moves in this
mode (no `theta_history` is even written), and the array job's checkpoint schedule
(`CKPT` in `04_train_vade_pinned_prior.sbatch`) only saves intermediate models for the
`em` arm, so there's no saved state between epoch 0 (the frozen prior weights) and
epoch 500 (final labels) to inspect. The aggregate loss curve doesn't help either —
`reconstruction_loss`/`kl_loss` in `trials/vade_pin73_15729713_4.out` flatten out by
epoch ~2 and stay flat (±0.1) for the rest of the run, with no visible inflection at
the point where cluster assignments must be drifting. The mechanism is presumably the
same rich-get-richer dynamic acting through the encoder/responsibilities instead of
through `theta_p` directly (a frozen non-uniform `theta_p` still weights the KL term's
`log(pi_c)` asymmetrically per component, biasing the encoder toward the
already-larger ones) but this is inference, not measurement. Pinning down the epoch
would need `gmm_fixed` checkpointed the same way `em` is and rerun — not done, since
the decision below made it moot.

### Decision

**Going forward, `-pi_mode uniform_fixed` (the original default) is the one in use.**
`gmm_fixed` and `em` are not being pursued further for now — both reproduce component
collapse reliably across seeds and priors, and a real fix (e.g. damping `em`'s `rho`
much harder or flattening the frozen prior toward uniform before use in `gmm_fixed`)
was not attempted. If a data-informed (non-uniform) `theta_p` is wanted again later,
start from that flattening idea rather than re-running the current `em`/`gmm_fixed` as
implemented — they are confirmed, not suspected, to collapse.

## 2026-09-03: n_init in GMM candidate fitting — raising it hurts, not helps

**Question:** `_fit_gmm_candidates` (`src/model/vade_model.py`) fits each of its 10
outer `random_states` with `n_init=1`. Would raising `n_init` (sklearn re-runs EM from
that many k-means starts per call and keeps only the best-by-log-likelihood one)
improve model selection?

**Tested directly**, no retraining needed — pure `sklearn`, refitting the same 10
`random_states` at `n_init` in `{1, 5, 10}` on the real production latent
(`scripts/gmm_metric_validation/results/real_latent_relu/latent.npz`, the relu-AE
latent the rs=73/rs=13 priors above came from; `data/03_prior/relu_rs73/z.npy` itself
turned out to be truncated — 333,792 of the expected 448,280 floats — so this
equivalent full latent was used instead):

| n_init | min_weight mean (range) | below 0.02 floor | separation mean (range) | log-likelihood mean, std | distinct ll values / 10 seeds |
|---|---|---|---|---|---|
| 1 | 0.0223 (0.0121-**0.0322**) | 4/10 | 1.2814 (1.126-**1.365**) | 14.2153, std 0.0053 | 2 |
| 5 | 0.0209 (0.0166-0.0249) | 5/10 | 1.2913 (1.247-1.334) | 14.2217, std 0.0016 | 1 |
| 10 | 0.0210 (0.0178-0.0249) | 5/10 | 1.2928 (1.255-1.334) | 14.2221, std 0.0013 | 1 |

**Answer: no, `n_init=1` should stay.** Two findings:

1. There is almost nothing to gain on log-likelihood — even at `n_init=1` the 10
   outer seeds already span only 2 distinct likelihood values (std 0.005 out of
   ~14.2); `n_init=5` moves the mean by 0.05% relative and `n_init=10` moves it
   nothing further. The space is already saturated.
2. Raising `n_init` collapses candidate diversity and makes the pool *worse* on the
   metrics actually used for selection. At `n_init>=5` all 10 outer seeds converge to
   the same local optimum (1 distinct ll value) — every candidate becomes a
   near-copy of the others — and that shared optimum is worse than what a lucky
   single-init seed finds: best `min_weight` in the pool drops from 0.0322 (rs=101,
   `n_init=1`) to 0.0249, best separation drops from 1.3651 (rs=73) to 1.3339, and
   candidates failing the `min_weight >= 0.02` survivor floor rises from 4/10 to 5/10.

Mechanism: `n_init>1` reruns EM from multiple k-means starts within a single call and
keeps only the highest-log-likelihood run — pure log-likelihood optimization pressure
before balance/separation ever get a look. That is exactly what `select_balanced_gmm`
exists to counteract (see its docstring: sklearn's single best-by-log-likelihood pick
"tends to favor one tight component plus several near-empty ones"), and it lines up
with the 2026-08-27 entry's finding that this latent is only ~3 effective dimensions
(5/10 alive dims, confirmed again here) asked to hold 10 components — a low-rank space
has a narrow, easily-reached likelihood optimum, and pushing every seed harder toward
it just walks them all into the same over-concentrated basin instead of surfacing the
more balanced structures a diverse candidate pool can find.

If within-seed k-means noise is a concern, the better lever is more/different *outer*
`random_states` (cheap — extend the tuple), which preserves cross-candidate diversity,
rather than `n_init`, which collapses it.

**Decision: `n_init=1` stays as the default.** `_fit_gmm_candidates`'s docstring
updated with a pointer to this entry so the reasoning isn't re-litigated later.

## 2026-09-03: direct behavioral confirmation — rs=13 is a fine prior under uniform_fixed

Follow-up to the pi_mode entry above, which argued (via component matching) that rs=13
and rs=73 are structurally the same prior and the earlier collapse was pi_mode, not
prior quality. Direct test: `src/jobs/05_train_vade_rs13_uniform_check.sbatch` (job
`15730705`) takes the exact rs=13 prior pickle that produced 5/6/2-dominant-cluster
collapse under `gmm_fixed` (`data/03_prior/relu_rs13/gmm_prior.pkl`, copied verbatim
from `.../pi_gmminit_fixed_run1/gmm_models/`) and trains it under `pi_mode=uniform_fixed`,
seeds 48/73/101.

**Result: balanced clusters in all 3 seeds** (10/8/9 clusters, largest fraction
10.6-20.5%, no dominant component) — confirms the collapse was pi_mode, not the prior.

Pairwise NMI/AMI/ARI between the 3 seeds:

| pair | NMI | AMI | ARI |
|---|---|---|---|
| run1 vs run2 | 0.599 | 0.598 | 0.478 |
| run1 vs run3 | 0.716 | 0.716 | 0.659 |
| run2 vs run3 | 0.637 | 0.637 | 0.553 |

For context, the rs=73 pinned-prior `uniform_fixed` runs (job `15729713`) show the same
spread (NMI 0.568-0.648 pairwise) — so rs=13's run-to-run agreement is unremarkable,
just normal VaDE seed-to-seed variation (different encoder init + batch order) on top
of a shared prior, not evidence of anything wrong with rs=13.

Cross-prior, same seed (rs13 vs rs73): NMI 0.628-0.732, ARI 0.556-0.747 — **as high as
or higher than** the within-prior across-seed numbers above. Swapping which of the two
structurally-equivalent priors you train changes the result about as much as just
changing the seed does. Independent, behavioral confirmation (not just static
component-matching) that separation-only selection is picking sound, interchangeable
priors, and that the 2026-08-31 collapse was entirely a `pi_mode` problem.

## 2026-09-07: a component can go permanently empty even under uniform_fixed — the GMM prior is fully frozen after init

Surfaced while verifying the `loopbin/` package integration end-to-end: job `15773343`
(`loopbin/jobs/experiments/06_train_vade_k6_k7.sbatch`, `-mw 1,0,0`, `pi_mode=uniform_fixed`,
seed 48, fresh GMM candidate selection per k — the pinned rs=73/rs=13 10-component
priors above don't apply at k=6/7) trained cleanly (500 epochs, stable converging loss,
exit 0) but `all_clusters.pdf` showed only 5 clusters for the k=6 run and 6 for the
k=7 run.

**Not a training failure or a plotting bug.** `loopbin/cli.py`'s `cluster_data_inner_func`
takes `cluster = argmax(gmm(z_mean), axis=1)` and intentionally drops any GMM component
that never wins the argmax for a single loop, renumbering what's left to stay
contiguous (`# remove non-existing cluster` — deliberate, to avoid confusing users with
gapped cluster IDs). Checking the *raw*, pre-drop argmax counts over all requested
components confirmed one component captured exactly 0 of 44,828 loops in each run:
component 4 of 6 (k=6) and component 0 of 7 (k=7).

**Mechanism: under every `pi_mode`, `u_p` and `lambda_p` (the GMM prior's per-component
means/variances) are `trainable=False` — frozen at their `load_pretrained_weights`
initialization for the entire run.** Verified empirically: the initial sklearn
`GaussianMixture` fit (pickled to `gmm_models/`) and the final trained model's
`vade.gmm.u_p`/`lambda_p` are bit-identical — L2 movement is `0.000` for every
component in both the k=6 and k=7 runs. Under `uniform_fixed`, `theta_p` is frozen too
(`GMM.build`'s `trainable=self.trainable_theta`, only `True` for `pi_mode='gradient'`).
So under `uniform_fixed` the entire GMM prior is static from the moment training starts
— only the encoder adapts, learning where to place each loop's `z` to jointly minimize
reconstruction loss and KL-to-this-fixed-prior.

The two dead components share a striking profile in their *initial* candidate fit —
before VaDE training even began:

| | weight | mean norm | covariance (diag mean) | rank among its siblings |
|---|---|---|---|---|
| k=6, component 4 | 0.2617 | 1.28 | 0.0334 | **largest** weight, **smallest** norm, **smallest** (tightest) covariance of all 6 |
| k=7, component 0 | 0.262 | 1.28 | 0.0335 | **largest** weight, **smallest** norm, **smallest** (tightest) covariance of all 7 |

In both runs — independently fit, different k, same `-mw 1,0,0` separation-only
selection — the doomed component is the single largest-weight candidate, but also by
far the tightest (3-30x smaller variance than every sibling) and closest to the latent
origin. That is exactly the "one tight component plus near-empty others" shape
`select_balanced_gmm` exists to steer away from (its own docstring's phrase for what
sklearn's default best-log-likelihood pick tends to produce) — except `-mw 1,0,0`
disables everything but the separation metric, so a component can still be tight enough
to score well on separation alone while being a bad target once the prior can no longer
move to accommodate it. Likely explanation: a narrow, fixed-position component only
earns KL benefit for the encoder in a very precise region of latent space; here, it was
evidently cheaper for the encoder to never place any of 44,828 loops that precisely and
let the broader neighboring components absorb everything instead.

This also explains a follow-on question about cross-condition consistency: since the
trained model (encoder + fully frozen prior) is identical every time it's applied,
predicting on different data subsets (e.g. `loopbin cluster` run separately per
condition) should keep routing away from the *same* dead component rather than a
different one each time — consistent with the user's own testing, where the same
cluster is missing between the control and degron subsets. The renumbering is
per-invocation, though, so the saved cluster ID for a given loop no longer necessarily
equals its GMM component index in `gmm_models/`'s saved `theta_p`/`u_p`/`lambda_p` —
harmless for the common case (nobody maps cluster IDs back to raw component index) but
worth knowing before ever doing so.

**Action taken:** `cluster_data_inner_func` now logs a warning whenever this happens —
which raw component(s) were dropped and the resulting saved-label -> original-component
map — instead of silently renumbering. The renumbering behavior itself is intentional
(kept as-is, not reverted) so cluster IDs stay contiguous for users. Not yet
investigated: whether this "tight, near-origin, high-weight" candidate shape is
specific to `-mw 1,0,0` at these untested k values (6, 7 — the validated k=10 runs
above never hit this), or would also show up with the combined metric weighting.

## 2026-09-07: trainable u_p/lambda_p reproducibly collapses to one cluster — confirms why they're frozen

Follow-up to the entry above. The user recalled that an earlier test with `u_p`/
`lambda_p` (the GMM prior's per-component means/variances) set trainable, instead of
the current `trainable=False` default, produced "one big cluster" -- asked to confirm
or refute this empirically rather than take it on memory alone.

Added a diagnostic-only `trainable_prior` kwarg to `GMM`/`VADE` (default `False`, not
wired to any CLI flag -- see `vade_model.py`) and ran a controlled A/B:
`scripts/gmm_metric_validation/07_test_trainable_prior.py`
(`loopbin/jobs/experiments/07_test_trainable_prior.sbatch`, job `15775433`). Both arms
share everything except `trainable_prior` -- same pinned k=6 GMM prior (the
`pi_uniform_run1` pickle from the 06_train_vade_k6_k7.sbatch verification run above),
same pretrained AE, same seed (48), same 300 epochs, `pi_mode` fixed at
`uniform_fixed` (`theta_p` frozen either way) -- isolating prior-parameter
trainability as the only variable. A callback recorded the raw (pre-drop) argmax
cluster distribution every 20 epochs.

**Result: confirmed.** `frozen` (current default) stays stable the whole run -- 5/6
components populated throughout, `max_frac` oscillating harmlessly in 0.26-0.42 (same
dead component 4 as the original verification run, expected). `trainable` thrashes --
briefly *healthier* than frozen at epoch 20 (6/6 populated, vs frozen's 5/6) -- then
destabilizes, partially recovers twice (epochs 80 and 160-180), and by epoch 200 has
fully collapsed: **all 44,828 loops in a single cluster**, where it stays through epoch
300.

| epoch | frozen: populated / max_frac | trainable: populated / max_frac |
|---|---|---|
| 0 (init, pre-training) | 1/6 / 1.000 | 1/6 / 1.000 (trivial artifact, both arms -- see below) |
| 20 | 5/6 / 0.422 | 6/6 / 0.511 |
| 60 | 5/6 / 0.321 | 2/6 / 0.881 |
| 100 | 5/6 / 0.294 | 6/6 / 0.649 |
| 140 | 5/6 / 0.260 | 5/6 / 0.951 |
| 180 | 5/6 / 0.261 | 4/6 / 0.437 |
| 200 | 5/6 / 0.273 | **1/6 / 1.000** |
| 300 | 5/6 / 0.283 | 1/6 / 1.000 |

Full per-checkpoint record (both arms, every 20 epochs): `results/trainable_prior_test/results.json`.

Two things worth flagging beyond the headline result:

1. **Epoch 0 is a trivial artifact in both arms**, not evidence either configuration
   starts collapsed. `_record(0)` runs right after `load_pretrained_weights`, before any
   training step -- VADE's `z_mean`/`z_log_var` projection heads are freshly initialized
   (the plain AE has no such heads to copy weights from), so they don't yet land in the
   region the pinned GMM prior was fit against. Once training starts, `frozen`
   differentiates into its stable 5-cluster split within 20 epochs.
2. **`trainable`'s collapse isn't a clean monotonic slide** into one cluster -- it
   oscillates for ~180 epochs (looking better than `frozen` at epoch 20, then worse,
   then recovering twice) before locking in. Once collapsed (epoch 200 on), *which* raw
   component holds everyone keeps changing (component 1 at epoch 200-220, then 0, then
   3, then 2 by epoch 300) even though `max_frac` stays pinned at exactly 1.000 --
   consistent with the encoder itself having collapsed to producing near-identical
   `z_mean` for every input, so which component "wins" becomes hypersensitive to
   whatever small step `u_p`/`lambda_p` (still moving) took that epoch.

**Conclusion: freezing `u_p`/`lambda_p` (current default, every `pi_mode`) is the right
call, not an arbitrary restriction.** Letting the GMM prior's means/variances train
jointly with the encoder reproduces the one-big-cluster failure mode the user recalled.
`trainable_prior` stays diagnostic-only -- no CLI flag planned.
