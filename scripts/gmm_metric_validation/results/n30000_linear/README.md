# Linear-bottleneck run at N=30000 (2026-08-27)

Archived output of `02_evaluate_metric_weights.py` + `03_plot_metric_selection.py`
with the AE bottleneck activation switched from ReLU to **linear**, submitted as
SLURM job `15546947` via `../../run_linear_bottleneck.sbatch`. See the
"2026-08-27: linear bottleneck" entry in `../../run_report.md` for the full writeup —
this file is a short index for what's in this folder.

## Controlled A/B against `../n30000/`

`01_generate_toy_data.py` was deliberately **not** rerun: `toy_data/` already held the
N=30000 datasets from job `15519814`, so both runs see byte-identical input and the
only difference is the bottleneck activation. The previous run's ReLU-pretrained AEs
and latent embeddings were moved to `../../pretrained_n30000_relu/` rather than
overwritten, so both latent spaces are still on disk and directly comparable.

`../n30000/latent_diagnostics.json` and `latent_diagnostics_summary.txt` were
backfilled post hoc from those saved ReLU latents (no AE retraining) so the "before"
side of every comparison below is measured the same way as the "after".

## What changed in the code

- `src/model/ae.py`: bottleneck activation is now an `AE(input_size, latent_activation=...)`
  constructor arg. It still **defaults to `'relu'`** — production `main.py:pretrain_ae`
  and `04_test_real_pretrain_gmm.py` are untouched, and every model under `trials/`
  remains reproducible. Only `02` opts in, via `LATENT_ACTIVATION = None`.
- `02_evaluate_metric_weights.py`: added `compute_latent_diagnostics` (alive/dead dims,
  effective rank, within-cluster vs. global dimension correlation) →
  `latent_diagnostics.json`, plus three headline fields on each `per_dataset_summary`
  row (`n_dead_dims`, `effective_rank`, `within_cluster_mean_abs_corr`).
- `03_plot_metric_selection.py`: new `plot_latent_diagnostics_page`, rendered first in
  the PDF, behind the same optional-file guard as the other added pages so the older
  archived runs still render.

## Headline: the dead-unit problem is solved

| | ReLU (`../n30000/`) | linear (here) |
|---|---|---|
| alive latent dims | 4.4 / 10 | **10.0 / 10** (9/9 datasets) |
| effective rank | 2.29 | **9.39** |
| within-cluster mean \|r\| | 0.356 | **0.086** |
| global mean \|r\| | 0.365 | 0.043 |
| AE reconstruction loss | 0.2952 | **0.2813** (better on 9/9) |
| datasets with 0/10 GMM survivors | 4 / 9 | **0 / 9** |
| datasets where `cov_health` is measurable | 4 / 9 | **9 / 9** |

Three runs of `N_SAMPLES` scaling (3000 → 10000 → 30000) moved the dead-dim count
from 8/10 to ~5.6/10. Changing one activation moved it to 0/10. The cause was
dying ReLU, not insufficient data.

Note the correlation result is not just "smaller": under ReLU, within-cluster \|r\|
(0.356) and global \|r\| (0.365) were *equal*, meaning the correlation was intrinsic
to each component — exactly what a diagonal covariance cannot represent. Under linear
both collapse toward zero, so `covariance_type='diag'` is now a well-specified choice
rather than a violated assumption.

## Secondary: ARI improves, but candidate selection still loses to random

Every strategy's mean ARI improved (oracle 0.1237 → 0.1414, random_pick
0.1163 → 0.1301). But `random_pick` (0.1301) still beats every actual selection
strategy, the best of which is `loglik_only`/`naive_loglik` at 0.1274.

With 10/10 survivors on all 9 datasets, this is the first run where the composite
scoring is exercised on its intended path rather than the degenerate no-survivor
fallback — and it underperforms picking at random. See the run report for why
`min_weight`'s exclusion from `metric_weights` is the likely explanation.

Within-dataset mean rho vs. ARI, ReLU → linear:

| metric | ReLU | linear | notes |
|---|---|---|---|
| min_weight | +0.333 | **+0.289** | only metric positive in all 3 runs |
| log_likelihood | -0.133 | +0.036 | ~0, unstable across runs |
| cov_health | -0.296 (4/9) | -0.068 (9/9) | first fair test; median -0.370 |
| separation | +0.015 | -0.179 | +0.232 → +0.015 → -0.179 across runs |
| assignment_confidence | -0.122 | -0.223 | negative in all 3 runs |

## Files

Same schema as `../n30000/`, plus `latent_diagnostics.json` (new). The PDF's first
page is the new latent-health page.
