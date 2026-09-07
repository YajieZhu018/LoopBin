# N_SAMPLES=10000 run (2026-08-26)

Archived output of `01_generate_toy_data.py` + `02_evaluate_metric_weights.py` (+
`03_plot_metric_selection.py`) with `N_SAMPLES` bumped from 3000 (the original run,
backed up at `../../n3000_backup/`) to 10000, submitted as SLURM job `15516809` via
`../run_bump_samples.sbatch`. See the "2026-08-26: N_SAMPLES bump" entry in
`../../run_report.md` for the full writeup — this file is a short index for what's
in this folder.

## Why this run happened

`run_report.md`'s original limitations section noted the AE's 10-D latent space was
70-90% dead (collapsed to ~zero variance) at `N_SAMPLES=3000`, and floated more
samples per pretrain epoch as the most direct lever to test. This run tests that.

## What changed vs. the N=3000 baseline

- **Dead latent units improved but didn't resolve**: spot-checked datasets went from
  8/10 dead dims (N=3000) to 6/10 dead (N=10000). The sample-size hypothesis was
  right, but only partially — the latent space is still majority-dead.
- **`cov_health` is still completely degenerate** (pinned constant, `NaN`
  correlation) — still not a fair test, since >50% of dims are still dead.
- **Absolute mean ARI dropped substantially** (~0.21 at N=3000 → ~0.085 at N=10000).
  This tracks the toy-data generation process changing scale with N, not a
  regression — treat it as a different, harder toy-data regime, not a worse fit.

## The more important finding: pooled correlations are confounded

The original `correlations.json` in this run (and in the N=3000 run before it)
computes each metric's Spearman rho against ARI **pooled across all 9 datasets'
candidates combined**. Checking this against the **within-dataset** correlation
(computed separately per dataset, then averaged — the number that actually matters,
since `select_balanced_gmm` only ever ranks candidates fit on the same data) shows
the pooled numbers are largely a cross-dataset difficulty confound, not real
within-dataset signal:

| metric | pooled rho | mean within-dataset rho |
|---|---|---|
| min_weight | 0.102 (n.s.) | **+0.280** (real, understated by pooling) |
| separation | 0.826 | **+0.232** (real, but far weaker than pooled suggests) |
| assignment_confidence | 0.466 | **-0.136** (near-zero/slightly negative — pooled figure is an artifact) |
| log_likelihood | -0.244 | **+0.016** (no real signal either way) |
| cov_health | NaN | NaN (untested — see above) |

`within_dataset_correlations.json` in this folder has the full per-dataset
breakdown. `02_evaluate_metric_weights.py` and `03_plot_metric_selection.py` were
patched after this run finished to compute/plot this automatically going forward
(new `within_dataset_correlations.json` output, new PDF page comparing pooled vs.
within-dataset rho); `within_dataset_correlations.json` and the regenerated PDF in
this folder were produced by re-running that new logic against this run's existing
`per_candidate_metrics.json` — the expensive AE pretraining step was not re-run.

## Files

- `per_dataset_summary.json`, `per_candidate_metrics.json`, `correlations.json`
  (pooled), `strategy_mean_ari.json` — as originally produced by `02`.
- `within_dataset_correlations.json` — added afterward (see above), the number to
  actually trust.
- `metric_selection_report.pdf`, `metric_selection_summary.csv` — regenerated with
  the updated `03` so the PDF includes the pooled-vs-within-dataset comparison page.
- No `training_curves.json` — this run predates the loss-vs-epoch tracking added to
  `02`/`03` afterward, so no AE training-curve page in the PDF for this run.
