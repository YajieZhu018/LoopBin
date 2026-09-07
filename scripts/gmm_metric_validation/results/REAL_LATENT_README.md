# Real-data latent health check (2026-08-27)

`real_latent_relu/` and `real_latent_linear/` — SLURM job `15548074`, via
`../run_real_latent_check.sbatch`. See the "2026-08-27: confirmed on the REAL data"
entry in `../run_report.md` for the full writeup.

Both arms replicate `main.py:pretrain_ae` exactly (full `merged_log_data.npy`,
44828 x 384, no subsample; 200 epochs; batch 256; Adam lr=0.002; seed 0;
`num_clusters=10`, `covariance_type='diag'`). The **only** difference is
`--latent-activation`, so `real_latent_relu/` is a faithful reproduction of the
model every clustering under `trials/` was built on.

## Headline

| | relu (= production) | linear |
|---|---|---|
| alive latent dims | **5 / 10** | **10 / 10** |
| effective rank | **2.93** | **9.42** |
| AE reconstruction loss | 0.1811 | **0.1779** |
| GMM candidates below the 0.02 `min_weight` floor | **4 / 10** | **0 / 10** |
| within-cluster mean \|r\| | 0.078 | 0.070 |
| global mean \|r\| | 0.406 | 0.049 |

Production's latent space is majority-dead: 5 of 10 dims have variance exactly
0.0000, and effective rank is 2.93. Published clusters were fit by a 10-component
GMM in an effectively ~3-dimensional space.

Note the correlation columns **contradict the toy harness**: on real data
global \|r\| (0.406) sits far above within-cluster \|r\| (0.078), which is the healthy
pattern, so `covariance_type='diag'` is well-specified here. The toy data's
within ≈ global result was an artifact of `01_generate_toy_data.py` sampling
elliptical correlated covariances by design.

## Files per arm

- `latent_diagnostics.json` — the table above plus per-dim variance and per-cluster
  correlations. On real data the within-cluster statistic is grouped by the selected
  GMM's own hard assignments (no ground truth exists), which makes it optimistic.
- `latent.npz` — the pretrained latent embedding (`z`), for re-running GMM
  experiments without repeating the 20-minute pretrain.
- `pretrained_ae/` — SavedModel dir.
- `candidate_metrics.json` / `.csv` — the 10 GMM candidates' metrics, pairwise
  ARI/NMI agreement matrices, and which candidate the balanced vs. naive rule picked.
- `balanced_gmm_test_report.pdf` — loss curve, per-candidate metrics, agreement
  heatmaps, PCA of the latent colored by assignment.
- `subsampled_data.npy` — despite the name, the full dataset here (`--n-samples 0`).
