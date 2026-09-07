# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

LoopBin clusters chromatin loops using a VaDE (Variational Deep Embedding) model. It learns a joint
latent representation of Micro-C interaction submatrices and Cut&Tag/ChIP-seq protein-binding profiles
(CTCF, cohesin/SMC1A, H3K27ac, H3K27me3, etc.), then fits a Gaussian Mixture Model in that latent space
to assign each loop to a cluster. Comparing cluster composition across conditions (e.g. control vs.
cohesin-degron) reveals how loop types shift with cellular context.

This is a research pipeline, not a library/application: there is no test suite. Correctness is judged
by whether the pipeline runs end-to-end and produces sensible clusters/plots, not by unit tests.

LoopBin is packaged as an installable Python package (`loopbin/`, with `setup.py`/`pyproject.toml`
providing a `loopbin` console script) rather than a loose collection of scripts.

## Environment setup

Two documented paths, scoped by use case — see README.md's `## Install` section for the exact commands:

- **conda** (`loopbin.yml`, Python 3.7 + TensorFlow 2.5) — the full pipeline, including the raw
  `preprocess`/`process` stages' dependencies (`cooler`, `cooltools`, `bioframe`, `pysam`, `pytables`,
  the `ucsc-bigwigaverageoverbed` binary).
- **`uv`-managed venv** (`requirements.txt`, Python 3.9) — leaner, faster to set up, verified working for
  the model/training stack (`pretrain`/`train`/`cluster`/`merge`/`calculateg`/`nmi`), but does **not**
  cover the raw preprocessing/process stages yet. This is what all the `loopbin/jobs/*.sbatch` training
  scripts actually use: `source .venv/bin/activate`.

`loopbin.yml`/`loopbin_core.yml` are earlier/leaner conda specs kept for reference; `requirements.txt`
carries the versions actually verified together (TensorFlow imports, forward pass, `sklearn`
`GaussianMixture` all tested) on this cluster.

**Important:** the venv lives at `.venv/`, a sibling of the `loopbin/` package directory, not inside it —
naming the venv `loopbin/` (as an earlier iteration of this repo did) collides with the package's own
import name and silently deletes/restores each other's files on every branch switch. Don't recreate that
collision.

There is no separate install step for running the pipeline beyond activating the venv and `pip install -e
.` — everything after that runs via the `loopbin` command or `python main.py`.

## Running the pipeline

Everything goes through the `loopbin` console script (or, for back-compatibility, `python main.py
<command> ...`) — an argparse subcommand CLI, not a flat `-f FUNCTION` flag. The seven pipeline stages,
in order:

```bash
# 1. Convert a bigwig track into per-chromosome bedgraphs
loopbin preprocess -b BIGWIG_FILE -g OUTPUT_FOLDER -n BIGWIG_NAME -res 8000 -cs CHROM_SIZES

# 2. Build the model input: for each loop anchor pair, extract a submatrix (loopbin/fn/processing.py)
#    plus the flanking epigenetic signal, filtering loops too close to the diagonal
loopbin process -l LOOPS.bedpe -c HIC.mcool -g BEDGRAPH_FOLDER -r N_CPU -u OUTPUT_FOLDER -res 8000

# Then co-normalize/merge conditions that were processed separately (log + min-max normalize together)
loopbin normalize -e control,degron -u OUTPUT_FOLDER

# 3. Pretrain a plain autoencoder (loopbin/model/ae.py) to initialize VaDE weights and seed GMM centers
loopbin pretrain -d MERGED_LOG_DATA.npy -u OUTPUT_FOLDER -s SEED -t THREADS

# 4. Train the VaDE model (loopbin/model/vade_model.py): encoder -> GMM prior in latent space -> decoder
loopbin train -num N_CLUSTERS -d DATA.npy -if_pre True -pre PRETRAINED_AE -ep N_EPOCHS -u OUTPUT_FOLDER \
        -p CTCF,H3K27ac,H3K27me3,SMC1A -s SEED -t THREADS
# See README.md's "The GMM prior and pi_mode" section for -mw/-gmm_prior/-pi_mode/... .

# 5. Assign clusters with a trained model
loopbin cluster -d DATA.npy -m MODEL_PATH -u OUTPUT_FOLDER -p CTCF,H3K27ac,H3K27me3,SMC1A

# 6. (optional) Merge small/spurious clusters (e.g. <2% of loops) into others
loopbin merge -k 2,3 -d DATA.npy -u OUTPUT_FOLDER -p CTCF,H3K27ac,H3K27me3,SMC1A
```

`loopbin --help` / `loopbin <command> --help` are the authoritative flag reference; README.md documents
the same table plus a reproducibility section (`-s/--seed`, `-t/--threads` — a run is bit-identical at a
fixed seed and thread count).

**Known gap:** `loopbin/cli.py` also defines `calculate_generalizability` (cross-validated
generalizability/silhouette/Calinski-Harabasz scan over cluster counts) and an NMI-heatmap-across-runs
helper — the old flat CLI exposed these as `-f calculateg`/`-f nmi`, but the subcommand parser
(`loopbin/cli.py`'s `parse_arguments`) currently only registers `preprocess/process/normalize/pretrain/
train/cluster/merge`. Both functions are reachable only by importing `loopbin.cli` and calling them
directly from a script — not from the command line. Treat this as a known rough edge, not something to
silently wire up without checking with the user first.

## Architecture

**`main.py`** is a thin back-compat shim (`python main.py <command> ...` -> `loopbin.cli.cli()`). All
real orchestration — argument parsing/validation, model construction, training loops, checkpointing, and
post-training cluster/plot generation — lives in **`loopbin/cli.py`**, one `..._<stage>` function per
pipeline stage plus the `train` subcommand's seed/threads/regularizer/pi_mode resolution logic. The
`loopbin/model` classes stay intentionally "dumb" (just the network + loss); `cli.py` handles the
experiment logic (seeds, callbacks, saving).

**`loopbin/fn/processing.py`** — turns raw inputs (`.bedpe` loop list, `.mcool` Hi-C matrix, per-
chromosome bedgraphs) into the model's flat input array. Key steps: `input_creator_r` multiprocesses over
loops, pulling a submatrix per anchor pair via `cool_to_matrix` and the flanking epigenetic signal via
`extract_signal_from_bedgraph`, discarding loop pairs whose anchors are within 2 bins of the diagonal.
`formatting` reassembles multiprocessing results back into original loop order. `log_epi`/`log_microc`/
`process_all_groups` log-transform and co-normalize data — critically, when comparing conditions, data
must be normalized *together* (`process_all_groups`/`normalize`) so values are on a comparable scale, not
normalized independently per condition. `create_data` reshapes the flat arrays back into the multichannel
image form used for the per-cluster average plots (not for model input, which stays flat). `cooler`/
`tqdm` are imported lazily inside `input_creator_r`/`process`, not at module top level — the `uv` venv
above deliberately omits them, and importing this module at all (needed for `create_data`, used by
`train`/`cluster`/`merge`) would otherwise crash on those stages even though they never touch `cooler`.

**`loopbin/model/ae.py`** — plain autoencoder (`500-500-2000-10` encoder/decoder, sigmoid output; encoder
bottleneck activation configurable via `latent_activation`, default `'relu'` — a linear bottleneck is a
useful diagnostic comparison, see the GMM-selection validation work below), used only to pretrain weights
and get an initial GMM fit before VaDE training. Supports optional per-feature loss weighting
(`loss_weights`) in its weighted-BCE `train_step`/`test_step`.

**`loopbin/model/vade_model.py`** — the core model. `GMM` is a custom Keras layer holding GMM parameters
(`theta_p`, `u_p`, `lambda_p`); `theta_p` is non-trainable by default but can be made trainable
(`trainable_theta`, used by `pi_mode='gradient'`). `select_balanced_gmm`/`_fit_gmm_candidates` fit several
GMM candidates (fixed `random_state`s, `n_init=1` — deliberately not raised, see
`_fit_gmm_candidates`'s docstring and `scripts/gmm_metric_validation/run_report.md`) and score them on
separation/confidence/fit quality rather than sklearn's default best-log-likelihood pick, to avoid a
degenerate "one tight cluster + near-empty others" solution. `VADE.call` returns both `gmm(z)` (cluster
responsibilities) and the decoder reconstruction. `train_step`/`test_step` compute reconstruction loss
(weighted binary cross-entropy) plus the closed-form GMM-prior KL loss (`calculate_kl_loss` — the
standard VaDE ELBO), with optional marginal-entropy and marginal-KL-to-target regularizer terms.
`load_pretrained_weights` accepts `prior_gmm_path` (pin a specific fitted GMM instead of re-selecting
one), `metric_weights` (reweight the selection metrics), and `init_theta_from_gmm`. `vade_mnist.py` is a
reference/prototype implementation against MNIST kept for comparison, not part of the pipeline.

**`loopbin/plot/plotting.py`** — all diagnostic plots (elbow/silhouette for k selection — silhouette is
computed on a subsample, `sample_size=min(2000, len(lat_space))`, for tractability on large runs; training
loss curves, per-cluster average Micro-C + epigenetic images across two PDF pages, pie charts of cluster
sizes, t-SNE of latent space, `theta` history over training). Called from `loopbin/cli.py` after
training/clustering, not invoked directly by users.

**`loopbin/fn/init_data.py`** — thin data loading/train-test-split helpers; mostly used by the exploratory
scripts in `scripts/`, not the main CLI path.

**`data/`** — pipeline stage outputs live in numbered subfolders matching the CLI stages:
`00_raw` (bigwig/mcool/bedpe inputs) -> `01_preprocess` (per-condition bedgraphs) -> `02_process`
(per-condition raw + merged/log-normalized `.npy` arrays, one subfolder per condition/protein-set
combination) -> `03_prior` (pinned GMM prior pickles, one subfolder per pretrained latent space, for
`-gmm_prior`).

**`trials/`** — saved experiment outputs (trained models, labels, plots), one folder per run, typically
named `vade_<n>clusters_<conditions>_<protein set>_run<N>` — a naming convention worth following for new
runs since several downstream `scripts/` (e.g. NMI/consensus clustering) glob multiple run folders by
this pattern.

**`scripts/`** — one-off downstream analysis scripts (not part of the `loopbin` CLI), organized by
analysis type: `cluster_shift/` (cluster composition changes across conditions + NMI stability),
`degs/` and `unique/` (linking loop clusters to differentially expressed genes / RNA-seq FPKM),
`intersect/` (overlap of clusters with ChIP peaks, metaplots via deepTools), `metaplot/`,
`probability/` (filtering loops by GMM assignment confidence), `merge_clusters/` (consensus clustering
across repeated runs), `interfere_cluster_number/` (model-selection diagnostics: BIC/AIC, silhouette,
loss vs. cluster count), `gmm_metric_validation/` (validation of `select_balanced_gmm`'s candidate-
selection metrics against ground-truth ARI/NMI on toy data plus real pretrained latents — see its
`run_report.md` for the running log of findings, including the `pi_mode` cluster-collapse investigation).
Many of these have hardcoded paths to specific trial run folders — check the script's path constants
before rerunning it on new data rather than assuming it's parameterized.

## Running on the cluster

`loopbin/jobs/*.sbatch` holds one job script per pipeline stage — copy the nearest one rather than
writing a submission from scratch. `loopbin/jobs/experiments/` holds one-off/experiment-specific job
scripts (pinned-prior comparisons, seed sweeps, debug scripts) tied to this account's paths/partitions —
it's gitignored, kept for local reuse only, not meant for other users. `scripts/gmm_metric_validation/
run_*.sbatch` similarly has one script per validation experiment.

The SLURM account is `scc_ukla_papantonis` and the only usable partitions are `scc-gpu` (A100s) and
`scc-cpu`; the generic `standard96`/`medium96s` partitions show up in `sinfo` but are rejected at
submission with "Invalid account or account/partition combination". GPU jobs need the CUDA 11 runtime
shim every existing GPU script carries — see `loopbin/jobs/experiments/cuda11_runtime_libs.tar` and
`scripts/gmm_metric_validation/run_report.md`'s 2026-08-26 entry for why it's a tarball.

Prefer sbatch over running things on the login node for anything that imports TensorFlow or matplotlib.
The venv lives on the ceph mount, and login-node reads of individual package files intermittently hang in
uninterruptible `D` state for tens of minutes — on 2026-08-27 `import matplotlib` timed out repeatedly at
120s while unrelated cold reads from the same mount returned in 0.1s, and each hung process needed `kill
-9`. Compute nodes were unaffected. The same mount has also been observed to make even small git
operations (`git checkout`, `git status`) hang for minutes under load from other users' jobs — if a git
command seems stuck, check `ps`/`fuser` on `.git/index.lock` before assuming it's a real conflict; a
stale lock with no owning process is safe to remove.

**Never switch git branches in this working directory while a job is running against it** — `main.py`/
`loopbin/` get swapped out from under the running process. Pull specific files across branches with
scoped `git checkout <branch> -- <path>` instead of a full branch switch when something might be running.

## Conventions to know

- Random seeds are fixed per training function (`train_vade` uses seed 48 by default, `-s/--seed`
  overrides it; precedence is flag > `$LOOPBIN_SEED` > built-in default) for reproducibility of published
  results — don't change the default without a reason.
- `-if_pre True` (string, not a Python bool) is the expected way to enable pretrained-weight loading;
  `args.if_pretrain == 'True'` is a literal string comparison throughout `loopbin/cli.py`.
- GMM cluster centers computed during `load_pretrained_weights` are also pickled to a hardcoded path
  (`/usr/users/yzhu1/LoopBin/trials/saved_models/gmm_models/`) — this is stale/user-specific; treat it as
  a known rough edge, not an issue to silently "fix" without checking with the user first.
- Multi-page diagnostic PDFs built with `PdfPages` must rasterize any large point cloud:
  `ax.scatter(..., rasterized=True)` together with `pdf.savefig(fig, dpi=200)`. Left as vectors, the
  PCA page in `scripts/gmm_metric_validation/04_test_real_pretrain_gmm.py` emitted ~45k markers per
  panel into a 9.8 MB content stream — 95% of the file — which VS Code's PDF preview took minutes to
  render, making the report look like an empty file.
- Nothing persists the AE pretraining history (`04_test_real_pretrain_gmm.py` passes the `fit` return
  value straight to the plot), so a report's loss page cannot be rebuilt without retraining. The other
  pages can: `latent.npz` plus `candidate_metrics.json` hold everything, and per-candidate cluster
  labels are recoverable by refitting `GaussianMixture` at the recorded `random_state`.
- Unresolved: `AE.train_step` (`loopbin/model/ae.py`) returns `{"loss": ...}` from a `keras.metrics.Mean`
  tracker named `total_loss`, and the first epoch of `history.history['loss']` does not match the loss
  Keras printed for that epoch (0.209 plotted vs 0.2643 logged in the 2026-08-27 `real_latent_linear`
  run); later epochs agree to 4 decimals. Cause unconfirmed — treat epoch 1 of any AE loss curve as
  suspect rather than as a real training signal.
- `pi_mode` (see README.md's "The GMM prior and pi_mode" section): `gradient`/`gmm_fixed`/`em` are all
  prone to a rich-get-richer cluster collapse; `uniform_fixed` is the one that's held up in testing. This
  was root-caused to `pi_mode` handling itself, not GMM prior selection quality — see
  `scripts/gmm_metric_validation/run_report.md`'s 2026-09-03 entries for the full investigation
  (component-matching two "different" priors as structurally identical, a clean pinned-prior 3-arm×3-seed
  experiment, and a targeted confirmation run).
