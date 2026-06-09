# LoopBin example — DLD-1 (GSE178593), reproducing Fig 1D

This runs LoopBin end-to-end on the published DLD-1 data and produces the per-cluster
**Micro-C + CUT&Tag** average plots (`all_clusters.pdf`) that correspond to **Fig 1D** of the paper.

## Layout
```
example/
  scripts/
    0_download_GSE178593.sh   # download the bigWigs + Micro-C mcools from NCBI GEO
    1_run_pipeline.sh         # end-to-end: 8kb mcool -> preprocess -> ... -> train -> cluster
  references/hg38/            # hg38 chrom.sizes (optional override; the pipeline derives chroms from the mcool)
  example_data/              # (gitignored) the downloaded data + your loop .bedpe go here
  run_output/                # (gitignored) all pipeline outputs
```

## 1. Environment
```bash
conda env create -f ../loopbin.yml   # from the repo root: conda env create -f loopbin.yml
conda activate loopbin
```

## 2. Download the data
```bash
bash scripts/0_download_GSE178593.sh
```
Downloads **14 CUT&Tag bigWig tracks** (CTCF, SMC1A, H3K27ac, H3K27me3 — Ctrl + Aux/degron) and the
**2 merged-replicate Micro-C `.mcool`** matrices (`GSM5394172_control…`, `GSM5394173_degron…`) into
`example_data/`. Source: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178593>.
> Inside mainland China the direct NCBI download can stall — use a proxy (a `…_proxy.sh` variant) there.

## 3. Supply the loop calls (not redistributed here)
Place the two loop-anchor files in `example_data/`:
```
example_data/control_loops_labeled.bedpe
example_data/degron_loops_labeled.bedpe
```
These are the paper's loop calls — loops identified at **5 kb + 10 kb** (HiCCUPS / cooltools / mustache,
merged with pgltools), one BEDPE per condition (`chr1 s1 e1 chr2 s2 e2 …`).

## 4. Run the pipeline
```bash
bash scripts/1_run_pipeline.sh
```
Steps: (0) build a balanced **8 kb** mcool (LoopBin requires `/resolutions/8000`) → (1) preprocess bigWigs →
(2) process → (3) normalize/merge control+degron → (4) pretrain AE → (5) train VaDE (`num=7`, `ep=2000`) →
(6) cluster. **Output:** `run_output/05_cluster/{control,degron}/all_clusters.pdf`.

## ⚠️ Seeds & reproducibility — read this
LoopBin clustering is **stochastic**. VaDE has many local optima, and the **punctate** active marks
(CTCF / SMC1A / H3K27ac) are a *weak* attractor versus the **broad** H3K27me3, so the **active** loop
classes only resolve on some runs. Training is also **non-deterministic** (`TF_DETERMINISTIC_OPS` off and
the pretrain AE is unseeded), so even a fixed seed does not reproduce a result exactly.

**Performance of different seeds** (our 6-seed sweep at `num=7, ep=2000`):

| seed | result |
|---|---|
| **1** | **clean** — 2 active + 1 repressive + 3 lonely, no straggler ✅ (used for the Fig-1D reproduction) |
| 2 | clean — 1 large active + 2 repressive + lonely ✅ |
| 3 | clean — 2 active + 1 repressive + lonely ✅ |
| 4 | degenerate — 7 clusters, 3 tiny stragglers (incl. a ~40-loop noise cluster) ✗ |
| 5 | degenerate — active collapsed to a ~0.2 % straggler ✗ |
| 73 | mixed — active resolved (~14 %) but with a ~0.5 % straggler |

→ **~3 of 6 draws are clean.** The pipeline defaults to `LOOPBIN_SEED=1`, but **run a few seeds and keep a
clean draw** — one with **no cluster < ~2 %** and **all three axes** (active / repressive / lonely) present:
```bash
for s in 1 2 3 4 5; do LOOPBIN_SEED=$s bash scripts/1_run_pipeline.sh; done
```
A robust fix (per-channel normalization so the punctate active marks compete, + optional determinism) is planned.

## Notes
- The pipeline derives the chromosome set directly from the mcool (chr1–22, X). `references/hg38/` is an
  optional override (set `REF_SIZES` in `1_run_pipeline.sh`).
- Knobs are env-overridable: `NCLUST`, `EPOCHS`, `NCPU`, `LOOPBIN_SEED`.
