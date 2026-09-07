# LoopBin

LoopBin is an unsupervised neural network — adapted from the **Variational Deep Embedding (VaDE)** model —
that **clusters chromatin loops** by their genome-interaction (Micro-C) and protein-binding (CUT&Tag /
ChIP-seq) profiles. It embeds each loop's Micro-C contact map together with its mark signal into a latent
space and separates them with a Gaussian-mixture clustering head, so you can quantify loop *types* and how
they shift between cellular conditions.

![model](images/model.png)

## Install

LoopBin is an installable package that provides a `loopbin` command.

```bash
git clone https://github.com/YajieZhu018/LoopBin
cd LoopBin
conda env create --file loopbin.yml      # the `loopbin` env (Python 3.7, TensorFlow 2.5)
conda activate loopbin
pip install -e .                          # installs the `loopbin` CLI
```

`loopbin --help` lists the commands; `loopbin <command> --help` shows a command's flags.
(For back-compatibility, `python main.py <command> …` also works.)

**Alternative: `uv`-managed venv.** `requirements.txt` is a leaner, pinned, verified-working env on
Python 3.9 — faster to set up than conda, and covers everything except `preprocess`
(`normalize`/`pretrain`/`train`/`cluster`/`merge`/`calculateg`/`nmi`, plus `process` via `cooler`).
Package versions are kept in sync with `loopbin.yml`'s (see its header for why `cooler` is pinned to
`0.9.3` rather than latest). What it can't do on its own: `preprocess` needs the `bigWigAverageOverBed`
binary specifically, which has no PyPI wheel — the conda env above gets it automatically via bioconda,
but if you're on the `uv` venv, download it yourself from UCSC's tools directory,
<https://hgdownload.soe.ucsc.edu/admin/exe/> (e.g. `linux.x86_64/bigWigAverageOverBed` for Linux,
`macOSX.x86_64/bigWigAverageOverBed` for macOS/Intel or `macOSX.arm64/bigWigAverageOverBed` for
Apple Silicon), then `chmod +x` it and put it on your `PATH`.

```bash
module load uv   # or: pip install uv
uv venv --python 3.9 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
pip install -e . --no-deps   # installs the `loopbin` CLI without re-resolving deps
# preprocess only: fetch bigWigAverageOverBed for your platform and put it on PATH, e.g.
#   curl -o bigWigAverageOverBed https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/bigWigAverageOverBed
#   chmod +x bigWigAverageOverBed && mv bigWigAverageOverBed ~/.local/bin/   # or anywhere on $PATH
```

## Inputs

| Input | What |
|---|---|
| `.bedpe` | loop-anchor coordinates |
| `.mcool` | Micro-C contact matrix — **must contain a balanced `/resolutions/<res>`** (default 8000; the example builds it) |
| `.bw`   | one bigWig per mark (e.g. CTCF, SMC1A, H3K27ac, H3K27me3) |

Training data for the paper: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178593>.

## The pipeline

LoopBin runs as a sequence of commands; intermediate files pass between steps by folder convention.

```bash
# 1. preprocess each mark: bigWig -> per-chromosome bedgraph
loopbin preprocess -b CTCF.bw -n CTCF -g bedgraph/ -res 8000 -cs hg38.chrom.sizes

# 2. process each condition: loops + mcool + marks -> npy
loopbin process -l loops.bedpe -c matrix_8kb.mcool -g bedgraph/ \
        -p CTCF,H3K27ac,H3K27me3,SMC1A -r 8 -u out/cond/ -res 8000

# 3. normalize + co-normalize the conditions together
loopbin normalize -e control,degron -u out/

# 4. pretrain the autoencoder
loopbin pretrain -d out/merged_log_data.npy -u pretrain/ -s 1 -t 16

# 5. train VaDE + cluster
loopbin train -num 7 -d out/merged_log_data.npy -if_pre True -pre pretrain/ \
        -ep 2000 -u train/ -p CTCF,H3K27ac,H3K27me3,SMC1A -s 1 -t 16

# 6. predict per-condition clusters -> all_clusters.pdf
loopbin cluster -d out/cond/log_data.npy -m train/ -u cluster/cond/ \
        -p CTCF,H3K27ac,H3K27me3,SMC1A

# optional: merge tiny clusters (e.g. <~2 % of loops)
loopbin merge -k 2,3 -d out/merged_log_data.npy -u merged/ -p CTCF,H3K27ac,H3K27me3,SMC1A
```

| command | role | key flags |
|---|---|---|
| `preprocess` | bigWig → bedgraph | `-b -n -g -res -cs` |
| `process` | loops+mcool+marks → npy | `-l -c -g -p -r -u -res` |
| `normalize` | merge + co-normalize conditions | `-e -u` |
| `pretrain` | pretrain the autoencoder | `-d -u -s -t` |
| `train` | train VaDE + cluster | `-num -d -ep -pre -if_pre -u -p -s -t -mw -gmm_prior -pi_mode -pi_em_every -pi_em_warmup -pi_em_rho -pi_floor -pi_floor_hold -pi_floor_ramp_end -ckpt_epochs` |
| `cluster` | predict with a trained model | `-d -m -u -p` |
| `merge` | merge small clusters | `-d -u -k -p` |

## The GMM prior and `pi_mode`

`train`'s GMM prior — the mixture the VaDE latent space is regularized toward — is normally re-selected
from the pretrained AE's latent space each run (`select_balanced_gmm`, scored on separation/confidence/
fit quality rather than sklearn's default best-log-likelihood pick; `-mw` reweights those three metrics).
Pin a specific fitted GMM instead with `-gmm_prior <path/to/gmm_prior.pkl>` for reproducibility across
machines (GPU/CPU numerical differences can select a different candidate at the same seed).

`-pi_mode` controls how the prior's mixing weights (`theta_p`) are handled during training —
`uniform_fixed` (default, frozen at `1/k`), `gmm_fixed` (frozen at the GMM's fitted weights), `em`
(periodic EM M-step, tuned by `-pi_em_every/-pi_em_warmup/-pi_em_rho` and floored via
`-pi_floor/-pi_floor_hold/-pi_floor_ramp_end` to keep a shrinking component recoverable early on), or
`gradient` (trained like any other weight). `gradient`/`gmm_fixed`/`em` are all prone to a
rich-get-richer cluster collapse; `uniform_fixed` is the one that's held up in testing — see
`scripts/gmm_metric_validation/run_report.md` for the comparison. `-ckpt_epochs` saves extra mid-training
checkpoints (e.g. around where `em`'s floor releases) rather than only the final epoch.

## Reproducibility — `-s/--seed`, `-t/--threads`

VaDE training is seed-sensitive. LoopBin makes a run **reproducible**: at a fixed seed the result is
**bit-identical** (every RNG is seeded — including the pretrain autoencoder — and TensorFlow deterministic
ops are enabled).

- `-s/--seed N` — random seed. Precedence: flag > `$LOOPBIN_SEED` > built-in default 73. (The example pins `-s 1`.)
- `-t/--threads N` — CPU threads. Precedence: flag > `$LOOPBIN_THREADS` > 16. Reproducibility holds at any
  **fixed** thread count — keep it the same across `pretrain` and `train`.

> The *effective* cluster count is stochastic across **different** seeds (the paper's runs return 5–7 even at
> a fixed `-num`). Run a few seeds and keep a clean draw.

## Consensus across a seed sweep — `example/scripts/consensus_*.py`

Because a single VaDE run is seed-sensitive, `example/scripts/` ships two helpers that combine several
fixed-seed runs into one **consensus** partition. Neither retrains anything: both read only the
`labels.npy` each seed run wrote.

```
run_seedsweep/
├── 02_process/           # the shared input matrix (merged_log_data.npy) — one copy for all seeds
├── seed1/ … seed5/       # one `loopbin train` run each: labels.npy, all_clusters.pdf, pie.pdf, …
├── consensus/            # consensus at k = the modal per-seed k
└── consensus_numselect/  # consensus at a data-driven k*
```

```bash
# k fixed to the modal per-seed k
python example/scripts/consensus_cluster.py   <sweep_dir> CTCF,H3K27ac,H3K27me3,H3K4me3

# k chosen from the data;  args: <sweep_dir> <marks> <seeds> [out_subdir] [n_sub] [kmin] [kmax]
python example/scripts/consensus_numselect.py <sweep_dir> CTCF,H3K27ac,H3K27me3,H3K4me3 1,2,3,4,5
```

**How `consensus_numselect/all_clusters.pdf` relates to `seed1…seed5/all_clusters.pdf`** — it is not a
further run; it is the five seed runs, post-processed:

1. load `seed<N>/labels.npy` for the listed seeds — and nothing else (not the model weights, `prob.npy`
   or `recon.npy`);
2. build the **co-association matrix** on a subsample (`n_sub`, default 8000 loops): `C[i,j]` = fraction
   of the seeds that put loops *i* and *j* in the same cluster;
3. **choose k** — average-linkage agglomerative clustering of `1 − C` for k = `kmin`..`kmax`
   (default 2..12), keeping the k with the best silhouette (PAC breaks ties); the scan is plotted to
   `k_selection.pdf`. `LOOPBIN_FORCE_K=<k>` overrides the choice;
4. take one **medoid** loop per consensus cluster and assign *every* loop to the medoid it co-clusters
   with most often across seeds → `labels.npy` plus `loop_confidence.npy` (that co-association fraction,
   per loop);
5. re-plot with the **same** `02_process/merged_log_data.npy` and the same `plot_all_clusters` a single
   run uses — only the label vector differs.

So `seedN/all_clusters.pdf` shows the clusters of *one* run (whose realized k varies across seeds), while
`consensus_numselect/all_clusters.pdf` shows the same loops and the same signal regrouped into the
partitions that are reproducible across the whole sweep.

`consensus_summary.txt` reports the selected k*, the cluster sizes, the confidence distribution, the
baseline pairwise-seed ARI and each seed's ARI **against the consensus**. Read that last line: if one
seed scores far higher than the others, the consensus is largely that seed rather than a balanced merge.

Other knobs: `LOOPBIN_MERGE_MIN_FRAC=<frac>` drops consensus clusters smaller than `frac` and reassigns
their loops to the nearest kept medoid. Companion plots for a sweep: `plot_pairwise_nmi.py` and
`plot_shared_loops_heatmap.py`.

## Per-cluster downstream analysis

Once a run (or a consensus) has assigned every loop a cluster, `example/scripts/` provides the
per-cluster views. All three take a **labels dir** — `seedN/`, `consensus/`, `consensus_numselect/`
— so nothing is specific to one clustering.

### Loops of each cluster as BED-PE

```bash
python example/scripts/labels_to_bedpe.py <labels_dir> [loops.bedpe] [min_conf]
```

Pastes `labels.npy[i]` (and `loop_confidence.npy[i]`, when present) onto line *i* of
`02_process/loop_file_analysis.bedpe` — the same row correspondence `loopbin cluster` uses for
`labels_loops.bedpe` — and splits by label into `<labels_dir>/bedpe/cluster_<k>.bedpe`, plus
`all_loops_labeled.bedpe`. A `min_conf` above 0 also writes `low_confidence.bedpe` with the loops
it excluded. Coordinates are the binned anchors in pipeline row order; `sort -k1,1 -k2,2n` first
if a downstream tool needs sorted input.

### Chromatin state of each cluster's anchors

Characterises the clusters against an annotation the model never saw — the Ernst-lab mouse
**full-stack ChromHMM** model (mm10, 100 states, 200 bp bins,
[github.com/ernstlab/mouse_fullStack_annotations](https://github.com/ernstlab/mouse_fullStack_annotations)).

```bash
# 0. annotation -> example/references/mm10_chromstates/  (verifies the 200 bp binning)
bash example/references/0_download_chrom_states.sh

# 1. one state per distinct anchor locus, plus a matched genome background   (~1 min)
python example/scripts/anchor_chrom_states.py <sweep_dir> example/references/mm10_chromstates

# 2. per-cluster composition figures
python example/scripts/plot_cluster_states.py <labels_dir> example/references/mm10_chromstates [rule]
```

**How an anchor gets a state.** An anchor is one bin wide (5 kb in the mESC example) and a
full-stack segment is a run of 200 bp bins, so an anchor typically spans ~10-25 segments. The
anchor takes the class covering the most base pairs of it, ties breaking on the lowest state id.
Two rules are computed from the same overlaps and both stored in `anchor_states.tsv`:

| `rule` | winner | why |
|---|---|---|
| `state` (default) | the single 100-state class with the largest overlap | the literal max-overlap rule |
| `group` | the state GROUP with the largest total overlap, summed over its states | the top *state* holds only ~a third of a 5 kb anchor, so aggregating first is more stable |

The two agree on the group for ~76 % of anchors in the mESC run; run both and check the
conclusions hold. Figures are plotted at the ~16-group level (Prom, Enh, Tx, ReprPC, Quies, …)
with the official ChromHMM colours — 100 slices is not a readable pie.

**Anchors are a set.** A locus used by several loops of one cluster counts once, so a hub anchor
does not inflate its state. A locus used by loops in *different* clusters appears in each of their
sets; `chrom_states_summary.txt` reports how many.

**Joining on the mnemonic, not the number.** The BED segment names are `<id>_<mnemonic>`
(`98_mTSS1`), and the `state` column of `state_annotation_processed.tsv` is a *different*
numbering (TSV state 98 is `mEnhA17`). `anchor_chrom_states.py` joins on the mnemonic and aborts
if any segment state is unmatched.

Outputs, in `<labels_dir>/chrom_states/` (`chrom_states_grouprule/` for `rule=group`):

| file | content |
|---|---|
| `cluster_state_counts.tsv` | cluster × group: n_anchors, %, background %, log2 obs/exp |
| `states_histogram.pdf/.png` | one panel per cluster, % of the cluster's anchors per group |
| `states_pie.pdf/.png` | the same composition as pies, groups under 1 % merged |
| `states_enrichment_log2.pdf/.png` | log2(cluster % / background %) — without it every pie is just dominated by Quies |
| `chrom_states_summary.txt` | anchor counts, shared loci, top enriched/depleted groups |

The background is every 5 kb window of chr1-19,chrX put through the same max-overlap rule
(`references/mm10_chromstates/genome_5kb_window_groups.tsv`), so observed and expected are
computed identically. Note the full-stack model is deliberately **cell-type agnostic** — one
annotation stacked over 901 mouse datasets — so it says what kind of element a locus generally is,
not what it is doing in the cell type being clustered.

### The mESC-specific annotation of Hsieh et al. 2020 (Fig 3E)

The full-stack model above is cell-type agnostic. For an mESC-specific alternative — the one Hsieh
et al. (*Mol Cell* 2020) use in Figure 3E — run:

```bash
bash example/references/0_prepare_hsieh_fig3E_states.sh
python example/scripts/anchor_chrom_states.py <sweep_dir> example/references/mESC_chromHMM_wei12 \
       example/references/mm10.main.chrom.sizes chrom_states_wei12
python example/scripts/plot_cluster_states.py <labels_dir> example/references/mESC_chromHMM_wei12 \
       state <sweep_dir>/chrom_states_wei12 chrom_states_hsieh_fig3E_all12
LOOPBIN_STATES_EXCLUDE=Intergenic \
python example/scripts/plot_cluster_states.py <labels_dir> example/references/mESC_chromHMM_wei12 \
       state <sweep_dir>/chrom_states_wei12 chrom_states_hsieh_fig3E
```

**Provenance** — the attribution is indirect, so the chain is worth recording. Hsieh's Fig S3F
legend: "The 11 ChromHMM states in mESCs were identified from (Pintacuda et al., 2017)". That
paper's STAR Methods in turn names the source by URL: "the pre-defined ChromHMM state
(https://github.com/guifengwei/ChromHMM_mESC_mm10) across chr2 and chr3 was used". The repo
(Guifeng Wei, Dec 2015) is ChromHMM on ENCODE E14 mESC ChIP-seq (H3K4me1/me3, H3K27me3, H3K27ac,
H3K9me3, H3K9ac, H3K36me3, CTCF, Nanog, Oct4), mm10, 200 bp bins, a **12**-state model.
(The main text's "(Ernst and Kellis, 2012)" is the ChromHMM *software* paper, not a state source.)

**Which repo file** — the repo ships the same segmentation twice, and only one of them is usable
here. Verified: 431,899 rows in both, identical intervals row for row, state labels 1:1.

| file | cols | state column | used |
|---|---|---|---|
| `mESC_E14_12_dense.annotated.bed.gz` | 9 | `2_Intergenic` + `itemRgb` colour | **yes** |
| `mESC_E14_12_segments.bed.gz` | 4 | `E2` | no — kept only as the raw ChromHMM output |

`segments.bed` is ChromHMM's raw output, whose states are anonymous `E1..E12`; `dense.bed` is the
browser track, and `.annotated` is Wei's addition of a **name and colour** per state. Both the
Fig-3E label mapping and the figure palette come from those names/colours, so the prep script
reads the dense file (`0_prepare_hsieh_fig3E_states.sh`, `zcat "$DENSE"`) and derives
`mESC_chromHMM_wei12/segments.sorted.bed` from it. That derived file is what
`anchor_chrom_states.py` intersects the anchors against.

Hsieh's Fig 3E shows **11** of those 12. The prep script maps them and records the mapping in
`mESC_chromHMM_wei12/annotation.tsv`:

| Wei state | Fig 3E label | | Wei state | Fig 3E label |
|---|---|---|---|---|
| `1_Insulator` | Insulator | | `7_ActivePromoter` | Active promoter |
| `3_Heterochromatin` | Heterochromatin | | `8_StrongEnhancer` | Strong Enhancer |
| `4_Enhancer` | Enhancer | | `10_TranscriptionElongation` | Elongation |
| `5_RepressedChromatin` | Repressive | | `11_WeakEnhancer` | Weak Enhancer |
| `6_BivalentChromatin` | Bivalent promoter | | `12_LowSignal/RepetitiveElements` | Repeats |
| `9_TranscriptionTransition` | **Weak promoter** (inferred) | | `2_Intergenic` | **not shown in Fig 3E** |

Ten map by name. `9_TranscriptionTransition` → "Weak promoter" is inferred by elimination — it is
the only state left, and Intergenic (76 % of the genome) cannot be a promoter class. `Repeats` is
40 kb genome-wide (42 segments), so it is absent from the plots for lack of any anchor; Hsieh
likewise print "Repeats (not shown)".

**Two output sets, on purpose.** `chrom_states_hsieh_fig3E_all12/` keeps Intergenic — the honest
denominator. `chrom_states_hsieh_fig3E/` drops it and renormalises to the Fig-3E vocabulary; the
panel titles then carry the retained fraction (`1389 of 6918 anchor loci = 20 %`), because the
excluded share differs hugely between clusters and the renormalised pies must not be read as
covering the whole cluster.

**This annotation fits 5 kb anchors far better than the 100-state model**: the winning state
covers a median 80 % of an anchor (vs 32 %), and only 1.8 % of anchors need a tie-break (vs 13.5 %).
Twelve broad mESC states are simply a better match to a 5 kb window than 100 fine-grained
universal ones.

## Example — DLD-1 (GSE178593)

`example/` reproduces the paper's per-cluster Micro-C + CUT&Tag figure (`all_clusters.pdf`) on the published
DLD-1 data:

```bash
bash example/scripts/0_download_GSE178593.sh     # bigWigs + Micro-C mcools from GEO -> example/example_data/
# put the loop calls in example/example_data/{control,degron}_loops_labeled.bedpe
bash example/scripts/1_run_pipeline.sh           # builds the 8 kb mcool, then runs the full pipeline
```

Output: `example/run_output/.../all_clusters.pdf`. See **`example/README.md`** for the data sources, the
8 kb-mcool requirement, and the seed/reproducibility notes.

## Citation

If you use LoopBin, please cite the LoopBin preprint (bioRxiv, 2026; DOI 10.64898/2026.01.13.699359).

## Authors

Yajie Zhu, Alexis Bel.
