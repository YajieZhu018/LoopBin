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
git clone https://github.com/astudentfromsustech/LoopBin
cd LoopBin
conda env create --file loopbin.yml      # the `loopbin` env (Python 3.7, TensorFlow 2.5)
conda activate loopbin
pip install -e .                          # installs the `loopbin` CLI
```

`loopbin --help` lists the commands; `loopbin <command> --help` shows a command's flags.
(For back-compatibility, `python main.py <command> …` also works.)

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
| `train` | train VaDE + cluster | `-num -d -ep -pre -if_pre -u -p -s -t` |
| `cluster` | predict with a trained model | `-d -m -u -p` |
| `merge` | merge small clusters | `-d -u -k -p` |

## Reproducibility — `-s/--seed`, `-t/--threads`

VaDE training is seed-sensitive. LoopBin makes a run **reproducible**: at a fixed seed the result is
**bit-identical** (every RNG is seeded — including the pretrain autoencoder — and TensorFlow deterministic
ops are enabled).

- `-s/--seed N` — random seed. Precedence: flag > `$LOOPBIN_SEED` > built-in default 73. (The example pins `-s 1`.)
- `-t/--threads N` — CPU threads. Precedence: flag > `$LOOPBIN_THREADS` > 16. Reproducibility holds at any
  **fixed** thread count — keep it the same across `pretrain` and `train`.

> The *effective* cluster count is stochastic across **different** seeds (the paper's runs return 5–7 even at
> a fixed `-num`). Run a few seeds and keep a clean draw.

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
