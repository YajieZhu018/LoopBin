#!/usr/bin/env bash
# 1_run_pipeline.sh — LoopBin end-to-end on the DLD-1 GSE178593 example (reproduces paper Fig 1D).
#
#   (0) build a balanced 8 kb mcool  -> (1) preprocess bigWigs -> (2) process loops+mcool+marks
#   -> (3) normalize/merge conditions -> (4) pretrain AE -> (5) train VaDE (num=7, ep=2000)
#   -> (6) cluster.   Output: run_output/05_cluster/<cond>/all_clusters.pdf
#         (per-cluster Micro-C + CTCF/H3K27ac/H3K27me3/SMC1A 2D averages; cf. paper Fig 1D).
#
# PREREQUISITES
#   - conda env from ../../loopbin.yml:   conda activate loopbin
#   - run  scripts/0_download_GSE178593.sh  first (downloads the bigWigs + mcools into example_data/)
#   - place the loop calls in  example_data/{control,degron}_loops_labeled.bedpe
#     (the merged 5 kb + 10 kb HiCCUPS/cooltools/mustache loop set from the paper; not redistributed here)
#
# =====================================================================================================
# >>> SEED / REPRODUCIBILITY NOTE — PLEASE READ <<<
#   LoopBin clustering is STOCHASTIC. VaDE has many local optima, and the *punctate* active marks
#   (CTCF / SMC1A / H3K27ac) are a WEAK attractor versus the *broad* H3K27me3 — so the "active" loop
#   classes only resolve on some runs. In our 6-seed sweep, ONLY ~3 of 6 runs were "clean" (active +
#   repressive + lonely classes all resolved as substantial clusters); the other ~3 produced degenerate
#   stragglers. Worse, training is NOT deterministic (TF_DETERMINISTIC_OPS is off and the pretrain AE is
#   unseeded), so even a fixed seed does not reproduce a given result exactly.
#
#   => DEFAULT seed is 1 (the seed of our clean reproduction), but EXPECT TO RUN A FEW SEEDS and keep a
#      CLEAN draw — one with no cluster < ~2 % AND all three axes (active / repressive / lonely) present:
#          for s in 1 2 3 4 5; do LOOPBIN_SEED=$s NCLUST=7 EPOCHS=2000 bash 1_run_pipeline.sh; done
#   (A robust fix — per-channel mark normalization so the punctate active marks compete — is planned.)
# =====================================================================================================
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LB=$(cd "$SCRIPT_DIR/../.." && pwd)              # repo root (…/LoopBin)
EX=$LB/example/example_data
OUT=$LB/example/run_output
PROT=CTCF,H3K27ac,H3K27me3,SMC1A
RES=8000
NCPU=${NCPU:-8}
NCLUST=${NCLUST:-7}            # paper: GMM of 7 components
EPOCHS=${EPOCHS:-2000}         # paper: 2000 epochs
export LOOPBIN_SEED=${LOOPBIN_SEED:-1}   # seed of our clean reproduction; override per run
REF_SIZES=""                  # optional chrom.sizes override; empty => derive from the mcool (chr1-22,X)

command -v cooler >/dev/null || { echo "ERROR: activate the LoopBin env first (conda activate loopbin)"; exit 1; }
mkdir -p "$OUT"
die(){ echo "ERROR: $*" >&2; exit 1; }

# condition -> downloaded merged mcool (500 bp base, has /resolutions/1000) and the 8 kb mcool we build
declare -A SRC=(   [control]=GSM5394172_control_mergeRep_500bp.mcool [degron]=GSM5394173_degron_mergeRep_500bp.mcool )
declare -A MCOOL=( [control]=control_8kb.mcool                       [degron]=degron_8kb.mcool )

# ---- STEP 0: ensure a hictk-valid, BALANCED 8 kb mcool (LoopBin hardcodes /resolutions/8000) ----
# The merged mcool follows the standard cooler ladder (no 8 kb). 8000 = 1000*8, so coarsen the 1 kb level
# by 8 then zoomify+balance into a valid mcool. (Do NOT `cooler cp` into a group — it omits MCOOL metadata.)
echo "===== STEP 0: ensure balanced 8 kb mcool ====="
for cond in control degron; do
  out=$EX/${MCOOL[$cond]}
  if cooler ls "$out" 2>/dev/null | grep -q "/resolutions/8000" && \
     python -c "import cooler,sys; sys.exit(0 if 'weight' in cooler.Cooler('$out::/resolutions/8000').bins().columns else 1)" 2>/dev/null; then
    echo "  [$cond] $out already balanced @8kb -> skip"; continue
  fi
  [ -s "$EX/${SRC[$cond]}" ] || die "missing $EX/${SRC[$cond]} (run scripts/0_download_GSE178593.sh first)"
  tmp=$(mktemp /tmp/${cond}_8000.XXXX.cool)
  echo "  [$cond] coarsen 1000 -> 8000 (k=8) ..."; cooler coarsen -k 8 -p "$NCPU" -o "$tmp" "$EX/${SRC[$cond]}::/resolutions/1000"
  echo "  [$cond] zoomify + balance -> $out ...";   rm -f "$out"; cooler zoomify -p "$NCPU" -r 8000 --balance --balance-args "-p $NCPU" -o "$out" "$tmp"; rm -f "$tmp"
done

# ---- chrom.sizes: derive from the mcool (= exactly chr1-22,X) unless REF_SIZES given ----
if [ -n "$REF_SIZES" ] && [ -s "$REF_SIZES" ]; then CHROM_SIZES=$REF_SIZES
else CHROM_SIZES=$EX/hg38.from_mcool.chrom.sizes
  python -c "import cooler; cooler.Cooler('$EX/control_8kb.mcool::/resolutions/$RES').chromsizes.to_csv('$CHROM_SIZES', sep='\t', header=False)"
fi
echo "[setup] chrom.sizes: $(wc -l < "$CHROM_SIZES") chromosomes"

# bigWig mapping (rep1; Ctrl=control, Aux=degron)
declare -A BW_control=( [CTCF]=GSM6245909_DLD1asyn_Ctrl_CTCF_rep1.bw [SMC1A]=GSM6245913_DLD1asyn_Ctrl_SMC1A_rep1.bw [H3K27ac]=GSM6245917_DLD1asyn_Ctrl_H3K27ac_rep1.bw [H3K27me3]=GSM6245919_DLD1asyn_Ctrl_H3K27me3_rep1.bw )
declare -A BW_degron=(  [CTCF]=GSM6245911_DLD1asyn_Aux_CTCF_rep1.bw  [SMC1A]=GSM6245915_DLD1asyn_Aux_SMC1A_rep1.bw  [H3K27ac]=GSM6245918_DLD1asyn_Aux_H3K27ac_rep1.bw  [H3K27me3]=GSM6245921_DLD1asyn_Aux_H3K27me3_rep1.bw )

# ---- STEP 1: preprocess (bigWig -> bedgraph) ----
echo "===== STEP 1: preprocess (res=$RES) ====="
for cond in control degron; do
  pg=$OUT/01_preprocess/$cond; mkdir -p "$pg"; declare -n BW="BW_$cond"
  for name in CTCF SMC1A H3K27ac H3K27me3; do
    bw=${BW[$name]}; [ -f "$EX/$bw" ] || die "missing bigwig $EX/$bw (run 0_download_GSE178593.sh)"
    python "$LB/main.py" -f preprocess -b "$EX/$bw" -g "$pg/" -n "$name" -res "$RES" -cs "$CHROM_SIZES" || die "preprocess $cond/$name"
  done
done

# ---- STEP 2: process (loops + 8 kb mcool + bedgraphs -> npy) ----
echo "===== STEP 2: process ====="
for cond in control degron; do
  loops=$EX/${cond}_loops_labeled.bedpe
  [ -s "$loops" ] || die "missing loop file $loops (see README — supply the loop calls)"
  po=$OUT/02_process/$cond; mkdir -p "$po"
  python "$LB/main.py" -f process -l "$loops" -c "$EX/${MCOOL[$cond]}" -g "$OUT/01_preprocess/$cond/" -p "$PROT" -r "$NCPU" -u "$po" -res "$RES" || die "process $cond"
done

# ---- STEP 3: normalize (merge control + degron) ----
echo "===== STEP 3: normalize ====="
python "$LB/main.py" -f normalize -e control,degron -u "$OUT/02_process/" || die "normalize"

# ---- STEP 4: pretrain AE ----
echo "===== STEP 4: pretrain ====="; mkdir -p "$OUT/03_pretrain"
python "$LB/main.py" -f pretrain -d "$OUT/02_process/merged_log_data.npy" -u "$OUT/03_pretrain/" || die "pretrain"

# ---- STEP 5: train VaDE (seed via LOOPBIN_SEED; default 1) ----
echo "===== STEP 5: train (num=$NCLUST ep=$EPOCHS seed=$LOOPBIN_SEED) ====="; mkdir -p "$OUT/04_train"
python "$LB/main.py" -f train -num "$NCLUST" -d "$OUT/02_process/merged_log_data.npy" -if_pre True -pre "$OUT/03_pretrain/" -ep "$EPOCHS" -u "$OUT/04_train/" -p "$PROT" || die "train"

# ---- STEP 6: cluster (per condition) ----
echo "===== STEP 6: cluster ====="
for cond in control degron; do
  co=$OUT/05_cluster/$cond; mkdir -p "$co"
  python "$LB/main.py" -f cluster -d "$OUT/02_process/$cond/log_data.npy" -m "$OUT/04_train/" -u "$co/" -p "$PROT" || die "cluster $cond"
done

echo "===== DONE — Fig-1D-style figures: $OUT/05_cluster/{control,degron}/all_clusters.pdf ====="
