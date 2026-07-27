#!/usr/bin/env bash
# 1_run_loopbin_pipeline.sh
# Run the LoopBin pipeline end-to-end on the hg38 DLD-1 example (preprocess -> cluster).
# Smoke test: reduced training epochs to confirm the pipeline runs on this CPU box.
#
# Uses the generalized code: --chrom-sizes (no hardcoded genome) + --resolution 8000
# (the LoopBin paper's native resolution). The merged mcool has no /resolutions/8000, so this
# uses the prebuilt example_data/{control,degron}_8kb.mcool (balanced /resolutions/8000;
# rebuild with 0_change_into_8kb_mcool.sh if absent). NO mm10/hg38 swap needed.
set -uo pipefail

ROOT=/media/desk16/tly2104/ypf/projects_7_loopbin
LB=$ROOT/LoopBin
EX=$LB/example/example_data
OUT=$LB/example/run_output
PROT=CTCF,H3K27ac,H3K27me3,SMC1A
RES=8000
NCPU=8
NCLUST=10
EPOCHS=50
REF_SIZES=""   # optional chrom.sizes override. EMPTY => derive from the mcool (= exactly the
               # chroms in the data: chr1-22,X = 23). The downloaded hg38.main.chrom.sizes (25,
               # incl chrY/chrM) is NOT used by default: this Micro-C mcool/loops have only 23,
               # so tiling chrY/chrM would just make bedgraphs `process` never reads.

source /media/desk16/tly2104/miniforge3/etc/profile.d/conda.sh
conda activate loopbin
mkdir -p "$OUT"
die(){ echo "ERROR: $*" >&2; exit 1; }

# ---- chrom.sizes: derive from the mcool by default (= exactly the chroms in the data), so
#      preprocess tiles precisely what `process` consumes. Set REF_SIZES above to override. ----
if [ -n "$REF_SIZES" ] && [ -s "$REF_SIZES" ]; then
  CHROM_SIZES=$REF_SIZES
  echo "[setup] using chrom.sizes override: $CHROM_SIZES"
else
  CHROM_SIZES=$EX/hg38.from_mcool.chrom.sizes
  echo "[setup] deriving chrom.sizes from the 8 kb mcool (the data's own chrom set) ..."
  python -c "import cooler; cooler.Cooler('$EX/control_8kb.mcool::/resolutions/$RES').chromsizes.to_csv('$CHROM_SIZES', sep='\t', header=False)"
fi
echo "[setup] chrom.sizes: $(wc -l < "$CHROM_SIZES") chromosomes ($CHROM_SIZES)"

# condition -> prebuilt 8 kb mcool (balanced /resolutions/8000; built by 0_change_into_8kb_mcool.sh)
declare -A MCOOL=( [control]=control_8kb.mcool [degron]=degron_8kb.mcool )
for cond in control degron; do
  [ -s "$EX/${MCOOL[$cond]}" ] || die "missing $EX/${MCOOL[$cond]} -- build it with: bash $(dirname "$0")/0_change_into_8kb_mcool.sh"
done
# bigwig mapping (rep1; Ctrl=control, Aux=degron)
declare -A BW_control=( [CTCF]=GSM6245909_DLD1asyn_Ctrl_CTCF_rep1.bw [SMC1A]=GSM6245913_DLD1asyn_Ctrl_SMC1A_rep1.bw [H3K27ac]=GSM6245917_DLD1asyn_Ctrl_H3K27ac_rep1.bw [H3K27me3]=GSM6245919_DLD1asyn_Ctrl_H3K27me3_rep1.bw )
declare -A BW_degron=(  [CTCF]=GSM6245911_DLD1asyn_Aux_CTCF_rep1.bw  [SMC1A]=GSM6245915_DLD1asyn_Aux_SMC1A_rep1.bw  [H3K27ac]=GSM6245918_DLD1asyn_Aux_H3K27ac_rep1.bw  [H3K27me3]=GSM6245921_DLD1asyn_Aux_H3K27me3_rep1.bw )

# ---- STEP 1: preprocess (bigwig -> bedgraph; pure Python, --chrom-sizes) ----
echo "===== STEP 1: preprocess (res=$RES) ====="
for cond in control degron; do
  pg=$OUT/01_preprocess/$cond; mkdir -p "$pg"
  declare -n BW="BW_$cond"
  for name in CTCF SMC1A H3K27ac H3K27me3; do
    bw=${BW[$name]}; [ -f "$EX/$bw" ] || die "missing bigwig $EX/$bw"
    echo "  [$cond/$name] $bw"
    python "$LB/main.py" preprocess -b "$EX/$bw" -g "$pg/" -n "$name" -res "$RES" -cs "$CHROM_SIZES" || die "preprocess $cond/$name"
  done
  n=$(ls "$pg"/*.bedgraph 2>/dev/null | wc -l); echo "  [$cond] bedgraphs: $n"
  [ "$n" -gt 0 ] || die "no bedgraphs for $cond"
done

# ---- STEP 2: process (loops + merged mcool @10kb + bedgraph -> npy) ----
echo "===== STEP 2: process ====="
for cond in control degron; do
  po=$OUT/02_process/$cond; mkdir -p "$po"
  echo "  [$cond] process (mcool ${MCOOL[$cond]} ::/resolutions/$RES)"
  python "$LB/main.py" process -l "$EX/${cond}_loops_labeled.bedpe" -c "$EX/${MCOOL[$cond]}" \
    -g "$OUT/01_preprocess/$cond/" -p "$PROT" -r "$NCPU" -u "$po" -res "$RES" || die "process $cond"
  [ -f "$po/raw/raw_micro_c.npy" ] || die "process: no $po/raw/raw_micro_c.npy"
done

# ---- STEP 3: normalize ----
echo "===== STEP 3: normalize ====="
python "$LB/main.py" normalize -e control,degron -u "$OUT/02_process/" || die "normalize"
[ -f "$OUT/02_process/merged_log_data.npy" ] || die "no merged_log_data.npy"

# ---- STEP 4: pretrain AE ----
echo "===== STEP 4: pretrain ====="
mkdir -p "$OUT/03_pretrain"
python "$LB/main.py" pretrain -d "$OUT/02_process/merged_log_data.npy" -u "$OUT/03_pretrain/" || die "pretrain"

# ---- STEP 5: train VADE ----
echo "===== STEP 5: train (num=$NCLUST ep=$EPOCHS) ====="
mkdir -p "$OUT/04_train"
python "$LB/main.py" train -num "$NCLUST" -d "$OUT/02_process/merged_log_data.npy" \
  -if_pre True -pre "$OUT/03_pretrain/" -ep "$EPOCHS" -u "$OUT/04_train/" -p "$PROT" || die "train"

# ---- STEP 6: cluster (per condition) ----
echo "===== STEP 6: cluster ====="
for cond in control degron; do
  co=$OUT/05_cluster/$cond; mkdir -p "$co"
  python "$LB/main.py" cluster -d "$OUT/02_process/$cond/log_data.npy" -m "$OUT/04_train/" -u "$co/" -p "$PROT" || die "cluster $cond"
  [ -f "$co/labels_loops.bedpe" ] || echo "  WARN: no labels_loops.bedpe for $cond"
done

echo "===== PIPELINE DONE -> $OUT ====="
find "$OUT/05_cluster" -type f | sort
