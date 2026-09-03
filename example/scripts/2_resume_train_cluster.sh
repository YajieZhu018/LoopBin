#!/usr/bin/env bash
# 2_resume_train_cluster.sh
# Resume the 8 kb baseline from the already-saved pretrain AE (skips re-running pretrain):
#   build the merged loop file (normalize doesn't) -> train (reuse AE) -> cluster per condition.
# Reuses run_output/03_pretrain (saved AE) + run_output/02_process (process + normalize outputs).
set -uo pipefail

ROOT=/media/desk16/tly2104/ypf/projects_7_loopbin
LB=$ROOT/LoopBin
OUT=$LB/example/run_output
P=$OUT/02_process
PROT=CTCF,H3K27ac,H3K27me3,SMC1A
NCLUST=10
EPOCHS=50

source /media/desk16/tly2104/miniforge3/etc/profile.d/conda.sh
conda activate loopbin
die(){ echo "ERROR: $*" >&2; exit 1; }

[ -f "$OUT/03_pretrain/saved_model.pb" ] || die "no saved AE at $OUT/03_pretrain"
[ -f "$P/merged_log_data.npy" ]          || die "no merged_log_data.npy"

# normalize concatenates the .npy arrays (control then degron) but NOT the loop file that
# train's cluster_data_inner_func reads. Build the merged bedpe in the SAME order (control, degron).
if [ ! -s "$P/loop_file_analysis.bedpe" ]; then
  echo "[setup] building merged $P/loop_file_analysis.bedpe (control + degron) ..."
  cat "$P/control/loop_file_analysis.bedpe" "$P/degron/loop_file_analysis.bedpe" > "$P/loop_file_analysis.bedpe"
fi
echo "[setup] merged loop file: $(wc -l < "$P/loop_file_analysis.bedpe") lines (expect 43744)"

echo "===== STEP 5: train (reuse saved AE; num=$NCLUST ep=$EPOCHS) ====="
mkdir -p "$OUT/04_train"
python "$LB/main.py" -f train -num "$NCLUST" -d "$P/merged_log_data.npy" \
  -if_pre True -pre "$OUT/03_pretrain/" -ep "$EPOCHS" -u "$OUT/04_train/" -p "$PROT" || die "train"
[ -f "$OUT/04_train/saved_model.pb" ] || die "train: no saved VADE model"

echo "===== STEP 6: cluster ====="
for cond in control degron; do
  co=$OUT/05_cluster/$cond; mkdir -p "$co"
  echo "  [$cond] cluster"
  python "$LB/main.py" -f cluster -d "$P/$cond/log_data.npy" -m "$OUT/04_train/" -u "$co/" -p "$PROT" || die "cluster $cond"
  [ -f "$co/labels_loops.bedpe" ] || echo "  WARN: no labels_loops.bedpe for $cond"
done

echo "===== RESUME DONE -> $OUT/05_cluster ====="
find "$OUT/05_cluster" -type f | sort
