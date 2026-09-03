#!/usr/bin/env bash
# 3_rerun_ep1000_figureD.sh
# "Pragmatic closer to Figure D" on CPU (no GPU): train VaDE 1 run @ ep=1000 reusing the
# already-saved pretrained AE, then cluster control+degron. The cluster step's all_clusters.pdf
# IS the Figure-D-style plot (each cluster a row; Micro-C + CTCF/SMC1A/H3K27ac/H3K27me3 columns;
# 16x16 2D averages). Reuses 03_pretrain (AE) + 02_process outputs. Detached/background-safe.
set -uo pipefail

ROOT=/media/desk16/tly2104/ypf/projects_7_loopbin
LB=$ROOT/LoopBin
OUT=$LB/example/run_output
P=$OUT/02_process
PROT=CTCF,H3K27ac,H3K27me3,SMC1A
NCLUST=10
EPOCHS=1000
TRAIN=$OUT/04_train_ep1000
CLU=$OUT/05_cluster_ep1000

source /media/desk16/tly2104/miniforge3/etc/profile.d/conda.sh
conda activate loopbin
die(){ echo "ERROR: $*" >&2; exit 1; }

[ -f "$OUT/03_pretrain/saved_model.pb" ] || die "no saved AE at 03_pretrain"
[ -f "$P/merged_log_data.npy" ]          || die "no merged_log_data.npy"
# train needs the merged loop file (normalize doesn't write it); build if missing (control+degron order)
[ -s "$P/loop_file_analysis.bedpe" ] || cat "$P/control/loop_file_analysis.bedpe" "$P/degron/loop_file_analysis.bedpe" > "$P/loop_file_analysis.bedpe"

echo "[$(date +%H:%M:%S)] ===== STEP A: train VaDE (num=$NCLUST, ep=$EPOCHS, reuse pretrained AE) ====="
mkdir -p "$TRAIN"
python "$LB/main.py" -f train -num "$NCLUST" -d "$P/merged_log_data.npy" \
  -if_pre True -pre "$OUT/03_pretrain/" -ep "$EPOCHS" -u "$TRAIN/" -p "$PROT" || die "train"
[ -f "$TRAIN/saved_model.pb" ] || die "train: no saved VADE model"

echo "[$(date +%H:%M:%S)] ===== STEP B: cluster control + degron (auto-writes all_clusters.pdf = Figure D) ====="
for cond in control degron; do
  co=$CLU/$cond; mkdir -p "$co"
  echo "  [$cond] cluster"
  python "$LB/main.py" -f cluster -d "$P/$cond/log_data.npy" -m "$TRAIN/" -u "$co/" -p "$PROT" || die "cluster $cond"
  [ -f "$co/all_clusters.pdf" ] || echo "  WARN: no all_clusters.pdf for $cond"
  python - "$co/labels.npy" "$cond" <<PY
import numpy as np, collections, sys
lab=np.load(sys.argv[1]); c=collections.Counter(lab.tolist()); t=len(lab)
print(f"  [{sys.argv[2]}] n={t}, {len(c)} clusters -> "+", ".join(f"c{k}:{v}({100*v/t:.0f}%)" for k,v in sorted(c.items())))
PY
done

echo "[$(date +%H:%M:%S)] ===== DONE -> Figure-D-style plots ====="
find "$CLU" -name "all_clusters.pdf" | sort
