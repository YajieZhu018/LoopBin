#!/usr/bin/env bash
# 5_figure1D_num7_ep2000.sh
# Faithful reproduction of LoopBin paper Figure 1D: a SINGLE VaDE run with the paper's exact knobs
# (Methods: "GMM of seven components" -> num=7; 2000 epochs), reusing the 200-epoch pretrained AE.
# NO consensus. Reuses 03_pretrain (AE) + 02_process outputs. Detached/background-safe.
set -uo pipefail

ROOT=/media/desk16/tly2104/ypf/projects_7_loopbin
LB=$ROOT/LoopBin
OUT=$LB/example/run_output
P=$OUT/02_process
PROT=CTCF,H3K27ac,H3K27me3,SMC1A
NCLUST=7        # paper: "GMM of seven components"
EPOCHS=2000     # paper Methods: 2000 epochs
TRAIN=$OUT/04_train_num7_ep2000
CLU=$OUT/05_cluster_num7_ep2000

# thread caps (shared box, ~39 load from others) -> keep us to ~20 cores
export TF_NUM_INTRAOP_THREADS=20 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=20

source /media/desk16/tly2104/miniforge3/etc/profile.d/conda.sh
conda activate loopbin
die(){ echo "ERROR: $*" >&2; exit 1; }

[ -f "$OUT/03_pretrain/saved_model.pb" ] || die "no saved AE at 03_pretrain"
[ -f "$P/merged_log_data.npy" ]          || die "no merged_log_data.npy"
[ -s "$P/loop_file_analysis.bedpe" ] || cat "$P/control/loop_file_analysis.bedpe" "$P/degron/loop_file_analysis.bedpe" > "$P/loop_file_analysis.bedpe"

echo "[$(date +%H:%M:%S)] ===== TRAIN VaDE (num=$NCLUST, ep=$EPOCHS, reuse AE) -- paper Fig 1D config ====="
mkdir -p "$TRAIN"
python "$LB/main.py" -f train -num "$NCLUST" -d "$P/merged_log_data.npy" \
  -if_pre True -pre "$OUT/03_pretrain/" -ep "$EPOCHS" -u "$TRAIN/" -p "$PROT" || die "train"
[ -f "$TRAIN/saved_model.pb" ] || die "train: no saved VADE model"

echo "[$(date +%H:%M:%S)] ===== CLUSTER control + degron (writes all_clusters.pdf = Fig 1D) ====="
for cond in control degron; do
  co=$CLU/$cond; mkdir -p "$co"
  echo "  [$cond] cluster"
  python "$LB/main.py" -f cluster -d "$P/$cond/log_data.npy" -m "$TRAIN/" -u "$co/" -p "$PROT" || die "cluster $cond"
  python - "$co/labels.npy" "$cond" <<PY
import numpy as np, collections, sys
lab=np.load(sys.argv[1]); c=collections.Counter(lab.tolist()); t=len(lab)
print(f"  [{sys.argv[2]}] n={t}, {len(c)} clusters -> "+", ".join(f"c{k}:{v}({100*v/t:.1f}%)" for k,v in sorted(c.items())))
PY
done

echo "[$(date +%H:%M:%S)] ===== FIG1D DONE ====="
find "$TRAIN" "$CLU" -name "all_clusters.pdf" | sort
