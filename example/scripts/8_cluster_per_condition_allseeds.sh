#!/usr/bin/env bash
# 8_cluster_per_condition_allseeds.sh
# Per-condition (control/degron) cluster prediction for all 6 seed models from the sweep.
# 12 runs (6 seeds x 2 conds), throttled to <=3 concurrent (box-overload lesson). Detached-safe.
set -uo pipefail

ROOT=/media/desk16/tly2104/ypf/projects_7_loopbin
LB=$ROOT/LoopBin
DIR=$LB/example/run_seedsweep_num7ep2000
PROT=CTCF,H3K27ac,H3K27me3,SMC1A
SEEDS="73 1 2 3 4 5"
CONDS="control degron"
MAXJOBS=3

source /media/desk16/tly2104/miniforge3/etc/profile.d/conda.sh
conda activate loopbin
die(){ echo "ERROR: $*" >&2; exit 1; }
[ -d "$DIR/02_process" ] || die "no $DIR/02_process symlink"

echo "[$(date +%H:%M:%S)] ===== per-condition cluster: seeds [$SEEDS] x [$CONDS], <=$MAXJOBS concurrent ====="
for s in $SEEDS; do
  [ -f "$DIR/seed$s/saved_model.pb" ] || { echo "  seed$s: no model -> skip"; continue; }
  for cond in $CONDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
    od="$DIR/seed$s/05_cluster/$cond"; mkdir -p "$od"
    echo "[$(date +%H:%M:%S)]  launch seed$s/$cond"
    ( TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=6 \
      python "$LB/main.py" -f cluster -d "$DIR/02_process/$cond/log_data.npy" -m "$DIR/seed$s/" -u "$od/" -p "$PROT" \
      > "$od/cluster.log" 2>&1 ) &
  done
done
wait
echo "[$(date +%H:%M:%S)] ===== ALL PER-CONDITION CLUSTERING DONE ====="

python - "$DIR" $SEEDS <<PY
import numpy as np, collections, os, sys
DIR=sys.argv[1]; seeds=sys.argv[2:]
for s in seeds:
  for cond in ["control","degron"]:
    f=os.path.join(DIR,f"seed{s}","05_cluster",cond,"labels.npy")
    if not os.path.isfile(f): print(f"  seed{s}/{cond}: no labels (check {cond}/cluster.log)"); continue
    lab=np.load(f); c=collections.Counter(lab.tolist()); t=len(lab)
    print(f"  seed{s}/{cond}: n={t} -> "+", ".join(f"c{k}:{v}({100*v/t:.0f}%)" for k,v in sorted(c.items())))
PY
echo "===== DONE -> $DIR/seed*/05_cluster/{control,degron}/ ====="
