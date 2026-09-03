#!/usr/bin/env bash
# 7_seedsweep_num7ep2000.sh
# Seed sweep to land a Fig-1D-like draw: 6 seeds (73,1,2,3,4,5) at num=7/ep=2000, reusing the existing
# lr=0.002 AE. Tagged dir + symlinked data (no copy/recompute). Detached/background-safe.
# seed 73 is re-run here as a determinism check vs run_output/04_train_num7_ep2000 (TF_DETERMINISTIC_OPS off).
set -uo pipefail

ROOT=/media/desk16/tly2104/ypf/projects_7_loopbin
LB=$ROOT/LoopBin
RUN=$LB/example/run_output
DIR=$LB/example/run_seedsweep_num7ep2000
PROT=CTCF,H3K27ac,H3K27me3,SMC1A
NCLUST=7; EPOCHS=2000
SEEDS="73 1 2 3 4 5"

source /media/desk16/tly2104/miniforge3/etc/profile.d/conda.sh
conda activate loopbin
die(){ echo "ERROR: $*" >&2; exit 1; }

# tagged dir + symlinks (reuse processed data + the existing lr=0.002 AE)
mkdir -p "$DIR"
[ -e "$DIR/02_process" ]  || ln -s "$RUN/02_process"  "$DIR/02_process"
[ -e "$DIR/03_pretrain" ] || ln -s "$RUN/03_pretrain" "$DIR/03_pretrain"
[ -f "$DIR/02_process/merged_log_data.npy" ]  || die "no merged_log_data via symlink"
[ -f "$DIR/02_process/loop_file_analysis.bedpe" ] || die "no merged loop_file_analysis.bedpe via symlink"
[ -f "$DIR/03_pretrain/saved_model.pb" ]      || die "no AE via symlink"

echo "[$(date +%H:%M:%S)] ===== SEED SWEEP num=$NCLUST ep=$EPOCHS seeds=[$SEEDS] ====="
for s in $SEEDS; do
  od="$DIR/seed$s"; mkdir -p "$od"
  echo "[$(date +%H:%M:%S)]  launch seed=$s -> $od (log: $od/train.log)"
  TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=6 LOOPBIN_SEED=$s \
    python "$LB/main.py" -f train -num "$NCLUST" -d "$DIR/02_process/merged_log_data.npy" \
    -if_pre True -pre "$DIR/03_pretrain/" -ep "$EPOCHS" -u "$od/" -p "$PROT" \
    > "$od/train.log" 2>&1 &
done
wait
echo "[$(date +%H:%M:%S)] ===== ALL SEEDS DONE ====="

# per-seed summary: cluster count + size distribution
python - "$DIR" $SEEDS <<PY
import numpy as np, collections, sys, os
DIR=sys.argv[1]; seeds=sys.argv[2:]
for s in seeds:
    f=os.path.join(DIR,f"seed{s}","labels.npy")
    if not os.path.isfile(f): print(f"  seed {s}: NO labels.npy (check seed{s}/train.log)"); continue
    lab=np.load(f); c=collections.Counter(lab.tolist()); t=len(lab)
    dist=", ".join(f"c{k}:{v}({100*v/t:.0f}%)" for k,v in sorted(c.items()))
    print(f"  seed {s}: {len(c)} clusters, n={t} -> {dist}")
PY
echo "===== SWEEP DONE -> $DIR ====="
