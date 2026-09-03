#!/usr/bin/env bash
# Full leave-one-out robustness test on DLD-1: drop CTCF, then H3K27ac, then H3K27me3 (SMC1A done).
# Each: slice -> pretrain -> 5-seed sweep (ep500, NO_PLOTS, num7) -> consensus+num-selection. Sequential.
set -u
PY=/media/desk16/tly2104/miniforge3/envs/loopbin/bin/python
LB=~/ypf/projects_7_loopbin/LoopBin
SRC=$LB/example/run_seedsweep_num7ep2000/02_process
EP=500; CONC=2

run_one(){
  local name=$1 drop=$2
  local DST=$LB/example/run_$name
  mkdir -p $DST/02_process $DST/03_pretrain
  echo "[$(date +%H:%M:%S)] ===== $name (drop $drop) ====="
  local marks
  marks=$($PY /tmp/slice_drop.py $SRC/merged_log_data.npy $DST/02_process/merged_log_data.npy $drop)
  echo "  remaining marks: $marks"
  cp $SRC/loop_file_analysis.bedpe $DST/02_process/
  echo "[$(date +%H:%M:%S)] $name pretrain..."
  TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=6 LOOPBIN_SEED=1 \
    $PY $LB/main.py pretrain -d $DST/02_process/merged_log_data.npy -u $DST/03_pretrain/ -t 6 > $DST/pretrain.log 2>&1 \
    || { echo "  $name pretrain FAILED"; tail -12 $DST/pretrain.log; return 1; }
  local running=0
  for s in 1 2 3 4 5; do
    local od=$DST/seed$s; mkdir -p $od
    ( TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=6 LOOPBIN_SEED=$s LOOPBIN_NO_PLOTS=1 \
        $PY $LB/main.py train -num 7 -d $DST/02_process/merged_log_data.npy \
        -if_pre True -pre $DST/03_pretrain/ -ep $EP -u $od/ -p $marks -t 6 > $od/train.log 2>&1 ) &
    running=$((running+1)); [ "$running" -ge "$CONC" ] && { wait -n; running=$((running-1)); }
  done
  wait
  echo "[$(date +%H:%M:%S)] $name consensus + num-selection:"
  $PY /tmp/consensus_numselect.py $DST $marks 1,2,3,4,5 2>&1 | grep -vE 'Axes3D|warnings.warn|FutureWarning|return f|UserWarning|ConvergenceWarning'
  echo "=== ${name}_DONE ==="
}

run_one drop_ctcf     CTCF
run_one drop_h3k27ac  H3K27ac
run_one drop_h3k27me3 H3K27me3
echo "[$(date +%H:%M:%S)] === LOO_ALL_DONE ==="
