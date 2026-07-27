#!/usr/bin/env bash
# Robustness test: drop SMC1A from the DLD-1 example (slice last 32 cols), redo the whole
# seed-sweep -> consensus -> num-selection pipeline, and compare stability to the 4-mark run.
set -u
PY=/media/desk16/tly2104/miniforge3/envs/loopbin/bin/python
LB=~/ypf/projects_7_loopbin/LoopBin
SRC=$LB/example/run_seedsweep_num7ep2000
DST=$LB/example/run_drop_smc1a
PROT=CTCF,H3K27ac,H3K27me3
EP=500; CONC=2
mkdir -p $DST/02_process
echo "[$(date +%H:%M:%S)] === drop-SMC1A test ==="
# 1. slice off SMC1A (cols 352:384) -> 3-mark data
$PY -c "import numpy as np; d=np.load('$SRC/02_process/merged_log_data.npy'); np.save('$DST/02_process/merged_log_data.npy', d[:,:352]); print('sliced',d.shape,'->',(d[:,:352]).shape)"
cp $SRC/02_process/loop_file_analysis.bedpe $DST/02_process/
# 2. pretrain AE on the 3-mark data (seed 1)
mkdir -p $DST/03_pretrain
echo "[$(date +%H:%M:%S)] pretrain (3-mark)..."
TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=6 LOOPBIN_SEED=1 \
  $PY $LB/main.py pretrain -d $DST/02_process/merged_log_data.npy -u $DST/03_pretrain/ -t 6 > $DST/pretrain.log 2>&1 \
  && echo "  pretrain DONE" || { echo "  pretrain FAILED"; tail -15 $DST/pretrain.log; exit 1; }
# 3. seed sweep (num=7, ep=500, NO_PLOTS)
running=0
for s in 1 2 3 4 5; do
  od=$DST/seed$s; mkdir -p $od
  echo "[$(date +%H:%M:%S)] train seed=$s"
  ( TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=6 LOOPBIN_SEED=$s LOOPBIN_NO_PLOTS=1 \
      $PY $LB/main.py train -num 7 -d $DST/02_process/merged_log_data.npy \
      -if_pre True -pre $DST/03_pretrain/ -ep $EP -u $od/ -p $PROT -t 6 > $od/train.log 2>&1 ) &
  running=$((running+1)); [ "$running" -ge "$CONC" ] && { wait -n; running=$((running-1)); }
done
wait
echo "[$(date +%H:%M:%S)] seed sweep done; per-seed k:"
$PY -c "import numpy as np,collections; [print(' seed%s k=%d sizes%%='%(s,len(set(np.load('$DST/seed%s/labels.npy'%s).tolist())), )+' '.join('%.1f'%(v/15261*100 if False else v/len(np.load('$DST/seed%s/labels.npy'%s))*100) for v in sorted(collections.Counter(np.load('$DST/seed%s/labels.npy'%s).tolist()).values(),reverse=True))) for s in [1,2,3,4,5]]" 2>/dev/null || true
# 4. consensus + num-selection
echo "[$(date +%H:%M:%S)] consensus + num-selection..."
$PY /tmp/consensus_numselect.py $DST CTCF,H3K27ac,H3K27me3 1,2,3,4,5 2>&1 | grep -vE 'Axes3D|warnings.warn|FutureWarning|return f|UserWarning|ConvergenceWarning'
echo "[$(date +%H:%M:%S)] === DROP_SMC1A_DONE ==="
