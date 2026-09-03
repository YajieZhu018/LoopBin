#!/usr/bin/env bash
# 0_change_into_8kb_mcool.sh
# -----------------------------------------------------------------------------
# Build a hictk-VALID, BALANCED 8 kb (/resolutions/8000) mcool for each condition,
# so the example reproduces the author's results at 8 kb resolution.
#
# WHY 8 kb:
#   LoopBin hardcodes the micro-C resolution at 8 kb: function.py:142 opens
#   cool_file + "::/resolutions/8000", and processing.py reads it BALANCED
#   (processing.py:135  hic.matrix(balance=True,...)). The authors' input mcool
#   (DLD1_async_14h*_matrix_1kb.mcool) was 1 kb-zoomified so it had 8 kb natively.
#
# WHY WE MUST BUILD IT:
#   Our merged-replicate mcool follows the standard cooler ladder
#   (500,1000,2500,5000,10000,...) and has NO 8 kb. 8000 = 1000*8 (integer), so we
#   coarsen the 1 kb level by factor 8; 2500/5000/10000 cannot reach 8 kb evenly.
#
# METHOD (two steps, both produce correct output):
#   1) cooler coarsen 1kb -> 8kb  (raw counts; aggregates 8x8 blocks of 1 kb bins)
#   2) cooler zoomify the 8 kb cool into a proper .mcool + --balance
#      zoomify writes valid MCOOL root metadata (hictk-valid); --balance adds the
#      'weight' column LoopBin needs. (A plain `cooler cp` into a group does NOT
#      write that metadata -> hictk rejects it; do not use cp.)
#
# NOTE: this uses the merged-replicate matrix. For a byte-faithful reproduction of
#   the authors' exact file, download DLD1_async_14h{control,degron}_matrix_1kb.mcool
#   instead (it already contains a balanced 8 kb level).
# -----------------------------------------------------------------------------
set -euo pipefail

EX=/media/desk16/tly2104/ypf/projects_7_loopbin/LoopBin/example/example_data
source /media/desk16/tly2104/miniforge3/etc/profile.d/conda.sh
conda activate loopbin

# condition -> source merged-replicate mcool (500 bp base, has /resolutions/1000)
declare -A SRC=(
  [control]=GSM5394172_control_mergeRep_500bp.mcool
  [degron]=GSM5394173_degron_mergeRep_500bp.mcool
)

is_balanced () {  # $1 = mcool path ; succeeds if /resolutions/8000 exists AND is balanced
  cooler ls "$1" 2>/dev/null | grep -q "/resolutions/8000" || return 1
  python -c "import cooler,sys; sys.exit(0 if 'weight' in cooler.Cooler('$1::/resolutions/8000').bins().columns else 1)"
}

for cond in control degron; do
  out=$EX/${cond}_8kb.mcool
  if is_balanced "$out"; then
    echo "[$cond] $out already has a balanced /resolutions/8000 -> skip"
    continue
  fi
  tmp=/tmp/${cond}_8000.cool

  # 1) coarsen 1 kb -> 8 kb (reuse a valid leftover tmp if present, else build it)
  if cooler info "$tmp" >/dev/null 2>&1; then
    echo "[$cond] (1/2) reusing existing coarsened cool: $tmp"
  else
    echo "[$cond] (1/2) coarsen 1000 -> 8000 (k=8) ..."
    cooler coarsen -k 8 -p 8 -o "$tmp" "$EX/${SRC[$cond]}::/resolutions/1000"
  fi

  # 2) wrap into a valid .mcool and balance the 8 kb level
  echo "[$cond] (2/2) zoomify into valid mcool + balance (8 kb) ..."
  rm -f "$out"
  cooler zoomify -p 8 -r 8000 --balance --balance-args "-p 8" -o "$out" "$tmp"
  echo "[$cond] done -> $out"
done

echo
echo "=== verify (valid MCOOL + balanced 8 kb) ==="
for cond in control degron; do
  out=$EX/${cond}_8kb.mcool
  echo "-- $cond --"
  cooler ls "$out"
  python - "$out" <<PY
import sys, cooler
c = cooler.Cooler(sys.argv[1] + "::/resolutions/8000")
print("  nbins:", c.info["nbins"], "| balanced(weight col):", "weight" in c.bins().columns)
PY
  command -v hictk >/dev/null 2>&1 && { echo -n "  hictk: "; hictk metadata "$out" 2>&1 | grep -E "format\"|format-version|8000" | tr -d ' '; } || true
done
echo "ALL DONE."
