#!/usr/bin/env bash
# download_GSE178593.sh
# Download ALL GSE178593 example inputs DIRECTLY from NCBI GEO (no proxy) for general users:
#   14 Cut&Tag bigWig tracks  +  2 merged-replicate Micro-C .mcool matrices.
# Saved into  <repo>/example/example_data  (override with:  OUT=/path ./download_GSE178593.sh)
# Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178593
#
# Resumable: aria2c -c (or wget -c) continues partial files if interrupted; just re-run.
# NOTE: from inside mainland China these direct NCBI downloads can be very slow/stall —
#       use download_GSE178593_proxy.sh there instead.
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT="${OUT:-$SCRIPT_DIR/../example_data}"
mkdir -p "$OUT"

FILES=(
  # --- Cut&Tag bigWig tracks (Ctrl + Aux/degron, hg38) ---
  GSM6245909_DLD1asyn_Ctrl_CTCF_rep1.bw
  GSM6245910_DLD1asyn_Ctrl_CTCF_rep2.bw
  GSM6245911_DLD1asyn_Aux_CTCF_rep1.bw
  GSM6245912_DLD1asyn_Aux_CTCF_rep2.bw
  GSM6245913_DLD1asyn_Ctrl_SMC1A_rep1.bw
  GSM6245914_DLD1asyn_Ctrl_SMC1A_rep2.bw
  GSM6245915_DLD1asyn_Aux_SMC1A_rep1.bw
  GSM6245916_DLD1asyn_Aux_SMC1A_rep2.bw
  GSM6245917_DLD1asyn_Ctrl_H3K27ac_rep1.bw
  GSM6245918_DLD1asyn_Aux_H3K27ac_rep1.bw
  GSM6245919_DLD1asyn_Ctrl_H3K27me3_rep1.bw
  GSM6245920_DLD1asyn_Ctrl_H3K27me3_rep2.bw
  GSM6245921_DLD1asyn_Aux_H3K27me3_rep1.bw
  GSM6245922_DLD1asyn_Aux_H3K27me3_rep2.bw
  # --- merged-replicate Micro-C matrices (~5.5 GB each) ---
  GSM5394172_control_mergeRep_500bp.mcool
  GSM5394173_degron_mergeRep_500bp.mcool
)

# pick a downloader: aria2c (fast, multi-connection) if present, else wget
have_aria2=0; command -v aria2c >/dev/null 2>&1 && have_aria2=1

t0=$(date +%s)
for f in "${FILES[@]}"; do
  gsm=${f%%_*}                 # GSM6245909 / GSM5394172
  bucket="${gsm%???}nnn"       # GEO dir bucket (last 3 digits -> nnn)
  url="https://ftp.ncbi.nlm.nih.gov/geo/samples/${bucket}/${gsm}/suppl/${f}"
  echo ">> $f"
  if [ "$have_aria2" = 1 ]; then
    aria2c -x8 -s8 -c -d "$OUT" -o "$f" "$url" || echo "FAILED: $f"
  else
    wget -c -O "$OUT/$f" "$url" || echo "FAILED: $f"
  fi
done
t1=$(date +%s); secs=$((t1-t0)); [ "$secs" -lt 1 ] && secs=1

echo "===================== download summary ====================="
echo "dir   : $OUT"
echo "bw    : $(ls "$OUT"/*.bw 2>/dev/null | wc -l)/14"
echo "mcool : $(ls "$OUT"/*mergeRep*.mcool 2>/dev/null | wc -l)/2"
bytes=$(du -sb "$OUT"/*.bw "$OUT"/*mergeRep*.mcool 2>/dev/null | awk '{s+=$1} END{print s+0}')
python3 -c "b=$bytes;s=$secs;print(f'size  : {b/1024/1024/1024:.2f} GB\ntime  : {s}s\nspeed : {b/s/1024/1024:.1f} MB/s')" 2>/dev/null || true
