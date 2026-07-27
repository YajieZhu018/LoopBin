#!/usr/bin/env bash
# download_GSE178593_proxy.sh
# Download ALL GSE178593 example inputs via the proxy into LoopBin/example/example_data:
#   14 Cut&Tag bigWig tracks  +  2 merged-replicate Micro-C .mcool matrices.
# Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178593
set -uo pipefail

OUT="$HOME/ypf/projects_7_loopbin/LoopBin/example/example_data"
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

# proxy ON (Japan node); aria2c uses https_proxy (all_proxy socks form unset to avoid a warning)
source ~/proxy/proxy.sh
proxy on
unset all_proxy ALL_PROXY

t0=$(date +%s)
for f in "${FILES[@]}"; do
  gsm=${f%%_*}                 # e.g. GSM6245909  /  GSM5394172
  bucket="${gsm%???}nnn"       # GEO dir bucket: last 3 digits -> nnn  (GSM6245nnn / GSM5394nnn)
  url="https://ftp.ncbi.nlm.nih.gov/geo/samples/${bucket}/${gsm}/suppl/${f}"
  echo ">> $f"
  aria2c -x8 -s8 -c -d "$OUT" -o "$f" "$url" || echo "FAILED: $f"
done
t1=$(date +%s); secs=$((t1-t0)); [ "$secs" -lt 1 ] && secs=1
proxy off

echo "===================== download summary ====================="
echo "dir   : $OUT"
echo "bw    : $(ls "$OUT"/*.bw 2>/dev/null | wc -l)/14"
echo "mcool : $(ls "$OUT"/*mergeRep*.mcool 2>/dev/null | wc -l)/2"
bytes=$(du -sb "$OUT"/*.bw "$OUT"/*mergeRep*.mcool 2>/dev/null | awk '{s+=$1} END{print s+0}')
python3 -c "b=$bytes;s=$secs;print(f'size  : {b/1024/1024/1024:.2f} GB\ntime  : {s}s\nspeed : {b/s/1024/1024:.1f} MB/s')"
