#!/usr/bin/env bash
# Download the Ernst-lab MOUSE FULL-STACK ChromHMM annotation (mm10, 100 states) and verify it.
#
#   Repo:  https://github.com/ernstlab/mouse_fullStack_annotations
#   Data:  https://public.hoffman2.idre.ucla.edu/ernst/2K9RS/full_stack/full_stack_annotation_public_release/mm10/
#
# The UCLA host is NOT reachable direct from the China boxes (DNS resolves, TCP times out), so the
# download goes through ~/proxy/smartdl.sh, which tries DIRECT first and falls back to the proxy.
#
# Produces, in <this dir>/mm10_chromstates/ :
#   mm10_100_segments_segments.bed.gz   as downloaded (36 M)
#   state_annotation_processed.tsv      state -> group / colour / mnemonic (26 K)
#   mm10_100_segments.main.sorted.bed   chr1-19 + chrX only, coordinate sorted (LC_ALL=C)
#
# Usage:  bash 0_download_chrom_states.sh
set -euo pipefail

BASE=https://public.hoffman2.idre.ucla.edu/ernst/2K9RS/full_stack/full_stack_annotation_public_release/mm10
REF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=$REF/mm10_chromstates
SEG_GZ=$OUT/mm10_100_segments_segments.bed.gz
TSV=$OUT/state_annotation_processed.tsv
SORTED=$OUT/mm10_100_segments.main.sorted.bed
mkdir -p "$OUT"

# ---------------------------------------------------------------- download --
for f in mm10_100_segments_segments.bed.gz state_annotation_processed.tsv; do
  if [ -s "$OUT/$f" ]; then echo "[have] $f"; continue; fi
  echo "[get ] $f"
  ~/proxy/smartdl.sh "$BASE/$f" "$OUT"
done
gzip -t "$SEG_GZ" || { echo "ERROR: $SEG_GZ is not a valid gzip (truncated download?)"; exit 1; }

# ------------------------------------------------- main-chromosome subset ---
# Keep only the chromosomes the loop calls use (chr1-19, chrX); LC_ALL=C sort so the order
# matches the anchor BED sorted the same way (needed for `bedtools intersect -sorted`).
if [ ! -s "$SORTED" ]; then
  echo "[make] $(basename "$SORTED")"
  zcat "$SEG_GZ" \
    | awk -v OFS="\t" '$1 ~ /^chr([1-9]|1[0-9]|X)$/ {print $1,$2,$3,$4}' \
    | LC_ALL=C sort -k1,1 -k2,2n > "$SORTED"
fi

# ------------------------------------------------------------------ verify --
echo
echo "=== VERIFY: 200 bp binning, state ids, chromosomes ==="
zcat "$SEG_GZ" | awk '
  NR==1 {ncol=NF}
  { n++
    if (($2 % 200) != 0 || ($3 % 200) != 0) offgrid++
    L=$3-$2; if (L % 200 != 0) offlen++
    if (min=="" || L<min) min=L
    if (L>max) max=L
    sum+=L
    st[$4]++
    chr[$1]++
  }
  END {
    printf "segments (all chroms) : %d   columns: %d\n", n, ncol
    printf "boundaries off the 200 bp grid : %d\n", offgrid+0
    printf "lengths not a multiple of 200  : %d\n", offlen+0
    printf "segment length  min=%d  max=%d  mean=%.0f  (min must be 200)\n", min, max, sum/n
    printf "distinct state labels : %d\n", length(st)
    printf "distinct chromosomes  : %d\n", length(chr)
  }'

echo
echo "--- example segment lines ---"
head -3 "$SORTED"   # from the sorted subset: `zcat | head` would SIGPIPE under `set -o pipefail`
echo "--- state annotation columns ---"
head -1 "$TSV"
echo "--- state groups (col 3) ---"
awk -F"\t" 'NR>1{g[$3]++} END{n=0; for (k in g){printf "%s(%d) ", k, g[k]; n++} printf "\n= %d groups over %d states\n", n, NR-1}' "$TSV"

echo
echo "--- main-chrom subset ---"
wc -l < "$SORTED" | xargs printf "segments on chr1-19,chrX : %s\n"
cut -f1 "$SORTED" | uniq | tr "\n" " "; echo
echo
echo "[done] $OUT"
