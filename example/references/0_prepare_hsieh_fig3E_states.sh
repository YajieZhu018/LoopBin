#!/usr/bin/env bash
# Prepare the mESC chromatin-state annotation used in Hsieh et al. Mol Cell 2020, Figure 3E.
#
# PROVENANCE (the attribution is indirect; this is the full chain):
#   Hsieh 2020, Fig. S3F legend: "The 11 ChromHMM states in mESCs were identified from
#   (Pintacuda et al., 2017)."  Pintacuda et al. 2017 (Mol Cell 68:955) STAR Methods names the
#   source by URL: "the pre-defined ChromHMM state (https://github.com/guifengwei/
#   ChromHMM_mESC_mm10) across chr2 and chr3 was used".  So the annotation is:
#     https://github.com/guifengwei/ChromHMM_mESC_mm10
#   (Hsieh's main-text "(Ernst and Kellis, 2012)" is the ChromHMM software paper, not a state set.)
#   Guifeng Wei, Dec 2015: ChromHMM on ENCODE E14 mESC ChIP-seq (H3K4me1, H3K4me3, H3K27me3,
#   H3K27ac, H3K9me3, H3K9ac, H3K36me3, CTCF, Nanog, Oct4); models 9-15 tested, 12-state kept.
#   Genome mm10, 200 bp bins.
#
# The model has TWELVE states; Hsieh's Fig 3E shows ELEVEN of them:
#   * 2_Intergenic is not displayed (it is the null state, 76 % of the genome);
#   * 12_LowSignal/RepetitiveElements is displayed as "Repeats" (Fig S3F: "Repeats (not shown)",
#     unsurprising — it is 42 segments / 42 kb genome-wide);
#   * 9_TranscriptionTransition is displayed as "Weak promoter" — this is the ONE inferred
#     rename: it is the only state left once the other ten match by name, and Intergenic is far
#     too large to be a promoter class.
# We keep Intergenic in the output (dropping 76 % of the genome would distort every percentage)
# and label it "Intergenic"; the other eleven carry the Fig-3E names.
#
# Usage:  bash 0_prepare_hsieh_fig3E_states.sh
set -euo pipefail

REF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=$REF/mESC_chromHMM_wei12
RAW=https://raw.githubusercontent.com/guifengwei/ChromHMM_mESC_mm10/master
DENSE=$OUT/mESC_E14_12_dense.annotated.bed.gz
SEG=$OUT/segments.sorted.bed
ANN=$OUT/annotation.tsv
mkdir -p "$OUT"

# ---------------------------------------------------------------- download --
# GitHub is throttled from the China boxes; smartdl tries direct then the proxy.
for f in mESC_E14_12_dense.annotated.bed.gz mESC_E14_12_segments.bed.gz; do
  if [ -s "$OUT/$f" ]; then echo "[have] $f"; continue; fi
  echo "[get ] $f"
  ~/proxy/smartdl.sh "$RAW/$f" "$OUT"
done
gzip -t "$DENSE"

# ------------------------------------------------- main-chromosome subset ---
# Columns 1-4 of the dense track: chrom, start, end, "<id>_<StateName>".
# chr1-19 + chrX only, LC_ALL=C sorted to match the anchor BED.
if [ ! -s "$SEG" ]; then
  echo "[make] $(basename "$SEG")"
  zcat "$DENSE" | tail -n +2 \
    | awk -v OFS="\t" '$1 ~ /^chr([1-9]|1[0-9]|X)$/ {print $1,$2,$3,$4}' \
    | LC_ALL=C sort -k1,1 -k2,2n > "$SEG"
fi

# ------------------------------------------------------------ annotation ---
# Same column layout as the full-stack TSV so the downstream scripts are unchanged.
#   mneumonics = the segment-name suffix (the join key)
#   group      = the label plotted (Hsieh Fig 3E)
#   comments   = Wei's original state name, kept for provenance
cat > "$ANN" <<'TSV'
state	comments	group	color	itemRgb	state_order_by_group	mneumonics
2	2_Intergenic (null state; NOT shown in Hsieh Fig 3E)	Intergenic	#0099cc	(0, 153, 204)	0	Intergenic
12	12_LowSignal/RepetitiveElements	Repeats	#ffffcc	(255, 255, 204)	1	LowSignal/RepetitiveElements
5	5_RepressedChromatin	Repressive	#669933	(102, 153, 51)	2	RepressedChromatin
3	3_Heterochromatin	Heterochromatin	#33ff99	(51, 255, 153)	3	Heterochromatin
10	10_TranscriptionElongation	Elongation	#cccc66	(204, 204, 102)	4	TranscriptionElongation
11	11_WeakEnhancer	Weak Enhancer	#ffff00	(255, 255, 0)	5	WeakEnhancer
9	9_TranscriptionTransition (Fig 3E name inferred)	Weak promoter	#cc99ff	(204, 153, 255)	6	TranscriptionTransition
6	6_BivalentChromatin	Bivalent promoter	#006600	(0, 102, 0)	7	BivalentChromatin
4	4_Enhancer	Enhancer	#663399	(102, 51, 153)	8	Enhancer
8	8_StrongEnhancer	Strong Enhancer	#ff00cc	(255, 0, 204)	9	StrongEnhancer
7	7_ActivePromoter	Active promoter	#cc0033	(204, 0, 51)	10	ActivePromoter
1	1_Insulator	Insulator	#0000ff	(0, 0, 255)	11	Insulator
TSV

# ------------------------------------------------------------------ verify --
echo
echo "=== VERIFY: 200 bp binning, states, chromosomes ==="
awk '{ n++
       if (($2 % 200) != 0 || ($3 % 200) != 0) offgrid++
       L=$3-$2; if (L % 200 != 0) offlen++
       if (min=="" || L<min) min=L
       st[$4] += L }
     END { printf "segments (chr1-19,X) : %d\n", n
           printf "boundaries off the 200 bp grid : %d\n", offgrid+0
           printf "lengths not a multiple of 200  : %d\n", offlen+0
           printf "min segment length : %d  (must be 200)\n", min
           printf "distinct states    : %d\n", length(st)
           for (k in st) printf "   %-34s %12d bp\n", k, st[k] }' "$SEG" | sort -k1,1

# every segment state must join to the annotation by mnemonic
cut -f4 "$SEG" | sed 's/^[0-9]*_//' | LC_ALL=C sort -u > /tmp/_seg_mnem.$$
tail -n +2 "$ANN" | cut -f7 | LC_ALL=C sort -u > /tmp/_ann_mnem.$$
if ! diff -q /tmp/_seg_mnem.$$ /tmp/_ann_mnem.$$ > /dev/null; then
  echo "ERROR: segment states and annotation mnemonics differ:"; diff /tmp/_seg_mnem.$$ /tmp/_ann_mnem.$$
  rm -f /tmp/_seg_mnem.$$ /tmp/_ann_mnem.$$; exit 1
fi
rm -f /tmp/_seg_mnem.$$ /tmp/_ann_mnem.$$
echo "all segment states join to the annotation by mnemonic"
echo
echo "[done] $OUT"
