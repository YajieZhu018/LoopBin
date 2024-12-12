#!/bin/bash

# find the intersect between each pair of clusters in control and degron 
# pglsort
il=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run1/
ol=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run1/intersect/01_sorted/
mkdir -p $ol
for cond in control degron
do
  file=${il}${cond}_rep1/labels_loops.bedpe
  #pgltools sort $file > ${ol}${cond}_labels_loops.bedpe
done

# pgl intersect
il=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run1/intersect/01_sorted/
ol=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run1/intersect/02_intersect/
mkdir -p "$ol"temp/
file1="${il}control_labels_loops.bedpe"
file2="${il}degron_labels_loops.bedpe"
# keep original in the first entry; keep annotation from the second file; keep unique; allow distance up to 8000bp
# Create temporary BED files for both anchors of degron; keep the line number as the fifth column
awk -v gap=8000 '{print $1, ($2-gap < 0 ? 0 : $2-gap), $3+gap, $9, NR}' OFS="\t" "$file2" > "$ol"temp/degron_anchor1.bed
awk -v gap=8000 '{print $4, ($5-gap < 0 ? 0 : $5-gap), $6+gap, $9, NR}' OFS="\t" "$file2" > "$ol"temp/degron_anchor2.bed
awk 'BEGIN {OFS=FS="\t"}
  {
    print $1, $2, $3, $4, $5, $6, $11, NR
  }' $file1 > "$ol"temp/control.bedpe
# find overlap with two anchors
bedtools pairtobed -type either -a "$ol"temp/control.bedpe -b "$ol"temp/degron_anchor1.bed > "$ol"temp/overlap_anchor1.bed
bedtools pairtobed -type either -a "$ol"temp/control.bedpe -b "$ol"temp/degron_anchor2.bed > "$ol"temp/overlap_anchor2.bed

# Combine overlaps from both anchors into BEDPE format with merged labels
awk 'BEGIN {OFS=FS="\t"}
  NR==FNR {dic[$8"|"$13]=0; next} $8"|"$13 in dic {print $1, $2, $3, $4, $5, $6, $7, $12}' "$ol"temp/overlap_anchor1.bed "$ol"temp/overlap_anchor2.bed | sort -u > "$ol"control_intersect.bedpe