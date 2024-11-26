#!/bin/bash
# assign the cluster number to loops
il=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/unique/01_sorted/
ol=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/unique/01_sorted/
gtf=/usr/users/yzhu1/Genome/hg38/hg38.ncbiRefSeq.5kb.upstreamTSS.bed
for cond in control degron 
do
    # sort
    #pgltools sort "$ol""$cond"_loops_labels.bed > "$ol""$cond"_loops_labels_sorted.bed
    # intersect the loops with promoters
    pgltools intersect1D -wa -allA -a "$ol""$cond"_loops_label_merged.bed -b $gtf | sort -u > "$ol""$cond"_loops_labels_genes.bed
done

