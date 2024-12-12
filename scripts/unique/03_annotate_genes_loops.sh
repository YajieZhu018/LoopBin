#!/bin/bash
# assign the cluster number to loops
il="$1"unique/01_unique/
gtf=/usr/users/yzhu1/Genome/hg38/hg38.ncbiRefSeq.5kb.upstreamTSS.bed
for cond in control degron
do
    # sort
    #pgltools sort "$ol""$cond"_loops_labels.bed > "$ol""$cond"_loops_labels_sorted.bed
    # intersect the loops with promoters
    pgltools intersect1D -wa -allA -a "$il""$cond"_unique_labels_loops.bedpe -b $gtf | sort -u > "$il""$cond"_unique_loops_labels_genes.bedpe
done

