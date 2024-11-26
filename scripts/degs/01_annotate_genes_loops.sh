#!/bin/bash
# assign the cluster number to loops
il=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/metaplot/temp/
ol=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/degs/
gtf=/usr/users/yzhu1/Genome/hg38/hg38.ncbiRefSeq.5kb.upstreamTSS.bed
for cond in control degron 
do
    #for i in 0 1 2 4 5 6
    #do 
    #  j=$i
    #  if [ $i -eq 0 ]; then
    #    j=3
    #  fi
    #  awk -v j=$j 'BEGIN {OFS=FS="\t"}
    #    {print $1,$2,$3,$4,$5,$6,j}' "$il""$cond"_loops_label_"$i".bed >> "$ol""$cond"_loops_labels.bed
    #done
    # sort
    #pgltools sort "$ol""$cond"_loops_labels.bed > "$ol""$cond"_loops_labels_sorted.bed
    # intersect the loops with promoters
    pgltools intersect1D -wa -allA -a "$ol""$cond"_loops_labels_sorted.bed -b $gtf | sort -u > "$ol""$cond"_loops_labels_genes.bed
    # get the regions that cannot be assigned
    pgltools intersect1D -wa -v -a "$ol""$cond"_loops_labels_sorted.bed -b $gtf | sort -u > "$ol""$cond"_loops_labels_no_genes.bed
done

