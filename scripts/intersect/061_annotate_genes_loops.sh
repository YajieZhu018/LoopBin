#!/bin/bash
# assign the cluster number to loops
folder=$1
il="$folder"intersect/02_intersect/
gtf=/usr/users/yzhu1/Genome/hg38/hg38.ncbiRefSeq.5kb.upstreamTSS.bed
degs_file=/usr/users/yzhu1/LoopBin/data/RNA-seq/GSE176285_DLD1_async_factory_gene_irnaseq_GRCh38_qval0.05n.csv
for cond in control degron 
do
    input_file="$il""$cond"_intersect.bedpe
    output_file="$il""$cond"_intersect_genes.bed
    # sort
    pgltools sort $input_file > "$il""$cond"_intersect_sorted.bedpe
    # intersect the loops with promoters
    pgltools intersect1D -wa -allA -a "$il""$cond"_intersect_sorted.bedpe -b $gtf | sort -u > $output_file
done

