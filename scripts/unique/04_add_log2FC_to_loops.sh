#!/bin/bash
ol="$1"unique/01_unique/
degs_file=/usr/users/yzhu1/LoopBin/data/RNA-seq/GSE176285_DLD1_async_factory_gene_irnaseq_GRCh38_qval0.05n.csv
for cond in control degron
do
    loop_file="$ol""$cond"_unique_loops_labels_genes.bedpe
    out_file="$ol""$cond"_unique_loops_labels_log2FC.bedpe
    awk 'BEGIN {OFS="\t"; FS="\t|,"; print "chrom1","start1","end1","chrom2","start2","end2","other","cluster","strand","gene","log2FC"} 
        NR==FNR {gsub(/"/,""); dic[$2]=$5; next} 
        $10 in dic {print $0,dic[$10]}' $degs_file $loop_file > $out_file
done