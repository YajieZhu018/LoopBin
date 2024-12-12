#!/bin/bash
ol="$1"unique/01_unique/
for cond in control degron
do
    loop_file="$ol""$cond"_unique_loops_labels_genes.bedpe
    fkpm_file=/usr/users/yzhu1/LoopBin/data/RNA-seq/GSE176285_norm_counts_FPKM_GRCh38.p13_NCBI_geneName.tsv
    out_file="$ol""$cond"_unique_loops_labels_fkpm.bedpe
    awk 'BEGIN {OFS=FS="\t"; print "chrom1","start1","end1","chrom2","start2","end2","other","cluster","strand","gene","fpkm_control", "fpkm_degron"} 
        NR==FNR {dic[$10]=($2+$3)/2"\t"($4+$5)/2; next} 
        $10 in dic {print $0,dic[$10]}' $fkpm_file $loop_file > $out_file
done