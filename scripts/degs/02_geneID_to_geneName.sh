#!/bin/bash
# add gene names to the FKPM File
fpkm_file='/usr/users/yzhu1/LoopBin/data/RNA-seq/GSE176285_norm_counts_FPKM_GRCh38.p13_NCBI.tsv'
gene_info='/usr/users/yzhu1/Genome/hg38/Homo_sapiens.gene_info'
out_file='/usr/users/yzhu1/LoopBin/data/RNA-seq/GSE176285_norm_counts_FPKM_GRCh38.p13_NCBI_geneName.tsv'
# match the gene info
awk 'BEGIN {OFS=FS="\t"}
    NR==FNR {dic[$2]=$3;next}   # dic[geneID] = geneName
    FNR==1 {print $0,"geneName"}
    $1 in dic {print $0, dic[$1]}' $gene_info $fpkm_file > $out_file