#!/bin/bash
folder=$1
il="$folder"intersect/02_intersect/
degs_file=/usr/users/yzhu1/LoopBin/data/RNA-seq/GSE176285_DLD1_async_factory_gene_irnaseq_GRCh38_qval0.05n.csv
pairs=("1 2" "3 4" "1 6" "2 6")
cond=control
out_file="$il""$cond"_loops_genes_log2FC.bed
awk 'BEGIN {OFS="\t"; print "chrom1","start1","end1","chrom2","start2","end2","strand","gene","log2FC", "cluster"}' $degs_file >> $out_file
for pair in "${pairs[@]}"
do
    # read pair to i, j
    read -r i j <<< "$pair"
    loop_file="$il""$cond"_loops_control_"$i"_degron_"$j"_genes.bed
    awk -v i=$i -v j=$j 'BEGIN {OFS="\t"; FS="\t|,"} 
        NR==FNR {gsub(/"/,""); dic[$2]=$5; next} 
        $13 in dic {print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$12"\t"$13,dic[$13],i"-"j}' $degs_file $loop_file >> $out_file
done
