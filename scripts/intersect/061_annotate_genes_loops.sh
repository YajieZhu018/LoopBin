#!/bin/bash
# assign the cluster number to loops
il=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/intersect/02_intersect/
gtf=/usr/users/yzhu1/Genome/hg38/hg38.ncbiRefSeq.5kb.upstreamTSS.bed
degs_file=/usr/users/yzhu1/LoopBin/data/RNA-seq/GSE176285_DLD1_async_factory_gene_irnaseq_GRCh38_qval0.05n.csv
mkdir "$il"sorted
for cond in control degron 
do
    # loop through 1-6 clusters
    for i in {1..6}
    do
        for j in {1..6}
        do
            # sort
            pgltools sort "$il""$cond"_loops_control_"$i"_degron_"$j".bed > "$il"sorted/"$cond"_loops_control_"$i"_degron_"$j".bed
            # intersect the loops with promoters
            pgltools intersect1D -wa -allA -a "$il"sorted/"$cond"_loops_control_"$i"_degron_"$j".bed -b $gtf | sort -u > "$il""$cond"_loops_control_"$i"_degron_"$j"_genes.bed
        done
    done
done

