#!/bin/bash
il=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_8clusters_control_rep1_H3K27ac_H3K27me3_SMC1A_H3K4me1_with_added_noises_
ol=/usr/users/yzhu1/LoopBin/trials/cluster_shift/vade_8clusters_control_rep1_H3K27ac_H3K27me3_SMC1A_H3K4me1_with_added_noises/
mkdir -p $ol
for i in {1..5}
do  
    for j in {1..5}
    do
        if [ $i -lt $j ]; then
            prob=0.95
            file1="$il"run"$i"/labels_loops_prob"$prob".bedpe
            file2="$il"run"$j"/labels_loops_prob"$prob".bedpe
            outfile="$ol"run"$i"_to_run"$j"_prob"$prob".bedpe
            pdf="$ol"heatmap_run"$i"_to_run"$j"_prob"$prob".pdf
            bash 01_pair_cluster.sh $file1 $file2 $outfile
            python 02_heatmap.py $outfile $pdf
        fi
    done 
done