#!/bin/bash
il=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_
ol=/usr/users/yzhu1/LoopBin/trials/cluster_shift/vade_10clusters_merged_control_degron_rep1_cov_dia/
mkdir -p $ol
for i in {1..5}
do  
    for j in {1..5}
    do
        if [ $i -lt $j ]; then
            #prob=0.95
            file1="$il"run"$i"/control_rep1/labels_loops.bedpe
            file2="$il"run"$j"/control_rep1/labels_loops.bedpe
            outfile="$ol"run"$i"_to_run"$j".bedpe
            pdf="$ol"heatmap_run"$i"_to_run"$j".pdf
            bash 01_pair_cluster.sh $file1 $file2 $outfile
            python 02_heatmap.py $outfile $pdf
        fi
    done 
done