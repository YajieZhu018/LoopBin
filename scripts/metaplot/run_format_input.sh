#!/bin/bash
fl=/usr/users/yzhu1/LoopBin/scripts/metaplot/
# Format the input bed files for the metaplot analysis
for i in 10
do
    il=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_"$i"clusters_merged_control_degron_rep1_cov_dia_run1/
    ol=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_"$i"clusters_merged_control_degron_rep1_cov_dia_run1/metaplot/
    mkdir -p "$ol"temp
    for cond in control degron 
    do
        loops_bedpe=/usr/users/yzhu1/LoopBin/data/00_raw/"$cond"_loops_labeled.bedpe
        index_file=/usr/users/yzhu1/LoopBin/data/02_process/"$cond"_rep1/loop_number.npy
        # get loop coordinate into a bed file
        loops_bed="$ol"/temp/"$cond"_loops.bed 
        python "$fl"get_loop_coordinate.py $loops_bedpe $index_file $loops_bed
        # separate the loops into different clusters
        labels_file="$il"prediction/"$cond"_rep1/labels.npy
        python "$fl"separate_loops.py $loops_bed $labels_file "$ol"temp/"$cond"_
        # format the bed files
        bash "$fl"format_bed.sh "$ol"temp/ $cond
    done
done
        
