#!/bin/bash

# find the intersect between each pair of clusters in control and degron 
# pglsort
il=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/metaplot/temp/
ol=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/intersect/01_sorted/
mkdir -p $ol
for cond in control degron
do
    for i in 0 1 2 4 5 6
    do
        j=$i
        # if i==0, j=3
        if [ $i -eq 0 ]; then
            j=3
        fi
        file=${il}${cond}_loops_label_${i}.bed
        pgltools sort $file > ${ol}${cond}_loops_label_${j}.bed
    done
done

# pgl intersect
il=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/intersect/01_sorted/
ol=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/intersect/02_intersect/
mkdir -p $ol
for i in {1..6}
do
    for j in {1..6}
    do
        # keep original in the first entry; keep unique; allow distance up to 8000bp
        pgltools intersect -wb -u -d 8000 -a ${il}control_loops_label_${i}.bed -b ${il}degron_loops_label_${j}.bed | sort -u > ${ol}control_loops_control_${i}_degron_${j}.bed
        pgltools intersect -wb -u -d 8000 -a ${il}degron_loops_label_${j}.bed -b ${il}control_loops_label_${i}.bed | sort -u > ${ol}degron_loops_control_${i}_degron_${j}.bed
    done
done