#!/bin/bash
il=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run1/intersect/02_intersect/
cd $il
mkdir temp/
inFile="$il"control_intersect_sorted.bedpe
l1=1
l2=4
awk -v l1=$l1 -v l2=$l2 'BEGIN {OFS=FS="\t"}
    $7==l1 && $8==l2 {print $1,($2+$3)/2,($2+$3)/2+1>"temp/"l1"_"l2"_anchor1.txt";
    print $4,($5+$6)/2,($5+$6)/2+1>"temp/"l1"_"l2"_anchor2.txt";}' $inFile
cat temp/"$l1"_"$l2"_anchor1.txt temp/"$l1"_"$l2"_anchor2.txt | sort -u > temp/"$l1"_"$l2".bed
rm temp/*anchor*.txt