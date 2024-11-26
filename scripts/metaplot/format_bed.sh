#!/bin/bash
il=$1
cond=$2
cd $il
for i in {0..6}
do
    inFile="$il""$cond"_loops_label_"$i".bed
    awk -v i=$i -v cond=$cond 'BEGIN {OFS=FS="\t"}
        {print $1,($2+$3)/2,($2+$3)/2+1>cond"_loops_label"i"_anchor1.bed";
        print $4,($5+$6)/2,($5+$6)/2+1>cond"_loops_label"i"_anchor2.bed";}' $inFile
    cat "$cond"_loops_label"$i"_anchor1.bed "$cond"_loops_label"$i"_anchor2.bed | sort -u > "$cond"_loop_anchors_label_"$i".bed
    #rm loops_label*.bed
done