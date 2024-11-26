#!/bin/bash
il=/usr/users/yzhu1/LoopBin/result/intersect/02_intersect
cd $il
mkdir temp/
for inFile in `ls rep1_loops_control_*_degron_*.bed`
do
    cond=${inFile%.bed}
    awk -v cond=$cond 'BEGIN {OFS=FS="\t"}
        {print $1,($2+$3)/2,($2+$3)/2+1>"temp/"cond"_anchor1.txt";
        print $4,($5+$6)/2,($5+$6)/2+1>"temp/"cond"_anchor2.txt";}' $inFile
    cat temp/"$cond"_anchor1.txt temp/"$cond"_anchor2.txt | sort -u > temp/"$cond".bed
    rm temp/*anchor*.txt
done