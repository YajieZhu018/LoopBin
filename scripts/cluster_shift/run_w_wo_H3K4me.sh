#!/bin/bash
file1=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run4/control_rep1/labels_loops.bedpe
file2=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_control_rep1_CTCF_H3K27ac_H3K27me3_SMC1A_H3K4me1_run4/labels_loops.bedpe
ol=/usr/users/yzhu1/LoopBin/trials/cluster_shift/vade_10clusters_control_rep1_w_wo_H3K4me1_run4/
mkdir -p $ol
outfile="$ol"w_wo_H3K4me1.bedpe
pdf="$ol"heatmap_w_wo_H3K4me1.pdf
bash 01_pair_cluster.sh $file1 $file2 $outfile
python 02_heatmap.py $outfile $pdf