#!/bin/bash
il=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run1/
fl=/usr/users/yzhu1/LoopBin/scripts/intersect/
cd $fl
#python 04_plot_length.py "$il"intersect/01_sorted/ "$il"intersect/03_plot/ 6
python 05_average_plot_shifted_clusters.py $il 
#bash 061_annotate_genes_loops.sh $il