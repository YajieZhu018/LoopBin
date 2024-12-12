#!/bin/bash
il=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run1/
num_clusters=6
#bash 01_unique.sh $il
#python 02_barplot_perc.py $il
#bash 03_annotate_genes_loops.sh $il
#bash 04_add_log2FC_to_loops.sh $il
python 05_violinplot_log2FC.py $il
#bash 04_add_fkpm_to_loops.sh $il
#python 05_violinplot_pkfm.py $il