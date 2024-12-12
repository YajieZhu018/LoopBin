#!/bin/bash
il=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run1/
#bash 01_annotate_genes_loops.sh $il
#bash 03_add_fkpm_to_loops.sh $il
#python 04_violinplot_pkfm.py $il
python 05_barplot_perc_loops_with_promoters.py $il