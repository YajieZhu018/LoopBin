#!/bin/bash
# Slurm Parameters
#SBATCH -p medium
#SBATCH --job-name=metaplot
#SBATCH --output=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run1/intersect/02_intersect/temp/metaplot_%A_%a.out
#SBATCH --error=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run1/intersect/02_intersect/temp/metaplot_%A_%a.err
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=32gb
#SBATCH --time=32:00:00
# #SBATCH --array=0-1

module load deeptools

# Input
il_bw=/usr/users/yzhu1/LoopBin/data/00_raw/
il_bed=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run1/intersect/02_intersect/temp/
ol=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run1/intersect/03_plot/
first=1
second=4
matrix="$ol"metaplot_shared_loops_control_c"$first"_degron_c"$second".gz
plot="$ol"metaplot_shared_loops_control_c"$first"_degron_c"$second".pdf

# Generate matrix
computeMatrix reference-point \
     --referencePoint  TSS \
     -p max \
     -b 32000 -a 32000 \
     -R "$il_bed""$first"_"$second".bed \
     -S "$il_bw"control_H3K27ac_rep1.bw "$il_bw"degron_H3K27ac_rep1.bw \
     --samplesLabel control_H3K27ac degron_H3K27ac \
     --skipZeros   \
     -o $matrix
     
# Plot
plotHeatmap -m $matrix -o $plot --refPointLabel loop_anchor --averageTypeSummaryPlot mean --regionsLabel control_c"$first"_degron_c"$second"
