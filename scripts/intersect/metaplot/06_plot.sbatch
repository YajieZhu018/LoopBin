#!/bin/bash
# Slurm Parameters
#SBATCH -p medium
#SBATCH --job-name=metaplot
#SBATCH --output=/usr/users/yzhu1/LoopBin/result/intersect/03_plot/metaplot_%A_%a.out
#SBATCH --error=/usr/users/yzhu1/LoopBin/result/intersect/03_plot/metaplot_%A_%a.err
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=32gb
#SBATCH --time=32:00:00
# #SBATCH --array=0-1

#module load deeptools

# Input
il_bw=/usr/users/yzhu1/LoopBin/Data/
il_bed=/usr/users/yzhu1/LoopBin/result/intersect/02_intersect/temp/
ol=/usr/users/yzhu1/LoopBin/result/intersect/03_plot/
first=1
second=5
matrix="$ol"metaplot_shared_loops_control_c"$first"_degron_c"$second".gz
plot="$ol"metaplot_shared_loops_control_c"$first"_degron_c"$second".pdf
# Plot
plotHeatmap -m $matrix -o $plot --refPointLabel loop_anchor --averageTypeSummaryPlot mean --regionsLabel control_c"$first"_degron_c"$second"
