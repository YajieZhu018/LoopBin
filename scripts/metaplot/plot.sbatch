#!/bin/bash
# Slurm Parameters
#SBATCH -p medium
#SBATCH --job-name=metaplot
#SBATCH --output=/usr/users/yzhu1/LoopBin/trials/saved_models/metaplot_%A_%a.out
#SBATCH --error=/usr/users/yzhu1/LoopBin/trials/saved_models/metaplot_%A_%a.err
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=64gb
#SBATCH --time=32:00:00
#SBATCH --array=0

#module load deeptools
# write conditions into an array
conditions=(control degron)
# Input
i=7
il_bw=/usr/users/yzhu1/LoopBin/data/00_raw/
il_bed=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_"$i"clusters_merged_control_degron_rep1/metaplot/temp/
ol=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_"$i"clusters_merged_control_degron_rep1/metaplot/

# loop through all conditions
cond=${conditions[$SLURM_ARRAY_TASK_ID]}
matrix="$ol"metaplot_"$cond"_rep1.gz
plot="$ol"metaplot_"$cond"_rep1.pdf
     
# Plot
plotHeatmap --yMax 1.8 -m $matrix -o $plot --refPointLabel loop_anchor --averageTypeSummaryPlot mean --regionsLabel 0 1 2 4 5 6
