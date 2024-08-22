#!/bin/bash

#SBATCH -p medium
#SBATCH --job-name=preprocess_input
#SBATCH --output=/usr/users/yzhu1/LoopBin/data/01_preprocess/prepro_%A_%a.out
#SBATCH --error=/usr/users/yzhu1/LoopBin/data/01_preprocess/prepro_%A_%a.err
#SBATCH -c 7
#SBATCH --nodes=1
#SBATCH --mem=48gb
#SBATCH --time=24:00:00
#SBATCH --array=0-3

#conda activate loophigh
# Input
il=/usr/users/yzhu1/LoopBin/data/00_raw/
arr_cond=(control degron)
arr_rep=(rep1 rep1)
arr_name=(CTCF SMC1A H3K27ac H3K27me3)

j="${SLURM_ARRAY_TASK_ID}"
for i in 0 1
do 
    cond=${arr_cond["$i"]}
    rep=${arr_rep["$i"]}
    name=${arr_name["$j"]}
    bigwig="$il""$cond"_"$name"_"$rep".bw
    ol=/usr/users/yzhu1/LoopBin/data/01_preprocess/"$cond"_"$rep"/
    mkdir -p $ol
    # preprocess bigwig into bedgraph
    python /usr/users/yzhu1/LoopBin/main.py -f 1 -b $bigwig -g $ol -n $name
done
