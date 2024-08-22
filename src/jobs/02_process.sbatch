#!/bin/bash

#SBATCH -p medium
#SBATCH --job-name=process
#SBATCH --output=/usr/users/yzhu1/LoopBin/data/02_process/process_%A_%a.out
#SBATCH --error=/usr/users/yzhu1/LoopBin/data/02_process/process_%A_%a.err
#SBATCH -c 8
#SBATCH --nodes=1
#SBATCH --mem=48gb
#SBATCH --time=8:00:00
#SBATCH --array=0-1


# Input
arr_cond=(control_rep1 degron_rep1)
i="${SLURM_ARRAY_TASK_ID}"
cond=${arr_cond[$i]}
# remove _rep* to get the experiment name
experiment=${cond%_rep*}
il=/usr/users/yzhu1/LoopBin/data/01_preprocess/"$cond"/
ol=/usr/users/yzhu1/LoopBin/data/02_process/"$cond"/
loop=/usr/users/yzhu1/LoopBin/data/00_raw/"$experiment"_loops_labeled.bedpe
cool=/usr/users/yzhu1/LoopBin/data/00_raw/DLD1_async_14h"$experiment"_matrix_1kb.mcool   #::/resolutions/8000

#conda activate loophigh
mkdir $ol
# preprocess into input for training
python /usr/users/yzhu1/LoopBin/main.py -f 2 -l $loop -c $cool -g $il -r 8 -u $ol
