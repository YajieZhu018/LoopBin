#!/bin/bash

#SBATCH -p medium
#SBATCH --job-name=process
#SBATCH --output=/usr/users/yzhu1/LoopBin/data/02_process/process_%A_%a.out
#SBATCH --error=/usr/users/yzhu1/LoopBin/data/02_process/process_%A_%a.err
#SBATCH -c 1
#SBATCH --nodes=1
#SBATCH --mem=8gb
#SBATCH --time=1:00:00
# #SBATCH --array=0-1


# Input
conditions=control_rep1,degron_rep1
ol=/usr/users/yzhu1/LoopBin/data/02_process/

#conda activate loophigh
mkdir $ol
# preprocess bigwig into bedgraph
python /usr/users/yzhu1/LoopBin/main.py -f 2.1 -e $conditions -u $ol
