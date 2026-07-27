#!/bin/bash
#SBATCH -p medium
#SBATCH --job-name=merge_clusters
#SBATCH --output=/usr/users/yzhu1/LoopBin/trials/saved_models/merge_clusters_%A_%a.out
#SBATCH --error=/usr/users/yzhu1/LoopBin/trials/saved_models/merge_clusters_%A_%a.err
#SBATCH -c 2
#SBATCH --nodes=1
#SBATCH --mem=4gb
#SBATCH --time=2:00:00
#SBATCH --array=4
run=$SLURM_ARRAY_TASK_ID
data_path=/usr/users/yzhu1/LoopBin/data/02_process/control_rep1_CTCF_H3K27ac_H3K27me3_SMC1A_H3K4me1/merged_log_data.npy
output_path=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_control_rep1_CTCF_H3K27ac_H3K27me3_SMC1A_H3K4me1_run"$run"/
python /usr/users/yzhu1/LoopBin/main.py -f 9 -k 2,3 -d $data_path -u $output_path -p CTCF,H3K27ac,H3K27me3,SMC1A,H3K4me1