#!/bin/bash
#SBATCH -p medium
#SBATCH --job-name=merge_clusters
#SBATCH --output=/usr/users/yzhu1/LoopBin/trials/find_consensus_cluster_%A_%a.out
#SBATCH --error=/usr/users/yzhu1/LoopBin/trials/find_consensus_cluster_%A_%a.err
#SBATCH -c 8
#SBATCH --nodes=1
#SBATCH --mem=16gb
#SBATCH --time=5:00:00
# #SBATCH --array=1
#run=$SLURM_ARRAY_TASK_ID
data_path=/usr/users/yzhu1/LoopBin/data/02_process/control_rep1_CTCF_H3K27ac_H3K27me3_SMC1A_H3K4me1/merged_log_data.npy
output_path=/usr/users/yzhu1/LoopBin/trials/consensus_cluster/vade_8clusters_control_rep1_H3K27ac_H3K27me3_SMC1A_H3K4me1/
python /usr/users/yzhu1/LoopBin/main.py -f 10 -d $data_path -u $output_path -p CTCF,H3K27ac,H3K27me3,SMC1A,H3K4me1
