#!/bin/bash
#SBATCH -p medium
#SBATCH --job-name=plot_nmi
#SBATCH --output=/usr/users/yzhu1/LoopBin/trials/cluster_shift/vade_10clusters_merged_control_degron_rep1_cov_dia/nmi_%A_%a.out
#SBATCH --error=/usr/users/yzhu1/LoopBin/trials/cluster_shift/vade_10clusters_merged_control_degron_rep1_cov_dia/nmi_%A_%a.err
#SBATCH -c 2
#SBATCH --nodes=1
#SBATCH --mem=4gb
#SBATCH --time=2:00:00
# #SBATCH --array=5-8
data_path=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia
output_path=/usr/users/yzhu1/LoopBin/trials/cluster_shift/vade_10clusters_merged_control_degron_rep1_cov_dia/
python /usr/users/yzhu1/LoopBin/main.py -f 8 -d $data_path -u $output_path