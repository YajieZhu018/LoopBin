#!/bin/bash

#SBATCH -p gpu
#SBATCH --job-name=vade
#SBATCH --output=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_%A_%a.out
#SBATCH --error=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_%A_%a.err
#SBATCH -c 8
#SBATCH --nodes=1
#SBATCH --gpus-per-node 1
#SBATCH --mem=48gb
#SBATCH --time=8:00:00
#SBATCH --array=10

#module load cuda cudnn
#conda activate loophigh
n_clusters=$SLURM_ARRAY_TASK_ID
data_path=/usr/users/yzhu1/LoopBin/data/02_process/merged_log_data.npy   
pretrain_model_path=/usr/users/yzhu1/LoopBin/trials/saved_models/pretrain_ae_merged_control_degron_rep1/
output_path=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_"$n_clusters"clusters_merged_control_degron_rep1/
if_pretrain=True
epochs=3000
python /usr/users/yzhu1/LoopBin/main.py -f 4 -num $n_clusters -d $data_path -pre $pretrain_model_path -u $output_path -if_pre $if_pretrain -ep $epochs
#python /usr/users/yzhu1/LoopBin/src/jobs/train_vade.py $n_clusters $data_path $pretrain_model_path $output_path $if_pretrain $epochs