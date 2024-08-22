#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=vade
#SBATCH --output=/usr/users/yzhu1/LoopBin/trials/saved_models/predict_%A_%a.out
#SBATCH --error=/usr/users/yzhu1/LoopBin/trials/saved_models/predict_%A_%a.err
#SBATCH -c 2
#SBATCH --nodes=1
#SBATCH --gpus-per-node 1
#SBATCH --mem=24gb
#SBATCH --time=1:00:00
#SBATCH --array=7

#module load cuda cudnn
#conda activate loophigh
n_clusters=$SLURM_ARRAY_TASK_ID
model_path=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_"$n_clusters"clusters_merged_control_degron_rep1/
for cond in control_rep1 degron_rep1
do
    data_path=/usr/users/yzhu1/LoopBin/data/02_process/$cond/log_data.npy
    output_path="$model_path"/prediction/"$cond"/
    mkdir -p $output_path
    python /usr/users/yzhu1/LoopBin/main.py -f 5 -d $data_path -m $model_path -u $output_path
done
