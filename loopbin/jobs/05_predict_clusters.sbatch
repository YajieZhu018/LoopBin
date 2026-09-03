#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=predict
#SBATCH --output=/usr/users/yzhu1/LoopBin/trials/saved_models/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run_%a/predict_%A_%a.out
#SBATCH --error=/usr/users/yzhu1/LoopBin/trials/saved_models/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run_%a/predict_%A_%a.err
#SBATCH -c 2
#SBATCH --nodes=1
#SBATCH --gpus-per-node 1
#SBATCH --mem=24gb
#SBATCH --time=2:00:00
#SBATCH --array=1-5

#module load cuda cudnn
#conda activate loophigh
run=$SLURM_ARRAY_TASK_ID
model_path=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run"$run"/
for cond in control_rep1 degron_rep1 
do
    data_path=/usr/users/yzhu1/LoopBin/data/02_process/"$cond"/log_data.npy
    output_path="$model_path"/"$cond"/
    # if output path does not exist, create it
    if [ ! -d $output_path ]; then
        mkdir -p $output_path
    fi
    python /usr/users/yzhu1/LoopBin/main.py -f 5 -d $data_path -m $model_path -u $output_path -p CTCF,H3K27ac,H3K27me3,SMC1A
done
