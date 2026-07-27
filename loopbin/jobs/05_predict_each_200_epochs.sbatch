#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=predict
#SBATCH --output=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run7/checkpoints/predict_%A_%a.out
#SBATCH --error=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run7/checkpoints/predict_%A_%a.err
#SBATCH -c 2
#SBATCH --nodes=1
#SBATCH --gpus-per-node 1
#SBATCH --mem=24gb
#SBATCH --time=2:00:00
#SBATCH --array=1-15

#module load cuda cudnn
#conda activate loophigh
epochs=$((SLURM_ARRAY_TASK_ID * 200))
run=7
model_path=/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run"$run"/checkpoints/model_epoch_"$epochs"/
data_path=/usr/users/yzhu1/LoopBin/data/02_process/merged_log_data.npy
python /usr/users/yzhu1/LoopBin/main.py -f 5 -d $data_path -m $model_path -u $model_path -p CTCF,H3K27ac,H3K27me3,SMC1A
