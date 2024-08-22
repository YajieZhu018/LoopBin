#!/bin/bash

#SBATCH -p gpu
#SBATCH --job-name=pretrain_ae
#SBATCH --output=/usr/users/yzhu1/LoopBin/trials/saved_models/ae_%A_%a.out
#SBATCH --error=/usr/users/yzhu1/LoopBin/trials/saved_models/ae_%A_%a.err
#SBATCH -c 8
#SBATCH --nodes=1
#SBATCH --gpus-per-node 1
#SBATCH --mem=48gb
#SBATCH --time=10:00:00
#SBATCH --array=0

#module load cuda cudnn
#conda activate loophigh
# merge input
in_file=/usr/users/yzhu1/LoopBin/data/02_process/merged_log_data.npy
out_file=/usr/users/yzhu1/LoopBin/trials/saved_models/pretrain_ae_merged_control_degron_rep1/
#python /usr/users/yzhu1/LoopBin/src/jobs/pretrain_ae.py $in_file $out_file
python /usr/users/yzhu1/LoopBin/main.py -f 3 -d $in_file -u $out_file