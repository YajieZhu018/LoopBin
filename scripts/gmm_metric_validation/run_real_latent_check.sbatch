#!/bin/bash

#SBATCH -p scc-gpu
#SBATCH --job-name=gmm_real_latent_check
#SBATCH --output=/mnt/ceph-hdd/projects/SCC_UKLA_papantonis/Yajie/LoopBin/scripts/gmm_metric_validation/run_real_latent_check_%A.out
#SBATCH --error=/mnt/ceph-hdd/projects/SCC_UKLA_papantonis/Yajie/LoopBin/scripts/gmm_metric_validation/run_real_latent_check_%A.err
#SBATCH -c 4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --mem=64gb
#SBATCH --time=03:00:00
#SBATCH --array=0

module load gcc
module load cuda

source /mnt/ceph-hdd/projects/SCC_UKLA_papantonis/Yajie/LoopBin/.venv/bin/activate

# CUDA 11 runtime shim -- see run_bump_samples.sbatch and run_report.md's 2026-08-26
# CUDA/GPU entry for why this is needed and why it's a tarball.
CUDA11_LOCAL=/tmp/${SLURM_JOB_ID}_cuda11_libs
mkdir -p $CUDA11_LOCAL
tar -xf /mnt/ceph-hdd/projects/SCC_UKLA_papantonis/Yajie/LoopBin/src/jobs/cuda11_runtime_libs.tar -C $CUDA11_LOCAL
export LD_LIBRARY_PATH=$(find $CUDA11_LOCAL -maxdepth 3 -type d -name lib | paste -sd: -):$LD_LIBRARY_PATH

BASE=/mnt/ceph-hdd/projects/SCC_UKLA_papantonis/Yajie/LoopBin
cd $BASE/scripts/gmm_metric_validation

# Does the REAL production latent space suffer the same dead-unit / effective-rank
# collapse the toy harness found? Never been measured. If it does, every published
# clustering under trials/ was computed in a latent space far smaller than its
# nominal 10 dimensions.
#
# Both arms replicate main.py:pretrain_ae EXACTLY -- full merged_log_data.npy
# (no subsample, --n-samples 0), 200 epochs, batch 256, Adam lr=0.002, seed 0 --
# so the 'relu' arm is a faithful reproduction of what production actually builds,
# not a scaled-down proxy. num_clusters=10 matches the existing
# trials/vade_10clusters_merged_control_degron_rep1_cov_dia_run1 run.
#
# The only difference between the two arms is --latent-activation.
for ACT in relu linear; do
    echo "############ latent_activation=$ACT ############"
    python 04_test_real_pretrain_gmm.py \
        --data $BASE/data/02_process/merged_log_data.npy \
        --n-samples 0 \
        --num-clusters 10 \
        --epochs 200 \
        --seed 0 \
        --covariance-type diag \
        --metric-weights 1.0,1.0,1.0 \
        --latent-activation $ACT \
        --output $BASE/scripts/gmm_metric_validation/results/real_latent_$ACT
done
