#!/bin/bash

#SBATCH -p scc-gpu
#SBATCH --job-name=gmm_metric_bump_samples
#SBATCH --output=/mnt/ceph-hdd/projects/SCC_UKLA_papantonis/Yajie/LoopBin/scripts/gmm_metric_validation/run_bump_samples_%A.out
#SBATCH --error=/mnt/ceph-hdd/projects/SCC_UKLA_papantonis/Yajie/LoopBin/scripts/gmm_metric_validation/run_bump_samples_%A.err
#SBATCH -c 4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --mem=32gb
#SBATCH --time=03:00:00
#SBATCH --array=0

module load gcc
module load cuda

source /mnt/ceph-hdd/projects/SCC_UKLA_papantonis/Yajie/LoopBin/.venv/bin/activate

# This cluster (scc-gpu) has no `cuda`/`nvhpc` environment module and only CUDA 12.8
# installed system-wide -- TensorFlow 2.5.0 (pinned in the loopbin venv) needs CUDA
# 11.x runtime libraries, which don't exist anywhere on this system otherwise. Fix:
# unpack a prebuilt set of pip-sourced CUDA 11 runtime .so files (nvidia-cuda-runtime-cu11,
# nvidia-cudnn-cu11==8.6.0.163, cublas/cufft/curand/cusolver/cusparse-cu11) to the
# compute node's local /tmp and point LD_LIBRARY_PATH at them. Extracting locally
# (rather than referencing the many small .so files directly on ceph-hdd) matters:
# ceph-hdd write/read throughput for many-small-file payloads on this cluster is
# extremely slow (a `pip install --target=<ceph-hdd path>` of these same packages
# hung for 10+ minutes with no progress); the tarball is a single file so this
# extraction takes seconds. See run_report.md's 2026-08-26 CUDA/GPU entry.
CUDA11_LOCAL=/tmp/${SLURM_JOB_ID}_cuda11_libs
mkdir -p $CUDA11_LOCAL
tar -xf /mnt/ceph-hdd/projects/SCC_UKLA_papantonis/Yajie/LoopBin/src/jobs/cuda11_runtime_libs.tar -C $CUDA11_LOCAL
export LD_LIBRARY_PATH=$(find $CUDA11_LOCAL -maxdepth 3 -type d -name lib | paste -sd: -):$LD_LIBRARY_PATH

cd /mnt/ceph-hdd/projects/SCC_UKLA_papantonis/Yajie/LoopBin/scripts/gmm_metric_validation

# N_SAMPLES bumped 10000 -> 30000 in 01_generate_toy_data.py to push further on
# whether more samples per AE-pretrain epoch keeps reducing the dead-latent-unit
# problem (8/10 dead at N=3000 -> 6/10 dead at N=10000). Pre-bump toy_data/pretrained
# backed up to n10000_backup/ (and n3000_backup/ before that). This is also the
# first run of this script with the GPU fix actually in effect (see above).
python 01_generate_toy_data.py
python 02_evaluate_metric_weights.py
python 03_plot_metric_selection.py
