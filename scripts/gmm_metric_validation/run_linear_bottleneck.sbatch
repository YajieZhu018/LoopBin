#!/bin/bash

#SBATCH -p scc-gpu
#SBATCH --job-name=gmm_linear_bottleneck
#SBATCH --output=/mnt/ceph-hdd/projects/SCC_UKLA_papantonis/Yajie/LoopBin/scripts/gmm_metric_validation/run_linear_bottleneck_%A.out
#SBATCH --error=/mnt/ceph-hdd/projects/SCC_UKLA_papantonis/Yajie/LoopBin/scripts/gmm_metric_validation/run_linear_bottleneck_%A.err
#SBATCH -c 4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --mem=32gb
#SBATCH --time=03:00:00
#SBATCH --array=0

module load gcc
module load cuda

source /mnt/ceph-hdd/projects/SCC_UKLA_papantonis/Yajie/LoopBin/.venv/bin/activate

# CUDA 11 runtime shim -- identical to run_bump_samples.sbatch; see that script's
# comment and run_report.md's 2026-08-26 CUDA/GPU entry for why this is needed
# (TF 2.5.0 needs CUDA 11.x, this cluster only has 12.8 system-wide and no cuda
# module; the libs are tarred because ceph-hdd is pathologically slow for
# many-small-file payloads).
CUDA11_LOCAL=/tmp/${SLURM_JOB_ID}_cuda11_libs
mkdir -p $CUDA11_LOCAL
tar -xf /mnt/ceph-hdd/projects/SCC_UKLA_papantonis/Yajie/LoopBin/src/jobs/cuda11_runtime_libs.tar -C $CUDA11_LOCAL
export LD_LIBRARY_PATH=$(find $CUDA11_LOCAL -maxdepth 3 -type d -name lib | paste -sd: -):$LD_LIBRARY_PATH

cd /mnt/ceph-hdd/projects/SCC_UKLA_papantonis/Yajie/LoopBin/scripts/gmm_metric_validation

# Linear-bottleneck experiment, at the existing N=30000 toy data.
#
# Changes under test vs. the N=30000 run (SLURM 15519814, results/n30000/):
#   1. src/model/ae.py's bottleneck activation is now a constructor arg, and
#      02_evaluate_metric_weights.py sets LATENT_ACTIVATION = None (linear).
#      ae.py still DEFAULTS to 'relu', so production main.py:pretrain_ae and
#      04_test_real_pretrain_gmm.py are untouched -- this is opt-in here only.
#      Hypothesis: the dead latent dims (5-6 of 10 across every run so far) are
#      dying-ReLU, not insufficient data, so a linear bottleneck should recover
#      most of them and finally make cov_health measurable.
#   2. 02 now computes latent diagnostics (alive dims, effective rank, and
#      within-cluster vs. global dimension correlation) -> latent_diagnostics.json,
#      and 03 renders them as the report's first page.
#
# 01_generate_toy_data.py is deliberately NOT rerun: toy_data/ already holds the
# N=30000 datasets used by run 15519814, so reusing them keeps this a controlled
# A/B on the activation alone. The previous run's ReLU-bottleneck AEs and latents
# were moved to pretrained_n30000_relu/ for before/after comparison.
python 02_evaluate_metric_weights.py
python 03_plot_metric_selection.py
