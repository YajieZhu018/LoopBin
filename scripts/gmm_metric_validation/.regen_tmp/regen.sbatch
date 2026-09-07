#!/bin/bash
#SBATCH -p scc-cpu
#SBATCH --job-name=regen_report_pdf
#SBATCH --output=/mnt/ceph-hdd/projects/SCC_UKLA_papantonis/Yajie/LoopBin/scripts/gmm_metric_validation/.regen_tmp/regen_%A.out
#SBATCH --error=/mnt/ceph-hdd/projects/SCC_UKLA_papantonis/Yajie/LoopBin/scripts/gmm_metric_validation/.regen_tmp/regen_%A.err
#SBATCH -c 4
#SBATCH --nodes=1
#SBATCH --mem=32gb
#SBATCH --time=00:30:00

module load gcc
source /mnt/ceph-hdd/projects/SCC_UKLA_papantonis/Yajie/LoopBin/.venv/bin/activate
cd /mnt/ceph-hdd/projects/SCC_UKLA_papantonis/Yajie/LoopBin/scripts/gmm_metric_validation/.regen_tmp
export PYTHONUNBUFFERED=1
python regen_report.py linear relu
