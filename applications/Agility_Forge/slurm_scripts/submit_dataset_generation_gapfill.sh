#!/bin/bash -l
#SBATCH --job-name=agility_forge_dataset_gapfill
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Submit with:  sbatch slurm_scripts/submit_dataset_generation_gapfill.sh
# (must be submitted with CWD at repo root)
#
# Follow-up to submit_dataset_generation_backfill.sh (job 3803844), which was
# cancelled mid-run at 14/31 replacement rollouts complete. All incomplete
# rollout dirs (the original 31 stragglers + the 17 never-started backfill
# slots 415-431) were then moved out to data/backup/incomplete_rollouts/, so
# data/dataset/ now holds 383 clean, complete rollouts with nothing for the
# backfill script's own straggler-scan to find. This job doesn't re-scan --
# it directly requests the fixed number of rollouts needed to restore the
# original target: 400 - 383 = 17.

set -euo pipefail

cd /home/exr343/CIRP_2027

module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

echo "Job started: $(date)"
echo "Host: $(hostname)"

python -m applications.Agility_Forge.generate_dataset \
    --n-rollouts 17 \
    --n-hits 5 \
    --min-compression-displacement 0.5 \
    --max-compression-displacement 2.0

echo "Job finished: $(date)"
