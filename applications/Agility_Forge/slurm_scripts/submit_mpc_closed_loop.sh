#!/bin/bash -l
#SBATCH --job-name=agility_forge_mpc_closed_loop
#SBATCH --partition=gpu
#SBATCH --constraint=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Submit with:  sbatch slurm_scripts/submit_mpc_closed_loop.sh <target_rollout_id>
# (must be submitted with CWD at repo root)
#
# One target rollout per job -- submit 5 times (once per target) to run all
# 5 closed-loop MPC demos in parallel across separate GPU nodes rather than
# serially in one job. Each: 5 rounds of (SQP replan ~10-16min + real hit
# ~5-15min), horizon shrinking 5->4->3->2->1 -- budget a few hours per job;
# --time=8h leaves headroom.

set -euo pipefail

TARGET_ROLLOUT="$1"
OUT_DIR="applications/Agility_Forge/control/mpc_target_rollout_${TARGET_ROLLOUT}"

cd /home/exr343/CIRP_2027

module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Target rollout: $TARGET_ROLLOUT -> $OUT_DIR"

python -m applications.Agility_Forge.control.run_closed_loop \
    --target-rollout "$TARGET_ROLLOUT" \
    --out-dir "$OUT_DIR"

echo "Job finished: $(date)"
