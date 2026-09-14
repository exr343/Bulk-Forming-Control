#!/bin/bash -l
#SBATCH --job-name=agility_forge_full_solve_timer
#SBATCH --partition=gpu
#SBATCH --constraint=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Submit with: sbatch slurm_scripts/submit_full_solve_timer.sh <message_passing_steps>
#
# Real, non-extrapolated full SQP-BFGS plan() solve time for one mp_sweep
# model, on a dedicated node (avoids the busy-login-node contention issue
# from earlier measurements).

set -euo pipefail
MP_STEPS="$1"
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

python -m applications.Agility_Forge.GNN.mp_sweep.full_solve_timer --message-passing-steps "$MP_STEPS"
