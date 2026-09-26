#!/bin/bash -l
#SBATCH --job-name=agility_forge_prism_target
#SBATCH --partition=gpu
#SBATCH --constraint=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Submit with:  sbatch slurm_scripts/submit_prism_target.sh
# (must be submitted with CWD at repo root)
#
# Separate job, does NOT touch the concurrently-running full-test-set sweep
# (job 3824244). Closed-loop GNN-MPC toward a synthetic rounded-square-prism
# target (apothem 6.4375mm, reachable within one hit's 2.0mm max depth --
# see control/eval_prism_target.py's docstring for the full sizing
# rationale). n_hits=8: within the "start small" 5-10 range, ~2 hits/face
# across the 4 sides. u_j_mm_min=0.0 (project default as of the mpc.py
# change).

set -euo pipefail

cd /home/exr343/CIRP_2027

module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

# See submit_extended_horizon_pilot.sh's comment -- JAX (real plant) and
# PyTorch (planner) share one GPU in this process; capping JAX's
# preallocation avoids the OOM already hit once at horizon=10.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "GPU assigned:"
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "(nvidia-smi not available)"

python -m applications.Agility_Forge.control.eval_prism_target \
    --n-hits 8 \
    --apothem-mm 6.4375 \
    --u-j-mm-min 0.0 \
    --out-dir applications/Agility_Forge/control/mpc_prism_target

echo "Job finished: $(date)"
