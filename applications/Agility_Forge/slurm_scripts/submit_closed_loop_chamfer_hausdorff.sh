#!/bin/bash -l
#SBATCH --job-name=agility_forge_closed_loop_chd
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
# Submit with:  sbatch slurm_scripts/submit_closed_loop_chamfer_hausdorff.sh
# (must be submitted with CWD at repo root)
#
# Separate job, does not touch the concurrently-running full-test-set sweep
# (job 3824244). Standard 5-hit closed-loop GNN-MPC (run_closed_loop.py,
# unchanged design -- re-plans from the real plant's true state before every
# hit) targeting rollout 415's true hit-5 geometry, same target used for the
# original M=15 validation (control/mpc_target_rollout_415/) and the
# extended-horizon pilots. Only the model differs here: checkpoint_mp_5_
# chamfer_hausdorff.pt (message_passing_steps=5), trained with Chamfer+
# Hausdorff loss terms added (weight 1.0 each) on top of the per-node MSE,
# on the full current dataset (401 rollouts, no pinned snapshot) -- see
# GNN/runs_mp5_chamfer_hausdorff/metrics.json for that training run's own
# numbers (best epoch 67, test NRMSE 0.298 -- close to the original MSE-only
# M=5 model's 0.298, despite the very different loss).
#
# run_closed_loop.py now takes --message-passing-steps/--device as CLI args
# (previously message_passing_steps=15 and device="cpu" were hardcoded) --
# added so this same script can target any checkpoint, not just checkpoint_
# 3dim.pt (M=15). device=cuda here (JAX preallocation capped below, same fix
# as the earlier OOM).

set -euo pipefail

cd /home/exr343/CIRP_2027

module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "GPU assigned:"
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "(nvidia-smi not available)"

python -m applications.Agility_Forge.control.run_closed_loop \
    --target-rollout 415 \
    --checkpoint-path applications/Agility_Forge/GNN/mp_sweep/checkpoint_mp_5_chamfer_hausdorff.pt \
    --message-passing-steps 5 \
    --out-dir applications/Agility_Forge/control/mpc_target_rollout_415_chamfer_hausdorff

echo "Job finished: $(date)"
