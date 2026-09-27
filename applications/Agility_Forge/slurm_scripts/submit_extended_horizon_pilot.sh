#!/bin/bash -l
#SBATCH --job-name=agility_forge_extended_horizon_pilot
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
# Submit with:  sbatch slurm_scripts/submit_extended_horizon_pilot.sh
# (must be submitted with CWD at repo root)
#
# Single-rollout pilot, separate from the concurrently-running full-test-set
# sweep (job 3824244, NOT cancelled by this): does giving closed-loop MPC
# more real hits than the training data's fixed 5-hit schedule let it
# converge closer to a fixed target, or does extrapolating the GNN surrogate
# past hit 5 (states it was never trained on) make things worse? Target
# stays fixed at results/before_2026-09-26_fixes/rollout_415's true hit-5 geometry throughout;
# hits 6-10 have no real ground truth for this rollout at all.
#
# Cheap relative to the full sweep: one rollout, ~10 real hits. At the
# ~32min/5-real-hits pace measured live on the concurrent 4090 run
# (~6.4min/hit average), 10 hits is roughly ~65min of real-plant time plus
# negligible SQP planning (single-digit seconds per replan observed on this
# same GPU type) -- --time=06:00:00 leaves generous headroom.
#
# --constraint=gpu4090 (not gpul40s/gpu2h100): matches the concurrent job's
# GPU choice (already confirmed faster to schedule) and its checkpoint
# (checkpoint_mp_5.pt), so hit 1-5 numbers from this run are directly
# comparable to that run's per-hit averages.
#
# --u-j-mm-min 0.0 (rerun, first attempt's data deleted): the first 10-hit
# run (job 3825955) showed u_j_mm pinned at the hard 0.5mm floor and
# d_j_frac pinned near the 0.78 edge for every hit from 4 onward -- Chamfer
# plateaued and Hausdorff crept upward hit-over-hit past hit 5, because the
# optimizer has no way to choose "no hit": every remaining hit is forced to
# apply >=0.5mm of real plastic deformation somewhere, even once already
# near target. Lowering the floor to 0.0mm lets u_j_frac=0 mean an
# effectively negligible stroke, testing whether the plateau/degradation
# was caused by that forced minimum rather than surrogate extrapolation
# (mpc.py's plan_vs_actual_rmse_mm stayed flat through hit 10 in the first
# run, arguing against a surrogate-breakdown explanation).

set -euo pipefail

cd /home/exr343/CIRP_2027

module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

# This process runs both JAX (jax_forge, the real FEM plant) and PyTorch
# (ForgeGNN/MPCController, the SQP planner) on the same GPU. JAX's default
# allocator preallocates most of the GPU upfront and never gives it back --
# fine at horizon=5 (the concurrent sweep, job 3824244, has run 24+ rollouts
# on this same GPU type with zero OOMs), but the 10-step single-shooting
# rollout here roughly doubles PyTorch's autograd activation memory (all
# horizon steps' GNN activations retained at once for one .backward() call),
# and there wasn't enough headroom left. Capping JAX to grow on demand
# instead of preallocating fixes this without touching any model/planner
# code (first OOM'd in job 3825633).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "GPU assigned:"
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "(nvidia-smi not available)"

python -m applications.Agility_Forge.control.eval_extended_horizon \
    --target-rollout 415 \
    --n-hits 10 \
    --u-j-mm-min 0.0 \
    --out-dir applications/Agility_Forge/control/results/before_2026-09-26_fixes/rollout_415_10hits

echo "Job finished: $(date)"
