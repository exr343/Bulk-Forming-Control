#!/bin/bash -l
#SBATCH --job-name=agility_forge_mp_sweep_eval
#SBATCH --partition=gpu
#SBATCH --constraint=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Full test-set (77 rollouts x 5 hits x 4 models = 1540 forward-only
# autoregressive evaluations) Chamfer/Hausdorff comparison across the
# message-passing sweep. Forward-only (no backward/autodiff), so none of the
# 135-step OOM concerns from the gradient-timing benchmark apply here.

set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

python -m applications.Agility_Forge.GNN.mp_sweep.full_test_set_eval
