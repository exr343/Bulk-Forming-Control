#!/bin/bash -l
#SBATCH --job-name=agility_forge_undeformed_baseline
#SBATCH --partition=gpu
#SBATCH --constraint=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# No-skill baseline: Chamfer/Hausdorff between true geometry and the
# undeformed billet, same 77-rollout test split as the mp_sweep comparison.
# No model inference -- just reading .vtu files and computing distances --
# so this should be much faster than the model-eval jobs.

set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

python -m applications.Agility_Forge.GNN.mp_sweep.undeformed_baseline_eval
