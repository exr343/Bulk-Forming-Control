#!/bin/bash -l
#SBATCH --job-name=ablation_multistart
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Checks + GNN-only comparison of the multi-start planner (control/ablation_multistart.py).
# Output: applications/Agility_Forge/control/results/ablation_multistart/
set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
python -u -m applications.Agility_Forge.control.ablation_multistart "$@"
