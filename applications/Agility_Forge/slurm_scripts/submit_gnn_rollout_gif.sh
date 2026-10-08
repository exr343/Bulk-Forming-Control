#!/bin/bash -l
#SBATCH --job-name=gnn_rollout_gif
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# GIF comparing the simulation with the coil GNN's open-loop rollout. Args: rollout name and folder (see animate_rollout.py).
# Needs rollout.py's npz/json first. Output: applications/Agility_Forge/output/gnn_figures/
set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
python -u applications/Agility_Forge/output/gnn_figures/animate_rollout.py "$@"   # <name> <folder>, e.g. square_M5 .../gnn_figures/heldout_square
