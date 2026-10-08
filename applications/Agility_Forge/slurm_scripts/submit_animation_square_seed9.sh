#!/bin/bash -l
#SBATCH --job-name=animate_square_seed9
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# GIF of square run jitter seed 9 (new simulator, keep-the-hotter reheat).
# Submit with --dependency=afterany:<forging job>_0
set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
python applications/Agility_Forge/output/die12_coil_animation/animate.py \
    applications/Agility_Forge/data/dataset_die12_coil/square_jitter_seed9 square_seed9
