#!/bin/bash -l
#SBATCH --job-name=animate_square_5pass
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# GIF of the 5-pass plain square run (pass-5 half-gap 5.0 mm), data/dataset_die12_coil_5pass/square.
set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
python applications/Agility_Forge/output/die12_coil_animation/animate.py \
    applications/Agility_Forge/data/dataset_die12_coil_5pass/square square_5pass
