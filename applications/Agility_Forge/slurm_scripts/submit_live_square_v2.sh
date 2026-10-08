#!/bin/bash -l
#SBATCH --job-name=live_square_v2
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Live PNG of the running plain square run (new simulator): re-rendered whenever the run saves a new hit or
# reheat (checked every 2 min); stops after the forging task leaves the queue.
# Output: applications/Agility_Forge/output/die12_coil_animation/square_new_live.png
set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
python -u applications/Agility_Forge/output/die12_coil_animation/live.py \
    applications/Agility_Forge/data/dataset_die12_coil/square square_new --job 3955595_0 --interval 120
