#!/bin/bash -l
#SBATCH --job-name=live_all_square
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Live multi-panel PNG of all 10 new-simulator square runs (one 3D bar per run + Hausdorff vs hit),
# re-rendered whenever any run saves a new hit or reheat (checked every 2 min); stops after the
# forging jobs have left the queue.
# Output: applications/Agility_Forge/output/die12_coil_animation/all_runs_live.png
set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
python -u applications/Agility_Forge/output/die12_coil_animation/live_all.py --jobs 3975150 --interval 120 --passes 5 \
    --data-dir applications/Agility_Forge/data/dataset_die12_coil_5pass --fallback-dir applications/Agility_Forge/data/dataset_die12_coil
