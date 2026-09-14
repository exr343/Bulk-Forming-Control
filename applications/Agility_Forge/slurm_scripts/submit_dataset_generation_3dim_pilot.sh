#!/bin/bash -l
#SBATCH --job-name=agility_forge_dataset_3dim_pilot
#SBATCH --partition=gpu
#SBATCH --constraint=gpu2h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=96:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Submit with:  sbatch submit_dataset_generation_3dim_pilot.sh
# (run from applications/Agility_Forge/, or adjust the `cd` below)
#
# Pilot batch for the 3D-actuation-space expansion: adds u_j (strike depth,
# "compression_displacement") as a third independently-randomized per-hit
# parameter, alongside the existing d_j (axial position) and R_j (rotation).
# Previously u_j was fixed at 2.5mm for the entire dataset -- the old
# 2D-actuation dataset (338 rollouts, all at u_j=2.5mm) has been preserved
# at data/dataset_2dim rather than overwritten; this job writes to the
# (now-empty) default data/dataset path instead, starting the new dataset
# fresh from rollout_01.
#
# --min/max-compression-displacement 0.5/2.0mm: widened slightly beyond the
# two forging depths (0.62mm, 1.63mm) reported for this same billet geometry
# (diameter 15.9mm round stock) in the JAX-FORGE paper (Wright et al., CIRP
# Annals 2026) -- see generate_dataset.py's module docstring. Uniform
# sampling, independently redrawn every hit, matching d_j/R_j's convention.
#
# --n-rollouts 50: a deliberate pilot, not the final dataset size -- the
# point is to confirm the new (wider) strike-depth range doesn't introduce
# solver convergence problems (untested combinations of d_j/R_j/u_j) before
# committing to a much larger, more expensive generation run. Inspect
# manifest.json's records afterward for any missing/failed hits before
# scaling up.
#
# --time=96:00:00: at the old fixed-u_j=2.5mm dataset's observed per-hit
# wall time (~600-1000s), 50 rollouts x 5 hits could take up to ~55-70
# hours; set generously above that since the new, wider/untested u_j range
# could plausibly need more Newton retries (partition allows up to 13 days,
# so this has ample headroom either way).

set -euo pipefail

cd /home/exr343/CIRP_2027

module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "GPU assigned:"
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "(nvidia-smi not available)"

python -m applications.Agility_Forge.generate_dataset \
    --n-rollouts 50 \
    --n-hits 5 \
    --min-compression-displacement 0.5 \
    --max-compression-displacement 2.0

echo "Job finished: $(date)"
