#!/bin/bash -l
#SBATCH --job-name=agility_forge_dataset_3dim_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=96:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Submit with:  sbatch slurm_scripts/submit_dataset_generation_3dim_gpu.sh
# (must be submitted with CWD at repo root)
#
# Same 3D-actuation config as submit_dataset_generation_3dim_pilot_l40s.sh /
# submit_dataset_generation_3dim_h100.sh (d_j, R_j, u_j all randomized,
# u_j in Uniform(0.5, 2.0)mm), but with NO --constraint -- requested on the
# generic `gpu` partition so SLURM can place it on whatever GPU node is free
# fastest, rather than waiting specifically for an L40S or H100. Appends to
# the same data/dataset dir as the other 3D-actuation runs; safe to run
# concurrently since generate_dataset.py's ManifestWriter uses flock-based
# locking (see generate_dataset.py) as of 2026-09-07.
#
# 50 rollouts, matching the other 3D pilot-sized runs.

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
echo "JAX_PLATFORMS=${JAX_PLATFORMS:-<unset>}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

python -m applications.Agility_Forge.generate_dataset \
    --n-rollouts 50 \
    --n-hits 5 \
    --min-compression-displacement 0.5 \
    --max-compression-displacement 2.0

echo "Job finished: $(date)"
