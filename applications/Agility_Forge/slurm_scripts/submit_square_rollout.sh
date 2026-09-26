#!/bin/bash -l
#SBATCH --job-name=agility_forge_square
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# 48-hit square-forming rollout (simple_square-inspired) for GNN fine-tuning data.
# Submit from the repo root:  sbatch applications/Agility_Forge/slurm_scripts/submit_square_rollout.sh
# Resubmitting with the same --dataset-dir resumes from the last completed hit.

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

python -m applications.Agility_Forge.generate_square_rollout \
    --dataset-dir applications/Agility_Forge/data/dataset_finetuning/square

echo "Job finished: $(date)"
