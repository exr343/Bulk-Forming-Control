#!/bin/bash -l
#SBATCH --job-name=agility_forge_square_jitter
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --array=1-3
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# 3 jittered variants of the 48-hit square rollout (array task id = jitter seed = folder suffix).
# Jitter: station +-3.7 mm, hit angle +-5 deg, pass half-gap +-0.2 mm (see jitter_square_schedule).
# Submit from the repo root:  sbatch --dependency=afterany:<square job id> applications/Agility_Forge/slurm_scripts/submit_square_jitter_rollouts.sh
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

python -u -m applications.Agility_Forge.generate_square_rollout --jitter-seed "${SLURM_ARRAY_TASK_ID}" \
    --dataset-dir "applications/Agility_Forge/data/dataset_finetuning/square_jitter_seed${SLURM_ARRAY_TASK_ID}"

echo "Job finished: $(date)"
