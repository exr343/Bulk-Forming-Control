#!/bin/bash -l
#SBATCH --job-name=agility_forge_dataset_h100
#SBATCH --partition=gpu
#SBATCH --constraint=gpu2h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Submit with:  sbatch submit_dataset_generation_h100.sh
# (run from applications/Agility_Forge/, or adjust the `cd` below)
#
# H100 variant of submit_dataset_generation.sh: appends 25 more rollouts to
# the existing 25-rollout dataset (-> 50 total). --seed is deliberately
# omitted (fresh OS entropy) so this run draws NEW d_j/R_j samples instead of
# replaying the first run's --seed 42 sequence.

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
    --n-rollouts 25 \
    --n-hits 5

echo "Job finished: $(date)"
