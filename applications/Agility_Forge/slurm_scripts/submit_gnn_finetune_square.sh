#!/bin/bash -l
#SBATCH --job-name=gnn_finetune_square
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Finetune GNN checkpoint_mp_5.pt on the square-rod rollouts + pretraining replay (see GNN/finetune.py).
# Clears GNN/finetune_square first -- the previous finetune's results are overwritten (by request).
# Submit from the repo root:  sbatch applications/Agility_Forge/slurm_scripts/submit_gnn_finetune_square.sh

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

rm -rf applications/Agility_Forge/GNN/finetune_square
python -u -m applications.Agility_Forge.GNN.finetune

echo "Job finished: $(date)"
