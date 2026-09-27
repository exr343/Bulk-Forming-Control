#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=exr343@case.edu
#SBATCH --job-name=mp_sweep_finetune
#SBATCH --time=12:00:00
#
# New message-passing sweep, stage 2: finetune each depth's uniform checkpoint
# on the square runs (square + seeds 1,2,4-8; auto-discovered) with uniform
# replay, early stopping on seed 9, test = seed 3. Settings otherwise match the
# production finetune (GNN/finetune.py defaults). Array task id = depth.
# Submit from the repo root:  sbatch --array=1,5,15 applications/Agility_Forge/slurm_scripts/submit_mp_sweep_finetune.sh

set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
echo "Job started: $(date) on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
M="${SLURM_ARRAY_TASK_ID}"
SWEEP=applications/Agility_Forge/GNN/mp_sweep
python -u -m applications.Agility_Forge.GNN.finetune \
    --init-checkpoint "$SWEEP/checkpoint_mp_$M.pt" \
    --val-dir applications/Agility_Forge/data/dataset_finetuning/square_jitter_seed9 \
    --out-dir "$SWEEP/finetune/mp_$M" \
    --checkpoint-path "$SWEEP/finetune/mp_$M/checkpoint.pt"
echo "Job finished: $(date)"
