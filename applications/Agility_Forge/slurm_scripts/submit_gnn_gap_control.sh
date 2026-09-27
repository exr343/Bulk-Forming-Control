#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=exr343@case.edu
#SBATCH --job-name=gnn_gap_control
#SBATCH --time=12:00:00
#SBATCH --array=0-1
# Absolute gap control (GNN/data.py, 2026-09-27): retrain the M=3 GNN with the
# half-gap as the 4th control, same recipe as mp_sweep's M=3 model (pretrain
# on rollout_snapshot_383, then finetune on the square runs, val seed 9,
# test seed 3). Array task 0: gap only; task 1: gap + 10% no-change hits.
# Usage (repo root): sbatch applications/Agility_Forge/slurm_scripts/submit_gnn_gap_control.sh
set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
echo "Job started: $(date) on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader || true

NAMES=(gap_only gap_nochange10)
FRACS=(0.0 0.1)
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
FRAC="${FRACS[$SLURM_ARRAY_TASK_ID]}"
OUT=applications/Agility_Forge/GNN/gap_control/$NAME
echo "variant $NAME (no-change fraction $FRAC) -> $OUT"

python -u -m applications.Agility_Forge.GNN.train \
    --rollout-allowlist applications/Agility_Forge/GNN/mp_sweep/rollout_snapshot_383.json \
    --message-passing-steps 3 \
    --control gap --no-change-frac "$FRAC" \
    --out-dir "$OUT/pretrain" \
    --checkpoint-path "$OUT/pretrain/checkpoint.pt"

python -u -m applications.Agility_Forge.GNN.finetune \
    --init-checkpoint "$OUT/pretrain/checkpoint.pt" \
    --no-change-frac "$FRAC" \
    --val-dir applications/Agility_Forge/data/dataset_finetuning/square_jitter_seed9 \
    --out-dir "$OUT/finetune" \
    --checkpoint-path "$OUT/finetune/checkpoint.pt"
echo "Job finished: $(date)"
