#!/bin/bash -l
#SBATCH --job-name=agility_forge_gnn_mp_sweep
#SBATCH --partition=gpu
#SBATCH --constraint=gpu2h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Submit with:  sbatch slurm_scripts/submit_gnn_train_mp_sweep.sh <message_passing_steps>
# (must be submitted with CWD at repo root)
#
# Message-passing-depth comparison (5/15/45/135 steps). checkpoint_3dim.pt
# already exists at 15 steps -- only 5, 45, 135 need training here. All runs
# (including the existing 15-step one) MUST train on the identical 383-
# rollout snapshot for depth to be the only varying factor: the dataset has
# grown to 400 rollouts since checkpoint_3dim.pt was trained, so
# --rollout-allowlist pins every new run to GNN/mp_sweep/rollout_snapshot_383.json
# (reconstructed via file mtimes -- see chat history -- and verified to
# reproduce checkpoint_3dim.pt's exact original 306/77 train/test split).
#
# Everything else at train.py's defaults (latent=128, num_layers=2, batch=4,
# epochs=100, patience=20) -- only message_passing_steps varies.

set -euo pipefail

MP_STEPS="$1"
OUT_DIR="applications/Agility_Forge/GNN/mp_sweep/mp_${MP_STEPS}"

cd /home/exr343/CIRP_2027

module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "message_passing_steps=$MP_STEPS -> $OUT_DIR"

python -m applications.Agility_Forge.GNN.train \
    --rollout-allowlist applications/Agility_Forge/GNN/mp_sweep/rollout_snapshot_383.json \
    --message-passing-steps "$MP_STEPS" \
    --out-dir "$OUT_DIR" \
    --checkpoint-path "applications/Agility_Forge/GNN/mp_sweep/checkpoint_mp_${MP_STEPS}.pt"

echo "Job finished: $(date)"
