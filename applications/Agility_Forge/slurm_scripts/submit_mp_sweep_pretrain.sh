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
#SBATCH --job-name=mp_sweep_pretrain
#SBATCH --time=12:00:00
#
# New message-passing sweep, stage 1: pretrain the depths that have no uniform
# checkpoint yet on the pinned 383-rollout snapshot, with train.py's defaults
# (same as GNN/old/mp_sweep). Array task id = message-passing steps.
# Submit from the repo root:  sbatch --array=2,3,8 applications/Agility_Forge/slurm_scripts/submit_mp_sweep_pretrain.sh

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
python -u -m applications.Agility_Forge.GNN.train \
    --rollout-allowlist "$SWEEP/rollout_snapshot_383.json" \
    --message-passing-steps "$M" \
    --out-dir "$SWEEP/pretrain/mp_$M" \
    --checkpoint-path "$SWEEP/checkpoint_mp_$M.pt"
echo "Job finished: $(date)"
