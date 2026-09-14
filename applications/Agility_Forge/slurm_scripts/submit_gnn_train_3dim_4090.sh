#!/bin/bash -l
#SBATCH --job-name=agility_forge_gnn_train_3dim
#SBATCH --partition=gpu
#SBATCH --constraint=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Submit with:  sbatch slurm_scripts/submit_gnn_train_3dim_4090.sh
# (must be submitted with CWD at repo root)
#
# First GNN training run against the 3D-actuation dataset (data/dataset,
# u_j now a third independently-randomized control dim alongside d_j/R_j).
# Prerequisite done just before this run: data.py/model.py updated to encode
# u_j as u_j_frac = (u_j_mm - min)/(max - min), appended raw/unnormalized to
# the control broadcast (matching d_j_frac's existing convention) --
# NODE_IN_DIM 10 -> 11. Settled by explicit interview (see chat), not a
# silent default.
#
# Architecture kept at the existing reference sizing (latent=128,
# num_layers=2, message_passing_steps=15, ~2.33M params, train.py's
# defaults) -- deliberately not scaled up despite having more data than the
# 2D-only run it was originally sized against (~2,000 pairs from 383+
# rollouts here vs. ~1,690 before); data is still modest, so more capacity
# risks overfitting more than it helps.
#
# --batch-size left at train.py's default (4), NOT the 16 the earlier 2D-only
# GPU run (submit_gnn_train.sh) used for throughput -- this was an explicit
# choice to keep to the settled reference config; if per-epoch wall-clock
# time reported by this run looks GPU-underutilized, that's worth revisiting.
#
# --epochs/--patience left at train.py's defaults (100 / 20): patience=20
# specifically chosen (not patience=1 / stop-at-first-uptick) because the
# 2D-only run showed two sharp early loss spikes attributed to normalizer
# statistics still shifting -- a strict first-rise stop risks tripping on
# that exact known instability rather than real overfitting.
#
# Dataset note: data/dataset currently has 383 complete rollouts, with a
# gap-fill job (targeting 400) running in the background against the same
# directory. This run reads whatever's on disk at start (load_examples()
# is a one-time snapshot, not live-updating) -- not blocked on the gap-fill
# finishing first.

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

python -m applications.Agility_Forge.GNN.train \
    --dataset-dir applications/Agility_Forge/data/dataset \
    --epochs 100 \
    --patience 20 \
    --out-dir applications/Agility_Forge/GNN/runs_3dim \
    --checkpoint-path applications/Agility_Forge/GNN/checkpoint_3dim.pt

echo "Job finished: $(date)"
