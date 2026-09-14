#!/bin/bash -l
#SBATCH --job-name=agility_forge_koopman_train
#SBATCH --partition=gpu
#SBATCH --constraint=gpu2h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Submit with:  sbatch submit_koopman_train.sh
# (run from applications/Agility_Forge/, or adjust the `cd` below)
#
# Full training run with the current architecture (train.py defaults -- no
# override needed): state -> POD-project (fixed, r=75 modes fit on train
# only, see applications/Agility_Forge/SVD/) -> in_proj -> 4 residual blocks
# (constant width 250, LayerNorm'd, 8 layers each, chained directly with no
# transition layers since width doesn't change) -> out_proj -> z (n_z=250,
# > r=75 -- Koopman lifts to a *higher*-dim space). Decoder: learned linear
# n_z->r, then fixed POD reconstruction back to n_x. ~2.19M learned params
# total (down from ~2.4B in the pre-POD design), once
# applications/Agility_Forge/SVD/'s cumulative-energy analysis showed that
# scale wasn't justified by the data. No --gradclip (removed from train.py
# entirely).
#
# Trains against data/dataset_snapshot_r25 -- a static copy of the 25-rollout
# dataset taken before submit_dataset_generation_h100.sh (appending 25 more
# rollouts, still running as of this job's submission) started, so this read
# doesn't race that job's non-atomic manifest.json rewrites (see CLAUDE.md:
# don't read/write dataset/ while a generate_dataset.py job is actively
# running against it).
#
# --patience 50, --epochs 5000 as a safety cap: real early stopping this
# time (the previous --patience 5000 run was a deliberate diagnostic on the
# much smaller pre-residual-encoder architecture, not meant to carry over).
# At ~0.3s/epoch measured on this architecture, 5000 epochs is ~25 min if
# patience never triggers -- cheap relative to the 24h time limit.
#
# --checkpoint-path/--out-dir left at train.py defaults (checkpoint.pt,
# runs/) -- overwritten each invocation instead of accumulating.

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

python -m applications.Agility_Forge.koopman.train \
    --dataset-dir applications/Agility_Forge/data/dataset_snapshot_r25 \
    --epochs 5000 \
    --patience 50

echo "Job finished: $(date)"
