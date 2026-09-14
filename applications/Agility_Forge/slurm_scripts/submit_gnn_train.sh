#!/bin/bash -l
#SBATCH --job-name=agility_forge_gnn_train
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
# Submit with:  sbatch submit_gnn_train.sh
# (run from applications/Agility_Forge/, or adjust the `cd` below)
#
# First real training run of the MeshGraphNets forging surrogate
# (applications/Agility_Forge/GNN/) at full reference sizing (latent=128,
# num_layers=2, message_passing_steps=15, ~2.33M params -- see
# GNN/README.md). Previously only smoke-tested (tiny architecture, a few
# epochs, CPU-only) -- this is the first run meant to produce a real
# checkpoint and loss curve, and the first real measurement of per-epoch
# wall-clock time on a GPU (the only number available before this was
# ~9.4s/batch-of-4 on a 4-thread CPU, i.e. GPU timing was an unmeasured
# estimate, not a fact -- check slurm_logs/%x_%j.out for the actual
# per-epoch timings this run reports).
#
# Trains against the live data/dataset/ (338 complete rollouts as of this
# submission), not a frozen snapshot -- confirmed via `squeue -u exr343`
# that no generate_dataset.py job is currently running against it, so there
# is no risk of reading a torn manifest.json (see CLAUDE.md's note on
# dataset/'s non-atomic writes).
#
# --batch-size 16, up from train.py's default of 4: the default was sized
# for a CPU smoke test, not GPU throughput. Not benchmarked against
# alternatives -- if per-epoch time reported by this run suggests the GPU is
# underutilized (or, conversely, out-of-memory), adjust and resubmit rather
# than assuming 16 is correct.
#
# --epochs/--patience left at train.py's defaults (100 / 20) -- untested
# against real training dynamics, since this is the first real run. Early
# stopping will end the job sooner than the 24h cap if test loss plateaus.

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
    --batch-size 16 \
    --epochs 100 \
    --patience 20

echo "Job finished: $(date)"
