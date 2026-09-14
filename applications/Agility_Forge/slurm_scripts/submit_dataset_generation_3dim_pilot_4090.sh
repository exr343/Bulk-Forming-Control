#!/bin/bash -l
#SBATCH --job-name=agility_forge_dataset_3dim_pilot_4090
#SBATCH --partition=gpu
#SBATCH --constraint=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=96:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Submit with:  sbatch submit_dataset_generation_3dim_pilot_4090.sh
# (run from applications/Agility_Forge/, or adjust the `cd` below)
#
# 4090 variant of submit_dataset_generation_3dim_pilot_l40s.sh, submitted in
# its place (job 3794387, which was still PENDING on gpu2h100/gpul40s
# resource contention) once a free RTX 4090 (gput074, `mix` state -- has
# idle GPUs) became available. 24GB VRAM is ample for this workload: the
# surface mesh is only 4,355 nodes, and the JAX-FORGE paper itself reports
# running comparable simulations on an RTX 6000 (same consumer/workstation
# class, less VRAM than an L40S/H100). Picks up wherever the highest
# existing rollout_XX directory is at reservation time -- does not retry
# rollout 170's incomplete hit 3 (see agility_forge_dataset_3dim_pilot_l40s
# job 3793722's failure), just continues appending new rollouts.
#
# --n-rollouts/--n-hits/--min/max-compression-displacement: unchanged from
# the l40s pilot -- see that script's comments for the full rationale
# (3D-actuation-space pilot, u_j range grounded in the JAX-FORGE paper's
# reported forging depths).
#
# --time=96:00:00: carried over from the l40s script; 4090 throughput on
# this workload is untested, so kept generous (partition allows up to 13
# days).

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

python -m applications.Agility_Forge.generate_dataset \
    --n-rollouts 50 \
    --n-hits 5 \
    --min-compression-displacement 0.5 \
    --max-compression-displacement 2.0

echo "Job finished: $(date)"
