#!/bin/bash -l
#SBATCH --job-name=agility_forge_dataset_3dim_h100
#SBATCH --partition=gpu
#SBATCH --constraint=gpu2h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=120:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Submit with:  sbatch slurm_scripts/submit_dataset_generation_3dim_h100.sh
# (must be submitted with CWD at repo root)
#
# H100 variant of submit_dataset_generation_3dim_pilot_l40s.sh: same
# 3D-actuation-space config (u_j randomized Uniform(0.5, 2.0)mm alongside
# d_j/R_j) so it appends to the SAME data/dataset_pretraining dir rather than tripping
# ManifestWriter's config-mismatch check. Adds 100 more rollouts (not a
# pilot-sized batch -- the l40s/l40s-pilot runs already validated that this
# actuation range doesn't cause solver convergence problems).
#
# --time=120:00:00: at the ~34 min/rollout wall-clock pace observed on the
# l40s pilot (job 3793718: 50 rollouts / 28.47h), 100 rollouts is ~57h;
# generous headroom kept in case H100 throughput differs. Partition MaxTime
# is 13-08:00:00.
#
# Safe to run concurrently with the l40s job (3794407, rollouts 201-250)
# already targeting this same dataset dir: generate_dataset.py picked up its
# flock-based ManifestWriter locking fix at 16:48 on 09-07, before 3794407
# started (17:42) -- both processes will load that locked code and
# serialize manifest.json writes correctly. (The earlier failed job,
# 3793718, started before that fix landed and ran unlocked the whole time --
# that's the likely cause of the rollout-count undercount seen after it
# crashed.)

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
    --n-rollouts 100 \
    --n-hits 5 \
    --min-compression-displacement 0.5 \
    --max-compression-displacement 2.0

echo "Job finished: $(date)"
