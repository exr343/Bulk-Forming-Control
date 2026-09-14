#!/bin/bash -l
#SBATCH --job-name=agility_forge_dataset_batch
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s|gpu4v100|gpu2h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# One shard of a large parallel dataset-generation batch. NOT meant to be
# submitted directly -- launched by a loop (see chat/session history) that
# sets DATASET_SUFFIX per submission via `sbatch --export=ALL,DATASET_SUFFIX=NN`
# purely for --job-name/log-naming; all shards write into the SAME
# data/dataset/ dir. Safe to do so as of the ManifestWriter/rollout-index
# locking fix (fcntl.flock-based, see generate_dataset.py) -- stress-tested
# with 12 concurrent workers writing the same dir (no lost records, no
# overlapping rollout-index ranges) before pointing real jobs at it.
#
# CAVEAT: the lock is advisory and only protects processes that go through
# it. Job 3747184 (appending to data/dataset/, started before this fix
# existed) is running OLD code with no knowledge of the lock -- it can't be
# retroactively patched without restarting it and losing its progress, so
# there's a narrow (low-probability -- 3747184 only touches manifest.json
# every ~10-40 min, and the locked critical section here takes milliseconds)
# residual race against it specifically until it finishes (was at 48/50
# rollouts as of this batch's launch, so expected soon).
#
# --constraint spans exactly the GPU families empirically verified this
# session to run the jax-fem-env CUDA 12.8 stack correctly: L40S, the
# 4-GPU V100 nodes, and H100. Deliberately excludes P100/RTX2080/4090 --
# not checked this session, don't assume.
#
# --cpus-per-task=4 / --mem=16G (down from the single-job default of 8/48G):
# real usage on a prior single-job run was ~9.2G RSS (see sacct MaxRSS on
# job 3742390), and the V100 4-GPU nodes only have 24 CPUs total -- 4 jobs
# packed on one such node at 8 CPUs each wouldn't fit. 4 CPUs/16G leaves
# headroom for other users already running on these shared nodes.

set -euo pipefail

cd /home/exr343/CIRP_2027

module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Shard: ${DATASET_SUFFIX:-<unset>}"
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "(nvidia-smi not available)"

python -m applications.Agility_Forge.generate_dataset \
    --n-rollouts 25 \
    --n-hits 5

echo "Job finished: $(date)"
