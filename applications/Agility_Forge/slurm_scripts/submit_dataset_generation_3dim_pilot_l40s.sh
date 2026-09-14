#!/bin/bash -l
#SBATCH --job-name=agility_forge_dataset_3dim_pilot_l40s
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=96:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Submit with:  sbatch submit_dataset_generation_3dim_pilot_l40s.sh
# (run from applications/Agility_Forge/, or adjust the `cd` below)
#
# L40S variant of submit_dataset_generation_3dim_pilot.sh, submitted in its
# place (that job, 3791298, was cancelled while still queued on the H100
# partition -- resources on gpu2h100 nodes were harder to come by than on
# gpul40s at submission time). L40S is confirmed compatible with this exact
# generate_dataset.py workload: submit_dataset_generation_l40s_a.sh/_b.sh
# already exist in this repo and were run successfully against the same
# pipeline. Note: L40S is generally less powerful than H100 for this kind of
# dense compute, so per-hit wall time here may end up slower than the
# ~600-1000s/hit observed on H100 for the old fixed-u_j=2.5mm dataset --
# traded deliberately for shorter queue wait, not because it's expected to
# be faster once running.
#
# Same pilot for the 3D-actuation-space expansion as the H100 version: adds
# u_j (strike depth) as a third independently-randomized per-hit parameter,
# Uniform(0.5, 2.0)mm -- range grounded in the JAX-FORGE paper's reported
# forging depths (0.62mm, 1.63mm) for this same billet geometry (diameter
# 15.9mm round stock). Writes to the default data/dataset path (fresh; the
# old 2D-actuation dataset is preserved at data/dataset_2dim).
#
# --n-rollouts 50: a deliberate pilot, not the final dataset size -- confirms
# the new (wider) strike-depth range doesn't introduce solver convergence
# problems before committing to a larger, more expensive generation run.
#
# --time=96:00:00: generous given L40S's unknown-relative-to-H100 throughput
# on this workload and the untested u_j range's unknown effect on Newton
# retries; partition allows up to 13 days, so ample headroom either way.

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
