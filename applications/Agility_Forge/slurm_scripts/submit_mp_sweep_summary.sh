#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=exr343@case.edu
#SBATCH --job-name=mp_sweep_summary
#SBATCH --time=02:00:00
#
# New message-passing sweep, stage 3: 10-hit gradient timing for every depth on
# one GPU, plus the test-run Hausdorff/Chamfer summary and plots.
# Submit from the repo root:  sbatch applications/Agility_Forge/slurm_scripts/submit_mp_sweep_summary.sh

set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
echo "Job started: $(date) on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader || true

SWEEP=applications/Agility_Forge/GNN/mp_sweep
# summarize.py imports control/mpc.py, which pulls in the JAX simulator stack;
# JAX would otherwise reserve most of the GPU at import (first run OOM'd at M=15).
export JAX_PLATFORMS=cpu
python -u -m applications.Agility_Forge.GNN.mp_sweep.summarize
echo "Job finished: $(date)"
