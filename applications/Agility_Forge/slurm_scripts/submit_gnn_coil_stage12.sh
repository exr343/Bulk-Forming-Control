#!/bin/bash -l
#SBATCH --job-name=gnn_coil_stage12
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --array=1-10%2
#
# GNN for the new simulator, stages 1-2 (old random hits, then old square runs), one task per
# message-passing steps M = array id; at most 2 at a time so the forging runs keep 4 GPUs.
# See GNN/coil.py. Submit from the repo root.
set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
echo "Job started: $(date) on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
python -u -m applications.Agility_Forge.GNN.coil --stage 12 --mp "${SLURM_ARRAY_TASK_ID}" "$@"   # extra args, e.g. --test-run square --out ...
echo "Job finished: $(date)"
