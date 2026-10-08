#!/bin/bash -l
#SBATCH --job-name=gnn_coilT_finetune_mpc
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=24:00:00
#
# Finetune the M = 5 temperature-state coil GNN on the MPC runs' own hits (GNN/finetune_mpc.py).
# Submit after the MPC run ends: sbatch --dependency=afterany:<mpc job> slurm_scripts/submit_gnn_coilT_finetune_mpc.sh
set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
echo "Job started: $(date) on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
python -u -m applications.Agility_Forge.GNN.finetune_mpc "$@"
echo "Job finished: $(date)"
