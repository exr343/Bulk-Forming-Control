#!/bin/bash -l
#SBATCH --job-name=gnn_coilT_stage3
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --array=1-10
#
# Coil GNN with temperature in the nodal state (GNN/coil_T.py), stage 3 (new 5-pass square runs, cycle chains),
# one task per M. Submit with --dependency=aftercorr:<stage12 job>.
set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
echo "Job started: $(date) on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
python -u -m applications.Agility_Forge.GNN.coil_T --stage 3 --mp "${SLURM_ARRAY_TASK_ID}" "$@"
echo "Job finished: $(date)"
