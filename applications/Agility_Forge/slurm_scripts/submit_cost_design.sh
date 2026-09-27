#!/bin/bash -l
#SBATCH --job-name=cost_design
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#
# Surrogate closed-loop cost-design screen (control/cost_design.py): one array
# task per config in the JSON file given as the first argument.
# Submit from the repo root:
#   sbatch --array=0-6 applications/Agility_Forge/slurm_scripts/submit_cost_design.sh \
#       applications/Agility_Forge/control/cost_design_configs/exp1_control_change.json
set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
# cost_design imports eval_square_target -> the JAX simulator stack; keep JAX
# off the GPU so it doesn't reserve the memory PyTorch needs.
export JAX_PLATFORMS=cpu
python -u -m applications.Agility_Forge.control.cost_design --configs "$1" --index "${SLURM_ARRAY_TASK_ID}"
