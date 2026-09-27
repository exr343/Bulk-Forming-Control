#!/bin/bash -l
#SBATCH --job-name=mpc_square_cost
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Real-simulator check of a cost design from control/cost_design.py: 50-hit
# closed-loop GNN-MPC toward the square run's final geometry with the M = 3
# finetuned GNN and the given penalty weights. Resubmitting resumes.
# Submit from the repo root:
#   sbatch applications/Agility_Forge/slurm_scripts/submit_mpc_square_cost_design.sh <name> '<penalty json>'
# <name> is the output folder under control/results/real_simulator/, e.g. e5_w30_effort0.3.
set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
echo "Job started: $(date) on $(hostname); config $1 penalty $2"
python -u -m applications.Agility_Forge.control.eval_square_target \
    --checkpoint-path applications/Agility_Forge/GNN/mp_sweep/finetune/mp_3/checkpoint.pt \
    --penalty "$2" \
    --out-dir "applications/Agility_Forge/control/results/real_simulator/$1"
echo "Job finished: $(date)"
