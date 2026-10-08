#!/bin/bash -l
#SBATCH --job-name=gnn_coilT_ensemble
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --array=1-4
#
# Deep-ensemble members for the coil MPC: the temperature-state coil GNN (GNN/coil_T.py), M = 5 message-passing
# steps, plain square held out, trained through all three stages from scratch (stage 1 NOT reused, so the members
# stay diverse), one task per seed = array id. Member 0 is the existing GNN/coil_T_sweep_test_square/mp_5/stage3.pt
# (seed 0). Each member lands in GNN/coil_T_ensemble_test_square/seed_<k>/mp_5/stage3.pt. Submit from the repo root:
#   sbatch applications/Agility_Forge/slurm_scripts/submit_gnn_coilT_ensemble.sh
set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
echo "Job started: $(date) on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
SEED="${SLURM_ARRAY_TASK_ID}"
OUT=applications/Agility_Forge/GNN/coil_T_ensemble_test_square/seed_${SEED}
python -u -m applications.Agility_Forge.GNN.coil_T --stage 12 --mp 5 --test-run square --seed "${SEED}" --out "${OUT}" "$@"
python -u -m applications.Agility_Forge.GNN.coil_T --stage 3 --mp 5 --test-run square --seed "${SEED}" --out "${OUT}" "$@"
echo "Job finished: $(date)"
