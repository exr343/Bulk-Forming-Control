#!/bin/bash -l
#SBATCH --job-name=mpc_coil
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# MPC on the new simulator (control/mpc_coil.py) with the temperature-state coil GNN (GNN/coil_T.py): each cycle
# scans shape + surface temperature, plans 6 hits through the GNN, reheats (coil at the average planned die
# centre), applies all 6 to the simulator; stops when every width from the first station is <= 10.8 mm or after
# 20 cycles. Resubmit with the same OUT to resume. Submit from the repo root, e.g.:
#   CKPT=applications/Agility_Forge/GNN/coil_T_sweep_test_square/mp_5/stage3.pt \
#   OUT=applications/Agility_Forge/control/results/coil_mpc_tstate_M5 \
#     sbatch --export=ALL applications/Agility_Forge/slurm_scripts/submit_mpc_coil.sh
set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
echo "Job started: $(date) on $(hostname)"
python -u -m applications.Agility_Forge.control.mpc_coil --checkpoint "${CKPT}" --out-dir "${OUT}"
echo "Job finished: $(date)"
