#!/bin/bash -l
#SBATCH --job-name=mpc_coil_ens
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --array=0-3
#
# lambda sweep of the coil MPC (control/mpc_coil.py) with the 5-member deep ensemble of the temperature-state coil
# GNN (M = 5, plain square held out): cost = cross-section error of the members' mean forecast + lambda * the
# members' y,z variance (both mm^2). One task per lambda in {0, 0.1, 1, 10}; lambda = 0 is the ensemble-mean-only
# baseline. Otherwise the multistart run's settings (N = 100, K = 5, B = 50, seed 0, SLSQP maxiter 100, 20 cycles).
# Train the members first (submit_gnn_coilT_ensemble.sh) and check GNN/ensemble_calibration.py. Resubmit to resume.
# Submit from the repo root (SUFFIX is appended to the out dir; extra args go to mpc_coil.py), e.g.:
#   sbatch applications/Agility_Forge/slurm_scripts/submit_mpc_coil_ensemble.sh
#   SUFFIX=_single_zone11 sbatch --export=ALL --array=3 applications/Agility_Forge/slurm_scripts/submit_mpc_coil_ensemble.sh \
#       --n-samples 1 --coil-zone-mm 11          (lambda 10, run 1's planner: single start + coil zone; --array=4 for lambda 100)
set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
# JAX (the simulator) otherwise takes ~75% of the GPU at start-up; the SLSQP gradient through 5 members needs >10 GB
# (job 3994571 ran out of memory in its first plan).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "Job started: $(date) on $(hostname)"
LAMBDAS=(0 0.1 1 10 100)            # default array 0-3 = the planned sweep; index 4 = 100
LAM=${LAMBDAS[${SLURM_ARRAY_TASK_ID}]}
G=applications/Agility_Forge/GNN
CKPTS="${G}/coil_T_sweep_test_square/mp_5/stage3.pt"
for k in 1 2 3 4; do CKPTS="${CKPTS} ${G}/coil_T_ensemble_test_square/seed_${k}/mp_5/stage3.pt"; done
python -u -m applications.Agility_Forge.control.mpc_coil --checkpoints ${CKPTS} --var-weight "${LAM}" \
    --out-dir "applications/Agility_Forge/control/results/coil_mpc_tstate_M5_ens_lam${LAM}${SUFFIX:-}" "$@"
echo "Job finished: $(date)"
