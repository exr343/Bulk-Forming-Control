#!/bin/bash -l
#SBATCH --job-name=agility_forge_mpc_open_loop
#SBATCH --partition=gpu
#SBATCH --constraint=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Submit with:  sbatch slurm_scripts/submit_mpc_open_loop_validation.sh
# (must be submitted with CWD at repo root)
#
# First-ever real execution of control/plant_interface.ForgingPlant (was a
# 100% stub until this session). Replays the already-computed 5-step MPC
# plan open-loop (no re-planning) through the real JAX-FORGE simulator --
# cheaper than mpc.MPCController.run()'s full receding-horizon loop, while
# still exercising the plant across all 5 real hits (including the thermal
# relaxation ramp). ~5-15 min/hit observed during dataset generation, so
# budget up to ~1.5h; --time=4h leaves headroom for a first run of untested
# code.

set -euo pipefail

cd /home/exr343/CIRP_2027

module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

echo "Job started: $(date)"
echo "Host: $(hostname)"

python -m applications.Agility_Forge.control.validate_open_loop

echo "Job finished: $(date)"
