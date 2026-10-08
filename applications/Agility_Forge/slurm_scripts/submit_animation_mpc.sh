#!/bin/bash -l
#SBATCH --job-name=animate_mpc
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Progress GIFs of an MPC run (control/mpc_coil.py output), written into the run's own folder:
# <name>_progress.gif (simulation) and <name>_progress_with_gnn.gif (simulation + the GNN's forecast under it),
# each with a last frame and sample frames. Make them after every MPC run. ONLY_GNN=1 skips the first.
# Submit from the repo root:  RUN=applications/Agility_Forge/control/results/<run> \
#   sbatch --export=ALL applications/Agility_Forge/slurm_scripts/submit_animation_mpc.sh
set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
if [ "${ONLY_GNN:-0}" != "1" ]; then
    python -u applications/Agility_Forge/output/die12_coil_animation/animate.py "${RUN}" "$(basename "${RUN}")" "${RUN}"
fi
python -u applications/Agility_Forge/output/die12_coil_animation/animate_mpc.py "${RUN}" "$(basename "${RUN}")" "${RUN}"
