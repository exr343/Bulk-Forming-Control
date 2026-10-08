#!/bin/bash -l
#SBATCH --job-name=ensemble_calibration
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Does the deep ensemble's spread track its real error? (GNN/ensemble_calibration.py: per hit, the 5 members'
# y,z variance vs the error of their mean, on the held-out plain square and the MPC runs' own hits.) Run after the
# members are trained, e.g.:
#   sbatch --dependency=afterok:<submit_gnn_coilT_ensemble job> applications/Agility_Forge/slurm_scripts/submit_ensemble_calibration.sh
set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
echo "Job started: $(date) on $(hostname)"
python -u -m applications.Agility_Forge.GNN.ensemble_calibration "$@"
echo "Job finished: $(date)"
