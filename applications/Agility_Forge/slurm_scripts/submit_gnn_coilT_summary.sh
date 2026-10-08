#!/bin/bash -l
#SBATCH --job-name=gnn_coilT_summary
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Collects <out>/mp_*/stage3_eval.json of the coil_T sweep into <out>/summary.json.
# Submit with --dependency=afterany:<stage3 job>.
set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
echo "Job started: $(date) on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
python -u -m applications.Agility_Forge.GNN.coil_T --summarize "$@"
