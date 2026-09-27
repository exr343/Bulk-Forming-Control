#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=gput063
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=exr343@case.edu
#SBATCH --job-name=mp_sweep_finished_summary
#SBATCH --time=02:00:00
#
# Summary of the second message-passing-steps sweep (M = 1-10, finetuned on the
# finished square runs): gradient time of a 10-hit horizon on one GPU, and the
# seed-3 test rollout errors, into GNN/mp_sweep_finished/.
# Submit from the repo root after the finetune array:
#   sbatch --dependency=afterok:<finetune job> applications/Agility_Forge/slurm_scripts/submit_mp_sweep_finished_summary.sh
set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export JAX_PLATFORMS=cpu  # keep JAX off the GPU the timing uses
echo "Job started: $(date) on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
python -u -m applications.Agility_Forge.GNN.mp_sweep.summarize --depths 1,2,3,4,5,6,7,8,9,10 \
    --finetune-root applications/Agility_Forge/GNN/mp_sweep_finished/finetune \
    --out-dir applications/Agility_Forge/GNN/mp_sweep_finished
echo "Job finished: $(date)"
