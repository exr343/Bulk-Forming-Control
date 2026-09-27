#!/bin/bash -l
#SBATCH --job-name=agility_forge_square_finish
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=exr343@case.edu
#SBATCH --array=0-9
#SBATCH --exclude=gput063
#
# Finishing phase for the 10 square runs (generate_square_rollout.py --finish-from):
# each completed 48-hit run is copied to data/dataset_finetuning/finished/<name> and
# continued until every station is within 10.6 + 0.2 mm across at 0 and 90 deg, or
# 52 extra hits. Task 0 = the unjittered square run; task N (1-9) = square_jitter_seedN,
# rebuilt with the same --jitter-seed so its schedule (and stations) match.
# The originals are left untouched (training data, open-loop reference, reports).
# Submit from the repo root:
#   sbatch applications/Agility_Forge/slurm_scripts/submit_square_finish.sh
# Resubmitting resumes each run from its last completed hit.

set -euo pipefail

cd /home/exr343/CIRP_2027

module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

echo "Job started: $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "(nvidia-smi not available)"

FT=applications/Agility_Forge/data/dataset_finetuning
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    NAME=square
    JITTER=()
else
    NAME="square_jitter_seed${SLURM_ARRAY_TASK_ID}"
    JITTER=(--jitter-seed "${SLURM_ARRAY_TASK_ID}")
fi
echo "Finishing $NAME"

python -u -m applications.Agility_Forge.generate_square_rollout ${JITTER[@]+"${JITTER[@]}"} \
    --finish-from "$FT/$NAME" \
    --finish-tol-mm 0.2 --finish-max-hits 52 \
    --dataset-dir "$FT/finished/$NAME"

echo "Job finished: $(date)"
