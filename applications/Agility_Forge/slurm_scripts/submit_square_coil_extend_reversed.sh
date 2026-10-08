#!/bin/bash -l
#
# HISTORICAL (old overwrite reheat rule): these runs were cancelled on 2026-10-04 and moved to
# data/backup/die12_coil_overwrite_reheat/; the paths below are the original ones.
#SBATCH --job-name=agility_forge_square_coil_4pass_rev
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --array=1,2,4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Rerun of extension runs whose first pass-3 reheat failed (the coil jumped from the free end back to
# the clamp end and cooled the far end ~300 C): pass 3 now runs from the free end back toward the clamp.
# Resumes each run in place from its last checkpoint. Set the array to the failed task ids.
# Original extension notes:
# Continue the 10 finished 2-pass square runs (data/dataset_die12_coil) with two more passes at the
# final half-gap (5.3 mm, +-0.2 mm jitter for seeds 1-9), into copies in data/dataset_die12_coil_4pass.
# The originals are not touched. Task 0 = plain run, tasks 1-9 = jitter seeds.
# Submit from the repo root:  sbatch applications/Agility_Forge/slurm_scripts/submit_square_coil_extend.sh
# Resubmitting resumes from the last completed hit or reheat.

set -euo pipefail

cd /home/exr343/CIRP_2027

module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

echo "Job started: $(date)"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "(nvidia-smi not available)"

SRC=applications/Agility_Forge/data/dataset_die12_coil
OUT=applications/Agility_Forge/data/dataset_die12_coil_4pass
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    NAME=square; JITTER=()
else
    NAME=square_jitter_seed${SLURM_ARRAY_TASK_ID}; JITTER=(--jitter-seed "${SLURM_ARRAY_TASK_ID}")
fi
python -u -m applications.Agility_Forge.generate_square_coil_rollout "${JITTER[@]}" --extra-passes 2 --reverse-passes 3 \
    --continue-from "${SRC}/${NAME}" --dataset-dir "${OUT}/${NAME}"

echo "Job finished: $(date)"
