#!/bin/bash -l
#SBATCH --job-name=agility_forge_square_coil_pass5
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --array=0-9
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Pass 5 for the 10 finished 4-pass square runs (data/dataset_die12_coil/<run>): each is copied to
# data/dataset_die12_coil_5pass/<run> and continued from its final state. Same stations (fresh per-pass
# jitter as before), direct linear solver. Pass-5 half-gap differs per run, evenly 5.0-5.3 mm
# (task 0 = plain square = 5.0, task 9 = seed 9 = 5.3), so the data spans just above to slightly past the
# 10.6 mm target. Submit from the repo root:
#   sbatch applications/Agility_Forge/slurm_scripts/submit_square_coil_pass5.sh
# Resubmitting resumes from the last completed hit or reheat.

set -euo pipefail

cd /home/exr343/CIRP_2027

module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

echo "Job started: $(date)"
echo "Host: $(hostname)"

T=${SLURM_ARRAY_TASK_ID}
GAP=$(awk -v t="$T" 'BEGIN { printf "%.4f", 5.0 + 0.3 * t / 9 }')
if [ "$T" -eq 0 ]; then RUN=square; JIT=""; else RUN=square_jitter_seed$T; JIT="--jitter-seed $T"; fi
echo "Run ${RUN}: pass-5 half-gap ${GAP} mm"
python -u -m applications.Agility_Forge.generate_square_coil_rollout $JIT --extra-passes 3 --last-pass-half-gap "$GAP" \
    --linear-solver scipy --continue-from "applications/Agility_Forge/data/dataset_die12_coil/${RUN}" \
    --dataset-dir "applications/Agility_Forge/data/dataset_die12_coil_5pass/${RUN}"

echo "Job finished: $(date)"
