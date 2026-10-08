#!/bin/bash -l
#SBATCH --job-name=agility_forge_square_coil_direct
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --array=0-5,7,8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Resume the 8 runs of submit_square_coil_rollouts.sh (job 3955595) that stopped on a failed reheat at the
# start of pass 4, with the direct linear solver (--linear-solver scipy). BiCGSTAB stops converging there;
# a direct solve passes (output/reheat_failure_diag/). Each run resumes from its last saved hit or reheat.
# Submit from the repo root:  sbatch applications/Agility_Forge/slurm_scripts/submit_square_coil_resume_direct.sh

set -euo pipefail

cd /home/exr343/CIRP_2027

module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

echo "Job started: $(date)"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "(nvidia-smi not available)"

OUT=applications/Agility_Forge/data/dataset_die12_coil
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    python -u -m applications.Agility_Forge.generate_square_coil_rollout --extra-passes 2 --linear-solver scipy --dataset-dir "${OUT}/square"
else
    python -u -m applications.Agility_Forge.generate_square_coil_rollout --jitter-seed "${SLURM_ARRAY_TASK_ID}" --extra-passes 2 --linear-solver scipy \
        --dataset-dir "${OUT}/square_jitter_seed${SLURM_ARRAY_TASK_ID}"
fi

echo "Job finished: $(date)"
