#!/bin/bash -l
#SBATCH --job-name=agility_forge_dataset_l40s_a
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Submit with:  sbatch submit_dataset_generation_l40s_a.sh
#
# Writes to its OWN fresh dataset dir (data/dataset_l40s_a), not the live
# data/dataset_pretraining/ -- generate_dataset.py's ManifestWriter isn't safe for
# concurrent writers (reads the whole manifest.json, overwrites it whole on
# every save), and job 3747184 (appending to data/dataset_pretraining/) is still
# running as of this job's submission. Running this alongside
# submit_dataset_generation_l40s_b.sh (also its own separate dir) on two
# L40S GPUs avoids any shared-file race between all three jobs.
#
# rollout_l40s_a/ starts numbering at 1 (fresh dir) -- merging into the main
# data/dataset_pretraining/ corpus is a deliberate later step, not automatic.

set -euo pipefail

cd /home/exr343/CIRP_2027

module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "GPU assigned:"
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "(nvidia-smi not available)"

python -m applications.Agility_Forge.generate_dataset \
    --n-rollouts 25 \
    --n-hits 5 \
    --dataset-dir applications/Agility_Forge/data/dataset_l40s_a

echo "Job finished: $(date)"
