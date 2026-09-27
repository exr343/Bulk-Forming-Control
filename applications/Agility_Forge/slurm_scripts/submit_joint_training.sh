#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=exr343@case.edu
#SBATCH --job-name=gnn_joint_training
#SBATCH --time=12:00:00
#
# Joint-training baseline for the pretrain-then-finetune comparison: train from
# random weights on ALL uniform-train hits + the same 8 square episodes every
# epoch (lr 1e-4, normalizers learn online), early stopping on seed 9, test on
# seed 3 -- same splits as the mp_sweep finetunes. Array task id = depth.
# Submit from the repo root:  sbatch --array=3,5 applications/Agility_Forge/slurm_scripts/submit_joint_training.sh

set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
echo "Job started: $(date) on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
M="${SLURM_ARRAY_TASK_ID}"
OUT=applications/Agility_Forge/GNN/joint_training
python -u -m applications.Agility_Forge.GNN.finetune \
    --from-scratch-mp "$M" --lr 1e-4 --replay-ratio -1 \
    --val-dir applications/Agility_Forge/data/dataset_finetuning/square_jitter_seed9 \
    --out-dir "$OUT/mp_$M" \
    --checkpoint-path "$OUT/mp_$M/checkpoint.pt"
echo "Job finished: $(date)"
