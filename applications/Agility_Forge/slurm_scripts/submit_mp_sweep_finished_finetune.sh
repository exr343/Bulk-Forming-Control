#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=gput063
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=exr343@case.edu
#SBATCH --job-name=mp_sweep_finished_ft
#SBATCH --time=12:00:00
#
# Message-passing-steps sweep, M = 1-10 (array task id = M), second round:
# finetune each pretrained M checkpoint (GNN/mp_sweep/checkpoint_mp_<M>.pt, same
# pretraining recipe as the first sweep) on the FINISHED square runs
# (data/dataset_finetuning/finished/, i.e. carried on until every station is within
# 10.6 + 0.2 mm). Same split as before: train = square + jitter seeds 1,2,4-8;
# early stopping on seed 9; test = seed 3. Everything else = finetune.py defaults
# (stroke input, replay ratio 1.0, lr 1e-5, frozen normalizers, patience 20).
# Submit from the repo root, after the finishing runs and the missing pretraining:
#   sbatch --array=1-10 --dependency=afterok:<pretrain job>,afterany:<finishing jobs> \
#       applications/Agility_Forge/slurm_scripts/submit_mp_sweep_finished_finetune.sh
set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
echo "Job started: $(date) on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader || true

M="${SLURM_ARRAY_TASK_ID}"
FIN=applications/Agility_Forge/data/dataset_finetuning/finished
OUT=applications/Agility_Forge/GNN/mp_sweep_finished/finetune/mp_$M
TRAIN="$FIN/square"
for s in 1 2 4 5 6 7 8; do TRAIN="$TRAIN,$FIN/square_jitter_seed$s"; done

python -u -m applications.Agility_Forge.GNN.finetune \
    --init-checkpoint "applications/Agility_Forge/GNN/mp_sweep/checkpoint_mp_$M.pt" \
    --train-dirs "$TRAIN" \
    --val-dir "$FIN/square_jitter_seed9" \
    --test-dir "$FIN/square_jitter_seed3" \
    --out-dir "$OUT" \
    --checkpoint-path "$OUT/checkpoint.pt"
echo "Job finished: $(date)"
