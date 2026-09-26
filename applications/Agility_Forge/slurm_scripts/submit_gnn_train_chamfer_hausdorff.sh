#!/bin/bash -l
#SBATCH --job-name=agility_forge_gnn_chamfer_hausdorff
#SBATCH --partition=gpu
#SBATCH --constraint=gpu2h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Submit with:  sbatch slurm_scripts/submit_gnn_train_chamfer_hausdorff.sh
# (must be submitted with CWD at repo root)
#
# New M=5 model trained with Chamfer + Hausdorff terms added to the loss
# (GNN/model.py's ForgeGNN.loss_and_predict, opt-in via --chamfer-weight/
# --hausdorff-weight). message_passing_steps=5, everything else at train.py's
# defaults (latent=128, num_layers=2, batch=4, epochs=100, patience=20).
#
# No --rollout-allowlist: trains on the FULL current dataset (401 rollouts),
# not the 383-rollout pinned snapshot submit_gnn_train_mp_sweep.sh used for
# checkpoint_mp_5.pt -- more data, but the train/test split differs from
# checkpoint_mp_5.pt's, so this run's numbers aren't an exact apples-to-apples
# comparison against it (dataset AND loss both differ now, not just the loss).
#
# --chamfer-weight/--hausdorff-weight 1.0 each (up from an initial 0.01
# guess, per explicit instruction) -- normalized per-node MSE loss runs
# ~0.4-2.2 over training per prior runs' curves, physical Chamfer/Hausdorff
# are O(mm)/O(mm^2), so at weight=1 these terms will likely dominate the
# loss rather than gently nudge it. That's the intent here, not a mistake --
# but expect very different training dynamics than the MSE-only run.
#
# Real added cost vs. the original M=5 run: an (B,N,N) cdist matrix (N~4,355)
# computed WITH gradients every training step, not just occasionally for
# reporting -- batch=4 (train.py's default, same as the original M=5 run,
# NOT the 16 used for the full-reference-sizing run) chosen partly to keep
# this within memory headroom on a first attempt.

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

python -m applications.Agility_Forge.GNN.train \
    --message-passing-steps 5 \
    --chamfer-weight 1.0 \
    --hausdorff-weight 1.0 \
    --out-dir applications/Agility_Forge/GNN/runs_mp5_chamfer_hausdorff \
    --checkpoint-path applications/Agility_Forge/GNN/mp_sweep/checkpoint_mp_5_chamfer_hausdorff.pt

echo "Job finished: $(date)"
