#!/bin/bash -l
#SBATCH --job-name=agility_forge_prism_target_chd
#SBATCH --partition=gpu
#SBATCH --constraint=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Submit with:  sbatch slurm_scripts/submit_prism_target_chamfer_hausdorff.sh
# (must be submitted with CWD at repo root)
#
# Same rounded-square-prism target/pilot as submit_prism_target.sh (apothem
# 6.4375mm, n_hits=8, u_j_mm_min=0.0), rerun with checkpoint_mp_5_chamfer_
# hausdorff.pt instead of the original MSE-only checkpoint_mp_5.pt -- direct
# comparison of whether training with Chamfer/Hausdorff loss terms changes
# behavior on the prism target specifically, given the MSE-only model mostly
# gave up (near-zero strokes, ended up worse than the undeformed billet).
# Separate output dir, does not touch the original run or the concurrently-
# running full-test-set sweep (job 3824244).

set -euo pipefail

cd /home/exr343/CIRP_2027

module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "GPU assigned:"
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "(nvidia-smi not available)"

python -m applications.Agility_Forge.control.eval_prism_target \
    --n-hits 8 \
    --apothem-mm 6.4375 \
    --u-j-mm-min 0.0 \
    --checkpoint-path applications/Agility_Forge/GNN/mp_sweep/checkpoint_mp_5_chamfer_hausdorff.pt \
    --out-dir applications/Agility_Forge/control/mpc_prism_target_chamfer_hausdorff

echo "Job finished: $(date)"
