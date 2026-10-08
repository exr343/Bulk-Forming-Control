#!/bin/bash -l
#SBATCH --job-name=gnn_figs_heldout_square
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Figures for the sweep with the plain square held out (GNN/coil_sweep_test_square): loss curves, then the
# M = 5 rollouts on the plain square (test) and seed 3 (now training). The two GIFs are separate jobs
# (submit_gnn_rollout_gif.sh) that depend on this one. Output: output/gnn_figures/heldout_square/
set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
F=applications/Agility_Forge/output/gnn_figures
SW=applications/Agility_Forge/GNN/coil_sweep_test_square
python -u $F/plot_losses.py --sweep $SW --out-dir $F/heldout_square --test-run "plain square"
for RUN in square 3; do
    python -u $F/rollout.py --mp 5 --run $RUN --sweep $SW --test-run square --out-dir $F/heldout_square
done
