#!/bin/bash -l
#SBATCH --job-name=mpc_e9_gap
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --exclude=gput063
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# 50-hit closed-loop GNN-MPC toward the idealized square target
# (control/targets/ideal_square_10.6: round to the 800 C point, 7.5 mm taper,
# 10.6 x 10.6 mm square, 158.2 mm total). Experiment 9: same cost as
# experiment 8 (25% cross-section share, w = 91.37, no stroke effort), but the
# M = 3 GNN retrained with the absolute half-gap control and 10% no-change hits
# (GNN/gap_control/gap_nochange10). Starts after that training finishes. On a simulator failure the run
# resumes from its last completed hit, up to 2 times (3 attempts in total).
# Submit from the repo root:
#   sbatch --dependency=afterok:<training job>_1 applications/Agility_Forge/slurm_scripts/submit_mpc_e9_gap.sh
set -uo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
echo "Job started: $(date) on $(hostname)"
for attempt in 1 2 3; do
    echo "=== attempt $attempt: $(date)"
    if python -u -m applications.Agility_Forge.control.eval_square_target \
        --checkpoint-path applications/Agility_Forge/GNN/gap_control/gap_nochange10/finetune/checkpoint.pt \
        --target-vtu applications/Agility_Forge/control/targets/ideal_square_10.6/target_on_billet.vtu \
        --penalty '{"transverse": 91.37195, "effort_s": 0.0}' \
        --n-hits 50 \
        --out-dir applications/Agility_Forge/control/results/real_simulator/e9_ideal_share25_gap; then
        echo "Finished on attempt $attempt"; break
    fi
    echo "Attempt $attempt failed; resuming from the last completed hit"
done
echo "Job finished: $(date)"
