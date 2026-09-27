#!/bin/bash -l
#SBATCH --job-name=mpc_e13_stroke05
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
# 10.6 x 10.6 mm square, 158.2 mm total). Experiment 13: experiment 8's cost
# (M = 3 stroke GNN, 25% cross-section share, w = 91.37, no stroke effort) with
# the stroke limited to 0.5-2 mm (best surrogate design, experiments 11-12). On a simulator failure the run
# resumes from its last completed hit, up to 2 times (3 attempts in total).
# Submit from the repo root:
#   sbatch applications/Agility_Forge/slurm_scripts/submit_mpc_e13_stroke05.sh
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
        --checkpoint-path applications/Agility_Forge/GNN/mp_sweep/finetune/mp_3/checkpoint.pt \
        --target-vtu applications/Agility_Forge/control/targets/ideal_square_10.6/target_on_billet.vtu \
        --penalty '{"transverse": 91.37195, "effort_s": 0.0}' \
        --u-j-mm-min 0.5 \
        --n-hits 50 \
        --out-dir applications/Agility_Forge/control/results/real_simulator/e13_ideal_share25_stroke0.5; then
        echo "Finished on attempt $attempt"; break
    fi
    echo "Attempt $attempt failed; resuming from the last completed hit"
done
echo "Job finished: $(date)"
