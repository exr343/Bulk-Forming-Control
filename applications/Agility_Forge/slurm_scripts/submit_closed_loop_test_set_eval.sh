#!/bin/bash -l
#SBATCH --job-name=agility_forge_closed_loop_test_set_eval
#SBATCH --partition=gpu
#SBATCH --constraint=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#
# Submit with:  sbatch slurm_scripts/submit_closed_loop_test_set_eval.sh
# (must be submitted with CWD at repo root)
#
# Single serial job, one L40S, covering all 77 test rollouts (the pinned
# 383-rollout snapshot's test split) in one closed-loop GNN-MPC sweep -- see
# control/eval_closed_loop_test_set.py's docstring for the full design.
# Uses checkpoint_mp_5.pt (message_passing_steps=5), not the control loop's
# prior M=15 default, per the depth sweep's finding that M=5 matches or
# beats M=15 on accuracy at ~1/3 the planning cost.
#
# --constraint=gpu4090 (not gpu2h100 or gpul40s): switched twice now --
# first off H100 (job 3822009, cancelled) which sat PENDING with 0 free
# H100s cluster-wide, then off L40S (job 3822098, cancelled) which had 1
# free GPU but sat behind several other users' higher-priority pending jobs
# in the fair-share queue with no realistic start for ~10h. At submission
# time, gpu4090 had 4 free GPUs (2 each on gput074/gput075) and nothing else
# queued against that constraint, so this should dispatch immediately.
#
# Budget: no measured 4090 number exists for this workload (the 4090 pilot
# dataset-generation script's own comment calls 4090 throughput here
# "untested" -- see submit_dataset_generation_3dim_pilot_4090.sh). Carrying
# forward the same ~56.5h (~2.4 days) estimate used for L40S (L40S's
# measured ~34.2min/rollout FEM pace, job 3793718, plus ~10min/rollout for
# M=5 SQP replanning) as a rough guess, not a confirmed number -- the two
# chips are close enough on paper (Ada Lovelace generation, similar FP32
# throughput, both equally poor at the FP64 jax_forge's Newton solve
# actually needs) that this is a reasonable placeholder pending an actual
# measurement, which slurm_logs/%x_%j.out's per-rollout elapsed-time prints
# will provide once this run is underway. --time=96:00:00 (4 days) keeps
# ~1.7x headroom over that estimate. Any rollout that fails to converge is
# dropped (not retried) per eval_closed_loop_test_set.py's fail-fast policy,
# so a bad rollout costs at most the time already sunk into it, not the
# whole job.

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

python -m applications.Agility_Forge.control.eval_closed_loop_test_set \
    --out-dir applications/Agility_Forge/control/eval_closed_loop_test_set

echo "Job finished: $(date)"
