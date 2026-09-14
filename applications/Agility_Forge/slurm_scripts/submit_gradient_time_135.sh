#!/bin/bash -l
#SBATCH --job-name=agility_forge_gradient_time_135
#SBATCH --partition=gpu
#SBATCH --constraint=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Clean, uncontended measurement of the 135-step model's gradient-computation
# time (horizon=5) -- the same benchmark timed out (>300s, didn't even finish
# one call) on the shared login node (26 other users, load ~8/32 cores, swap
# 91% full). Random init, not the trained checkpoint -- timing depends only
# on architecture (message_passing_steps), not learned weights.

set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

python3 -c "
import time
import torch
from applications.Agility_Forge.GNN.data import build_surface_mesh_info
from applications.Agility_Forge.GNN.model import ForgeGNN
from applications.Agility_Forge.control.mpc import _rollout_cost

mesh_info = build_surface_mesh_info('applications/Agility_Forge/data/dataset/rollout_01/undeformed.vtu')
model = ForgeGNN(mesh_info, latent_size=128, num_layers=2, message_passing_steps=135)
model.eval()
H_mm = float(mesh_info.rest_pos[:, 0].max().item())

x0 = torch.zeros(mesh_info.rest_pos.shape[0], 3)
target = torch.zeros(mesh_info.rest_pos.shape[0], 3)
horizon = 5
u0 = torch.tile(torch.tensor([0.4, 90.0, 0.5]), (horizon,))

def one_call():
    u_t = u0.clone().requires_grad_(True)
    cost = _rollout_cost(u_t, model, mesh_info, x0, H_mm, 0.2, target, horizon)
    cost.backward()
    return u_t.grad

t_warm = time.time()
one_call()
print(f'warm-up call: {time.time()-t_warm:.3f}s', flush=True)

t0 = time.time()
for _ in range(5):
    one_call()
elapsed = (time.time() - t0) / 5
print(f'135-step model, horizon=5: {elapsed:.3f}s/gradient-call (avg of 5)')
"
