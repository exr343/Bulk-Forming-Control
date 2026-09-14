"""Times a full MPCController.plan() SQP solve for a given mp_sweep checkpoint
-- the real, non-extrapolated measurement (not the isolated single-gradient-
call benchmark in benchmark_gradient_time.py). Same target/setup as the
original 15-step plan() runs (rollout_339's true hit-5 geometry, undeformed
x0, horizon=5) for a direct, apples-to-apples comparison.

Usage:
    python -m applications.Agility_Forge.GNN.mp_sweep.full_solve_timer --message-passing-steps 5
"""

import argparse
import time

import meshio
import torch

from applications.Agility_Forge.GNN.data import build_surface_mesh_info
from applications.Agility_Forge.GNN.model import ForgeGNN
from applications.Agility_Forge.control.mpc import MPCController


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--message-passing-steps", type=int, required=True)
    p.add_argument("--checkpoint-path", default=None,
                   help="Defaults to GNN/mp_sweep/checkpoint_mp_<N>.pt")
    args = p.parse_args()
    ckpt_path = args.checkpoint_path or (
        f"applications/Agility_Forge/GNN/mp_sweep/checkpoint_mp_{args.message_passing_steps}.pt")

    mesh_info = build_surface_mesh_info(
        "applications/Agility_Forge/data/dataset/rollout_01/undeformed.vtu")
    model = ForgeGNN(mesh_info, latent_size=128, num_layers=2,
                      message_passing_steps=args.message_passing_steps)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    true_m = meshio.read(
        "applications/Agility_Forge/GNN/runs_3dim/eval/rollout_339/hit_05_true.vtu")
    target = torch.tensor(true_m.point_data["Displacement"], dtype=torch.float32)

    controller = MPCController(model, mesh_info, horizon=5, target_state=target,
                                band_width_frac=0.2, device="cpu")
    x0 = torch.zeros(mesh_info.rest_pos.shape[0], 3)

    print(f"Starting plan() for message_passing_steps={args.message_passing_steps}...", flush=True)
    t0 = time.time()
    u_seq, result = controller.plan(x0)
    elapsed = time.time() - t0

    print(f"\n=== M={args.message_passing_steps} full SQP-BFGS solve ===")
    print(f"elapsed: {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print(f"success: {result.success}  message: {result.message}")
    print(f"nit (reported iterations): {result.nit}")
    print(f"nfev (function evals): {getattr(result, 'nfev', 'n/a')}")
    print(f"final cost: {result.fun:.4f}")
    if result.nit:
        print(f"effective s/iteration: {elapsed/result.nit:.3f}")
    print(f"planned u sequence:\n{u_seq}")


if __name__ == "__main__":
    main()
