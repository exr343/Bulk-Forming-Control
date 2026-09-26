"""Benchmarks the wall-clock cost of one autodiff gradient computation
(forward through the single-shooting rollout + one backward() call -- exactly
mpc.py's cost_and_grad) at each horizon length 5,4,3,2,1, for a given
checkpoint. This is the per-SQP-iteration cost MPCController.plan() actually
pays at each step of the receding-horizon loop (step k plans over horizon
6-k), so results are reported per forging step, matching the accuracy table.

Usage:
    python -m applications.Agility_Forge.GNN.mp_sweep.benchmark_gradient_time \\
        --checkpoint-path applications/Agility_Forge/GNN/mp_sweep/checkpoint_mp_5.pt \\
        --message-passing-steps 5
"""

import argparse
import json
import time

import torch

from applications.Agility_Forge.GNN.data import build_surface_mesh_info
from applications.Agility_Forge.GNN.model import ForgeGNN
from applications.Agility_Forge.control.mpc import N_CONTROLS, _rollout_cost

N_REPEATS = 5  # timed repeats per horizon, after 1 untimed warm-up call


def benchmark(model, mesh_info, H_mm, band_width_frac, device="cpu"):
    """Returns {horizon: mean_seconds} for horizon in 5,4,3,2,1 -- same fixed
    x0 (undeformed) and a fixed dummy control guess for every horizon/model,
    so the comparison isolates message-passing depth, not input variation."""
    x0 = torch.zeros(mesh_info.rest_pos.shape[0], 3, device=device)
    target = torch.zeros(mesh_info.rest_pos.shape[0], 3, device=device)
    results = {}
    for horizon in [5, 4, 3, 2, 1]:
        u0 = torch.tile(torch.tensor([0.4, 90.0, 0.5]), (horizon,)).to(device)

        def one_call():
            u_t = u0.clone().requires_grad_(True)
            cost = _rollout_cost(u_t, model, mesh_info, x0, H_mm, band_width_frac, target, horizon)
            cost.backward()
            return u_t.grad

        one_call()  # untimed warm-up
        t0 = time.time()
        for _ in range(N_REPEATS):
            one_call()
        elapsed = (time.time() - t0) / N_REPEATS
        results[horizon] = elapsed
        print(f"horizon={horizon}: {elapsed:.3f}s/gradient-call (avg of {N_REPEATS})", flush=True)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--message-passing-steps", type=int, required=True)
    p.add_argument("--out-path", default=None,
                   help="Defaults to <checkpoint-dir>/gradient_time_benchmark.json")
    args = p.parse_args()

    mesh_info = build_surface_mesh_info(
        "applications/Agility_Forge/data/dataset_pretraining/rollout_01/undeformed.vtu")
    model = ForgeGNN(mesh_info, latent_size=128, num_layers=2,
                      message_passing_steps=args.message_passing_steps)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    H_mm = float(mesh_info.rest_pos[:, 0].max().item())
    results = benchmark(model, mesh_info, H_mm, band_width_frac=0.2)

    out_path = args.out_path or args.checkpoint_path.rsplit(".", 1)[0] + "_gradient_time_benchmark.json"
    with open(out_path, "w") as f:
        json.dump({"message_passing_steps": args.message_passing_steps,
                    "seconds_per_horizon": results}, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
