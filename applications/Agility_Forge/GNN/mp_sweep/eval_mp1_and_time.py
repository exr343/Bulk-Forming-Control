"""Full test-set Chamfer/Hausdorff eval (merged into full_test_set_eval.json)
plus clean gradient-timing benchmark, for the M=1 model -- run separately
from the other four since it was trained after the initial sweep.

Usage: python -m applications.Agility_Forge.GNN.mp_sweep.eval_mp1_and_time
"""

import json
import time

import torch

from applications.Agility_Forge.GNN.data import build_surface_mesh_info, load_examples, split_examples_by_rollout
from applications.Agility_Forge.GNN.mp_sweep.full_test_set_eval import eval_model, DATASET_DIR, SNAPSHOT_PATH
from applications.Agility_Forge.GNN.model import ForgeGNN
from applications.Agility_Forge.control.mpc import _rollout_cost


def main():
    mesh_info = build_surface_mesh_info(f"{DATASET_DIR}/rollout_01/undeformed.vtu")
    with open(SNAPSHOT_PATH) as f:
        allowed = json.load(f)
    examples, manifest = load_examples(DATASET_DIR, allowed_rollouts=allowed)
    _, test_ex = split_examples_by_rollout(examples, train_frac=0.8)

    print("=== evaluating mp_steps=1 (full test set) ===", flush=True)
    avg_chamfer, avg_hausdorff = eval_model(1, mesh_info, test_ex)
    for h in sorted(avg_chamfer):
        print(f"  hit {h}: chamfer={avg_chamfer[h]:.4f}mm^2 hausdorff={avg_hausdorff[h]:.4f}mm", flush=True)

    eval_path = "applications/Agility_Forge/GNN/mp_sweep/full_test_set_eval.json"
    with open(eval_path) as f:
        full = json.load(f)
    full["results"]["1"] = {"chamfer_mm2": avg_chamfer, "hausdorff_mm": avg_hausdorff}
    with open(eval_path, "w") as f:
        json.dump(full, f, indent=2)

    print("=== gradient timing, mp_steps=1, horizon=5 ===", flush=True)
    model = ForgeGNN(mesh_info, latent_size=128, num_layers=2, message_passing_steps=1)
    model.eval()
    H_mm = float(mesh_info.rest_pos[:, 0].max().item())
    x0 = torch.zeros(mesh_info.rest_pos.shape[0], 3)
    target = torch.zeros(mesh_info.rest_pos.shape[0], 3)
    u0 = torch.tile(torch.tensor([0.4, 90.0, 0.5]), (5,))

    def one_call():
        u_t = u0.clone().requires_grad_(True)
        cost = _rollout_cost(u_t, model, mesh_info, x0, H_mm, 0.2, target, 5)
        cost.backward()
        return u_t.grad

    one_call()
    t0 = time.time()
    for _ in range(5):
        one_call()
    elapsed = (time.time() - t0) / 5
    print(f"1-step model, horizon=5: {elapsed:.3f}s/gradient-call (avg of 5)")


if __name__ == "__main__":
    main()
