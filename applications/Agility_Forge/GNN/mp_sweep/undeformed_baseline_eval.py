"""No-skill baseline: Chamfer/Hausdorff between the TRUE geometry at each hit
and the UNDEFORMED billet (zero displacement everywhere) -- i.e. "the model
predicted nothing moves at all." No model involved; pure data statistic,
same 77-rollout test split and per-hit averaging as full_test_set_eval.py,
for direct comparison against the trained models' numbers.

Usage: python -m applications.Agility_Forge.GNN.mp_sweep.undeformed_baseline_eval
"""

import json
from collections import defaultdict

import numpy as np
import torch

from applications.Agility_Forge.GNN.data import (
    build_surface_mesh_info, load_examples, split_examples_by_rollout, ForgeGNNDataset,
)
from applications.Agility_Forge.GNN.train import _chamfer_hausdorff_mm

DATASET_DIR = "applications/Agility_Forge/data/dataset_pretraining"
SNAPSHOT_PATH = "applications/Agility_Forge/GNN/mp_sweep/rollout_snapshot_383.json"


def main():
    mesh_info = build_surface_mesh_info(f"{DATASET_DIR}/rollout_01/undeformed.vtu")
    with open(SNAPSHOT_PATH) as f:
        allowed = json.load(f)
    examples, manifest = load_examples(DATASET_DIR, allowed_rollouts=allowed)
    _, test_ex = split_examples_by_rollout(examples, train_frac=0.8)
    n_test_rollouts = len({e.rollout for e in test_ex})
    print(f"test rollouts: {n_test_rollouts}", flush=True)

    ds = ForgeGNNDataset(DATASET_DIR, test_ex, mesh_info)
    n_surface = mesh_info.rest_pos.shape[0]
    undeformed_pts = mesh_info.rest_pos  # zero displacement -> just rest position

    per_hit_chamfer, per_hit_hausdorff, per_hit_rmse = (defaultdict(list) for _ in range(3))
    for ex in test_ex:
        true_next = ds._load_surface_displacement(ex.x_next_vtu)
        true_pts = mesh_info.rest_pos + true_next
        chamfer, hausdorff = _chamfer_hausdorff_mm(true_pts, undeformed_pts)
        rmse = (true_next ** 2).mean().sqrt().item()  # true_pts - undeformed_pts == true_next
        per_hit_chamfer[ex.hit].append(chamfer)
        per_hit_hausdorff[ex.hit].append(hausdorff)
        per_hit_rmse[ex.hit].append(rmse)

    results = {}
    for h in sorted(per_hit_chamfer):
        c, ha, r = np.mean(per_hit_chamfer[h]), np.mean(per_hit_hausdorff[h]), np.mean(per_hit_rmse[h])
        results[h] = {"chamfer_mm2": float(c), "hausdorff_mm": float(ha), "rmse_mm": float(r)}
        print(f"hit {h}: chamfer={c:.4f}mm^2 hausdorff={ha:.4f}mm rmse={r:.4f}mm", flush=True)

    out_path = "applications/Agility_Forge/GNN/mp_sweep/undeformed_baseline_eval.json"
    with open(out_path, "w") as f:
        json.dump({"n_test_rollouts": n_test_rollouts, "results": results}, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
