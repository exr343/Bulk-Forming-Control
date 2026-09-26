"""Full test-set Chamfer/Hausdorff evaluation across all 4 mp_sweep models
(message_passing_steps = 5, 15, 45, 135). For each model, runs an
autoregressive rollout (model's own prediction feeds the next hit, matching
train.py's _rollout_eval convention) on EVERY test rollout in the 383-rollout
snapshot's 77-rollout test split, computes per-hit Chamfer/Hausdorff
(absolute positions, the corrected convention) against ground truth, and
averages across all 77 rollouts per hit -- this is the real, full-test-set
version of the single-rollout eval table used earlier.

Usage: python -m applications.Agility_Forge.GNN.mp_sweep.full_test_set_eval
"""

import json
from collections import defaultdict

import numpy as np
import torch

from applications.Agility_Forge.GNN.data import (
    build_surface_mesh_info, load_examples, split_examples_by_rollout,
    build_node_features, ForgeGNNDataset,
)
from applications.Agility_Forge.GNN.model import ForgeGNN
from applications.Agility_Forge.GNN.train import _chamfer_hausdorff_mm

MP_STEPS_LIST = [5, 15, 45, 135]
DATASET_DIR = "applications/Agility_Forge/data/dataset_pretraining"
SNAPSHOT_PATH = "applications/Agility_Forge/GNN/mp_sweep/rollout_snapshot_383.json"


def eval_model(mp_steps, mesh_info, test_ex):
    ckpt_path = f"applications/Agility_Forge/GNN/mp_sweep/checkpoint_mp_{mp_steps}.pt"
    model = ForgeGNN(mesh_info, latent_size=128, num_layers=2, message_passing_steps=mp_steps)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    by_rollout = defaultdict(list)
    for ex in test_ex:
        by_rollout[ex.rollout].append(ex)

    ds = ForgeGNNDataset(DATASET_DIR, test_ex, mesh_info)
    n_surface = mesh_info.rest_pos.shape[0]
    per_hit_chamfer, per_hit_hausdorff = defaultdict(list), defaultdict(list)

    with torch.no_grad():
        for rollout_id, hits in by_rollout.items():
            hits_sorted = sorted(hits, key=lambda e: e.hit)
            x_k_pred = torch.zeros(1, n_surface, 3)
            for ex in hits_sorted:
                true_next = ds._load_surface_displacement(ex.x_next_vtu)
                nf = build_node_features(mesh_info, x_k_pred[0], ex.d_j_frac, ex.d_j_mm,
                                          ex.x_max_band_mm, ex.R_j_deg, ex.u_j_frac).unsqueeze(0)
                delta = model.predict_delta(nf, x_k_pred, accumulate=False)
                x_next_pred = x_k_pred + delta

                true_pts = mesh_info.rest_pos + true_next
                pred_pts = mesh_info.rest_pos + x_next_pred[0]
                chamfer, hausdorff = _chamfer_hausdorff_mm(true_pts, pred_pts)
                per_hit_chamfer[ex.hit].append(chamfer)
                per_hit_hausdorff[ex.hit].append(hausdorff)

                x_k_pred = x_next_pred

    avg_chamfer = {h: float(np.mean(v)) for h, v in per_hit_chamfer.items()}
    avg_hausdorff = {h: float(np.mean(v)) for h, v in per_hit_hausdorff.items()}
    return avg_chamfer, avg_hausdorff


def main():
    mesh_info = build_surface_mesh_info(f"{DATASET_DIR}/rollout_01/undeformed.vtu")
    with open(SNAPSHOT_PATH) as f:
        allowed = json.load(f)
    examples, manifest = load_examples(DATASET_DIR, allowed_rollouts=allowed)
    train_ex, test_ex = split_examples_by_rollout(examples, train_frac=0.8)
    n_test_rollouts = len({e.rollout for e in test_ex})
    print(f"test rollouts: {n_test_rollouts}", flush=True)

    results = {}
    for mp_steps in MP_STEPS_LIST:
        print(f"=== evaluating mp_steps={mp_steps} ===", flush=True)
        avg_chamfer, avg_hausdorff = eval_model(mp_steps, mesh_info, test_ex)
        results[mp_steps] = {"chamfer_mm2": avg_chamfer, "hausdorff_mm": avg_hausdorff}
        for h in sorted(avg_chamfer):
            print(f"  hit {h}: chamfer={avg_chamfer[h]:.4f}mm^2 hausdorff={avg_hausdorff[h]:.4f}mm",
                  flush=True)

    out_path = "applications/Agility_Forge/GNN/mp_sweep/full_test_set_eval.json"
    with open(out_path, "w") as f:
        json.dump({"n_test_rollouts": n_test_rollouts, "results": results}, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
