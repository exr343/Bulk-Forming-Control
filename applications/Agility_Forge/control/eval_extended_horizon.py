"""Extended-horizon closed-loop MPC pilot: does allowing MORE real hits than
the training data's fixed 5-hit schedule let receding-horizon MPC converge
closer to a fixed target, or does pushing the GNN surrogate past the
deformation states it was ever trained on make things worse past hit 5?

Single rollout (default: mpc_target_rollout_415's own true hit-5 final
geometry -- the same target already used for
control/mpc_target_rollout_415/'s earlier 5-hit M=15 validation), but run
for up to --n-hits (default 10) real hits instead of 5. The target stays
FIXED at the true hit-5 geometry throughout the whole run -- there is no
real ground truth beyond hit 5 for this rollout (it was only ever forged
for 5 hits in generate_dataset.py), so hits 6+ are the surrogate
extrapolating into states it never saw during training. That extrapolation
is the actual thing this pilot measures -- whether Chamfer/Hausdorff keep
improving past hit 5, plateau, or get worse.

Uses checkpoint_mp_5.pt, matching the concurrent full-test-set closed-loop
run (control/eval_closed_loop_test_set.py) so hit 1-5 numbers here are
directly comparable to that run's per-hit averages.

Writes result.json incrementally (after every hit, atomically), plus a
Chamfer/Hausdorff-vs-hit plot and per-step surface .vtu files for animation
(same convention as run_closed_loop.py).

Usage:
    python -m applications.Agility_Forge.control.eval_extended_horizon \\
        --target-rollout 415 --n-hits 10 \\
        --out-dir applications/Agility_Forge/control/mpc_target_rollout_415_10hits
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
import torch

from applications.Agility_Forge.GNN.data import build_surface_mesh_info
from applications.Agility_Forge.GNN.model import ForgeGNN
from applications.Agility_Forge.GNN.train import _chamfer_hausdorff_mm
from applications.Agility_Forge.control.mpc import MPCController
from applications.Agility_Forge.control.plant_interface import ForgingPlant, load_default_billet, extract_surface_state

DATASET_DIR = "applications/Agility_Forge/data/dataset_pretraining"
CHECKPOINT_PATH = "applications/Agility_Forge/GNN/mp_sweep/checkpoint_mp_5.pt"
MP_STEPS = 5
BAND_WIDTH_FRAC = 0.2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-rollout", type=int, default=415,
                   help="Rollout id (from data/dataset_pretraining) whose hit_05_final.vtu is the fixed MPC target.")
    p.add_argument("--n-hits", type=int, default=10, help="Real hits to run (target stays fixed at hit 5's geometry).")
    p.add_argument("--u-j-mm-min", type=float, default=0.5,
                   help="Physical mm floor u_j_frac=0 maps to. Training data's actual floor is 0.5mm "
                        "(the default); lowering this (e.g. to 0.0) lets the optimizer choose an "
                        "effectively-negligible stroke once already near target, instead of being forced "
                        "to keep disturbing the shape by at least this much on every remaining hit.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _save_surface_vtu(mesh_info, displacement, out_path):
    out_mesh = meshio.Mesh(
        points=mesh_info.rest_pos.numpy(),
        cells=[("triangle", mesh_info.surf_faces_local)],
        point_data={"Displacement": np.asarray(displacement)},
    )
    out_mesh.write(out_path)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    mesh_info = build_surface_mesh_info(os.path.join(DATASET_DIR, "rollout_01", "undeformed.vtu"))
    rest_pos = mesh_info.rest_pos.to(args.device)

    model = ForgeGNN(mesh_info, latent_size=128, num_layers=2, message_passing_steps=MP_STEPS)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    target_vtu = os.path.join(DATASET_DIR, f"rollout_{args.target_rollout:02d}", "hit_05_final.vtu")
    true_m = meshio.read(target_vtu)
    target_full = torch.tensor(true_m.point_data["Displacement"], dtype=torch.float32)
    target = target_full[mesh_info.full_to_surface].to(args.device)
    _save_surface_vtu(mesh_info, target.cpu(), os.path.join(args.out_dir, "target.vtu"))
    _save_surface_vtu(mesh_info, torch.zeros(mesh_info.rest_pos.shape[0], 3),
                       os.path.join(args.out_dir, "step_00.vtu"))

    controller = MPCController(model, mesh_info, horizon=args.n_hits, target_state=target,
                                band_width_frac=BAND_WIDTH_FRAC, device=args.device,
                                u_j_mm_min=args.u_j_mm_min)

    mesh, R, H, T_linear_fn = load_default_billet()
    plant = ForgingPlant(mesh, R, H, T_linear_fn)

    per_hit = {"chamfer_mm2": {}, "hausdorff_mm": {}, "rmse_mm": {}}

    def _save_partial():
        tmp = os.path.join(args.out_dir, "result.json.tmp")
        with open(tmp, "w") as f:
            json.dump({"target_rollout": args.target_rollout, "n_hits": args.n_hits,
                       "u_j_mm_min": args.u_j_mm_min, "per_hit": per_hit}, f, indent=2)
        os.replace(tmp, os.path.join(args.out_dir, "result.json"))

    def step_callback(step_idx, state, hit):
        k = step_idx + 1
        surf = extract_surface_state(state.sol_u, mesh_info).to(args.device)
        c, h = _chamfer_hausdorff_mm(rest_pos + target, rest_pos + surf)
        rmse = ((surf - target) ** 2).mean().sqrt().item()
        per_hit["chamfer_mm2"][k] = c
        per_hit["hausdorff_mm"][k] = h
        per_hit["rmse_mm"][k] = rmse
        _save_surface_vtu(mesh_info, surf.cpu(), os.path.join(args.out_dir, f"step_{k:02d}.vtu"))
        print(f"hit {k}/{args.n_hits}: chamfer={c:.4f}mm^2 hausdorff={h:.4f}mm rmse={rmse:.4f}mm", flush=True)
        _save_partial()

    applied_controls, plan_vs_actual_rmse_mm, plan_time_s, final_state = controller.run(
        plant, n_hits=args.n_hits, step_callback=step_callback)

    result = {
        "target_rollout": args.target_rollout,
        "n_hits": args.n_hits,
        "u_j_mm_min": args.u_j_mm_min,
        "per_hit": per_hit,
        "applied_controls": [
            {"d_j_frac": c[0], "R_j_deg": c[1], "u_j_mm": c[2]} for c in applied_controls
        ],
        "plan_vs_actual_rmse_mm": plan_vs_actual_rmse_mm,
        "plan_time_s": plan_time_s,
    }
    with open(os.path.join(args.out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    hits = sorted(per_hit["chamfer_mm2"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(hits, [per_hit["chamfer_mm2"][k] for k in hits], marker="o")
    ax1.axvline(5, color="gray", linestyle="--", linewidth=1, label="hit 5 (training horizon)")
    ax1.set_xlabel("hit"); ax1.set_ylabel("Chamfer (mm^2)"); ax1.set_title("Chamfer vs. hit"); ax1.legend()
    ax2.plot(hits, [per_hit["hausdorff_mm"][k] for k in hits], marker="o", color="tab:orange")
    ax2.axvline(5, color="gray", linestyle="--", linewidth=1, label="hit 5 (training horizon)")
    ax2.set_xlabel("hit"); ax2.set_ylabel("Hausdorff (mm)"); ax2.set_title("Hausdorff vs. hit"); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "error_vs_hit.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved: {args.out_dir}/result.json, error_vs_hit.png, "
          f"step_00..{args.n_hits:02d}.vtu, target.vtu")


if __name__ == "__main__":
    main()
