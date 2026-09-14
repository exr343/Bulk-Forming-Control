"""Full receding-horizon MPC demo: JAX-FORGE (ForgingPlant) as the real plant,
ForgeGNN as the planning surrogate, targeting one held-out rollout's true
hit-5 geometry. At each of 5 hits, re-plans over the REMAINING horizon from
the plant's TRUE post-hit state (not the GNN's own prediction), applies only
the first control for real, and repeats.

Saves every step's real state (undeformed + each post-hit state) plus the
target geometry, all as surface-mesh .vtu in --out-dir, as it happens (not
just at the end) via MPCController.run()'s step_callback -- so a crash/
timeout partway through a multi-hour run still leaves usable partial output.

Usage:
    python -m applications.Agility_Forge.control.run_closed_loop \\
        --target-rollout 415 --out-dir applications/Agility_Forge/control/mpc_target_rollout_415
"""

import argparse
import json
import os

import meshio
import numpy as onp
import torch

from applications.Agility_Forge.GNN.data import build_surface_mesh_info
from applications.Agility_Forge.GNN.model import ForgeGNN
from applications.Agility_Forge.GNN.train import _chamfer_hausdorff_mm
from applications.Agility_Forge.control.mpc import MPCController
from applications.Agility_Forge.control.plant_interface import ForgingPlant, load_default_billet, extract_surface_state


def _save_surface_vtu(mesh_info, displacement, out_path):
    out_mesh = meshio.Mesh(
        points=mesh_info.rest_pos.numpy(),
        cells=[("triangle", mesh_info.surf_faces_local)],
        point_data={"Displacement": onp.array(displacement)},
    )
    out_mesh.write(out_path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-rollout", type=int, required=True,
                   help="Rollout id (from data/dataset) whose hit_05_final.vtu is the MPC target.")
    p.add_argument("--dataset-dir", default="applications/Agility_Forge/data/dataset")
    p.add_argument("--checkpoint-path", default="applications/Agility_Forge/GNN/checkpoint_3dim.pt")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    mesh_info = build_surface_mesh_info(
        os.path.join(args.dataset_dir, "rollout_01", "undeformed.vtu"))
    model = ForgeGNN(mesh_info, latent_size=128, num_layers=2, message_passing_steps=15)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    target_vtu = os.path.join(args.dataset_dir, f"rollout_{args.target_rollout:02d}", "hit_05_final.vtu")
    true_m = meshio.read(target_vtu)
    target_full = torch.tensor(true_m.point_data["Displacement"], dtype=torch.float32)
    target = target_full[mesh_info.full_to_surface]
    _save_surface_vtu(mesh_info, target, os.path.join(args.out_dir, "target.vtu"))

    controller = MPCController(model, mesh_info, horizon=5, target_state=target,
                                band_width_frac=0.2, device="cpu")

    mesh, R, H, T_linear_fn = load_default_billet()
    plant = ForgingPlant(mesh, R, H, T_linear_fn)

    # step_00 = undeformed, saved immediately so the animation has a start frame
    # even if the run fails on hit 1.
    _save_surface_vtu(mesh_info, torch.zeros_like(target), os.path.join(args.out_dir, "step_00.vtu"))

    def step_callback(step_idx, state, hit):
        surf = extract_surface_state(state.sol_u, mesh_info)
        _save_surface_vtu(mesh_info, surf, os.path.join(args.out_dir, f"step_{step_idx + 1:02d}.vtu"))
        print(f"Saved step_{step_idx + 1:02d}.vtu", flush=True)

    applied_controls, plan_vs_actual_rmse_mm, final_state = controller.run(
        plant, n_hits=5, step_callback=step_callback)

    actual_final = extract_surface_state(final_state.sol_u, mesh_info)
    final_rmse = ((actual_final - target) ** 2).mean().sqrt().item()
    # Absolute positions (rest + displacement), not raw displacement vectors
    # -- see train.py's _chamfer_hausdorff_mm docstring / the earlier bugfix
    # in this same session for why that distinction matters.
    final_chamfer_mm2, final_hausdorff_mm = _chamfer_hausdorff_mm(
        mesh_info.rest_pos + target, mesh_info.rest_pos + actual_final)

    print(f"\n=== target rollout {args.target_rollout} ===")
    for i, ((d_j_frac, R_j_deg, u_j_mm), mismatch) in enumerate(
            zip(applied_controls, plan_vs_actual_rmse_mm), 1):
        print(f"hit {i}: d_j_frac={d_j_frac:.4f} R_j_deg={R_j_deg:.2f} u_j_mm={u_j_mm:.3f}  "
              f"| plan-vs-actual RMSE this step: {mismatch:.4f}mm")
    print(f"Final real state vs. target -- RMSE: {final_rmse:.4f} mm, "
          f"Chamfer: {final_chamfer_mm2:.4f}mm^2, Hausdorff: {final_hausdorff_mm:.4f}mm")

    with open(os.path.join(args.out_dir, "result.json"), "w") as f:
        json.dump({
            "target_rollout": args.target_rollout,
            "applied_controls": applied_controls,
            "plan_vs_actual_rmse_mm": plan_vs_actual_rmse_mm,
            "final_rmse_vs_target_mm": final_rmse,
            "final_chamfer_mm2": final_chamfer_mm2,
            "final_hausdorff_mm": final_hausdorff_mm,
        }, f, indent=2)
    print(f"Saved: {args.out_dir}/step_00..05.vtu, target.vtu, result.json")


if __name__ == "__main__":
    main()
