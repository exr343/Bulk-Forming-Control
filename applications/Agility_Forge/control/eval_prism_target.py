"""Closed-loop GNN-MPC toward a synthetic rounded-square-prism target (round
billet -> box-ish cross-section), instead of another rollout's real
achieved geometry.

Target construction: superellipse cross-section (|y/a|^n + |z/a|^n = 1,
n=4 -- a "rounded square", not sharp 90deg corners) with apothem
`--apothem-mm` (default 6.4375mm = billet radius 7.9375mm minus 1.5mm
indentation depth, chosen to sit within a single hit's max 2.0mm reach --
the exact inscribed-square apothem, 5.61mm/2.32mm depth, exceeds that,
so it's not reachable even in principle). Held constant along the axially
reachable band (d_j_frac/band_width_frac's actual sampled range, [0.02,
0.98] of H -- generate_dataset.py can never compress outside this).
Bottom-face/pin nodes (hard BCs) and the unreachable axial tips keep their
original rest position (zero target displacement) -- no control sequence
can ever move them, so targeting them there would be chasing the
impossible.

Same MPCController/ForgingPlant/checkpoint_mp_5.pt machinery as
eval_extended_horizon.py -- only the target construction differs (no real
ground-truth rollout exists for this target, so no per-hit "true
trajectory" baseline is available here, unlike eval_closed_loop_test_set.py).

Usage:
    python -m applications.Agility_Forge.control.eval_prism_target \\
        --n-hits 8 --apothem-mm 6.4375 --out-dir applications/Agility_Forge/control/results/before_2026-09-26_fixes/prism_target
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
REACHABLE_LO_FRAC, REACHABLE_HI_FRAC = 0.02, 0.98  # d_j_frac's own sampled range +/- band width


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-hits", type=int, default=8)
    p.add_argument("--apothem-mm", type=float, default=6.4375,
                   help="Center-to-flat-face distance of the target square cross-section.")
    p.add_argument("--corner-exponent", type=float, default=4.0,
                   help="Superellipse exponent n: higher = sharper corners, 2 = circle.")
    p.add_argument("--u-j-mm-min", type=float, default=0.0)
    p.add_argument("--checkpoint-path", default=CHECKPOINT_PATH,
                   help="Must be message_passing_steps=5 (MP_STEPS is not itself a CLI arg here).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_prism_target(mesh_info, apothem_mm, n_exp):
    """Returns (n_surface, 3) target displacement: radial projection of each
    lateral-wall node onto a superellipse cross-section at its own axial
    position, angle preserved (no swirl, no axial redistribution). Hard-BC
    nodes and axially-unreachable nodes get zero target displacement."""
    rest = mesh_info.rest_pos
    x, y, z = rest[:, 0], rest[:, 1], rest[:, 2]
    H = float(x.max())
    theta = torch.atan2(z, y)
    rho = apothem_mm / (torch.abs(torch.cos(theta)) ** n_exp + torch.abs(torch.sin(theta)) ** n_exp) ** (1.0 / n_exp)
    y_t, z_t = rho * torch.cos(theta), rho * torch.sin(theta)

    reachable = (x >= REACHABLE_LO_FRAC * H) & (x <= REACHABLE_HI_FRAC * H)
    hard_bc = (mesh_info.is_bottom_face[:, 0] > 0.5) | (mesh_info.is_pin_node[:, 0] > 0.5)
    apply = reachable & (~hard_bc)

    target_disp = torch.zeros_like(rest)
    target_disp[apply, 1] = y_t[apply] - y[apply]
    target_disp[apply, 2] = z_t[apply] - z[apply]
    return target_disp


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
    ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    target_cpu = build_prism_target(mesh_info, args.apothem_mm, args.corner_exponent)
    target = target_cpu.to(args.device)
    _save_surface_vtu(mesh_info, target_cpu, os.path.join(args.out_dir, "target.vtu"))
    _save_surface_vtu(mesh_info, torch.zeros(mesh_info.rest_pos.shape[0], 3),
                       os.path.join(args.out_dir, "step_00.vtu"))

    controller = MPCController(model, mesh_info, horizon=args.n_hits, target_state=target,
                                band_width_frac=BAND_WIDTH_FRAC, device=args.device,
                                u_j_mm_min=args.u_j_mm_min)

    mesh, R, H, T_linear_fn = load_default_billet()
    plant = ForgingPlant(mesh, R, H, T_linear_fn)

    per_hit = {"chamfer_mm2": {}, "hausdorff_mm": {}, "rmse_mm": {}}

    # hit 0 = undeformed billet vs. target, for the plot's starting point.
    undeformed = torch.zeros_like(rest_pos)
    c0, h0 = _chamfer_hausdorff_mm(rest_pos + target, rest_pos + undeformed)
    per_hit["chamfer_mm2"][0] = c0
    per_hit["hausdorff_mm"][0] = h0
    per_hit["rmse_mm"][0] = ((undeformed - target) ** 2).mean().sqrt().item()
    print(f"hit 0/{args.n_hits} (undeformed): chamfer={c0:.4f}mm^2 hausdorff={h0:.4f}mm", flush=True)

    def _save_partial():
        tmp = os.path.join(args.out_dir, "result.json.tmp")
        with open(tmp, "w") as f:
            json.dump({"n_hits": args.n_hits, "apothem_mm": args.apothem_mm,
                       "corner_exponent": args.corner_exponent, "u_j_mm_min": args.u_j_mm_min,
                       "per_hit": per_hit}, f, indent=2)
        os.replace(tmp, os.path.join(args.out_dir, "result.json"))

    _save_partial()

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
        "n_hits": args.n_hits,
        "apothem_mm": args.apothem_mm,
        "corner_exponent": args.corner_exponent,
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

    hits = sorted(per_hit["chamfer_mm2"])  # includes 0 = undeformed
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(hits, [per_hit["chamfer_mm2"][k] for k in hits], marker="o")
    ax1.set_xlabel("hit (0 = undeformed)"); ax1.set_ylabel("Chamfer (mm^2)"); ax1.set_title("Chamfer vs. hit")
    ax2.plot(hits, [per_hit["hausdorff_mm"][k] for k in hits], marker="o", color="tab:orange")
    ax2.set_xlabel("hit (0 = undeformed)"); ax2.set_ylabel("Hausdorff (mm)"); ax2.set_title("Hausdorff vs. hit")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "error_vs_hit.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved: {args.out_dir}/result.json, error_vs_hit.png, "
          f"step_00..{args.n_hits:02d}.vtu, target.vtu")


if __name__ == "__main__":
    main()
