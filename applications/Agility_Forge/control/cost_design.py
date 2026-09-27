"""Cost-function design experiments for the square-rod MPC, run as a
SURROGATE closed loop: the MPC plans with the GNN, and the same GNN stands in
for the simulator (each applied hit's outcome is the GNN's prediction). A
50-hit run takes minutes instead of the ~10 h a real-simulator run needs, so
many cost designs can be screened; the promising ones then go to the real
plant (eval_square_target.py).

Everything except the cost matches eval_square_target.py: target = the square
run's hit-48 geometry, horizon 10, 50 hits, station >= the 800 C point, stroke
0-2 mm, warm start from the previous plan. The node-distance error term is
always kept; each config adds MPCController `penalty` terms (see mpc.py).

Per config, into <out-root>/<experiment>/<name>/ (experiment = the name's prefix, e.g. e7):
  results.json -- per-hit controls, errors, plan times; summary metrics
  error_vs_hit.png, controls_*.png -- same plots as eval_square_target.py
  thickness_profile.png -- final bar thickness along its length vs. target
  final_state.npy -- surface displacement (n_surface, 3) after the last hit

Summary metrics (all measured on the surrogate's final state):
  hausdorff/chamfer/rmse to target at the last hit and averaged over hits;
  cross_section_err_mm -- mean |width - target width| (both across-flats
    widths, 5 mm slices of the pressable region x = 20-90 mm): is the bar
    actually squared?
  tip_axial_mm -- free-end lengthwise displacement (target: see target_tip);
  n_active_hits -- hits with stroke > 0.1 mm (the MPC may choose ~0 = no hit);
  n_stations -- distinct station positions among active hits (2 mm rounding);
  longest_repeat -- most consecutive active hits at the same station.

Usage:
    python -m applications.Agility_Forge.control.cost_design --configs <json file> --index <i>
    (or --name baseline with no penalty flags)
"""

import argparse
import json
import os
import re
import time

import meshio
import numpy as np
import torch

from applications.Agility_Forge.GNN.data import (
    band_half_thickness, build_node_features, build_surface_mesh_info, gap_frac)
from applications.Agility_Forge.GNN.model import ForgeGNN
from applications.Agility_Forge.control.eval_square_target import (
    BAND_WIDTH_FRAC, SKIP_BELOW_MM, SQUARE_DIR, geometry_errors, load_reference, make_plots, surface_disp, _style,
    SURFACE, INK, INK_2, GRID, BLUE, REF_GRAY)
from applications.Agility_Forge.control.mpc import GAP_BOUNDS_MM, TRAVEL_CAP_MM, MPCController, gnn_u_frac
from applications.Agility_Forge.generate_square_rollout import D_J_MAX, X_800C_MM

CHECKPOINT_M3 = "applications/Agility_Forge/GNN/mp_sweep/finetune/mp_3/checkpoint.pt"
PROFILE_BINS = np.arange(20.0, 95.0, 5.0)  # rest-x slices of the pressable region
ACTIVE_STROKE_MM = 0.1  # below this a planned hit is treated as 'no hit'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", default=None, help="JSON list of {name, penalty} configs.")
    p.add_argument("--index", type=int, default=None, help="Which config in --configs to run.")
    p.add_argument("--name", default="baseline")
    p.add_argument("--penalty", default="{}", help="JSON dict of MPCController penalty weights.")
    p.add_argument("--n-hits", type=int, default=50)
    p.add_argument("--init-noise", type=float, default=0.0,
                   help="Robustness test: Gaussian noise (fraction of each control's range) added to every "
                        "re-plan's starting guess. 0 = deterministic.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--checkpoint-path", default=CHECKPOINT_M3)
    p.add_argument("--target-vtu", default=None,
                   help="Surface-mesh target (billet surface points + Displacement), e.g. "
                        "control/targets/ideal_square_10.6/target_on_billet.vtu. Default: square run hit 48.")
    p.add_argument("--cs-share", type=float, default=None,
                   help="Set the cross-section weight w from the target instead of by hand: w such that the "
                        "cross-section term is this share of the starting cost (billet vs. target), i.e. "
                        "w = s/(1-s) * E_len0/E_cs0 - 1. Overrides penalty['transverse'].")
    p.add_argument("--out-root", default="applications/Agility_Forge/control/results/surrogate",
                   help="Runs go to <out-root>/<experiment>/<name>, where <experiment> is the name's prefix (e.g. e7).")
    p.add_argument("--min-travel-mm", type=float, default=0.0,
                   help="Gap control only: minimum die travel for every planned hit (0 = misses allowed).")
    p.add_argument("--plan-variable", choices=("gap", "travel"), default="gap",
                   help="Gap-trained models only: what the optimizer varies -- the half-gap (with the travel "
                        "constraints), or the travel in [--min-travel-mm, 2] mm converted to a gap per hit.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def widths(rest, u):
    """Both across-flats widths per 5 mm slice of the pressable region."""
    pos = rest + u
    out = []
    for a, b in zip(PROFILE_BINS[:-1], PROFILE_BINS[1:]):
        s = (rest[:, 0] >= a) & (rest[:, 0] < b)
        out.append((np.ptp(pos[s, 1]), np.ptp(pos[s, 2])))
    return np.array(out)


def plot_thickness(out_dir, rest, x_final, x_target, name, n_hits):
    import matplotlib.pyplot as plt
    centers = 0.5 * (PROFILE_BINS[:-1] + PROFILE_BINS[1:])
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    for u, label, kw in [(np.zeros_like(x_target), "Billet", dict(color=REF_GRAY, linestyle=":")),
                         (x_target, "Target", dict(color=INK, linestyle="--")),
                         (x_final, f"MPC, hit {n_hits}", dict(color=BLUE, linestyle="-"))]:
        w = widths(rest, u)
        ax.plot(centers, w.max(1), marker="o", markersize=4, linewidth=2, label=label, **kw)
    _style(ax, f"Largest Across-Flats Width ({name})", "Position along the original bar (mm from x = 0)",
           "Width (mm)")
    leg = ax.legend(frameon=True, fontsize=9, labelcolor=INK_2)
    leg.get_frame().set_edgecolor(GRID)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "thickness_profile.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def replay_final_state(run_dir, device="cpu", checkpoint_path=CHECKPOINT_M3):
    """Rebuilds a finished run's final surrogate state by re-applying its
    logged hits through the same GNN (deterministic) -- for runs made before
    final_state.npy was saved."""
    with open(os.path.join(run_dir, "results.json")) as f:
        steps = json.load(f)["steps"]
    mesh_info = build_surface_mesh_info(os.path.join(SQUARE_DIR, "rollout_01", "undeformed.vtu"))
    H_mm = float(mesh_info.rest_pos[:, 0].max().item())
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = ForgeGNN(mesh_info, latent_size=a["latent_size"], num_layers=a["num_layers"],
                     message_passing_steps=a["message_passing_steps"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    x = torch.zeros(mesh_info.rest_pos.shape[0], 3, device=device)
    with torch.no_grad():
        for s in steps:
            d = s["d_j_frac"]
            nf = build_node_features(mesh_info, x, d, d * H_mm, (d + BAND_WIDTH_FRAC) * H_mm, s["R_j_deg"],
                                     gnn_u_frac(s["u_j_mm"])).unsqueeze(0)
            x = x + model.predict_delta(nf, x.unsqueeze(0), accumulate=False)[0]
    xf = x.cpu().numpy()
    np.save(os.path.join(run_dir, "final_state.npy"), xf)
    return xf


def main(args):
    if args.configs is not None:
        with open(args.configs) as f:
            cfg = json.load(f)[args.index]
        name, penalty = cfg["name"], cfg.get("penalty", {})
        args.init_noise = cfg.get("init_noise", args.init_noise)
        args.seed = cfg.get("seed", args.seed)
        args.target_vtu = cfg.get("target_vtu", args.target_vtu)
        args.cs_share = cfg.get("cs_share", args.cs_share)
        args.checkpoint_path = cfg.get("checkpoint_path", args.checkpoint_path)
        args.min_travel_mm = cfg.get("min_travel_mm", args.min_travel_mm)
        args.plan_variable = cfg.get("plan_variable", args.plan_variable)
    else:
        name, penalty = args.name, json.loads(args.penalty)
    prefix = name.split("_")[0]
    out_dir = os.path.join(args.out_root, prefix, name) if re.fullmatch(r"e\d+", prefix) else os.path.join(args.out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    device = args.device

    mesh_info = build_surface_mesh_info(os.path.join(SQUARE_DIR, "rollout_01", "undeformed.vtu"))
    rest_t = mesh_info.rest_pos.to(device)
    rest = mesh_info.rest_pos.numpy()
    H_mm = float(mesh_info.rest_pos[:, 0].max().item())
    if args.target_vtu:
        t = meshio.read(args.target_vtu)
        if t.points.shape[0] != rest.shape[0] or np.abs(t.points - rest).max() > 1e-4:
            raise ValueError(f"{args.target_vtu} is not on this billet's surface mesh")
        x_target = torch.tensor(t.point_data["Displacement"], dtype=torch.float32).to(device)
    else:
        x_target = surface_disp(os.path.join(SQUARE_DIR, "rollout_01", "hit_48_final.vtu"), mesh_info).to(device)
    xt_np = x_target.cpu().numpy()
    e_len0, e_cs0 = float((xt_np[:, 0] ** 2).sum()), float((xt_np[:, 1:] ** 2).sum())
    if args.cs_share is not None:
        s_ = args.cs_share
        penalty = {**penalty, "transverse": s_ / (1.0 - s_) * e_len0 / e_cs0 - 1.0}
    ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = ForgeGNN(mesh_info, latent_size=a["latent_size"], num_layers=a["num_layers"],
                     message_passing_steps=a["message_passing_steps"])
    model.load_state_dict(ckpt["state_dict"])
    control = a.get("control", "stroke")
    if control == "gap" and args.plan_variable == "travel":
        control = "gap_travel"
    mpc = MPCController(model, mesh_info, horizon=args.horizon, target_state=x_target,
                        band_width_frac=BAND_WIDTH_FRAC, device=device,
                        d_j_bounds=(X_800C_MM / H_mm, D_J_MAX), penalty=penalty, control=control,
                        min_travel_mm=args.min_travel_mm if control == "gap" else 0.0,
                        **({"u_j_mm_min": args.min_travel_mm} if control in ("gap_travel", "stroke")
                           and args.min_travel_mm > 0 else {}))
    ref = load_reference(SQUARE_DIR, mesh_info, x_target, rest_t, H_mm)
    print(f"Config '{name}': penalty {mpc.penalty} | M={a['message_passing_steps']} | {args.n_hits} hits, "
          f"horizon {args.horizon} | control {control}, min travel {args.min_travel_mm} mm", flush=True)

    x = torch.zeros_like(x_target)
    steps, u_init, u_prev = [], None, None
    rng = np.random.default_rng(args.seed)
    lo = np.array([b[0] for b in mpc.control_bounds] * args.horizon)
    hi = np.array([b[1] for b in mpc.control_bounds] * args.horizon)
    for step in range(1, args.n_hits + 1):
        t0 = time.perf_counter()
        if args.init_noise > 0:
            guess = u_init if u_init is not None else np.tile(
                [min(max(0.4, lo[0]), hi[0]), 90.0, 0.5], args.horizon)
            u_init = np.clip(guess + rng.normal(0.0, args.init_noise, guess.shape) * (hi - lo), lo, hi)
        u_seq, opt = mpc.plan(x, horizon=args.horizon, u_init=u_init, u_prev=u_prev)
        plan_time = time.perf_counter() - t0
        d, R, uf = (float(v) for v in u_seq[0])
        gap_info, skipped = {}, False
        if control == "gap":
            # Applied exactly as eval_square_target.py does: travel from the
            # current (surrogate) state, clipped to [0, cap]; a miss leaves the
            # bar unchanged -- NOT the GNN's own prediction for a miss, which
            # is wrong on thin sections (experiment 9).
            half_gap = GAP_BOUNDS_MM[0] + uf * (GAP_BOUNDS_MM[1] - GAP_BOUNDS_MM[0])
            with torch.no_grad():
                ht = float(band_half_thickness(mesh_info, x, d * H_mm, (d + BAND_WIDTH_FRAC) * H_mm, R))
            u_mm = min(max(ht - half_gap, 0.0), TRAVEL_CAP_MM)
            skipped = u_mm < SKIP_BELOW_MM
            gnn_ctrl = gap_frac(ht - u_mm)  # the gap actually reached (differs only if the cap binds)
            gap_info = {"half_gap_mm": half_gap, "band_half_thickness_mm": ht, "skipped": skipped}
        elif control == "gap_travel":
            u_mm = mpc.u_j_mm_min + uf * (mpc.u_j_mm_max - mpc.u_j_mm_min)
            with torch.no_grad():
                ht = float(band_half_thickness(mesh_info, x, d * H_mm, (d + BAND_WIDTH_FRAC) * H_mm, R))
            skipped = u_mm < SKIP_BELOW_MM
            gnn_ctrl = gap_frac(ht - u_mm)
            gap_info = {"half_gap_mm": ht - u_mm, "band_half_thickness_mm": ht, "skipped": skipped}
        else:
            u_mm = mpc.u_j_mm_min + uf * (mpc.u_j_mm_max - mpc.u_j_mm_min)
            gnn_ctrl = gnn_u_frac(u_mm)
        if not skipped:
            with torch.no_grad():
                nf = build_node_features(mesh_info, x, d, d * H_mm, (d + BAND_WIDTH_FRAC) * H_mm, R,
                                         gnn_ctrl).unsqueeze(0)
                x = x + mpc.model.predict_delta(nf, x.unsqueeze(0), accumulate=False)[0]
        c, h, r = geometry_errors(x, x_target, rest_t)
        steps.append({"step": step, "d_j_frac": d, "d_j_mm": d * H_mm, "R_j_deg": R, "u_j_mm": u_mm,
                      "plan_time_s": plan_time, "sqp_nit": int(opt.nit), "error_cost": opt.error_cost,
                      "penalty_cost": opt.penalty_cost,
                      "transverse_cost": opt.transverse_cost, "chamfer_mm2": c, "hausdorff_mm": h, "rmse_to_target_mm": r,
                      **gap_info})
        u_init = np.concatenate([u_seq[1:], u_seq[-1:]]).flatten()
        u_prev = u_seq[0]
        print(f"hit {step:2d}: d {d * H_mm:5.1f} mm  R {R:6.1f}  u {u_mm:.2f} mm | H {h:6.2f} C {c:7.2f} "
              f"| plan {plan_time:.1f}s nit {opt.nit}", flush=True)

    xf, xt = x.cpu().numpy(), x_target.cpu().numpy()
    np.save(os.path.join(out_dir, "final_state.npy"), xf)  # surface displacement after the last hit
    wf, wt = widths(rest, xf), widths(rest, xt)
    stations = np.round(np.array([s["d_j_mm"] for s in steps]) / 2.0) * 2.0
    active = np.array([s["u_j_mm"] > ACTIVE_STROKE_MM for s in steps])
    runs, longest = 0, 0  # consecutive ACTIVE hits (stroke > 0.1 mm) at the same station
    for i in range(len(stations)):
        if not active[i]:
            runs = 0
            continue
        runs = runs + 1 if (i > 0 and active[i - 1] and stations[i] == stations[i - 1]) else 1
        longest = max(longest, runs)
    summary = {
        "final_hausdorff_mm": steps[-1]["hausdorff_mm"], "final_chamfer_mm2": steps[-1]["chamfer_mm2"],
        "final_rmse_mm": steps[-1]["rmse_to_target_mm"],
        "mean_hausdorff_mm": float(np.mean([s["hausdorff_mm"] for s in steps])),
        "mean_chamfer_mm2": float(np.mean([s["chamfer_mm2"] for s in steps])),
        "cross_section_err_mm": float(np.abs(wf - wt).mean()),
        "billet_cross_section_err_mm": float(np.abs(widths(rest, np.zeros_like(xt)) - wt).mean()),
        "tip_axial_mm": float(xf[:, 0].max()), "target_tip_axial_mm": float(xt[:, 0].max()),
        "n_active_hits": int(active.sum()),
        "n_stations": int(len(np.unique(stations[active]))) if active.any() else 0,
        "longest_repeat": int(longest),
        "mean_stroke_mm": float(np.mean([s["u_j_mm"] for s in steps])),
        "mean_plan_time_s": float(np.mean([s["plan_time_s"] for s in steps])),
        "n_skipped": int(sum(s.get("skipped", False) for s in steps)),
        "n_travel_below_0.6mm": int(sum((not s.get("skipped", False)) and s["u_j_mm"] < 0.6 for s in steps)),
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "penalty": mpc.penalty, "config": {**vars(args), "surrogate_plant": True},
                   "start_error_split": {"lengthwise": e_len0, "cross_section": e_cs0,
                                         "cs_share_of_start_cost": (1 + mpc.penalty["transverse"]) * e_cs0 /
                                         (mpc.penalty["lengthwise"] * e_len0
                                          + (1 + mpc.penalty["transverse"]) * e_cs0)},
                   "summary": summary, "steps": steps, "reference": ref}, f, indent=2)
    make_plots(out_dir, steps, ref, args.n_hits)
    plot_thickness(out_dir, rest, xf, xt, name, args.n_hits)
    print("SUMMARY " + json.dumps({"name": name, **summary}), flush=True)


if __name__ == "__main__":
    main(parse_args())
