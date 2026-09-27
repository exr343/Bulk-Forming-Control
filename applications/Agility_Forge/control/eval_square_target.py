"""Closed-loop GNN-MPC toward the (non-jittered) square rod: can the MPC form
the billet incrementally into the square run's final geometry?

Settled by interview (2026-09-26):
  - Target: data/dataset_finetuning/square's hit-48 geometry (the square run's
    final state; that run is in the finetuning training set -- accepted).
  - Model: GNN/finetune_square/checkpoint_mp_5_finetuned_square.pt.
  - Receding horizon, ALWAYS 10 hits ahead (plans near the end look past hit
    50; only each plan's first hit is ever applied). Exactly 50 real hits.
    Warm start: previous plan shifted by one, last hit repeated.
  - Cost: MPCController's sum over the horizon of node-by-node squared
    distance to the target (unchanged).
  - d_j_frac bounded to [800 C point, 0.78] -- same floor the square data
    obeyed (x = 17.73 mm on the initial temperature profile).
  - Stroke 0-2 mm (mpc.U_J_MM_MIN/MAX), fed to the GNN on its training scale
    (mpc.gnn_u_frac -- the fix for the old frac mismatch).
  - Gap control (2026-09-27, used when the checkpoint was trained with
    `--control gap`): the MPC plans the half-gap the dies close to
    (mpc.GAP_BOUNDS_MM, travel capped at mpc.TRAVEL_CAP_MM per hit). The
    applied stroke is the true band half-thickness minus that half-gap,
    clipped to [0, cap]; below SKIP_BELOW_MM the dies miss the bar and the
    hit is skipped (state unchanged, no simulation).
  - Stop rule (--stop-tol-mm; off by default, settled 2026-09-27): the same
    test as the finishing phase of the square data runs
    (generate_square_rollout.across_by_station / within_tolerance). After
    every real hit, measure the bar's thickness at 0 and 90 deg in the six
    square-run check windows (19.3 mm die bands at stations d_j 0.17-0.78),
    counting only the part of each window that the target makes square
    (original x >= the square section's start, from target_spec.json). The
    run stops once every measurement is at most 10.6 + tol mm.
  - After every real hit: full plant state checkpointed, results.json and all
    plots rewritten. A plant failure is logged to failed_steps.jsonl, plots
    are made for the completed hits, and the job exits non-zero;
    resubmitting resumes from the last completed hit (re-planning there).

Plots (rewritten every hit):
  error_vs_hit.png -- Hausdorff, Chamfer (MPC solid, reference dotted) and
    SQP planning time per hit. "Reference" (open-loop) = the square run's own
    hits replayed, each compared with its own final (hit-48) geometry.
  controls_station.png / controls_angle.png / controls_stroke.png -- the
    three applied inputs per hit, MPC solid vs. the square run's own
    schedule dotted. Angle folded to [0, 180) (the dies press both sides).

Usage:
    python -m applications.Agility_Forge.control.eval_square_target \\
        --out-dir applications/Agility_Forge/control/results/real_simulator/e0_original_cost
"""

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
import jax.numpy as jnp
import torch

from applications.Agility_Forge.GNN.data import (
    band_half_thickness, build_node_features, build_surface_mesh_info, gap_frac, save_surface_vtu)
from applications.Agility_Forge.GNN.model import ForgeGNN
from applications.Agility_Forge.GNN.train import _chamfer_hausdorff_mm
from applications.Agility_Forge.control.mpc import (
    GAP_BOUNDS_MM, TRAVEL_CAP_MM, MPCController, TOTAL_TIME_S, gnn_u_frac)
from applications.Agility_Forge.control.plant_interface import (
    ForgingPlant, ForgingState, extract_surface_state, load_default_billet)
from applications.Agility_Forge.generate_dataset import _locked
from applications.Agility_Forge.generate_square_rollout import (
    D_J_MAX, N_INT_VARS, SQUARE_STATIONS_D_J, X_800C_MM, across_by_station, within_tolerance)
from applications.Agility_Forge.hit_config import Hit

SQUARE_DIR = "applications/Agility_Forge/data/dataset_finetuning/square"
CHECKPOINT = "applications/Agility_Forge/GNN/finetune_square/checkpoint_mp_5_finetuned_square.pt"
BAND_WIDTH_FRAC = 0.2
SKIP_BELOW_MM = 0.05  # gap control: a stroke below this is treated as a miss (no hit simulated)

# Reference palette (dataviz skill references/palette.md, light mode).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, REF_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-hits", type=int, default=50)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--checkpoint-path", default=CHECKPOINT)
    p.add_argument("--square-dir", default=SQUARE_DIR)
    p.add_argument("--target-vtu", default=None,
                   help="Surface-mesh target (points = the billet's surface nodes, point_data Displacement), e.g. "
                        "control/targets/ideal_square_10.6/target_on_billet.vtu. Default: the square run's hit-48 "
                        "state. The open-loop reference (square run replayed) is always scored against this target.")
    p.add_argument("--out-dir", default="applications/Agility_Forge/control/results/real_simulator/square_target_run")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--u-j-mm-min", type=float, default=None,
                   help="Stroke control: smallest allowed stroke (mm); default mpc.U_J_MM_MIN (0). Experiment 13 "
                        "uses 0.5 (no idle hits; the GNN predicts phantom stretch for near-zero strokes).")
    p.add_argument("--stop-tol-mm", type=float, default=None,
                   help="Enable the stop rule: stop once every check window is at most 2 * --stop-half-gap-mm + "
                        "this across at 0 and 90 deg (the data runs' finishing test). Default: off.")
    p.add_argument("--stop-half-gap-mm", type=float, default=5.3)
    p.add_argument("--penalty", default="{}",
                   help="JSON dict of MPCController penalty weights (cost-design experiments; default: none).")
    return p.parse_args()


def surface_disp(vtu_path, mesh_info):
    d = meshio.read(vtu_path).point_data["Displacement"].astype(np.float32)
    return torch.tensor(d[mesh_info.full_to_surface])


def geometry_errors(x, x_target, rest_pos):
    chamfer, hausdorff = _chamfer_hausdorff_mm(rest_pos + x_target, rest_pos + x)
    rmse = ((x - x_target) ** 2).mean().sqrt().item()
    return chamfer, hausdorff, rmse


def load_reference(square_dir, mesh_info, x_target, rest_pos, H_mm):
    """The square run's own trajectory: per-hit errors vs. its final state and
    its applied controls."""
    with open(os.path.join(square_dir, "manifest.json")) as f:
        recs = sorted((r for r in json.load(f)["records"] if r["kind"] == "hit_final"), key=lambda r: r["hit"])
    ref = {"hit": [], "chamfer_mm2": [], "hausdorff_mm": [], "d_j_mm": [], "R_j_deg": [], "u_j_mm": []}
    for r in recs:
        x = surface_disp(os.path.join(square_dir, r["vtu_path"]), mesh_info).to(rest_pos.device)
        c, h, _ = geometry_errors(x, x_target, rest_pos)
        ref["hit"].append(r["hit"])
        ref["chamfer_mm2"].append(c)
        ref["hausdorff_mm"].append(h)
        ref["d_j_mm"].append(r["d_j_frac"] * H_mm)
        ref["R_j_deg"].append(r["R_j_deg"])
        ref["u_j_mm"].append(r["u_j_mm"])
    return ref


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")
    ax.tick_params(colors=INK_2)


def _legend(ax):
    leg = ax.legend(frameon=True, fontsize=9, labelcolor=INK_2)
    leg.get_frame().set_edgecolor(GRID)


def make_plots(out_dir, steps, ref, n_hits):
    if not steps:
        return
    hits = [s["step"] for s in steps]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), facecolor=SURFACE)
    for ax, key, color, title, ylabel in [
        (axes[0], "hausdorff_mm", BLUE, "Hausdorff Error", "Hausdorff error (mm)"),
        (axes[1], "chamfer_mm2", ORANGE, "Chamfer Error", "Chamfer error (mm²)"),
    ]:
        ax.plot(ref["hit"], ref[key], ":", color=REF_GRAY, linewidth=2, label="Open-loop (square run replayed)")
        ax.plot(hits, [s[key] for s in steps], "-", color=color, linewidth=2, label="Closed-loop GNN-MPC")
        _style(ax, title, "Hit", ylabel)
        ax.set_ylim(bottom=0)
        _legend(ax)
    axes[2].plot(hits, [s["plan_time_s"] for s in steps], "-", color=AQUA, linewidth=2, label="Closed-loop GNN-MPC")
    _style(axes[2], "SQP Planning Time", "Hit", "Planning time per step (s)")
    axes[2].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlim(0, n_hits + 1)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_vs_hit.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    for fname, key, title, ylabel, fold, cap in [
        ("controls_station.png", "d_j_mm", "Station Position", "Die band start (mm from x = 0)", False, None),
        ("controls_angle.png", "R_j_deg", "Hit Angle", "Angle, folded to [0, 180) (deg)", True, None),
        ("controls_stroke.png", "u_j_mm", "Stroke", "Stroke per die (mm)", False, 2.0),
    ]:
        f = (lambda v: v % 180.0) if fold else (lambda v: v)
        fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
        ax.plot(ref["hit"], [f(v) for v in ref[key]], ":", color=REF_GRAY, linewidth=2,
                marker="o", markersize=3, label="Square run's own schedule")
        ax.plot(hits, [f(s[key]) for s in steps], "-", color=BLUE, linewidth=2,
                marker="o", markersize=3, label="Closed-loop GNN-MPC")
        if cap is not None:
            ax.axhline(cap, color=INK_2, linewidth=0.8, linestyle="--")
            ax.text(n_hits + 0.5, cap, "2 mm cap", color=INK_2, fontsize=8, va="bottom", ha="right")
        _style(ax, title, "Hit", ylabel)
        ax.set_xlim(0, n_hits + 1)
        if fold:
            ax.set_ylim(-5, 185)
            ax.set_yticks([0, 45, 90, 135, 180])
        _legend(ax)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=150, facecolor=SURFACE)
        plt.close(fig)


def save_state(path, state):
    tmp = path + ".tmp.npz"
    np.savez(tmp, sol_u=np.asarray(state.sol_u), sol_dT=np.asarray(state.sol_dT),
             **{f"iv_{k}": np.asarray(v) for k, v in enumerate(state.int_vars)})
    os.replace(tmp, path)


def load_state(path):
    z = np.load(path)
    return ForgingState(jnp.array(z["sol_u"]), jnp.array(z["sol_dT"]),
                        [jnp.array(z[f"iv_{k}"]) for k in range(N_INT_VARS)])


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    mesh_info = build_surface_mesh_info(os.path.join(args.square_dir, "rollout_01", "undeformed.vtu"))
    rest_pos = mesh_info.rest_pos.to(device)
    H_mm = float(mesh_info.rest_pos[:, 0].max().item())
    with open(os.path.join(args.square_dir, "manifest.json")) as f:
        final_hit = max(r["hit"] for r in json.load(f)["records"])
    if args.target_vtu:
        target_vtu = args.target_vtu
        t = meshio.read(target_vtu)
        if t.points.shape[0] != mesh_info.rest_pos.shape[0] or \
                np.abs(t.points - mesh_info.rest_pos.numpy()).max() > 1e-4:
            raise ValueError(f"{target_vtu} is not on this billet's surface mesh (points must match rest_pos)")
        x_target = torch.tensor(t.point_data["Displacement"], dtype=torch.float32).to(device)
    else:
        target_vtu = os.path.join(args.square_dir, "rollout_01", f"hit_{final_hit:02d}_final.vtu")
        x_target = surface_disp(target_vtu, mesh_info).to(device)
    save_surface_vtu(mesh_info, x_target.cpu().numpy(), os.path.join(args.out_dir, "target.vtu"))

    # Stop-rule windows start where the target's square section starts, mapped
    # back to the original bar (the target's lengthwise map is monotonic).
    stop_x_min = X_800C_MM
    spec_path = os.path.join(os.path.dirname(target_vtu), "target_spec.json")
    if args.target_vtu and os.path.exists(spec_path):
        with open(spec_path) as f:
            sq_start = json.load(f)["square_section_mm"][0]
        rest_x = mesh_info.rest_pos[:, 0].numpy()
        order = np.argsort(rest_x)
        tgt_x = rest_x[order] + x_target[:, 0].cpu().numpy()[order]
        stop_x_min = float(np.interp(sq_start, tgt_x, rest_x[order]))

    ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = ForgeGNN(mesh_info, latent_size=a["latent_size"], num_layers=a["num_layers"],
                     message_passing_steps=a["message_passing_steps"])
    model.load_state_dict(ckpt["state_dict"])
    d_j_bounds = (X_800C_MM / H_mm, D_J_MAX)
    control = a.get("control", "stroke")
    mpc = MPCController(model, mesh_info, horizon=args.horizon, target_state=x_target,
                        band_width_frac=BAND_WIDTH_FRAC, device=device, d_j_bounds=d_j_bounds,
                        penalty=json.loads(args.penalty), control=control,
                        **({"u_j_mm_min": args.u_j_mm_min} if args.u_j_mm_min is not None else {}))
    print(f"Control input: {control}")

    ref = load_reference(args.square_dir, mesh_info, x_target, rest_pos, H_mm)

    with _locked(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "msh", "jax_forge", "tmp")):
        mesh, R, H, T_linear_fn = load_default_billet()
    plant = ForgingPlant(mesh, float(R), float(H), T_linear_fn)

    results_path = os.path.join(args.out_dir, "results.json")
    state_path = os.path.join(args.out_dir, "plant_state.npz")
    if os.path.exists(results_path) and os.path.exists(state_path):
        with open(results_path) as f:
            results = json.load(f)
        state = load_state(state_path)
        u_init = np.asarray(results["next_u_init"]) if results.get("next_u_init") else None
        u_prev = np.asarray(results["last_applied_u"]) if results.get("last_applied_u") else None
        print(f"Resuming after hit {len(results['steps'])}")
    else:
        state = plant.reset()
        x0 = extract_surface_state(state.sol_u, mesh_info).to(device)
        c0, h0, r0 = geometry_errors(x0, x_target, rest_pos)
        results = {"config": {**vars(args), "target_vtu": target_vtu, "d_j_bounds": d_j_bounds,
                              "u_j_mm_range": [mpc.u_j_mm_min, mpc.u_j_mm_max], "H_mm": H_mm,
                              "model_args": {k: a[k] for k in ("latent_size", "num_layers", "message_passing_steps")},
                              "control": control},
                   "undeformed_vs_target": {"chamfer_mm2": c0, "hausdorff_mm": h0, "rmse_mm": r0},
                   "reference": ref, "steps": [], "next_u_init": None, "last_applied_u": None}
        results["config"]["penalty_weights"] = mpc.penalty
        if args.stop_tol_mm is not None:
            results["config"]["stop_rule"] = {
                "half_gap_mm": args.stop_half_gap_mm, "tol_mm": args.stop_tol_mm,
                "max_across_mm": 2 * args.stop_half_gap_mm + args.stop_tol_mm,
                "stations_d_j": [float(d) for d in SQUARE_STATIONS_D_J], "window_x_min_mm": stop_x_min}
        if control == "gap":
            results["config"].update(gap_bounds_mm=list(GAP_BOUNDS_MM), travel_cap_mm=TRAVEL_CAP_MM,
                                     skip_below_mm=SKIP_BELOW_MM)
        save_surface_vtu(mesh_info, x0.cpu().numpy(), os.path.join(args.out_dir, "step_00.vtu"))
        u_init = None
        u_prev = None

    if results.get("stopped"):
        print(f"Run already stopped by the stop rule after hit {results['stopped']['hit']}.")
        return
    while len(results["steps"]) < args.n_hits:
        step = len(results["steps"]) + 1
        x0 = extract_surface_state(state.sol_u, mesh_info).to(device)
        t0 = time.perf_counter()
        u_seq, opt = mpc.plan(x0, horizon=args.horizon, u_init=u_init, u_prev=u_prev)
        plan_time = time.perf_counter() - t0
        d_j_frac, R_j_deg, u_j_frac = (float(v) for v in u_seq[0])
        d_mm, x_hi_mm = d_j_frac * H_mm, (d_j_frac + BAND_WIDTH_FRAC) * H_mm
        gap_info, skipped = {}, False
        if control == "gap":
            half_gap = GAP_BOUNDS_MM[0] + u_j_frac * (GAP_BOUNDS_MM[1] - GAP_BOUNDS_MM[0])
            with torch.no_grad():
                ht = float(band_half_thickness(mesh_info, x0, d_mm, x_hi_mm, R_j_deg))
            u_j_mm = min(max(ht - half_gap, 0.0), TRAVEL_CAP_MM)
            skipped = u_j_mm < SKIP_BELOW_MM
            gnn_ctrl = gap_frac(half_gap)
            gap_info = {"half_gap_mm": half_gap, "band_half_thickness_mm": ht, "skipped": skipped,
                        "planned_travel_mm": opt.planned_travel_mm[0]}
        else:
            u_j_mm = mpc.u_j_mm_min + u_j_frac * (mpc.u_j_mm_max - mpc.u_j_mm_min)
            gnn_ctrl = gnn_u_frac(u_j_mm)

        with torch.no_grad():
            nf = build_node_features(mesh_info, x0, d_j_frac, d_mm, x_hi_mm, R_j_deg, gnn_ctrl).unsqueeze(0)
            predicted = x0 + mpc.model.predict_delta(nf, x0.unsqueeze(0), accumulate=False)[0]

        hit = Hit(x_min_band=d_j_frac, x_max_band=d_j_frac + BAND_WIDTH_FRAC, compression_displacement=u_j_mm,
                  rotation_euler_x=R_j_deg, total_time=TOTAL_TIME_S)
        print(f"\n=== Hit {step}/{args.n_hits}: d_j={d_j_frac:.4f} ({d_j_frac * H_mm:.1f} mm) R={R_j_deg:.1f} "
              f"u={u_j_mm:.3f} mm{' (half-gap %.3f mm; SKIPPED)' % gap_info['half_gap_mm'] if skipped else ''} | "
              f"plan {plan_time:.1f}s, SQP nit={opt.nit} success={opt.success}", flush=True)
        t1 = time.perf_counter()
        try:
            if not skipped:
                # First real hit = no earlier hit actually simulated (gap control can skip hits).
                n_real = sum(not s.get("skipped", False) for s in results["steps"])
                state = plant.step(state, hit, is_first_hit=(n_real == 0))
        except Exception as exc:
            with open(os.path.join(args.out_dir, "failed_steps.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "d_j_frac": d_j_frac, "R_j_deg": R_j_deg, "u_j_mm": u_j_mm,
                                    "error": repr(exc), "traceback": traceback.format_exc(),
                                    "timestamp": datetime.now(timezone.utc).isoformat()}) + "\n")
            print(f"[FAILED] plant step {step}: {exc!r}. Stopping; plots cover hits 1-{step - 1}.", flush=True)
            make_plots(args.out_dir, results["steps"], ref, args.n_hits)
            raise SystemExit(1)
        plant_time = time.perf_counter() - t1

        x1 = extract_surface_state(state.sol_u, mesh_info).to(device)
        chamfer, hausdorff, rmse = geometry_errors(x1, x_target, rest_pos)
        results["steps"].append({
            "step": step, "d_j_frac": d_j_frac, "d_j_mm": d_j_frac * H_mm, "R_j_deg": R_j_deg, "u_j_mm": u_j_mm,
            "plan_time_s": plan_time, "plant_time_s": plant_time, "sqp_nit": int(opt.nit),
            "sqp_success": bool(opt.success), "sqp_message": str(opt.message), "plan_cost": float(opt.fun),
            "error_cost": opt.error_cost, "penalty_cost": opt.penalty_cost,
            "chamfer_mm2": chamfer, "hausdorff_mm": hausdorff, "rmse_to_target_mm": rmse,
            "gnn_pred_vs_actual_rmse_mm": ((predicted - x1) ** 2).mean().sqrt().item(),
            **gap_info,
        })
        if args.stop_tol_mm is not None:
            across = across_by_station(mesh, float(R), float(H), np.asarray(state.sol_u), SQUARE_STATIONS_D_J,
                                       stop_x_min)
            results["steps"][-1]["across_by_station_mm"] = {str(k): v for k, v in across.items()}
            results["steps"][-1]["max_across_0_90_mm"] = [max(v[0] for v in across.values()),
                                                          max(v[1] for v in across.values())]
        u_init = np.concatenate([u_seq[1:], u_seq[-1:]]).flatten()
        results["next_u_init"] = u_init.tolist()
        u_prev = u_seq[0]
        results["last_applied_u"] = u_prev.tolist()
        save_surface_vtu(mesh_info, x1.cpu().numpy(), os.path.join(args.out_dir, f"step_{step:02d}.vtu"))
        save_state(state_path, state)
        write_json(results_path, results)
        make_plots(args.out_dir, results["steps"], ref, args.n_hits)
        print(f"    -> Hausdorff {hausdorff:.3f} mm, Chamfer {chamfer:.4f} mm^2, RMSE {rmse:.3f} mm | "
              f"plant {plant_time:.0f}s", flush=True)
        if args.stop_tol_mm is not None:
            print(f"    stop rule: widest window {results['steps'][-1]['max_across_0_90_mm'][0]:.2f} mm at 0 deg, "
                  f"{results['steps'][-1]['max_across_0_90_mm'][1]:.2f} mm at 90 deg "
                  f"(limit {2 * args.stop_half_gap_mm + args.stop_tol_mm:.2f})", flush=True)
            if within_tolerance(across, args.stop_half_gap_mm, args.stop_tol_mm):
                results["stopped"] = {"hit": step, "reason": "part within tolerance (stop rule)"}
                write_json(results_path, results)
                print(f"\nSTOP RULE MET after hit {step}: every window within "
                      f"{2 * args.stop_half_gap_mm + args.stop_tol_mm:.2f} mm at 0 and 90 deg.", flush=True)
                break

    print(f"\nDone: {len(results['steps'])} hits. Final Hausdorff {results['steps'][-1]['hausdorff_mm']:.3f} mm, "
          f"Chamfer {results['steps'][-1]['chamfer_mm2']:.4f} mm^2. Results -> {args.out_dir}")


if __name__ == "__main__":
    main(parse_args())
