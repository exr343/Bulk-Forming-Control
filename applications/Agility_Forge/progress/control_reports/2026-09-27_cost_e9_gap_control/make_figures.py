"""Experiment 9 (gap-control GNN, same cost and target as experiment 8, real
simulator, 50 hits):
  errors_and_time.png -- Hausdorff, Chamfer and planning time per hit:
    experiment 9 vs. experiment 8 (32 hits) vs. the open-loop square run
  controls.png -- station, angle (folded to [0, 180)) and applied die travel
    per hit; skipped hits (dies missed the bar) marked
  gap_scan.png -- at the state before hit 33 (a skipped hit): the GNN's one-
    hit cost and predicted free-end stretch vs. the half-gap, same station
    and angle as applied
  width_along_bar.png -- thickness at 0 deg and width at 90 deg along the bar
    after hit 50 vs. the target, the billet and the open loop's hit 48
  summary.json -- the numbers behind the report

Usage (repo root):
    PYTHONPATH=. JAX_PLATFORMS=cpu python applications/Agility_Forge/progress/control_reports/2026-09-27_cost_e9_gap_control/make_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
import torch

from applications.Agility_Forge.GNN.data import (
    band_half_thickness, build_node_features, build_surface_mesh_info, gap_frac)
from applications.Agility_Forge.GNN.model import ForgeGNN

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = "applications/Agility_Forge/control/results/real_simulator/e9_ideal_share25_gap"
E8 = "applications/Agility_Forge/control/results/real_simulator/e8_ideal_share25"
SQUARE = "applications/Agility_Forge/data/dataset_finetuning/square"
CKPT = "applications/Agility_Forge/GNN/gap_control/gap_nochange10/finetune/checkpoint.pt"
W_TRANS = 91.37195
SCAN_HIT = 33
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, REF_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"


def style(ax, title, xlabel, ylabel):
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


def legend(ax, loc="best"):
    leg = ax.legend(frameon=True, fontsize=9, labelcolor=INK_2, loc=loc)
    leg.get_frame().set_edgecolor(GRID)


def main():
    with open(os.path.join(RUN, "results.json")) as f:
        res = json.load(f)
    with open(os.path.join(E8, "results.json")) as f:
        e8 = json.load(f)["steps"]
    steps, ref, H = res["steps"], res["reference"], res["config"]["H_mm"]
    hits = [s["step"] for s in steps]
    skipped = [s["step"] for s in steps if s["skipped"]]
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

    # Errors and planning time.
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), facecolor=SURFACE)
    for ax, key, title, ylabel in [(axes[0], "hausdorff_mm", "Hausdorff Error to Ideal Target", "Hausdorff error (mm)"),
                                   (axes[1], "chamfer_mm2", "Chamfer Error to Ideal Target", "Chamfer error (mm²)")]:
        ax.plot(ref["hit"], ref[key], ":", color=REF_GRAY, linewidth=2, label="Open loop: square run replayed")
        ax.plot([s["step"] for s in e8], [s[key] for s in e8], "-", color=ORANGE, linewidth=2,
                label="Experiment 8 (stroke control; ended at hit 32)")
        ax.plot(hits, [s[key] for s in steps], "-", color=BLUE, linewidth=2, label="Experiment 9 (gap control)")
        ax.scatter(skipped, [steps[h - 1][key] for h in skipped], s=18, color=BLUE, marker="x", zorder=3,
                   label="Experiment 9: skipped hit")
        style(ax, title, "Hit", ylabel)
        ax.set_ylim(bottom=0)
        legend(ax, "lower left")
    axes[2].plot([s["step"] for s in e8], [s["plan_time_s"] for s in e8], "-", color=ORANGE, linewidth=2,
                 label="Experiment 8")
    axes[2].plot(hits, [s["plan_time_s"] for s in steps], "-", color=BLUE, linewidth=2, label="Experiment 9")
    style(axes[2], "MPC Planning Time per Hit", "Hit", "SQP planning time (s)")
    axes[2].set_ylim(bottom=0)
    legend(axes[2], "upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "errors_and_time.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Controls.
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4), facecolor=SURFACE)
    axes[0].plot(hits, [s["d_j_mm"] for s in steps], "-", color=BLUE, linewidth=2, marker="o", markersize=3)
    axes[0].axhline(17.7, color=INK_2, linewidth=0.8, linestyle="--")
    style(axes[0], "Station", "Hit", "Die band start (mm from x = 0)")
    axes[0].set_ylim(10, 80)
    axes[1].plot(hits, [s["R_j_deg"] % 180 for s in steps], "-", color=BLUE, linewidth=2, marker="o", markersize=3)
    for a in (0, 90, 180):
        axes[1].axhline(a, color=GRID, linewidth=1.2)
    style(axes[1], "Angle (target flats at 0° and 90°)", "Hit", "Angle, folded to [0, 180) (deg)")
    axes[1].set_ylim(-5, 185)
    axes[1].set_yticks([0, 45, 90, 135, 180])
    travel = [s["u_j_mm"] for s in steps]
    axes[2].bar(hits, travel, color=BLUE, width=0.8, label="Applied die travel")
    axes[2].scatter(skipped, [0.05] * len(skipped), color=ORANGE, marker="x", s=30, zorder=3,
                    label="Skipped (dies missed the bar)")
    axes[2].axhline(2.0, color=INK_2, linewidth=0.8, linestyle="--")
    style(axes[2], "Die Travel per Hit", "Hit", "Travel per die (mm)")
    axes[2].set_ylim(0, 2.4)
    legend(axes[2], "upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "controls.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Gap scan at the state before a skipped hit.
    mi = build_surface_mesh_info(os.path.join(SQUARE, "rollout_01", "undeformed.vtu"))
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = ForgeGNN(mi, latent_size=a["latent_size"], num_layers=a["num_layers"],
                     message_passing_steps=a["message_passing_steps"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    t = meshio.read(os.path.join(RUN, "target.vtu"))
    rest, xt_np = t.points, t.point_data["Displacement"]
    xt = torch.tensor(xt_np)
    x0 = torch.tensor(meshio.read(os.path.join(RUN, f"step_{SCAN_HIT - 1:02d}.vtu")).point_data["Displacement"])
    s = steps[SCAN_HIT - 1]
    d_mm, R = s["d_j_mm"], s["R_j_deg"]
    ht = float(band_half_thickness(mi, x0, d_mm, d_mm + 0.2 * H, R))
    gaps = np.linspace(ht - 2.0, ht + 1.0, 25)
    scan = []
    with torch.no_grad():
        for g in gaps:
            nf = build_node_features(mi, x0, d_mm / H, d_mm, d_mm + 0.2 * H, R, gap_frac(float(g))).unsqueeze(0)
            x1 = x0 + model.predict_delta(nf, x0.unsqueeze(0), accumulate=False)[0]
            J = ((x1 - xt) ** 2).sum() + W_TRANS * ((x1[:, 1:] - xt[:, 1:]) ** 2).sum()
            scan.append({"half_gap_mm": float(g), "travel_mm": ht - float(g), "one_hit_cost": float(J),
                         "predicted_stretch_mm": float(x1[:, 0].max() - x0[:, 0].max())})
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), facecolor=SURFACE)
    for ax, key, color, title, ylabel, text_frac in [
        (axes[0], "one_hit_cost", BLUE, "GNN's One-Hit Cost", "Cost J after this hit (lower is better)", 0.85),
        (axes[1], "predicted_stretch_mm", ORANGE, "GNN's Predicted Free-End Stretch", "Stretch (mm)", 0.05),
    ]:
        ax.plot([r["half_gap_mm"] for r in scan], [r[key] for r in scan], "-", color=color, linewidth=2,
                marker="o", markersize=3)
        ax.axvline(ht, color=INK_2, linewidth=1, linestyle="--")
        ax.axvspan(ht, gaps[-1], color=GRID, alpha=0.6, linewidth=0)
        ax.text(ht + 0.05, ax.get_ylim()[0] + text_frac * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                "dies miss the bar\n(real result: no change)", color=INK_2, fontsize=8)
        style(ax, title, f"Half-gap (mm); bar's half-thickness here = {ht:.2f} mm", ylabel)
    fig.suptitle(f"State before hit {SCAN_HIT} (skipped): station {d_mm:.1f} mm, angle {R:.0f}°",
                 color=INK_2, fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "gap_scan.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Width along the bar.
    bins = np.arange(-5.0, 97.0, 5.0)
    centers = 0.5 * (bins[:-1] + bins[1:])

    def widths(u, axis):
        pos = rest + u
        return [float(np.ptp(pos[(rest[:, 0] >= lo) & (rest[:, 0] < hi), axis])) for lo, hi in zip(bins[:-1], bins[1:])]

    x50 = meshio.read(os.path.join(RUN, f"step_{hits[-1]:02d}.vtu")).point_data["Displacement"]
    with open(os.path.join(SQUARE, "manifest.json")) as f:
        last = max((r for r in json.load(f)["records"] if r["kind"] == "hit_final"), key=lambda r: r["hit"])
    ol = meshio.read(os.path.join(SQUARE, last["vtu_path"])).point_data["Displacement"][mi.full_to_surface]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), facecolor=SURFACE)
    for ax, axis, name in [(axes[0], 1, "Thickness at 0° (y)"), (axes[1], 2, "Width at 90° (z)")]:
        ax.plot(centers, widths(np.zeros_like(xt_np), axis), ":", color=REF_GRAY, linewidth=2, marker="o",
                markersize=3, label="Billet")
        ax.plot(centers, widths(xt_np, axis), "--", color=INK, linewidth=2, marker="o", markersize=3,
                label="Ideal target")
        ax.plot(centers, widths(ol, axis), "-", color=AQUA, linewidth=2, marker="o", markersize=3,
                label=f"Open loop after hit {last['hit']}")
        ax.plot(centers, widths(x50, axis), "-", color=BLUE, linewidth=2, marker="o", markersize=3,
                label=f"Experiment 9 after hit {hits[-1]}")
        style(ax, name, "Position along the original bar (mm from x = 0)", "Across-flats size (mm)")
        ax.set_ylim(8, 20)
        legend(ax, "upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "width_along_bar.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    z = np.load(os.path.join(RUN, "plant_state.npz"))
    J = np.linalg.det(z["iv_0"].reshape(-1, 3, 3))
    applied = [s for s in steps if not s["skipped"]]
    pt = np.array([s["plan_time_s"] for s in steps])
    summary = {
        "final": {"hausdorff_mm": steps[-1]["hausdorff_mm"], "chamfer_mm2": steps[-1]["chamfer_mm2"],
                  "free_end_mm": float(x50[:, 0].max()), "target_free_end_mm": float(xt_np[:, 0].max())},
        "open_loop_hit48": {"hausdorff_mm": ref["hausdorff_mm"][-1], "chamfer_mm2": ref["chamfer_mm2"][-1],
                            "free_end_mm": float(ol[:, 0].max())},
        "e8_hit32": {"hausdorff_mm": e8[-1]["hausdorff_mm"], "chamfer_mm2": e8[-1]["chamfer_mm2"]},
        "e9_hit32": {"hausdorff_mm": steps[31]["hausdorff_mm"], "chamfer_mm2": steps[31]["chamfer_mm2"]},
        "n_skipped": len(skipped), "skipped_hits": skipped,
        "n_applied_below_0.6mm": sum(s["u_j_mm"] < 0.6 for s in applied),
        "n_full_travel_ge_1.9mm": sum(s["u_j_mm"] >= 1.9 for s in applied),
        "stations_mm": [s["d_j_mm"] for s in steps],
        "n_stations_38_to_54mm": sum(38.0 <= s["d_j_mm"] <= 54.0 for s in steps),
        "angles_folded_deg": [s["R_j_deg"] % 180 for s in steps],
        "plan_time_s": {"mean": float(pt.mean()), "median": float(np.median(pt)), "max": float(pt.max())},
        "n_plans_not_converged": sum(not s["sqp_success"] for s in steps),
        "plant_time_min_mean_applied": float(np.mean([s["plant_time_s"] for s in applied]) / 60),
        "gnn_pred_vs_actual_rmse_mm": {"skipped_mean": float(np.mean([steps[h - 1]["gnn_pred_vs_actual_rmse_mm"]
                                                                      for h in skipped])),
                                       "applied_mean": float(np.mean([s["gnn_pred_vs_actual_rmse_mm"]
                                                                      for s in applied]))},
        "gap_scan": {"hit": SCAN_HIT, "station_mm": d_mm, "angle_deg": R, "band_half_thickness_mm": ht,
                     "points": scan},
        "final_state_det_F": {"min": float(J.min()), "p1": float(np.percentile(J, 1))},
    }
    with open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("-> figures/errors_and_time.png, controls.png, gap_scan.png, width_along_bar.png, summary.json")


if __name__ == "__main__":
    main()
