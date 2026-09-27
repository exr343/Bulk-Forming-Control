"""Absolute gap control vs. relative stroke, M = 3 GNN (all pretrain-then-
finetune, same data, splits and seeds):
  stroke     -- GNN/mp_sweep/finetune/mp_3 (the model experiments 7-8 used)
  gap only   -- GNN/gap_control/gap_only
  gap + 10% no-change -- GNN/gap_control/gap_nochange10 (experiment 9's model)

Outputs (in this folder):
  figures/rollout_seed3.png -- Hausdorff and Chamfer per hit of the 48-hit
    autoregressive rollout on the held-out square run (seed 3)
  figures/no_contact.png -- predicted change for hits that miss the bar,
    from every true pre-hit state of seed 3
  figures/e8_replay.png -- experiment 8's hits 26-32 replayed from the
    simulator's true states: band thickness before, predicted, actual
  summary.json -- all numbers behind the report

"Miss" for the stroke model = stroke 0 (the closest it can express); for the
gap models = a half-gap 0.5 mm above the band half-thickness.

Usage (repo root):
    PYTHONPATH=. JAX_PLATFORMS=cpu python applications/Agility_Forge/progress/GNN_reports/2026-09-27_gap_control/make_figures.py
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
    ForgeGNNDataset, band_half_thickness, build_node_features, build_surface_mesh_info, gap_frac, load_examples)
from applications.Agility_Forge.GNN.model import ForgeGNN
from applications.Agility_Forge.control.mpc import gnn_u_frac

HERE = os.path.dirname(os.path.abspath(__file__))
GNN = "applications/Agility_Forge/GNN"
MODELS = {
    "Stroke (current)": (f"{GNN}/mp_sweep/finetune/mp_3", "stroke"),
    "Gap only": (f"{GNN}/gap_control/gap_only/finetune", "gap"),
    "Gap + 10% no-change": (f"{GNN}/gap_control/gap_nochange10/finetune", "gap"),
}
SEED3 = "applications/Agility_Forge/data/dataset_finetuning/square_jitter_seed3"
E8 = "applications/Agility_Forge/control/results/real_simulator/e8_ideal_share25"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
REF_GRAY = "#8a8984"
MISS_MARGIN_MM = 0.5


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


def legend(ax, loc="upper left"):
    leg = ax.legend(frameon=True, fontsize=9, labelcolor=INK_2, loc=loc)
    leg.get_frame().set_edgecolor(GRID)


def load_model(run_dir, mi):
    ck = torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location="cpu", weights_only=False)
    a = ck["args"]
    m = ForgeGNN(mi, latent_size=a["latent_size"], num_layers=a["num_layers"],
                 message_passing_steps=a["message_passing_steps"])
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def predict(model, control, mi, x0, d_mm, H, R_deg, u_mm=None, half_gap=None):
    df = d_mm / H
    ctrl = gnn_u_frac(u_mm) if control == "stroke" else gap_frac(half_gap)
    with torch.no_grad():
        nf = build_node_features(mi, x0, df, d_mm, d_mm + 0.2 * H, R_deg, ctrl).unsqueeze(0)
        return x0 + model.predict_delta(nf, x0.unsqueeze(0), accumulate=False)[0]


def main():
    mi = build_surface_mesh_info(os.path.join(SEED3, "rollout_01", "undeformed.vtu"))
    H = float(mi.rest_pos[:, 0].max())
    rest = mi.rest_pos.numpy()
    models = {name: (load_model(d, mi), c) for name, (d, c) in MODELS.items()}
    summary = {"models": {n: d for n, (d, _) in MODELS.items()}}
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

    # 1. Held-out metrics + rollout curves (from each run's metrics.json).
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), facecolor=SURFACE)
    summary["seed3"] = {}
    for (name, (d, _)), color in zip(MODELS.items(), COLORS):
        with open(os.path.join(d, "metrics.json")) as f:
            met = json.load(f)
        ft, roll = met["finetuned"], met["finetuned"]["seed3_rollout"]
        summary["seed3"][name] = {
            "one_hit_nrmse": ft["test"]["nrmse"], "one_hit_chamfer_mm2": ft["test"]["chamfer_mm2"],
            "one_hit_hausdorff_mm": ft["test"]["hausdorff_mm"], "pretrain_test_nrmse": ft["pretrain_test"]["nrmse"],
            "rollout_hit48_hausdorff_mm": roll["per_hit_hausdorff_mm"][-1],
            "rollout_hit48_chamfer_mm2": roll["per_hit_chamfer_mm2"][-1],
            "rollout_mean_hausdorff_mm": float(np.mean(roll["per_hit_hausdorff_mm"])),
            "rollout_mean_chamfer_mm2": float(np.mean(roll["per_hit_chamfer_mm2"])),
            "rollout_hit48_rmse_mm": roll["per_hit_rmse_mm"][-1], "finetune_best_epoch": met["best_epoch"]}
        hits = range(1, len(roll["per_hit_hausdorff_mm"]) + 1)
        axes[0].plot(hits, roll["per_hit_hausdorff_mm"], color=color, linewidth=2, label=name)
        axes[1].plot(hits, roll["per_hit_chamfer_mm2"], color=color, linewidth=2, label=name)
    xl = "Hit (48-hit autoregressive rollout, test run seed 3)"
    style(axes[0], "Test Hausdorff (GNN vs. simulator)", xl, "Hausdorff error (mm)")
    style(axes[1], "Test Chamfer (GNN vs. simulator)", xl, "Chamfer error (mm²)")
    for ax in axes:
        ax.set_ylim(bottom=0)
        legend(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "rollout_seed3.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # 2. Missed hits: from every true pre-hit state of seed 3, same station
    # and angle as the real hit, dies not reaching the bar.
    exs, _ = load_examples(SEED3)
    ds = ForgeGNNDataset(SEED3, exs, mi, control="gap")
    miss = {n: {"mean_abs_mm": [], "max_abs_mm": [], "stretch_mm": []} for n in models}
    for ex in ds.examples:
        x0 = ds._load_surface_displacement(ex.x_k_vtu)
        ht = ex.half_gap_mm + ex.u_j_mm
        for name, (model, control) in models.items():
            xp = predict(model, control, mi, x0, ex.d_j_mm, H, ex.R_j_deg, u_mm=0.0, half_gap=ht + MISS_MARGIN_MM)
            dx = (xp - x0).numpy()
            miss[name]["mean_abs_mm"].append(float(np.abs(dx).mean()))
            miss[name]["max_abs_mm"].append(float(np.abs(dx).max()))
            miss[name]["stretch_mm"].append(float(xp[:, 0].max() - x0[:, 0].max()))
    summary["no_contact"] = {n: {k: {"mean": float(np.mean(v)), "max": float(np.max(v))} for k, v in d.items()}
                             for n, d in miss.items()}
    summary["no_contact"]["n_states"] = len(ds.examples)
    summary["no_contact"]["miss_margin_mm"] = MISS_MARGIN_MM
    # Reference scale: the real hits' own mean |displacement change|.
    real = [float(np.abs((ds._load_surface_displacement(e.x_next_vtu) - ds._load_surface_displacement(e.x_k_vtu))
                         .numpy()).max()) for e in ds.examples]
    summary["no_contact"]["real_hits_max_abs_mm_mean"] = float(np.mean(real))

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), facecolor=SURFACE)
    hits = np.arange(1, len(ds.examples) + 1)
    for (name, d), color in zip(miss.items(), COLORS):
        axes[0].plot(hits, d["max_abs_mm"], color=color, linewidth=2, label=name)
        axes[1].plot(hits, d["stretch_mm"], color=color, linewidth=2, label=name)
    axes[0].plot(hits, real, ":", color=REF_GRAY, linewidth=2, label="Real hit at that state (for scale)")
    style(axes[0], "Missed Hit: Largest Predicted Node Movement", "Pre-hit state (seed 3, hit number)",
          "Max |predicted change| (mm); should be 0")
    style(axes[1], "Missed Hit: Predicted Free-End Stretch", "Pre-hit state (seed 3, hit number)",
          "Predicted stretch (mm); should be 0")
    axes[1].axhline(0, color=INK_2, linewidth=0.8)
    axes[0].set_ylim(-0.05, 2.9)  # headroom so the legend clears the data
    axes[1].set_ylim(-0.1, 1.45)
    legend(axes[0], loc="upper right")
    legend(axes[1], loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "no_contact.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # 3. Experiment 8 replay, hits 26-32: each model's one-hit prediction
    # from the simulator's true state, with the hit actually applied
    # (converted to the half-gap for the gap models).
    with open(os.path.join(E8, "results.json")) as f:
        steps = json.load(f)["steps"]
    band = (rest[:, 0] >= 20.0) & (rest[:, 0] < 35.0)

    def disp(h):
        return torch.tensor(meshio.read(os.path.join(E8, f"step_{h:02d}.vtu")).point_data["Displacement"])

    def thick(x, R_deg):
        p = rest[band] + x.numpy()[band]
        a = np.deg2rad(R_deg)
        return float(np.ptp(np.cos(a) * p[:, 1] - np.sin(a) * p[:, 2]))

    rows = []
    for h in range(26, len(steps) + 1):
        s = steps[h - 1]
        x0, x1 = disp(h - 1), disp(h)
        x_hi = s["d_j_mm"] + 0.2 * H
        ht = float(band_half_thickness(mi, x0, s["d_j_mm"], x_hi, s["R_j_deg"]))
        row = {"hit": h, "d_j_mm": s["d_j_mm"], "R_j_deg": s["R_j_deg"], "u_j_mm": s["u_j_mm"],
               "half_gap_mm": ht - s["u_j_mm"], "thickness_before_mm": thick(x0, s["R_j_deg"]),
               "thickness_actual_mm": thick(x1, s["R_j_deg"]),
               "stretch_actual_mm": float(x1[:, 0].max() - x0[:, 0].max()), "predicted": {}}
        for name, (model, control) in models.items():
            xp = predict(model, control, mi, x0, s["d_j_mm"], H, s["R_j_deg"], u_mm=s["u_j_mm"],
                         half_gap=ht - s["u_j_mm"])
            row["predicted"][name] = {"thickness_mm": thick(xp, s["R_j_deg"]),
                                      "stretch_mm": float(xp[:, 0].max() - x0[:, 0].max()),
                                      "rmse_vs_actual_mm": float(((xp - x1) ** 2).mean().sqrt())}
        rows.append(row)
    summary["e8_replay"] = {"band_x_mm": [20.0, 35.0],
                            "thickness_note": "extent along the press direction of that hit", "hits": rows}

    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor=SURFACE)
    hs = [r["hit"] for r in rows]
    ax.plot(hs, [r["thickness_before_mm"] for r in rows], ":", color=REF_GRAY, linewidth=2, marker="o",
            markersize=4, label="Before the hit")
    ax.plot(hs, [r["thickness_actual_mm"] for r in rows], "-", color=INK, linewidth=2.5, marker="o",
            markersize=5, label="After the hit: simulator")
    for name, color in zip(models, COLORS):
        ax.plot(hs, [r["predicted"][name]["thickness_mm"] for r in rows], "--", color=color, linewidth=2,
                marker="o", markersize=4, label=f"After the hit: {name}")
    style(ax, "Experiment 8's Failing Hits, Replayed", "Hit (experiment 8)",
          "Band thickness along the press direction (mm)")
    legend(ax, loc="lower left")
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "e8_replay.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    with open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("-> figures/rollout_seed3.png, figures/no_contact.png, figures/e8_replay.png, summary.json")


if __name__ == "__main__":
    main()
