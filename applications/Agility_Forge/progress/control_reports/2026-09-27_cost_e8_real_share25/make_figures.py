"""Experiment 8 (25% cross-section share, no effort, real simulator, ended at
hit 32 by a simulator failure):
  width_along_bar.png -- thickness at 0 deg (y) and width at 90 deg (z) along
    the bar after hit 27 (before the clamp-end run of hits) and hit 32 (the
    last completed), vs. the ideal target
  gnn_check.json -- for hits 26-32: the M = 3 GNN's one-hit prediction from
    the TRUE pre-hit state with the applied control, vs. what the simulator
    did: band thickness/width (x = 20-35 mm) and free-end stretch
  min det F of the final saved state (worst element volume ratio)

Usage (repo root):
    PYTHONPATH=. JAX_PLATFORMS=cpu python applications/Agility_Forge/progress/control_reports/2026-09-27_cost_e8_real_share25/make_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
import torch

from applications.Agility_Forge.GNN.data import build_node_features, build_surface_mesh_info
from applications.Agility_Forge.GNN.model import ForgeGNN
from applications.Agility_Forge.control.mpc import gnn_u_frac

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = "applications/Agility_Forge/control/results/real_simulator/e8_ideal_share25"
CKPT = "applications/Agility_Forge/GNN/mp_sweep/finetune/mp_3/checkpoint.pt"
BILLET = "applications/Agility_Forge/data/dataset_finetuning/square/rollout_01/undeformed.vtu"
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


def disp(step):
    return meshio.read(os.path.join(RUN, f"step_{step:02d}.vtu")).point_data["Displacement"]


def main():
    with open(os.path.join(RUN, "results.json")) as f:
        res = json.load(f)
    steps, H = res["steps"], res["config"]["H_mm"]
    last = steps[-1]["step"]
    t = meshio.read(os.path.join(RUN, "target.vtu"))
    rest, xt = t.points, t.point_data["Displacement"]
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

    # Width along the bar, both press directions.
    bins = np.arange(-5.0, 97.0, 5.0)
    centers = 0.5 * (bins[:-1] + bins[1:])

    def widths(u, axis):
        pos = rest + u
        return [np.ptp(pos[(rest[:, 0] >= a) & (rest[:, 0] < b), axis]) for a, b in zip(bins[:-1], bins[1:])]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), facecolor=SURFACE)
    for ax, axis, name in [(axes[0], 1, "Thickness at 0° (y)"), (axes[1], 2, "Width at 90° (z)")]:
        ax.plot(centers, widths(xt, axis), "--", color=INK, linewidth=2, marker="o", markersize=4, label="Ideal target")
        ax.plot(centers, widths(disp(27), axis), "-", color=BLUE, linewidth=2, marker="o", markersize=4,
                label="After hit 27")
        ax.plot(centers, widths(disp(last), axis), "-", color=ORANGE, linewidth=2, marker="o", markersize=4,
                label=f"After hit {last} (last completed)")
        ax.axvspan(17.7, 17.7 + 0.2 * H, color=GRID, alpha=0.6, linewidth=0)
        ax.text(18.5, 7.2, "die band,\nhits 28-32", color=INK_2, fontsize=8)
        style(ax, name, "Position along the original bar (mm from x = 0)", "Across-flats size (mm)")
        ax.set_ylim(6, 27)
        leg = ax.legend(frameon=True, fontsize=9, labelcolor=INK_2, loc="upper right")
        leg.get_frame().set_edgecolor(GRID)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "width_along_bar.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # GNN one-hit predictions vs. the simulator, hits 26-last.
    mi = build_surface_mesh_info(BILLET)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = ForgeGNN(mi, latent_size=a["latent_size"], num_layers=a["num_layers"],
                     message_passing_steps=a["message_passing_steps"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    band = (rest[:, 0] >= 20.0) & (rest[:, 0] < 35.0)

    def yz(u):
        p = rest[band] + u[band]
        return float(np.ptp(p[:, 1])), float(np.ptp(p[:, 2]))

    rows = []
    for h in range(26, last + 1):
        s = steps[h - 1]
        x0, x1 = disp(h - 1), disp(h)
        df = s["d_j_frac"]
        with torch.no_grad():
            x0t = torch.tensor(x0)
            nf = build_node_features(mi, x0t, df, df * H, (df + 0.2) * H, s["R_j_deg"],
                                     gnn_u_frac(s["u_j_mm"])).unsqueeze(0)
            xp = (x0t + model.predict_delta(nf, x0t.unsqueeze(0), accumulate=False)[0]).numpy()
        rows.append({"hit": h, "d_j_mm": s["d_j_mm"], "R_j_deg": s["R_j_deg"], "u_j_mm": s["u_j_mm"],
                     "band_yz_before_mm": yz(x0), "band_yz_gnn_mm": yz(xp), "band_yz_actual_mm": yz(x1),
                     "stretch_gnn_mm": float(xp[:, 0].max() - x0[:, 0].max()),
                     "stretch_actual_mm": float(x1[:, 0].max() - x0[:, 0].max())})

    z = np.load(os.path.join(RUN, "plant_state.npz"))
    J = np.linalg.det(z["iv_0"].reshape(-1, 3, 3))
    with open(os.path.join(HERE, "gnn_check.json"), "w", encoding="utf-8") as f:
        json.dump({"band_x_mm": [20.0, 35.0], "hits": rows,
                   "final_state_det_F": {"min": float(J.min()), "p1": float(np.percentile(J, 1)),
                                         "n_below_0.8": int((J < 0.8).sum()), "n": int(J.size)}}, f, indent=2)
    print("-> figures/width_along_bar.png, gnn_check.json")


if __name__ == "__main__":
    main()
