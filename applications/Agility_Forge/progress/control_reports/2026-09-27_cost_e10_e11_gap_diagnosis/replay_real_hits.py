"""Replay the real hit sequences of experiments 8 (32 hits) and 9 (50 hits)
through both M = 3 GNNs (stroke-trained, gap + no-change-trained), each
autoregressively from the billet, and compare with the real simulator's
states: which GNN tracks reality on real MPC trajectories?

The stroke GNN gets each hit's applied stroke; the gap GNN gets the half-gap
= (band half-thickness of ITS OWN predicted state) - applied stroke. Skipped
hits (experiment 9) leave both unchanged. Writes replay_real_hits.json and
figures/replay_real_hits.png.

Usage (repo root):
    PYTHONPATH=. JAX_PLATFORMS=cpu python applications/Agility_Forge/progress/control_reports/2026-09-27_cost_e10_e11_gap_diagnosis/replay_real_hits.py
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
from applications.Agility_Forge.GNN.train import _chamfer_hausdorff_mm
from applications.Agility_Forge.control.mpc import gnn_u_frac

HERE = os.path.dirname(os.path.abspath(__file__))
RS = "applications/Agility_Forge/control/results/real_simulator"
RUNS = {"Experiment 8 (stroke MPC)": f"{RS}/e8_ideal_share25", "Experiment 9 (gap MPC)": f"{RS}/e9_ideal_share25_gap"}
MODELS = {"Stroke GNN": ("applications/Agility_Forge/GNN/mp_sweep/finetune/mp_3/checkpoint.pt", "stroke"),
          "Gap + no-change GNN": ("applications/Agility_Forge/GNN/gap_control/gap_nochange10/finetune/checkpoint.pt",
                                  "gap")}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"Stroke GNN": "#2a78d6", "Gap + no-change GNN": "#eb6834"}


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")
    ax.tick_params(colors=INK_2)


def main():
    mi = build_surface_mesh_info("applications/Agility_Forge/data/dataset_finetuning/square/rollout_01/undeformed.vtu")
    H = float(mi.rest_pos[:, 0].max())
    rest = mi.rest_pos
    models = {}
    for name, (path, control) in MODELS.items():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        a = ck["args"]
        m = ForgeGNN(mi, latent_size=a["latent_size"], num_layers=a["num_layers"],
                     message_passing_steps=a["message_passing_steps"])
        m.load_state_dict(ck["state_dict"])
        models[name] = (m.eval(), control)

    out = {}
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), facecolor=SURFACE)
    for col, (run, d) in enumerate(RUNS.items()):
        with open(os.path.join(d, "results.json")) as f:
            steps = json.load(f)["steps"]
        real = [torch.tensor(meshio.read(os.path.join(d, f"step_{s['step']:02d}.vtu")).point_data["Displacement"])
                for s in steps]
        out[run] = {"real_free_end_mm": [float(x[:, 0].max()) for x in real], "models": {}}
        for name, (model, control) in models.items():
            x = torch.zeros_like(real[0])
            rmse, hd, tip = [], [], []
            with torch.no_grad():
                for s, xr in zip(steps, real):
                    if not s.get("skipped", False):
                        d_mm, R, u = s["d_j_mm"], s["R_j_deg"], s["u_j_mm"]
                        if control == "gap":
                            ctrl = gap_frac(float(band_half_thickness(mi, x, d_mm, d_mm + 0.2 * H, R)) - u)
                        else:
                            ctrl = gnn_u_frac(u)
                        nf = build_node_features(mi, x, d_mm / H, d_mm, d_mm + 0.2 * H, R, ctrl).unsqueeze(0)
                        x = x + model.predict_delta(nf, x.unsqueeze(0), accumulate=False)[0]
                    rmse.append(float(((x - xr) ** 2).mean().sqrt()))
                    hd.append(float(_chamfer_hausdorff_mm(rest + xr, rest + x)[1]))
                    tip.append(float(x[:, 0].max()))
            out[run]["models"][name] = {"rmse_vs_real_mm": rmse, "hausdorff_vs_real_mm": hd, "free_end_mm": tip}
            hits = [s["step"] for s in steps]
            axes[0, col].plot(hits, hd, color=COLORS[name], linewidth=2, label=name)
            axes[1, col].plot(hits, tip, color=COLORS[name], linewidth=2, label=f"{name} replay")
        axes[1, col].plot(hits, out[run]["real_free_end_mm"], color=INK, linewidth=2.5, label="Real simulator")
        style(axes[0, col], f"{run}: GNN replay vs. real", "Hit", "Hausdorff, GNN replay vs. real state (mm)")
        style(axes[1, col], f"{run}: free end", "Hit", "Free-end lengthwise displacement (mm)")
        for ax in axes[:, col]:
            ax.set_ylim(bottom=0)
            leg = ax.legend(frameon=True, fontsize=9, labelcolor=INK_2, loc="upper left")
            leg.get_frame().set_edgecolor(GRID)
    plt.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "replay_real_hits.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    with open(os.path.join(HERE, "replay_real_hits.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    for run, r in out.items():
        for name, m in r["models"].items():
            for h in (10, 20, len(m["rmse_vs_real_mm"])):
                print(f"{run} | {name} | hit {h}: RMSE {m['rmse_vs_real_mm'][h-1]:.3f} mm, Hausdorff "
                      f"{m['hausdorff_vs_real_mm'][h-1]:.3f} mm, free end {m['free_end_mm'][h-1]:.2f} "
                      f"(real {r['real_free_end_mm'][h-1]:.2f})")


if __name__ == "__main__":
    main()
