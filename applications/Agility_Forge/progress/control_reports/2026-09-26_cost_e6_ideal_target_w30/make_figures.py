"""Experiment 6 (ideal square target, lead cost design, stopped at hit 13):
  errors_and_time.png -- Hausdorff, Chamfer and planning time per hit; the
    open-loop reference (square run replayed) scored against the IDEAL target
  width_along_bar.png -- bar width along its length after the last hit vs.
    the ideal target and the billet
  stations.png -- station of every hit

Usage (repo root):
    python applications/Agility_Forge/progress/control_reports/2026-09-26_cost_e6_ideal_target_w30/make_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = "applications/Agility_Forge/control/results/real_simulator/e6_ideal_w30_effort0.3"
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


def main():
    with open(os.path.join(RUN, "results.json")) as f:
        res = json.load(f)
    steps, ref = res["steps"], res["reference"]
    hits = [s["step"] for s in steps]
    n = len(steps)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

    # Main figure. Reference drawn over the same hits for a like-for-like view.
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), facecolor=SURFACE)
    for ax, key, color, title, ylabel in [
        (axes[0], "hausdorff_mm", BLUE, "Hausdorff Error to Ideal Target", "Hausdorff error (mm)"),
        (axes[1], "chamfer_mm2", ORANGE, "Chamfer Error to Ideal Target", "Chamfer error (mm²)"),
    ]:
        ax.plot(ref["hit"][:n], ref[key][:n], ":", color=REF_GRAY, linewidth=2, marker="o", markersize=3,
                label="Open loop: square run replayed")
        ax.plot(hits, [s[key] for s in steps], "-", color=color, linewidth=2, marker="o", markersize=3,
                label="Closed-loop GNN-MPC")
        style(ax, title, "Hit", ylabel)
        leg = ax.legend(frameon=True, fontsize=9, labelcolor=INK_2)
        leg.get_frame().set_edgecolor(GRID)
    axes[2].plot(hits, [s["plan_time_s"] for s in steps], "-", color=AQUA, linewidth=2, marker="o", markersize=3)
    style(axes[2], "MPC Planning Time per Hit", "Hit", "SQP planning time (s)")
    axes[2].set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "errors_and_time.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Width along the bar after the last hit.
    t = meshio.read(os.path.join(RUN, "target.vtu"))
    rest, xt = t.points, t.point_data["Displacement"]
    u = meshio.read(os.path.join(RUN, f"step_{n:02d}.vtu")).point_data["Displacement"]
    bins = np.arange(-5.0, 97.0, 5.0)
    centers = 0.5 * (bins[:-1] + bins[1:])

    def max_width(disp):
        pos = rest + disp
        return [max(np.ptp(pos[s, 1]), np.ptp(pos[s, 2]))
                for s in ((rest[:, 0] >= a) & (rest[:, 0] < b) for a, b in zip(bins[:-1], bins[1:]))]

    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    ax.plot(centers, max_width(np.zeros_like(xt)), ":", color=REF_GRAY, linewidth=2, marker="o", markersize=4,
            label="Billet")
    ax.plot(centers, max_width(xt), "--", color=INK, linewidth=2, marker="o", markersize=4, label="Ideal target")
    ax.plot(centers, max_width(u), "-", color=BLUE, linewidth=2, marker="o", markersize=4, label=f"MPC after hit {n}")
    ax.axvline(17.7, color=INK_2, linewidth=0.8, linestyle="--")
    ax.text(18.2, 10.2, "800 °C point", color=INK_2, fontsize=8)
    style(ax, "Largest Across-Flats Width Along the Bar",
          "Position along the original bar (mm from x = 0)", "Width (mm)")
    leg = ax.legend(frameon=True, fontsize=9, labelcolor=INK_2, loc="center right")
    leg.get_frame().set_edgecolor(GRID)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "width_along_bar.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Stations.
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURFACE)
    ax.plot(hits, [s["d_j_mm"] for s in steps], "-", color=BLUE, linewidth=2, marker="o", markersize=5)
    ax.axhline(17.7, color=INK_2, linewidth=0.8, linestyle="--")
    ax.text(0.6, 18.4, "lowest allowed station (800 °C point)", color=INK_2, fontsize=8)
    style(ax, "Station of Each Hit", "Hit", "Die band start (mm from x = 0)")
    ax.set_ylim(10, 80)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "stations.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("-> figures/errors_and_time.png, figures/width_along_bar.png, figures/stations.png")


if __name__ == "__main__":
    main()
