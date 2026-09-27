"""Bar profile after hit 11: MPC vs. the square run's own hit 11 vs. the target.

Two panels along the bar (original, undeformed x position, so the same
material is compared across bars):
  - cross-section thickness (the larger of the two across-flats widths, in
    5 mm slices), showing where each bar was actually squeezed
  - average lengthwise displacement, showing how much each bar has stretched

Usage (repo root):
    python applications/Agility_Forge/progress/control_reports/2026-09-26_cost_e0_original/make_profile_figure.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = "applications/Agility_Forge/control/results/real_simulator/e0_original_cost"
SQUARE = "applications/Agility_Forge/data/dataset_finetuning/square/rollout_01"
BAND = (17.7, 37.0)  # the one band the MPC pressed (d_j = 0.1837, width 0.2 * H)

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, REF_GRAY = "#2a78d6", "#8a8984"


def surface_fields():
    target = meshio.read(os.path.join(RUN, "target.vtu"))
    rest = target.points
    mpc = meshio.read(os.path.join(RUN, "step_11.vtu")).point_data["Displacement"]
    # The square run's .vtu is the full volume mesh; match its nodes to the
    # surface mesh by nearest rest position (the two files store coordinates
    # at different float precision, so exact lookup misses).
    full = meshio.read(os.path.join(SQUARE, "hit_11_final.vtu"))
    dist, idx = cKDTree(full.points).query(rest)
    assert dist.max() < 1e-4, f"surface/volume node mismatch: {dist.max()}"
    ref = full.point_data["Displacement"][idx]
    return rest, {"Target (square run, hit 48)": target.point_data["Displacement"],
                  "Square run, hit 11": ref, "GNN-MPC, hit 11": mpc}


def profiles(rest, u, bins):
    pos = rest + u
    thick, axial, centers = [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        s = (rest[:, 0] >= a) & (rest[:, 0] < b)
        thick.append(max(np.ptp(pos[s, 1]), np.ptp(pos[s, 2])))
        axial.append(u[s, 0].mean())
        centers.append(0.5 * (a + b))
    return np.array(centers), np.array(thick), np.array(axial)


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel("Position along the original bar (mm from x = 0)", color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")
    ax.tick_params(colors=INK_2)


def main():
    rest, fields = surface_fields()
    bins = np.arange(-5, 97, 5)
    styles = {"Target (square run, hit 48)": dict(color=INK, linestyle="--"),
              "Square run, hit 11": dict(color=REF_GRAY, linestyle=":"),
              "GNN-MPC, hit 11": dict(color=BLUE, linestyle="-")}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), facecolor=SURFACE)
    for name, u in fields.items():
        x, thick, axial = profiles(rest, u, bins)
        axes[0].plot(x, thick, linewidth=2, marker="o", markersize=4, label=name, **styles[name])
        axes[1].plot(x, axial, linewidth=2, marker="o", markersize=4, label=name, **styles[name])
    for ax in axes:
        ax.axvspan(*BAND, color=BLUE, alpha=0.08, linewidth=0)
    axes[0].annotate("band the MPC\npressed 11 times", (np.mean(BAND), 18.4), ha="center", color=INK_2, fontsize=8)
    style(axes[0], "Cross-Section Thickness", "Largest across-flats width (mm)")
    style(axes[1], "Lengthwise Displacement", "Mean axial displacement (mm)")
    axes[0].set_ylim(9, 19.5)
    axes[1].set_ylim(bottom=0)
    for ax, loc in zip(axes, ("lower left", "upper left")):
        leg = ax.legend(frameon=True, fontsize=9, labelcolor=INK_2, loc=loc)
        leg.get_frame().set_edgecolor(GRID)
    plt.tight_layout()
    out = os.path.join(HERE, "figures", "bar_profile_hit11.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
