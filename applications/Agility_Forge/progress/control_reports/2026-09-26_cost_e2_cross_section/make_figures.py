"""Experiment 2 (cross-section term + stroke effort) figures:
  thickness_comparison.png -- final bar width along its length for the billet,
    the target, the baseline cost and the two best experiment-2 configs
  best_controls.png -- station / angle / stroke over 50 hits for the best config

Usage (repo root):
    python applications/Agility_Forge/progress/control_reports/2026-09-26_cost_e2_cross_section/make_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = "applications/Agility_Forge/control/results/surrogate"
TARGET_RUN = "applications/Agility_Forge/control/results/real_simulator/e0_original_cost"  # holds target.vtu (surface)
BINS = np.arange(20.0, 95.0, 5.0)
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


def max_width(rest, u):
    pos = rest + u
    return np.array([max(np.ptp(pos[s, 1]), np.ptp(pos[s, 2]))
                     for s in ((rest[:, 0] >= a) & (rest[:, 0] < b) for a, b in zip(BINS[:-1], BINS[1:]))])


def final_state(name, rest):
    """The run's final surrogate state (control/cost_design.py saves it as
    final_state.npy; older runs were rebuilt with replay_final_state)."""
    return np.load(os.path.join(R, (name).split("_")[0], name, "final_state.npy"))


def main():
    t = meshio.read(os.path.join(TARGET_RUN, "target.vtu"))
    rest, xt = t.points, t.point_data["Displacement"]
    centers = 0.5 * (BINS[:-1] + BINS[1:])
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    ax.plot(centers, max_width(rest, np.zeros_like(xt)), ":", color=REF_GRAY, linewidth=2, marker="o",
            markersize=4, label="Billet")
    ax.plot(centers, max_width(rest, xt), "--", color=INK, linewidth=2, marker="o", markersize=4, label="Target")
    for name, label, color in [("e1_baseline", "Baseline cost (error term only)", BLUE),
                               ("e2_trans_30", "+ cross-section term, w = 30", ORANGE),
                               ("e2_trans_100_effort_0.1", "+ cross-section w = 100, stroke effort 0.1", AQUA)]:
        ax.plot(centers, max_width(rest, final_state(name, rest)), "-", color=color, linewidth=2, marker="o",
                markersize=4, label=label)
    style(ax, "Largest Across-Flats Width After 50 Hits (surrogate loop)",
          "Position along the original bar (mm from x = 0)", "Width (mm)")
    leg = ax.legend(frameon=True, fontsize=9, labelcolor=INK_2, loc="upper right")
    leg.get_frame().set_edgecolor(GRID)
    plt.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "thickness_comparison.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    with open(os.path.join(R, ("e2_trans_100_effort_0.1").split("_")[0], "e2_trans_100_effort_0.1", "results.json")) as f:
        steps = json.load(f)["steps"]
    hits = [s["step"] for s in steps]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), facecolor=SURFACE, sharex=True)
    kw = dict(color=AQUA, linewidth=2, marker="o", markersize=3)
    axes[0].plot(hits, [s["d_j_mm"] for s in steps], **kw)
    axes[1].plot(hits, [s["R_j_deg"] % 180.0 for s in steps], **kw)
    axes[2].plot(hits, [s["u_j_mm"] for s in steps], **kw)
    style(axes[0], "Station Position", "", "Die band start (mm)")
    style(axes[1], "Hit Angle", "", "Angle, folded to [0, 180) (deg)")
    style(axes[2], "Stroke", "Hit", "Stroke per die (mm)")
    axes[1].set_yticks([0, 45, 90, 135, 180])
    fig.suptitle("Best config: cross-section w = 100, stroke effort 0.1 (surrogate loop)", color=INK, fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "best_controls.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("-> figures/thickness_comparison.png, figures/best_controls.png")


if __name__ == "__main__":
    main()
