"""Experiment 1 (control-change penalties): station, angle and stroke over the
50 surrogate hits for the baseline cost and the two strongest penalties.

Usage (repo root):
    python applications/Agility_Forge/progress/control_reports/2026-09-26_cost_e1_control_change/make_controls_figure.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = "applications/Agility_Forge/control/results/surrogate"
RUNS = [("e1_baseline", "Baseline (error term only)", "#2a78d6"),
        ("e1_delta_all_1", "Change penalty on all controls, weight 1", "#eb6834"),
        ("e1_delta_d_1", "Change penalty on station only, weight 1", "#1baf7a")]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel("Hit", color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")
    ax.tick_params(colors=INK_2)


def main():
    fig, axes = plt.subplots(3, 1, figsize=(11, 10.5), facecolor=SURFACE, sharex=True)
    for name, label, color in RUNS:
        with open(os.path.join(R, (name).split("_")[0], name, "results.json")) as f:
            steps = json.load(f)["steps"]
        hits = [s["step"] for s in steps]
        kw = dict(color=color, linewidth=2, marker="o", markersize=3, label=label)
        axes[0].plot(hits, [s["d_j_mm"] for s in steps], **kw)
        axes[1].plot(hits, [s["R_j_deg"] % 180.0 for s in steps], **kw)
        axes[2].plot(hits, [s["u_j_mm"] for s in steps], **kw)
    style(axes[0], "Station Position", "Die band start (mm from x = 0)")
    style(axes[1], "Hit Angle", "Angle, folded to [0, 180) (deg)")
    style(axes[2], "Stroke", "Stroke per die (mm)")
    axes[1].set_yticks([0, 45, 90, 135, 180])
    for ax in axes[:2]:
        ax.set_xlabel("")
    leg = axes[0].legend(frameon=True, fontsize=9, labelcolor=INK_2, loc="lower right")
    leg.get_frame().set_edgecolor(GRID)
    plt.tight_layout()
    out = os.path.join(HERE, "figures", "controls_comparison.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
