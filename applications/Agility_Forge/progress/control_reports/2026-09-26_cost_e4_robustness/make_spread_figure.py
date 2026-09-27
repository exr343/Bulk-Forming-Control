"""Experiment 4 (robustness): final Hausdorff, cross-section error and free-end
displacement for 5 noisy-planning seeds of each design (dots), with the
noise-free run from experiments 2/3 as an open diamond.

Usage (repo root):
    python applications/Agility_Forge/progress/control_reports/2026-09-26_cost_e4_robustness/make_spread_figure.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = "applications/Agility_Forge/control/results/surrogate"
DESIGNS = [("t30_e0.1", "e3_trans_30_effort_0.1", "w = 30\neffort 0.1"),
           ("t30_e0.3", "e3_trans_30_effort_0.3", "w = 30\neffort 0.3"),
           ("t100_e0.1", "e2_trans_100_effort_0.1", "w = 100\neffort 0.1")]
METRICS = [("final_hausdorff_mm", "Hausdorff, Hit 50", "Hausdorff error (mm)", None),
           ("cross_section_err_mm", "Cross-Section Error, Hit 50", "Cross-section error (mm)", None),
           ("tip_axial_mm", "Free-End Displacement", "Lengthwise displacement (mm)", 35.04)]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")
    ax.tick_params(colors=INK_2)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURFACE)
    for ax, (key, title, ylabel, ref) in zip(axes, METRICS):
        for i, (tag, det, label) in enumerate(DESIGNS):
            seeds = [json.load(open(os.path.join(R, (f"e4_{tag}_seed{s}").split("_")[0], f"e4_{tag}_seed{s}", "results.json")))["summary"][key]
                     for s in range(5)]
            noiseless = json.load(open(os.path.join(R, (det).split("_")[0], det, "results.json")))["summary"][key]
            jitter = [-0.12, -0.06, 0.0, 0.06, 0.12]
            ax.scatter([i + j for j in jitter], seeds, s=64, color=COLORS[i], edgecolor=SURFACE, linewidth=1.5,
                       zorder=3, label="5 seeds, noisy planning" if i == 0 else None)
            ax.scatter([i + 0.3], [noiseless], s=90, marker="D", facecolor="none", edgecolor=INK, linewidth=1.5,
                       zorder=3, label="Noise-free run (exp. 2/3)" if i == 0 else None)
        if ref is not None:
            ax.axhline(ref, color=INK_2, linestyle="--", linewidth=1)
            ax.text(-0.45, ref, "target", color=INK_2, fontsize=8, va="bottom", ha="left")
        ax.set_xticks(range(3), [d[2] for d in DESIGNS])
        ax.set_xlim(-0.5, 2.6)
        style(ax, title, ylabel)
    leg = axes[0].legend(frameon=True, fontsize=9, labelcolor=INK_2, loc="upper left")
    leg.get_frame().set_edgecolor(GRID)
    plt.tight_layout()
    out = os.path.join(HERE, "figures", "robustness_spread.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
