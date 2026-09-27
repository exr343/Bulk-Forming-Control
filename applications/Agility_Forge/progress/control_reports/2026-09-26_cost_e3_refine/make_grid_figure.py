"""Experiment 3: final Hausdorff and cross-section error over the grid of
cross-section weight w (30, 50, 100) x stroke effort (0.03, 0.1, 0.3), from
the surrogate closed loop. (w = 100, effort 0.1 is experiment 2's run.)

Usage (repo root):
    python applications/Agility_Forge/progress/control_reports/2026-09-26_cost_e3_refine/make_grid_figure.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = "applications/Agility_Forge/control/results/surrogate"
WS, ES = (30, 50, 100), (0.03, 0.1, 0.3)
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"


def run_name(w, e):
    return "e2_trans_100_effort_0.1" if (w, e) == (100, 0.1) else f"e3_trans_{w}_effort_{e:g}"


def main():
    grids = {"final_hausdorff_mm": np.zeros((3, 3)), "cross_section_err_mm": np.zeros((3, 3))}
    for i, w in enumerate(WS):
        for j, e in enumerate(ES):
            with open(os.path.join(R, (run_name(w, e)).split("_")[0], run_name(w, e), "results.json")) as f:
                s = json.load(f)["summary"]
            for k in grids:
                grids[k][i, j] = s[k]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    for ax, (key, title, unit) in zip(axes, [("final_hausdorff_mm", "Hausdorff, Hit 50", "mm"),
                                            ("cross_section_err_mm", "Cross-Section Error, Hit 50", "mm")]):
        g = grids[key]
        im = ax.imshow(g, cmap="Blues", aspect="auto")  # darker = worse (larger error)
        for i in range(3):
            for j in range(3):
                dark = g[i, j] > g.min() + 0.6 * (g.max() - g.min())
                ax.text(j, i, f"{g[i, j]:.2f}", ha="center", va="center", fontsize=11,
                        color="#ffffff" if dark else INK)
        ax.set_xticks(range(3), [f"{e:g}" for e in ES])
        ax.set_yticks(range(3), [f"w = {w}" for w in WS])
        ax.set_xlabel("Stroke effort weight", color=INK_2)
        ax.set_ylabel("Cross-section weight", color=INK_2)
        ax.set_title(f"{title} ({unit})", color=INK, fontsize=12, fontweight="bold")
        ax.tick_params(colors=INK_2, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.suptitle("Surrogate closed loop, 50 hits: lower is better (darker = worse)", color=INK_2, fontsize=10)
    plt.tight_layout()
    out = os.path.join(HERE, "figures", "grid.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
