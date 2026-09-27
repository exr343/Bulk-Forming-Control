"""Per-hit test error on square_jitter_seed3 (48-hit autoregressive rollout):
pretrain-then-finetune vs. joint training, for M = 3 and M = 5 message-passing
steps. Rows: Hausdorff, Chamfer. Columns: M.

Usage (repo root):
    python applications/Agility_Forge/progress/GNN_reports/2026-09-26_pretrain_vs_joint/make_comparison_figure.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GNN = "applications/Agility_Forge/GNN"
RUNS = {"Pretrain, then finetune": f"{GNN}/mp_sweep/finetune/mp_{{M}}/metrics.json",
        "All at once (joint)": f"{GNN}/joint_training/mp_{{M}}/metrics.json"}

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"Pretrain, then finetune": "#2a78d6", "All at once (joint)": "#eb6834"}


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel("Hit (48-hit autoregressive rollout, test run seed 3)", color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")
    ax.tick_params(colors=INK_2)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), facecolor=SURFACE)
    for col, M in enumerate((3, 5)):
        for name, path in RUNS.items():
            with open(path.format(M=M)) as f:
                roll = json.load(f)["finetuned"]["seed3_rollout"]
            hits = range(1, len(roll["per_hit_hausdorff_mm"]) + 1)
            axes[0, col].plot(hits, roll["per_hit_hausdorff_mm"], color=COLORS[name], linewidth=2, label=name)
            axes[1, col].plot(hits, roll["per_hit_chamfer_mm2"], color=COLORS[name], linewidth=2, label=name)
        style(axes[0, col], f"Test Hausdorff, M = {M}", "Hausdorff error (mm)")
        style(axes[1, col], f"Test Chamfer, M = {M}", "Chamfer error (mm²)")
    for row in range(2):
        top = max(axes[row, c].get_ylim()[1] for c in range(2))
        for c in range(2):
            axes[row, c].set_ylim(0, top)  # same scale across M for a fair visual comparison
            leg = axes[row, c].legend(frameon=True, fontsize=9, labelcolor=INK_2, loc="upper left")
            leg.get_frame().set_edgecolor(GRID)
    plt.tight_layout()
    out = os.path.join(HERE, "figures", "per_hit_pretrain_vs_joint.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
