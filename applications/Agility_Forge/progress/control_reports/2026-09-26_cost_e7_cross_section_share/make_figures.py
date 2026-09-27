"""Experiment 7 (target-derived cross-section weight, ideal square target,
surrogate closed loop, 3 seeds per design). Figures:
  ranking.png -- final Chamfer and Hausdorff per design, one dot per seed,
    in the order set by the pre-registered rule (mean final Chamfer, then
    mean final Hausdorff); the open-loop reference's final value as a line
  errors_and_time.png -- Hausdorff, Chamfer and planning time per hit for the
    winner, the runner-up and the old design (mean over seeds, min-max band),
    with the open-loop reference scored against the ideal target
  winner_width.png -- width along the bar after hit 50, winner (median seed)
    vs. the ideal target and the billet
Also writes summary.json.

Usage (repo root):
    python applications/Agility_Forge/progress/control_reports/2026-09-26_cost_e7_cross_section_share/make_figures.py
"""

import collections
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = "applications/Agility_Forge/control/results/surrogate/e7"
TARGET = "applications/Agility_Forge/control/targets/ideal_square_10.6/target_on_billet.vtu"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, REF_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
LABELS = {"e7_share25_effort0": "25% share, no effort", "e7_share50_effort0": "50% share, no effort",
          "e7_share75_effort0": "75% share, no effort", "e7_share25_effort0.3": "25% share + effort 0.3",
          "e7_share50_effort0.3": "50% share + effort 0.3", "e7_share75_effort0.3": "75% share + effort 0.3",
          "e7_w30_effort0.3": "Old design: w = 30 + effort 0.3"}


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
    groups = collections.defaultdict(list)
    for d in sorted(glob.glob(os.path.join(R, "e7_*"))):
        with open(os.path.join(d, "results.json")) as f:
            groups[os.path.basename(d).rsplit("_seed", 1)[0]].append((d, json.load(f)))
    ref = next(iter(groups.values()))[0][1]["reference"]

    summary = {}
    for name, runs in groups.items():
        S = [r["summary"] for _, r in runs]
        summary[name] = {"label": LABELS[name], "w": runs[0][1]["penalty"]["transverse"],
                         "effort": runs[0][1]["penalty"]["effort_s"],
                         "cs_share_of_start_cost": runs[0][1]["start_error_split"]["cs_share_of_start_cost"],
                         **{k: [s[k] for s in S] for k in ("final_chamfer_mm2", "final_hausdorff_mm",
                                                         "cross_section_err_mm", "tip_axial_mm", "mean_plan_time_s",
                                                         "n_active_hits", "longest_repeat")}}
    order = sorted(summary, key=lambda n: (np.mean(summary[n]["final_chamfer_mm2"]),
                                           np.mean(summary[n]["final_hausdorff_mm"])))
    with open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"ranking_rule": "mean final Chamfer, then mean final Hausdorff (set before the runs)",
                   "order": order, "designs": summary,
                   "open_loop_reference_hit48": {"chamfer_mm2": ref["chamfer_mm2"][-1],
                                                 "hausdorff_mm": ref["hausdorff_mm"][-1]}}, f, indent=2)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

    # Ranking figure.
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), facecolor=SURFACE)
    for ax, key, ref_val, title, ylabel in [
        (axes[0], "final_chamfer_mm2", ref["chamfer_mm2"][-1], "Final Chamfer (hit 50)", "Chamfer error (mm²)"),
        (axes[1], "final_hausdorff_mm", ref["hausdorff_mm"][-1], "Final Hausdorff (hit 50)", "Hausdorff error (mm)"),
    ]:
        for i, n in enumerate(order):
            v = summary[n][key]
            ax.scatter([i] * len(v), v, s=55, color=BLUE if i == 0 else "#b5b4ae", edgecolor=SURFACE, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color=INK, linewidth=2, zorder=4)
        ax.axhline(ref_val, color=REF_GRAY, linestyle=":", linewidth=2)
        ax.text(len(order) - 0.5, ref_val, "open loop, hit 48", color=INK_2, fontsize=8, ha="right", va="bottom")
        ax.set_xticks(range(len(order)), [LABELS[n].replace(", ", ",\n").replace(" + ", "\n+ ") for n in order],
                      fontsize=8)
        style(ax, title, "", ylabel)
        ax.set_ylim(bottom=0)
    fig.suptitle("Ranked by mean final Chamfer (bar = mean, dots = 3 seeds); winner in blue", color=INK_2, fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "ranking.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Per-hit figure for three designs.
    show = [(order[0], BLUE), (order[1], ORANGE), ("e7_w30_effort0.3", AQUA)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), facecolor=SURFACE)
    for ax, key in zip(axes[:2], ("hausdorff_mm", "chamfer_mm2")):
        ax.plot(ref["hit"], ref[key], ":", color=REF_GRAY, linewidth=2, label="Open loop: square run replayed")
    for name, color in show:
        for ax, key in zip(axes, ("hausdorff_mm", "chamfer_mm2", "plan_time_s")):
            M = np.array([[s[key] for s in r["steps"]] for _, r in groups[name]])
            hits = np.arange(1, M.shape[1] + 1)
            ax.plot(hits, M.mean(0), color=color, linewidth=2, label=LABELS[name])
            ax.fill_between(hits, M.min(0), M.max(0), color=color, alpha=0.15, linewidth=0)
    style(axes[0], "Hausdorff Error to Ideal Target", "Hit", "Hausdorff error (mm)")
    style(axes[1], "Chamfer Error to Ideal Target", "Hit", "Chamfer error (mm²)")
    style(axes[2], "MPC Planning Time per Hit", "Hit", "SQP planning time (s)")
    for ax in axes:
        ax.set_ylim(bottom=0)
    for ax in axes[:2]:
        leg = ax.legend(frameon=True, fontsize=8, labelcolor=INK_2, loc="lower left")
        leg.get_frame().set_edgecolor(GRID)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "errors_and_time.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Winner's width along the bar (median-Chamfer seed).
    runs = groups[order[0]]
    med = sorted(runs, key=lambda x: x[1]["summary"]["final_chamfer_mm2"])[len(runs) // 2]
    t = meshio.read(TARGET)
    rest, xt = t.points, t.point_data["Displacement"]
    xf = np.load(os.path.join(med[0], "final_state.npy"))
    bins = np.arange(-5.0, 97.0, 5.0)
    centers = 0.5 * (bins[:-1] + bins[1:])

    def max_width(u):
        pos = rest + u
        return [max(np.ptp(pos[s, 1]), np.ptp(pos[s, 2]))
                for s in ((rest[:, 0] >= a) & (rest[:, 0] < b) for a, b in zip(bins[:-1], bins[1:]))]

    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    ax.plot(centers, max_width(np.zeros_like(xt)), ":", color=REF_GRAY, linewidth=2, marker="o", markersize=4,
            label="Billet")
    ax.plot(centers, max_width(xt), "--", color=INK, linewidth=2, marker="o", markersize=4, label="Ideal target")
    ax.plot(centers, max_width(xf), "-", color=BLUE, linewidth=2, marker="o", markersize=4,
            label=f"Winner after hit 50 ({os.path.basename(med[0]).rsplit('_', 1)[1]})")
    style(ax, "Width Along the Bar: 25% Share, No Effort (surrogate)",
          "Position along the original bar (mm from x = 0)", "Largest across-flats width (mm)")
    leg = ax.legend(frameon=True, fontsize=9, labelcolor=INK_2, loc="center right")
    leg.get_frame().set_edgecolor(GRID)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "winner_width.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("-> figures/ranking.png, figures/errors_and_time.png, figures/winner_width.png, summary.json")


if __name__ == "__main__":
    main()
