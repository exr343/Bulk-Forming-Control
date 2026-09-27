"""Experiments 10-11 (surrogate closed loop, ideal square target, 25% cross-
section share, no effort, 50 hits, 3 seeds each): why did the gap-control
MPC (experiment 9) do so badly?
  stroke            -- experiment 7's winner: stroke-trained M = 3 GNN, the
                       optimizer varies the stroke (0-2 mm)
  gap, misses       -- gap-trained GNN, optimizer varies the half-gap, travel
                       0-2 mm (misses allowed; experiment 9's setup)
  gap, min 0.5      -- as above, travel 0.5-2 mm (no misses)
  gap GNN, travel   -- gap-trained GNN, optimizer varies the travel
                       (0.5-2 mm), converted to the half-gap per hit
  stroke, 0.5-2     -- stroke-trained GNN, stroke 0.5-2 mm (same allowed
                       actions as the two rows above; no idle hits)
In the gap runs a miss leaves the bar unchanged (not the GNN's prediction).

Outputs:
  figures/errors_and_time.png -- Hausdorff, Chamfer and planning time per hit
    (mean over seeds, min-max band), open-loop square run dotted
  figures/final_errors.png -- final Chamfer and Hausdorff, one dot per seed
  figures/optimizer_and_angles.png -- share of plans stopped at the
    100-iteration limit, and share of hits within 10 deg of the target flats
  summary.json

Usage (repo root):
    python applications/Agility_Forge/progress/control_reports/2026-09-27_cost_e10_e11_gap_diagnosis/make_figures.py
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = "applications/Agility_Forge/control/results/surrogate"
GROUPS = {
    "Stroke GNN, stroke variable (exp. 7)": f"{R}/e7/e7_share25_effort0_seed*",
    "Gap GNN, gap variable, misses allowed (exp. 10)": f"{R}/e10/e10_gap_miss_seed*",
    "Gap GNN, gap variable, min travel 0.5 mm (exp. 10)": f"{R}/e10/e10_gap_min0.5_seed*",
    "Gap GNN, travel variable 0.5-2 mm (exp. 11)": f"{R}/e11/e11_gapgnn_travelvar_seed*",
    "Stroke GNN, stroke variable 0.5-2 mm (exp. 11)": f"{R}/e11/e11_strokegnn_min0.5_seed*",
}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
REF_GRAY = "#8a8984"


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
    runs = {}
    for name, pat in GROUPS.items():
        runs[name] = []
        for d in sorted(glob.glob(pat)):
            with open(os.path.join(d, "results.json")) as f:
                runs[name].append(json.load(f))
    ref = next(iter(runs.values()))[0]["reference"]
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

    summary = {"open_loop_hit48": {"chamfer_mm2": ref["chamfer_mm2"][-1], "hausdorff_mm": ref["hausdorff_mm"][-1]},
               "groups": {}}
    for name, rs in runs.items():
        st = [s for r in rs for s in r["steps"]]
        nit = np.array([s["sqp_nit"] for s in st])
        ang = np.array([s["R_j_deg"] % 180 for s in st])
        off = np.minimum.reduce([ang, np.abs(ang - 90), 180 - ang])
        S = [r["summary"] for r in rs]
        summary["groups"][name] = {
            "runs": [r["name"] for r in rs],
            **{k: [s[k] for s in S] for k in ("final_chamfer_mm2", "final_hausdorff_mm", "cross_section_err_mm",
                                             "tip_axial_mm", "mean_plan_time_s", "n_active_hits")},
            "n_skipped": [s.get("n_skipped", 0) for s in S],
            "share_plans_at_iteration_limit": float(np.mean(nit >= 100)),
            "share_hits_within_10deg_of_flats": float(np.mean(off <= 10)),
            "mean_travel_mm": float(np.mean([s["u_j_mm"] for s in st])),
            "share_hits_travel_ge_1.9mm": float(np.mean([s["u_j_mm"] >= 1.9 for s in st])),
        }
    with open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Per-hit errors and planning time.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0), facecolor=SURFACE)
    for ax, key in zip(axes[:2], ("hausdorff_mm", "chamfer_mm2")):
        ax.plot(ref["hit"], ref[key], ":", color=REF_GRAY, linewidth=2, label="Open loop: square run replayed")
    for (name, rs), color in zip(runs.items(), COLORS):
        for ax, key in zip(axes, ("hausdorff_mm", "chamfer_mm2", "plan_time_s")):
            M = np.array([[s[key] for s in r["steps"]] for r in rs])
            hits = np.arange(1, M.shape[1] + 1)
            ax.plot(hits, M.mean(0), color=color, linewidth=2, label=name)
            ax.fill_between(hits, M.min(0), M.max(0), color=color, alpha=0.12, linewidth=0)
    style(axes[0], "Hausdorff Error to Ideal Target", "Hit", "Hausdorff error (mm)")
    style(axes[1], "Chamfer Error to Ideal Target", "Hit", "Chamfer error (mm²)")
    style(axes[2], "MPC Planning Time per Hit", "Hit", "SQP planning time (s)")
    for ax in axes:
        ax.set_ylim(bottom=0)
    leg = axes[1].legend(frameon=True, fontsize=8, labelcolor=INK_2, loc="upper right")
    leg.get_frame().set_edgecolor(GRID)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "errors_and_time.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    short = ["Stroke GNN,\nstroke var.\n(exp. 7)", "Gap GNN, gap var.,\nmisses allowed\n(exp. 10)",
             "Gap GNN, gap var.,\nmin travel 0.5\n(exp. 10)", "Gap GNN,\ntravel var. 0.5-2\n(exp. 11)",
             "Stroke GNN,\nstroke var. 0.5-2\n(exp. 11)"]

    # Final errors per seed.
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), facecolor=SURFACE)
    for ax, key, olv, title, ylabel in [
        (axes[0], "final_chamfer_mm2", ref["chamfer_mm2"][-1], "Final Chamfer (hit 50)", "Chamfer error (mm²)"),
        (axes[1], "final_hausdorff_mm", ref["hausdorff_mm"][-1], "Final Hausdorff (hit 50)", "Hausdorff error (mm)"),
    ]:
        for i, (name, color) in enumerate(zip(runs, COLORS)):
            v = summary["groups"][name][key]
            ax.scatter([i] * len(v), v, s=55, color=color, edgecolor=SURFACE, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color=INK, linewidth=2, zorder=4)
        ax.axhline(olv, color=REF_GRAY, linestyle=":", linewidth=2)
        ax.text(len(runs) - 0.5, olv, "open loop, hit 48", color=INK_2, fontsize=8, ha="right", va="bottom")
        ax.set_xticks(range(len(runs)), short[:len(runs)], fontsize=8)
        style(ax, title, "", ylabel)
        ax.set_ylim(bottom=0)
    fig.suptitle("Bar = mean, dots = 3 seeds (surrogate closed loop)", color=INK_2, fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "final_errors.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Optimizer convergence and angles.
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), facecolor=SURFACE)
    for ax, key, title, ylabel in [
        (axes[0], "share_plans_at_iteration_limit", "Plans Stopped at the 100-Iteration Limit",
         "Share of plans (%)"),
        (axes[1], "share_hits_within_10deg_of_flats", "Hits Aligned With the Target's Flats",
         "Share of hits within 10° of 0°/90° (%)"),
    ]:
        vals = [100 * summary["groups"][n][key] for n in runs]
        bars = ax.bar(range(len(runs)), vals, color=COLORS[:len(runs)], width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", color=INK, fontsize=9)
        ax.set_xticks(range(len(runs)), short[:len(runs)], fontsize=8)
        style(ax, title, "", ylabel)
        ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "optimizer_and_angles.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("-> figures/errors_and_time.png, final_errors.png, optimizer_and_angles.png, summary.json")


if __name__ == "__main__":
    main()
