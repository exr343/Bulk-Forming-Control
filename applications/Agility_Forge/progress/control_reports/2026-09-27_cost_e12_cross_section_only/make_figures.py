"""Experiment 12 (surrogate closed loop, ideal square target, stroke GNN M = 3,
stroke 0.5-2 mm, 50 hits, 3 seeds): cross-section-only cost vs. experiment
11's 25%-share cost under the same rules.
  errors_and_time.png -- Hausdorff, Chamfer and planning time per hit (mean
    over seeds, min-max band), open-loop square run dotted
  controls.png -- stroke and station per hit (every seed)
  width_along_bar.png -- largest across-flats width along the bar after hit
    50 (median-Chamfer seed of each) vs. the target and the billet
  summary.json

Usage (repo root):
    python applications/Agility_Forge/progress/control_reports/2026-09-27_cost_e12_cross_section_only/make_figures.py
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = "applications/Agility_Forge/control/results/surrogate"
TARGET = "applications/Agility_Forge/control/targets/ideal_square_10.6/target_on_billet.vtu"
GROUPS = {"25% share: E + 91.4·E⊥ (exp. 11)": f"{R}/e11/e11_strokegnn_min0.5_seed*",
          "Cross-section only: E⊥ (exp. 12)": f"{R}/e12/e12_csonly_min0.5_seed*"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ["#2a78d6", "#eb6834"]
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


def legend(ax, loc="best"):
    leg = ax.legend(frameon=True, fontsize=9, labelcolor=INK_2, loc=loc)
    leg.get_frame().set_edgecolor(GRID)


def main():
    runs = {n: [(d, json.load(open(os.path.join(d, "results.json")))) for d in sorted(glob.glob(p))]
            for n, p in GROUPS.items()}
    ref = next(iter(runs.values()))[0][1]["reference"]
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

    summary = {"open_loop_hit48": {"chamfer_mm2": ref["chamfer_mm2"][-1], "hausdorff_mm": ref["hausdorff_mm"][-1]},
               "groups": {}}
    for name, rs in runs.items():
        st = [s for _, r in rs for s in r["steps"]]
        ang = np.array([s["R_j_deg"] % 180 for s in st])
        U = np.array([[s["u_j_mm"] for s in r["steps"]] for _, r in rs])
        D = np.array([[s["d_j_mm"] for s in r["steps"]] for _, r in rs])
        summary["groups"][name] = {
            "runs": [r["name"] for _, r in rs], "penalty": rs[0][1]["penalty"],
            **{k: [r["summary"][k] for _, r in rs] for k in (
                "final_chamfer_mm2", "final_hausdorff_mm", "cross_section_err_mm", "tip_axial_mm",
                "n_stations", "longest_repeat", "mean_plan_time_s")},
            "share_hits_within_10deg_of_flats": float(np.mean(np.minimum.reduce([ang, abs(ang - 90), 180 - ang]) <= 10)),
            "mean_stroke_by_10_hits_mm": [float(U[:, i:i + 10].mean()) for i in range(0, U.shape[1], 10)],
            "mean_station_by_10_hits_mm": [float(D[:, i:i + 10].mean()) for i in range(0, D.shape[1], 10)],
            "share_hits_at_station_ge_74mm": float(np.mean(D >= 74.0)),
        }
    with open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.9), facecolor=SURFACE)
    for ax, key in zip(axes[:2], ("hausdorff_mm", "chamfer_mm2")):
        ax.plot(ref["hit"], ref[key], ":", color=REF_GRAY, linewidth=2, label="Open loop: square run replayed")
    for (name, rs), color in zip(runs.items(), COLORS):
        for ax, key in zip(axes, ("hausdorff_mm", "chamfer_mm2", "plan_time_s")):
            M = np.array([[s[key] for s in r["steps"]] for _, r in rs])
            hits = np.arange(1, M.shape[1] + 1)
            ax.plot(hits, M.mean(0), color=color, linewidth=2, label=name)
            ax.fill_between(hits, M.min(0), M.max(0), color=color, alpha=0.15, linewidth=0)
    style(axes[0], "Hausdorff Error to Ideal Target", "Hit", "Hausdorff error (mm)")
    style(axes[1], "Chamfer Error to Ideal Target", "Hit", "Chamfer error (mm²)")
    style(axes[2], "MPC Planning Time per Hit", "Hit", "SQP planning time (s)")
    for ax in axes:
        ax.set_ylim(bottom=0)
    legend(axes[1], "upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "errors_and_time.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), facecolor=SURFACE)
    for (name, rs), color in zip(runs.items(), COLORS):
        for i, (_, r) in enumerate(rs):
            hits = [s["step"] for s in r["steps"]]
            kw = dict(color=color, linewidth=1.3, alpha=0.8, label=name if i == 0 else None)
            axes[0].plot(hits, [s["u_j_mm"] for s in r["steps"]], **kw)
            axes[1].plot(hits, [s["d_j_mm"] for s in r["steps"]], **kw)
    style(axes[0], "Stroke per Hit (all seeds)", "Hit", "Stroke per die (mm)")
    axes[0].set_ylim(0, 2.3)
    style(axes[1], "Station per Hit (all seeds)", "Hit", "Die band start (mm from x = 0)")
    axes[1].set_ylim(10, 85)
    axes[1].axhline(75.3, color=INK_2, linewidth=0.8, linestyle="--")
    axes[1].text(1, 76.5, "highest allowed station", color=INK_2, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, fontsize=9, labelcolor=INK_2)
    leg.get_frame().set_edgecolor(GRID)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(os.path.join(HERE, "figures", "controls.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    t = meshio.read(TARGET)
    rest, xt = t.points, t.point_data["Displacement"]
    bins = np.arange(-5.0, 97.0, 5.0)
    centers = 0.5 * (bins[:-1] + bins[1:])

    def max_width(u):
        pos = rest + u
        return [max(np.ptp(pos[s, 1]), np.ptp(pos[s, 2]))
                for s in ((rest[:, 0] >= a) & (rest[:, 0] < b) for a, b in zip(bins[:-1], bins[1:]))]

    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor=SURFACE)
    ax.plot(centers, max_width(np.zeros_like(xt)), ":", color=REF_GRAY, linewidth=2, marker="o", markersize=3,
            label="Billet")
    ax.plot(centers, max_width(xt), "--", color=INK, linewidth=2, marker="o", markersize=3, label="Ideal target")
    for (name, rs), color in zip(runs.items(), COLORS):
        med = sorted(rs, key=lambda x: x[1]["summary"]["final_chamfer_mm2"])[len(rs) // 2]
        xf = np.load(os.path.join(med[0], "final_state.npy"))
        ax.plot(centers, max_width(xf), "-", color=color, linewidth=2, marker="o", markersize=3,
                label=f"{name}, median seed")
    style(ax, "Largest Across-Flats Width After Hit 50", "Position along the original bar (mm from x = 0)",
          "Width (mm)")
    ax.set_ylim(8, 20)
    legend(ax, "upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "width_along_bar.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("-> figures/errors_and_time.png, controls.png, width_along_bar.png, summary.json")


if __name__ == "__main__":
    main()
