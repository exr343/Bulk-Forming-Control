"""Real-simulator validation of the cost designs. Figures:
  errors_and_time_real.png -- Hausdorff, Chamfer and MPC planning time per hit
    (from each run's results.json; square run replayed dotted)
  stretch_and_cross_section.png -- computed from each run's saved surface
    states (step_XX.vtu) vs. the target:
  - free-end lengthwise displacement (max x-displacement; target's is the line)
  - cross-section error (mean |width - target width|, both across-flats
    widths, 5 mm slices over x = 20-90 mm -- same metric as cost_design.py)
Also writes metrics.json with the per-hit series.

Usage (repo root):
    python applications/Agility_Forge/progress/control_reports/2026-09-26_cost_e5_real_simulator/make_figures.py
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
R = "applications/Agility_Forge/control/results"
RUNS = [("real_simulator/e5_w30_effort0.3", "w = 30 + effort 0.3", "#2a78d6"),
        ("real_simulator/e5_w100_effort0.1", "w = 100 + effort 0.1", "#eb6834"),
        ("real_simulator/e5_w30", "w = 30", "#1baf7a"),
        ("real_simulator/e5_w30_effort0.1", "w = 30 + effort 0.1", "#eda100")]
BINS = np.arange(20.0, 95.0, 5.0)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def widths(rest, u):
    pos = rest + u
    return np.array([(np.ptp(pos[s, 1]), np.ptp(pos[s, 2]))
                     for s in ((rest[:, 0] >= a) & (rest[:, 0] < b) for a, b in zip(BINS[:-1], BINS[1:]))])


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
    out = {}
    for run, label, _ in RUNS:
        d = os.path.join(R, run)
        t = meshio.read(os.path.join(d, "target.vtu"))
        rest, xt = t.points, t.point_data["Displacement"]
        wt = widths(rest, xt)
        with open(os.path.join(d, "results.json")) as f:
            res = json.load(f)
        tip, xsec = [], []
        for s in res["steps"]:
            u = meshio.read(os.path.join(d, f"step_{s['step']:02d}.vtu")).point_data["Displacement"]
            tip.append(float(u[:, 0].max()))
            xsec.append(float(np.abs(widths(rest, u) - wt).mean()))
        out[run] = {"label": label, "hits": [s["step"] for s in res["steps"]],
                    "hausdorff_mm": [s["hausdorff_mm"] for s in res["steps"]],
                    "chamfer_mm2": [s["chamfer_mm2"] for s in res["steps"]],
                    "tip_axial_mm": tip, "cross_section_err_mm": xsec,
                    "gnn_pred_vs_actual_rmse_mm": [s["gnn_pred_vs_actual_rmse_mm"] for s in res["steps"]],
                    "d_j_mm": [s["d_j_mm"] for s in res["steps"]], "u_j_mm": [s["u_j_mm"] for s in res["steps"]],
                    "R_j_deg": [s["R_j_deg"] for s in res["steps"]],
                    "target_tip_axial_mm": float(xt[:, 0].max()),
                    "billet_cross_section_err_mm": float(np.abs(widths(rest, np.zeros_like(xt)) - wt).mean()),
                    "reference": {k: res["reference"][k] for k in ("hit", "hausdorff_mm", "chamfer_mm2")}}
    with open(os.path.join(HERE, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Main figure: the three reported metrics -- Hausdorff, Chamfer, planning time.
    for run, label, _ in RUNS:
        with open(os.path.join(R, run, "results.json")) as f:
            out[run]["plan_time_s"] = [s["plan_time_s"] for s in json.load(f)["steps"]]
    with open(os.path.join(HERE, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    ref = next(iter(out.values()))["reference"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), facecolor=SURFACE)
    for ax, key in zip(axes[:2], ("hausdorff_mm", "chamfer_mm2")):
        ax.plot(ref["hit"], ref[key], ":", color="#8a8984", linewidth=2, label="Square run replayed (open loop)")
    for run, label, color in RUNS:
        m = out[run]
        axes[0].plot(m["hits"], m["hausdorff_mm"], color=color, linewidth=2, label=label)
        axes[1].plot(m["hits"], m["chamfer_mm2"], color=color, linewidth=2, label=label)
        axes[2].plot(m["hits"], m["plan_time_s"], color=color, linewidth=2, label=label)
    style(axes[0], "Hausdorff Error to Target", "Hausdorff error (mm)")
    style(axes[1], "Chamfer Error to Target", "Chamfer error (mm²)")
    style(axes[2], "MPC Planning Time per Hit", "SQP planning time (s)")
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, 51)
    for ax in axes[:2]:
        leg = ax.legend(frameon=True, fontsize=9, labelcolor=INK_2, loc="upper right")
        leg.get_frame().set_edgecolor(GRID)
    plt.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "errors_and_time_real.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Second figure: why it overshoots -- free-end stretch and cross-section error.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)
    for run, label, color in RUNS:
        m = out[run]
        axes[0].plot(m["hits"], m["tip_axial_mm"], color=color, linewidth=2, label=label)
        axes[1].plot(m["hits"], m["cross_section_err_mm"], color=color, linewidth=2, label=label)
    m0 = next(iter(out.values()))
    axes[0].axhline(m0["target_tip_axial_mm"], color=INK, linestyle="--", linewidth=1)
    axes[0].text(1, m0["target_tip_axial_mm"], "target", color=INK_2, fontsize=8, va="bottom")
    axes[1].axhline(m0["billet_cross_section_err_mm"], color="#8a8984", linestyle=":", linewidth=1.5)
    axes[1].text(1, m0["billet_cross_section_err_mm"], "untouched billet", color=INK_2, fontsize=8, va="bottom")
    style(axes[0], "Free-End Displacement", "Lengthwise displacement of the bar end (mm)")
    style(axes[1], "Cross-Section Error", "Mean width error, x = 20-90 mm (mm)")
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, 51)
    leg = axes[0].legend(frameon=True, fontsize=9, labelcolor=INK_2, loc="upper left")
    leg.get_frame().set_edgecolor(GRID)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "stretch_and_cross_section.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Lead design: width along the bar at its best hit vs. its last hit.
    run = RUNS[0][0]
    d = os.path.join(R, run)
    t = meshio.read(os.path.join(d, "target.vtu"))
    rest, xt = t.points, t.point_data["Displacement"]
    H = out[run]["hausdorff_mm"]
    best = out[run]["hits"][int(np.argmin(H))]
    last = out[run]["hits"][-1]
    centers = 0.5 * (BINS[:-1] + BINS[1:])
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    for u, label, kw in [(np.zeros_like(xt), "Billet", dict(color="#8a8984", linestyle=":")),
                         (xt, "Target", dict(color=INK, linestyle="--")),
                         (meshio.read(os.path.join(d, f"step_{best:02d}.vtu")).point_data["Displacement"],
                          f"Hit {best} (best Hausdorff)", dict(color="#2a78d6", linestyle="-")),
                         (meshio.read(os.path.join(d, f"step_{last:02d}.vtu")).point_data["Displacement"],
                          f"Hit {last} (last)", dict(color="#eb6834", linestyle="-"))]:
        ax.plot(centers, widths(rest, u).max(1), marker="o", markersize=4, linewidth=2, label=label, **kw)
    style(ax, "Lead Design (w = 30 + effort 0.3): Width Along the Bar", "Largest across-flats width (mm)")
    ax.set_xlabel("Position along the original bar (mm from x = 0)", color=INK_2)
    leg = ax.legend(frameon=True, fontsize=9, labelcolor=INK_2, loc="upper left")
    leg.get_frame().set_edgecolor(GRID)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "lead_width_best_vs_last.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("-> figures/errors_and_time_real.png, figures/stretch_and_cross_section.png, "
          "figures/lead_width_best_vs_last.png, metrics.json")


if __name__ == "__main__":
    main()
