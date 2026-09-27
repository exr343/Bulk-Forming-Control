"""Message-passing-steps (M) sweep, square-rod regime: summary of the reported
metrics (settled by interview, 2026-09-26):

  1. Gradient time of a 10-hit horizon -- one forward + backward pass through
     control/mpc.py's _rollout_cost (exactly one SLSQP iteration's gradient in
     MPCController.plan), from the undeformed billet toward the square run's
     hit-48 geometry, with a fixed control guess. 1 untimed warm-up + 10 timed
     repeats with CUDA synchronization. Same method as the old sweep's
     benchmark_gradient_time.py (GNN/old/mp_sweep/), at horizon 10.
  2. Hausdorff / Chamfer on the test run (square_jitter_seed3): the finetuned
     model's full 48-hit autoregressive rollout from the undeformed billet,
     per hit, read from each finetune run's metrics.json, plus the mean over
     all 48 hits.

Writes summary.json and plots (test_hausdorff_vs_hit.png,
test_chamfer_vs_hit.png, mean_error_vs_mp_steps.png, gradient_time_vs_mp_steps.png)
into GNN/mp_sweep/.

Usage (on a GPU node, after every finetune/mp_<d> run has finished):
    python -m applications.Agility_Forge.GNN.mp_sweep.summarize
    # second sweep (finetuned on the finished square runs, M = 1-10):
    python -m applications.Agility_Forge.GNN.mp_sweep.summarize --depths 1,2,3,4,5,6,7,8,9,10 \
        --finetune-root applications/Agility_Forge/GNN/mp_sweep_finished/finetune \
        --out-dir applications/Agility_Forge/GNN/mp_sweep_finished
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
import torch

from applications.Agility_Forge.GNN.data import build_surface_mesh_info
from applications.Agility_Forge.GNN.model import ForgeGNN
from applications.Agility_Forge.control.mpc import U_J_MM_MAX, U_J_MM_MIN, _rollout_cost

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
SQUARE_DIR = "applications/Agility_Forge/data/dataset_finetuning/square"
BAND_WIDTH_FRAC = 0.2
N_REPEATS = 10

# Reference palette (dataviz skill, light mode), validated for 6 series.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depths", default="1,2,3,5,8,15")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--finetune-root", default=os.path.join(SWEEP_DIR, "finetune"),
                   help="Folder holding finetune/mp_<M>/ runs (checkpoint.pt + metrics.json).")
    p.add_argument("--out-dir", default=SWEEP_DIR, help="Where summary.json and the plots go.")
    p.add_argument("--plots-only", action="store_true",
                   help="Redraw the plots from an existing summary.json (no GPU, no re-timing).")
    return p.parse_args()


def gradient_time(model, mesh_info, x_target, horizon, device):
    H_mm = float(mesh_info.rest_pos[:, 0].max().item())
    x0 = torch.zeros_like(x_target)
    u0 = torch.tile(torch.tensor([0.4, 90.0, 0.5]), (horizon,)).to(device)

    def one_call():
        u_t = u0.clone().requires_grad_(True)
        cost = _rollout_cost(u_t, model, mesh_info, x0, H_mm, BAND_WIDTH_FRAC, x_target, horizon,
                             U_J_MM_MIN, U_J_MM_MAX)
        cost.backward()
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    one_call()  # untimed warm-up
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        one_call()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def _style(ax, title, xlabel, ylabel):
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


def series_colors(n):
    """The fixed categorical order for up to 6 series; a sequential scale
    beyond that (M is ordinal, so lighter-to-darker reads as fewer-to-more)."""
    if n <= len(SERIES):
        return SERIES[:n]
    return [plt.cm.viridis(v) for v in np.linspace(0.0, 0.9, n)]


def plot_per_hit(summary, key, title, ylabel, fname):
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    ends = []
    n_hits = max(len(r[key]) for r in summary["depths"].values())
    for color, (d, r) in zip(series_colors(len(summary["depths"])), summary["depths"].items()):
        vals = r[key]
        hits = list(range(1, len(vals) + 1))
        ax.plot(hits, vals, "-", color=color, linewidth=2, label=f"M = {d}")
        ends.append([vals[-1], d, hits[-1]])
    # End labels, nudged apart so lines that finish close together stay readable.
    ymax = max(max(r[key]) for r in summary["depths"].values())
    gap = 0.045 * ymax
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, d, x in ends:
        ax.annotate(f"M = {d}", (x, y), xytext=(4, 0), textcoords="offset points",
                    color=INK_2, fontsize=8, va="center", annotation_clip=False)
    _style(ax, title, f"Hit ({n_hits}-hit autoregressive rollout, test run seed 3)", ylabel)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, n_hits + 6)
    leg = ax.legend(frameon=True, fontsize=9, labelcolor=INK_2, loc="upper left")
    leg.get_frame().set_edgecolor(GRID)
    plt.tight_layout()
    fig.savefig(os.path.join(summary["out_dir"], fname), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_mean_errors(summary):
    """Mean-over-hits Hausdorff and Chamfer vs. message-passing steps M -- two panels, since
    they're different quantities on different scales."""
    depths = list(summary["depths"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
    for ax, key, color, title, ylabel in [
        (axes[0], "test_mean_hausdorff_mm", SERIES[0], "Mean Test Hausdorff (all hits)", "Hausdorff error (mm)"),
        (axes[1], "test_mean_chamfer_mm2", SERIES[1], "Mean Test Chamfer (all hits)", "Chamfer error (mm²)"),
    ]:
        vals = [summary["depths"][d][key] for d in depths]
        ax.plot([int(d) for d in depths], vals, "-o", color=color, linewidth=2, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2)
        for d, v in zip(depths, vals):
            ax.annotate(f"{v:.2f}", (int(d), v), xytext=(0, 9), textcoords="offset points",
                        ha="center", color=INK_2, fontsize=8)
        _style(ax, title, "Message-passing steps M", ylabel)
        ax.set_xticks([int(d) for d in depths])
        ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(os.path.join(summary["out_dir"], "mean_error_vs_mp_steps.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_gradient_time(summary, horizon):
    depths = list(summary["depths"])
    means = [summary["depths"][d]["gradient_time_s_mean"] for d in depths]
    stds = [summary["depths"][d]["gradient_time_s_std"] for d in depths]
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURFACE)
    ax.errorbar([int(d) for d in depths], means, yerr=stds, fmt="-o", color=SERIES[0], linewidth=2,
                markersize=8, capsize=4, markeredgecolor=SURFACE, markeredgewidth=2)
    for d, m in zip(depths, means):
        ax.annotate(f"{m:.2f}s", (int(d), m), xytext=(0, 9), textcoords="offset points",
                    ha="center", color=INK_2, fontsize=8)
    _style(ax, f"Gradient Time, {horizon}-Hit Horizon", "Message-passing steps M", "Seconds per gradient")
    ax.set_xticks([int(d) for d in depths])
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(os.path.join(summary["out_dir"], "gradient_time_vs_mp_steps.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def make_plots(summary):
    plot_per_hit(summary, "test_hausdorff_mm_per_hit", "Test Hausdorff Error by Message-Passing Steps", "Hausdorff error (mm)",
                 "test_hausdorff_vs_hit.png")
    plot_per_hit(summary, "test_chamfer_mm2_per_hit", "Test Chamfer Error by Message-Passing Steps", "Chamfer error (mm²)",
                 "test_chamfer_vs_hit.png")
    plot_mean_errors(summary)
    plot_gradient_time(summary, summary["horizon"])


def main(args):
    if args.plots_only:
        with open(os.path.join(args.out_dir, "summary.json")) as f:
            make_plots({**json.load(f), "out_dir": args.out_dir})
        print(f"Plots redrawn -> {args.out_dir}")
        return
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device
    mesh_info = build_surface_mesh_info(os.path.join(SQUARE_DIR, "rollout_01", "undeformed.vtu"))
    disp = meshio.read(os.path.join(SQUARE_DIR, "rollout_01", "hit_48_final.vtu")).point_data["Displacement"]
    x_target = torch.tensor(disp[mesh_info.full_to_surface], dtype=torch.float32, device=device)

    summary = {"horizon": args.horizon, "n_repeats": N_REPEATS, "out_dir": args.out_dir,
               "finetune_root": args.finetune_root,
               "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu", "depths": {}}
    for d in args.depths.split(","):
        run_dir = os.path.join(args.finetune_root, f"mp_{d}")
        ckpt = torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location="cpu", weights_only=False)
        a = ckpt["args"]
        model = ForgeGNN(mesh_info, latent_size=a["latent_size"], num_layers=a["num_layers"],
                         message_passing_steps=a["message_passing_steps"]).to(device).eval()
        model.load_state_dict(ckpt["state_dict"])
        mean_s, std_s = gradient_time(model, mesh_info, x_target, args.horizon, device)
        with open(os.path.join(run_dir, "metrics.json")) as f:
            m = json.load(f)
        roll = m["finetuned"]["seed3_rollout"]
        summary["depths"][d] = {
            "gradient_time_s_mean": mean_s, "gradient_time_s_std": std_s,
            "test_hausdorff_mm_per_hit": roll["per_hit_hausdorff_mm"],
            "test_chamfer_mm2_per_hit": roll["per_hit_chamfer_mm2"],
            "test_hit48_hausdorff_mm": roll["per_hit_hausdorff_mm"][-1],
            "test_hit48_chamfer_mm2": roll["per_hit_chamfer_mm2"][-1],
            "test_mean_hausdorff_mm": float(np.mean(roll["per_hit_hausdorff_mm"])),
            "test_mean_chamfer_mm2": float(np.mean(roll["per_hit_chamfer_mm2"])),
            "best_epoch": m["best_epoch"], "early_stopping_on": m.get("early_stopping_on"),
            "n_params": sum(p.numel() for p in model.parameters()),
        }
        print(f"M={d}: gradient {mean_s:.3f}±{std_s:.3f}s | hit-48 Hausdorff "
              f"{roll['per_hit_hausdorff_mm'][-1]:.3f}mm Chamfer {roll['per_hit_chamfer_mm2'][-1]:.4f}mm^2 | "
              f"mean over hits Hausdorff {np.mean(roll['per_hit_hausdorff_mm']):.3f}mm "
              f"Chamfer {np.mean(roll['per_hit_chamfer_mm2']):.4f}mm^2",
              flush=True)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    make_plots(summary)
    print(f"Summary + plots -> {args.out_dir}")


if __name__ == "__main__":
    main(parse_args())
