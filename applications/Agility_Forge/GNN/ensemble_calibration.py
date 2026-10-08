"""Is the deep ensemble's spread a usable uncertainty? Run this before using the
spread penalty in the coil MPC (control/mpc_coil.py --var-weight): GNN
ensembles that differ only by seed can collapse to near-identical predictions
(Vieira et al. 2026, arXiv 2605.22593). If the spread barely correlates with
the error, the fallback is bootstrapping (each member trained on its own
resample of the training hits).

For every cycle of the held-out plain square run and of the MPC runs' own hits
(control/results/coil_mpc_tstate_M5*), each member chains the cycle from its
true start (scanned shape and measured temperature; its own reheat shape step,
the reheat temperature by the rule; coil_T.chain), as the MPC forecast does.
For every hit:
  spread = sum over surface nodes of the across-member variance (1/M) of the
           y and z displacement, mm^2 (the MPC's penalty term s_k);
  error  = sum over surface nodes of the squared y, z error of the members'
           mean shape vs the simulator, mm^2 (the MPC cost's units).
Reports Pearson (raw and log-log) and Spearman correlations per source and
pooled, the median error/spread ratio, and saves a spread-vs-error scatter (log
axes) and spread and error along each MPC run.

Usage (from the repo root):
    python -m applications.Agility_Forge.GNN.ensemble_calibration
    python -m applications.Agility_Forge.GNN.ensemble_calibration --checkpoints <.pt> ... --runs <dir> ... --out <dir>
"""

import argparse
import glob
import json
import logging
import os

import matplotlib
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from applications.Agility_Forge.GNN import coil_T as C

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("coil")

AF = C.AF
CHECKPOINTS = [f"{AF}/GNN/coil_T_sweep_test_square/mp_5/stage3.pt"] + \
              [f"{AF}/GNN/coil_T_ensemble_test_square/seed_{k}/mp_5/stage3.pt" for k in range(1, 5)]
RUNS = [f"{C.NEW_SQ}/square"] + sorted(glob.glob(f"{AF}/control/results/coil_mpc_tstate_M5*"))
OUT = f"{AF}/GNN/coil_T_ensemble_test_square/calibration"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def source_name(run):
    return "plain square (held out)" if run.rstrip("/").endswith("/square") else f"MPC {os.path.basename(run)}"


@torch.no_grad()
def per_hit(models, mesh, cache, dev, cycles):
    """One row per hit: spread and ensemble-mean error (mm^2), plus each member's own error."""
    rows = []
    for c in cycles:
        steps = c["steps"]
        outs = [C.chain(m, mesh, cache, steps, dev) for m in models]
        k = 0
        for j, s in enumerate(steps):
            if s.hit is None:
                continue
            k += 1
            P = torch.stack([o[0][j][0] for o in outs])            # (M, N, 3) predicted displacements
            t = outs[0][1][j][0]                                     # simulator
            rows.append({"hit": int(s.label.split()[-1]), "cycle_start_hit": c["start_hit"], "hit_in_cycle": k,
                         "spread_mm2": float(P[..., 1:].var(dim=0, unbiased=False).sum()),
                         "error_mm2": float(((P.mean(dim=0) - t)[:, 1:] ** 2).sum()),
                         "member_errors_mm2": [float(((p - t)[:, 1:] ** 2).sum()) for p in P]})
    return rows


def stats(rows):
    s = np.array([r["spread_mm2"] for r in rows]); e = np.array([r["error_mm2"] for r in rows])
    if len(rows) < 3:
        return {"n_hits": len(rows)}
    ok = (s > 0) & (e > 0)
    return {"n_hits": len(rows),
            "pearson": float(pearsonr(s, e)[0]),
            "pearson_log": float(pearsonr(np.log(s[ok]), np.log(e[ok]))[0]) if ok.sum() >= 3 else None,
            "spearman": float(spearmanr(s, e)[0]),
            "median_error_over_spread": float(np.median(e[ok] / s[ok])) if ok.any() else None,
            "median_spread_mm2": float(np.median(s)), "median_error_mm2": float(np.median(e))}


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color="#e4e3df", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot(res, out):
    srcs = list(res["sources"])
    mpc = [k for k in srcs if k.startswith("MPC")]
    fig, axes = plt.subplots(1, 1 + len(mpc), figsize=(5.2 * (1 + len(mpc)), 4.4), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    ax = axes[0, 0]; style(ax)
    lo, hi = np.inf, 0.0
    for i, k in enumerate(srcs):
        s = np.array([r["spread_mm2"] for r in res["sources"][k]["rows"]])
        e = np.array([r["error_mm2"] for r in res["sources"][k]["rows"]])
        st = res["sources"][k]["stats"]
        ax.scatter(s, e, s=14, alpha=0.75, color=COLORS[i % len(COLORS)],
                   label=f"{k}: Spearman {st.get('spearman', float('nan')):.2f} (n = {len(s)})")
        pos = np.concatenate([s[s > 0], e[e > 0]])
        if len(pos):
            lo, hi = min(lo, pos.min()), max(hi, pos.max())
    if hi > 0:
        ax.plot([lo, hi], [lo, hi], ls=":", color="#8a8984", lw=1.2, label="error = spread")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("ensemble spread, sum of y,z variance over nodes (mm²)")
    ax.set_ylabel("ensemble-mean error, sum of y,z sq. error (mm²)")
    ax.set_title(f"Per hit, {res['n_members']} members; pooled Spearman {res['pooled'].get('spearman', float('nan')):.2f}",
                 fontsize=10)
    ax.legend(fontsize=7.5, frameon=True, facecolor="white", edgecolor="#d0cfca", loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    for j, k in enumerate(mpc, start=1):
        ax = axes[0, j]; style(ax)
        rows = res["sources"][k]["rows"]
        h = [r["hit"] for r in rows]
        ax.plot(h, [r["error_mm2"] for r in rows], color=COLORS[0], lw=1.4, label="ensemble-mean error")
        ax.plot(h, [r["spread_mm2"] for r in rows], color=COLORS[1], lw=1.4, ls="--", label="spread")
        ax.set_yscale("log"); ax.set_xlabel("hit"); ax.set_ylabel("mm²")
        ax.set_title(k, fontsize=10)
        ax.legend(fontsize=8, frameon=True, facecolor="white", edgecolor="#d0cfca")
    fig.tight_layout()
    fig.savefig(f"{out}/spread_vs_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", default=CHECKPOINTS)
    p.add_argument("--runs", nargs="+", default=RUNS, help="Data or MPC run folders (generate_square_coil_rollout format).")
    p.add_argument("--out", default=OUT)
    p.add_argument("--limit", type=int, default=0, help="Smoke test: first n cycles per run.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = args.device
    mesh = C.Mesh(f"{C.PRE_DIR}/rollout_01/undeformed.vtu", dev)
    cache = C.VtuCache(mesh)
    models = [C.load(mesh, ck, dev).eval() for ck in args.checkpoints]
    res = {"checkpoints": args.checkpoints, "n_members": len(models), "sources": {}}
    for run in args.runs:
        cycles = [c for c in C.new_cycles(run, cache) if c["steps"]]
        if args.limit:
            cycles = cycles[:args.limit]
        rows = per_hit(models, mesh, cache, dev, cycles)
        res["sources"][source_name(run)] = {"run": run, "n_cycles": len(cycles), "stats": stats(rows), "rows": rows}
        log.info(f"{source_name(run)}: {json.dumps(res['sources'][source_name(run)]['stats'])}")
    res["pooled"] = stats([r for v in res["sources"].values() for r in v["rows"]])
    log.info(f"pooled: {json.dumps(res['pooled'])}")
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(f"{args.out}/calibration.json", "w"), indent=1)
    plot(res, args.out)
    log.info(f"wrote {args.out}/calibration.json and spread_vs_error.png")


if __name__ == "__main__":
    main()
