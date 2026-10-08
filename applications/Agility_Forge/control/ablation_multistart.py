"""Checks and comparison for the multi-start planner (mpc_coil.Planner, random
shooting + top-K SLSQP refinement), on the GNN only (no simulator).

Planning problems: 6 cycle-start scans (shape + measured surface temperature)
of the plain square 5-pass run, which is held out of the GNN's training, near
hits 1, 13, 25, 37, 49 and 61, each planned toward the ideal 10.6 mm square.

Checks:
  1. batched features/cost == per-sample features/cost (node_features, forecast);
  2. equivalence: N = K = 1 (the default start) == a reference copy of the
     previous single-start solve (same SLSQP call, no coil-zone constraint);
  3. determinism: the same seed twice gives the same selected starts and plan;
  4. gradients: refinement start gradients finite and nonzero; screening runs
     with gradients off (asserted inside Planner._screen).
Deterministic GPU algorithms are on (set in mpc_coil on import).
Comparison (per state): (a) top-K from N = 200, (b) K = 5 uniform random starts,
(c) the single default start: final model cost and wall time; plus time per
forward rollout (single and batched) and function evaluations per SLSQP solve.

    python -m applications.Agility_Forge.control.ablation_multistart [--out <dir>]
"""
import argparse
import json
import os
import time

import numpy as onp
import torch
import meshio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from applications.Agility_Forge.control.mpc_coil import (
    AF, MESH_VTU, TARGET_VTU, HITS_PER_CYCLE, Mesh, Planner, load_gnn, forecast, cross_section_cost,
    node_features_batch, forecast_cost_batch)
from applications.Agility_Forge.GNN.coil_T import node_features, Hit as GHit

RUN = f"{AF}/data/dataset_die12_coil_5pass/square"
CKPT = f"{AF}/GNN/coil_T_sweep_test_square/mp_5/stage3.pt"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def scans(mesh, dev):
    """[(label, cycle, start_hit, first, x_scan, T_scan)] near hits 1, 13, ..., 61."""
    m = json.load(open(f"{RUN}/manifest.json"))
    recs = {r["hit"]: r for r in m["records"] if r["kind"] == "hit_final"}
    starts = sorted({r["hit"] for r in m["reheats"]})
    out = []
    for want in (1, 13, 25, 37, 49, 61):
        h0 = min(starts, key=lambda h: abs(h - want))
        cyc = starts.index(h0) + 1
        vtu = f"{RUN}/rollout_01/undeformed.vtu" if h0 == 1 else f"{RUN}/{recs[h0 - 1]['vtu_path']}"
        p = meshio.read(vtu)
        x = torch.tensor(p.point_data["Displacement"][mesh.full_to_surface], dtype=torch.float32, device=dev)
        T = torch.tensor(p.point_data["Temperature"][mesh.full_to_surface, 0], dtype=torch.float32, device=dev)
        if h0 == 1:
            x = torch.zeros_like(x)
        out.append((f"cycle {cyc} (before hit {h0})", cyc, h0, h0 == 1, x, T))
    return out


def reference_single_start(planner, x_scan, T_scan, first, pointer):
    """The previous single-start solve, copied (minus the removed coil-zone constraint)."""
    b, x0, hi = planner.bounds(x_scan)
    lo = onp.array([v[0] for v in b]); span = onp.array([v[1] - v[0] for v in b])
    u0 = onp.clip(planner.initial_guess(pointer, x0, hi), lo, lo + span)

    def cost_grad(z):
        u = torch.tensor(lo + z * span, dtype=torch.float32, device=planner.device, requires_grad=True)
        shapes, _ = forecast(planner.model, planner.mesh, x_scan, T_scan, u.view(HITS_PER_CYCLE, 3), first)
        cost = cross_section_cost(shapes, planner.x_tgt)
        cost.backward()
        return cost.item(), u.grad.cpu().numpy().astype(onp.float64) * span

    z0 = (u0 - lo) / span
    c0 = max(cost_grad(z0)[0], 1e-12)
    res = minimize(lambda z: tuple(v / c0 for v in cost_grad(z)), z0, jac=True, method="SLSQP",
                   bounds=[(0.0, 1.0)] * len(z0), options={"maxiter": planner.maxiter})
    return lo + res.x * span, float(res.fun) * c0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{AF}/control/results/ablation_multistart")
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--sample-batch", type=int, default=50)
    ap.add_argument("--plot-only", action="store_true", help="Redraw comparison.png from results.json.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.plot_only:
        return plot(json.load(open(f"{args.out}/results.json")), args.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mesh = Mesh(MESH_VTU, dev)
    model = load_gnn(mesh, CKPT, dev); model.freeze_normalizers(); model.eval()
    x_tgt = torch.tensor(meshio.read(TARGET_VTU).point_data["Displacement"], dtype=torch.float32, device=dev)
    mk = lambda n, k: Planner(model, mesh, x_tgt, dev, 100, n, k, args.sample_batch, 0)
    states = scans(mesh, dev)
    res = {"device": dev, "checkpoint": CKPT, "run": RUN, "n_samples": args.n_samples, "top_k": args.top_k,
           "sample_batch": args.sample_batch, "checks": {}, "states": []}
    P = mk(args.n_samples, args.top_k)
    res["training_action_ranges"] = P.train_ranges
    print("training-data action ranges:", json.dumps(P.train_ranges), flush=True)

    # ---- check 1: batched features and cost vs per-sample
    lab, cyc, h0, first, x, T = states[2]
    b, x0, hi = P.bounds(x)
    lo = onp.array([v[0] for v in b]); span = onp.array([v[1] - v[0] for v in b])
    U = P._sample(8, lo, span, onp.random.default_rng(123))
    ctrl = torch.tensor(U, dtype=torch.float32, device=dev).view(-1, HITS_PER_CYCLE, 3)
    with torch.no_grad():
        fb = node_features_batch(mesh, x[None].expand(8, -1, -1), T[None].expand(8, -1), ctrl[:, 0, 0], ctrl[:, 0, 1], ctrl[:, 0, 2])
        fs = torch.stack([node_features(mesh, x, T, GHit(ctrl[i, 0, 1], ctrl[i, 0, 2], center=ctrl[i, 0, 0])) for i in range(8)])
        cb = forecast_cost_batch(model, mesh, x, T, ctrl, first, x_tgt)
        cs = torch.stack([cross_section_cost(forecast(model, mesh, x, T, ctrl[i], first)[0], x_tgt) for i in range(8)])
    res["checks"]["batched_features_max_abs_diff"] = float((fb - fs).abs().max())
    res["checks"]["batched_cost_max_rel_diff"] = float(((cb - cs).abs() / cs.abs()).max())
    print("check 1:", res["checks"], flush=True)

    # ---- timing: one forward rollout (6 hits + reheat), single vs batched
    with torch.no_grad():
        for _ in range(2):
            forecast(model, mesh, x, T, ctrl[0], first)
        if dev == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(8):
            forecast(model, mesh, x, T, ctrl[i], first)
        if dev == "cuda": torch.cuda.synchronize()
        t_single = (time.perf_counter() - t0) / 8
        Ub = torch.tensor(P._sample(args.sample_batch, lo, span, onp.random.default_rng(7)), dtype=torch.float32,
                          device=dev).view(-1, HITS_PER_CYCLE, 3)
        forecast_cost_batch(model, mesh, x, T, Ub, first, x_tgt)
        if dev == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        forecast_cost_batch(model, mesh, x, T, Ub, first, x_tgt)
        if dev == "cuda": torch.cuda.synchronize()
        t_batch = (time.perf_counter() - t0) / args.sample_batch
    res["timing"] = {"forward_rollout_single_s": t_single, "forward_rollout_batched_per_sample_s": t_batch}
    print("timing:", res["timing"], flush=True)

    # ---- check 2: N = K = 1 equals the previous single-start solve
    u_new, info_new = mk(1, 1).plan(x, T, first, 3 * (cyc - 1), cycle=cyc)
    u_ref, c_ref = reference_single_start(P, x, T, first, 3 * (cyc - 1))
    res["checks"]["equivalence"] = {"new_final_cost": info_new["final_cost"], "reference_final_cost": c_ref,
                                    "rel_diff": abs(info_new["final_cost"] - c_ref) / c_ref,
                                    "max_abs_plan_diff": float(onp.abs(u_new.reshape(-1) - u_ref).max())}
    print("check 2:", res["checks"]["equivalence"], flush=True)

    # ---- check 3: determinism (same seed twice)
    pa = mk(100, 5); u1, i1 = pa.plan(x, T, first, 0, cycle=cyc); u2, i2 = pa.plan(x, T, first, 0, cycle=cyc)
    res["checks"]["determinism"] = {"same_selected": i1["multistart"]["selected"] == i2["multistart"]["selected"],
                                    "max_abs_plan_diff": float(onp.abs(u1 - u2).max()),
                                    "max_abs_screen_cost_diff": float(onp.abs(onp.array(i1["multistart"]["screening_costs"])
                                                                              - onp.array(i2["multistart"]["screening_costs"])).max())}
    print("check 3:", res["checks"]["determinism"], flush=True)

    # ---- comparison
    arms = [("topk", f"Top-{args.top_k} of N = {args.n_samples}"), ("random", f"{args.top_k} random starts"),
            ("default", "Single default start")]
    for lab, cyc, h0, first, x, T in states:
        row = {"state": lab, "cycle": cyc, "start_hit": h0, "arms": {}}
        for mode, _ in arms:
            t0 = time.perf_counter()
            u, info = P.plan(x, T, first, 3 * (cyc - 1), cycle=cyc, mode=mode)
            d = info["multistart"]
            row["arms"][mode] = {"final_cost": info["final_cost"], "wall_s": time.perf_counter() - t0, "time_s": d["time_s"],
                                 "best_start": d["best_start"],
                                 "starts": [{k: s[k] for k in ("start", "screen_cost", "initial_cost", "final_cost", "nit",
                                                                 "nfev", "message", "time_s", "start_grad_norm",
                                                                 "start_grad_finite", "n_at_bound")} for s in d["starts"]],
                                 "plan": u.tolist()}
            print(f"[{lab}] {mode:7s}: final cost {info['final_cost']:.4g}, {time.perf_counter() - t0:.1f} s, "
                  f"nfev {[s['nfev'] for s in d['starts']]}", flush=True)
        res["states"].append(row)
    allstarts = [s for r in res["states"] for a in r["arms"].values() for s in a["starts"]]
    res["checks"]["gradients"] = {"all_start_grads_finite": all(s["start_grad_finite"] for s in allstarts),
                                  "min_start_grad_norm": min(s["start_grad_norm"] for s in allstarts)}
    res["summary"] = {}
    for mode, name in arms:
        c = onp.array([r["arms"][mode]["final_cost"] for r in res["states"]])
        cd = onp.array([r["arms"]["default"]["final_cost"] for r in res["states"]])
        w = onp.array([r["arms"][mode]["wall_s"] for r in res["states"]])
        nf = [s["nfev"] for r in res["states"] for s in r["arms"][mode]["starts"]]
        res["summary"][mode] = {"name": name, "mean_cost_vs_default": float((c / cd).mean()),
                                "wins_vs_default": int((c < cd * (1 - 1e-6)).sum()), "mean_wall_s": float(w.mean()),
                                "mean_nfev_per_solve": float(onp.mean(nf)),
                                "solves_ending_on_a_bound_var": int(sum(s["n_at_bound"] > 0 for r in res["states"]
                                                                       for s in r["arms"][mode]["starts"]))}
    json.dump(res, open(f"{args.out}/results.json", "w"), indent=1)
    print(json.dumps(res["summary"], indent=1))

    plot(res, args.out)


def plot(res, out):
    """Final GNN cost relative to the single default start (dots, one per planning
    problem and method) and planning wall time (bars)."""
    arms = [(m, res["summary"][m]["name"]) for m in ("topk", "random", "default")]
    plt.rcParams.update({"font.size": 9.5, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"wspace": 0.25})
    xs = onp.arange(len(res["states"])); off = 0.22
    rcs = []
    for j, (mode, name) in enumerate(arms):
        rc = [r["arms"][mode]["final_cost"] / r["arms"]["default"]["final_cost"] for r in res["states"]]
        rcs += rc
        a1.plot(xs + (j - 1) * off, rc, "o", ms=8, color=COLORS[j], mec="white", mew=1.2, label=name, zorder=3)
        a2.bar(xs + (j - 1) * off, [r["arms"][mode]["wall_s"] for r in res["states"]], off * 0.92, color=COLORS[j], label=name)
    a1.axhline(1.0, color="#8a8984", lw=1, ls=(0, (1, 2)), zorder=1)
    a1.set_ylim(min(rcs) - 0.03, max(rcs) + 0.03)
    for ax, t, yl in ((a1, "Final GNN Cost / Default-Start Cost (lower is better)", "Cost / default-start cost"),
                      (a2, "Planning Wall Time per Cycle", "Seconds")):
        ax.set_xticks(xs); ax.set_xticklabels([f"hit {r['start_hit']}" for r in res["states"]])
        ax.set_title(t, loc="left", fontsize=10.5); ax.set_ylabel(yl); ax.set_xlabel("Planning problem (scan before)")
        ax.grid(True, axis="y", color="#e6e5e0"); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    a2.legend(frameon=True, facecolor="white", edgecolor="#8a8984", fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 1.0))
    a2.set_ylim(0, max(r["arms"]["topk"]["wall_s"] for r in res["states"]) * 1.35)
    fig.suptitle("Multi-Start Planner vs Single Start (GNN only, held-out plain square states, M = 5)", x=0.07,
                 ha="left", fontsize=12.5, fontweight="bold")
    fig.savefig(f"{out}/comparison.png", dpi=110, bbox_inches="tight")
    print("wrote", f"{out}/results.json and comparison.png")


if __name__ == "__main__":
    main()
