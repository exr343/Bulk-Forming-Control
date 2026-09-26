"""Closed-loop GNN-MPC evaluation across the full pinned 77-rollout test set.

Research question: can receding-horizon GNN-MPC (MPCController.run() --
replanning from the real plant's TRUE state before every hit, horizon
shrinking 5->4->3->2->1) reproduce a part's final geometry, when that part
was originally produced *open-loop* -- generate_dataset.py's blind, randomly
drawn 5-hit schedule, no control optimization at all?

For every rollout in the 77-rollout test split of the pinned 383-rollout
snapshot (mp_sweep/rollout_snapshot_383.json -- the actual split
checkpoint_mp_5.pt was trained/evaluated against, NOT a re-split of the
live, now-401-rollout dataset, which would risk leakage), targets that
rollout's own true hit_05_final.vtu surface geometry and runs the real
closed-loop MPC controller against it.

Model: checkpoint_mp_5.pt (message_passing_steps=5) -- the depth sweep
(GNN/README.md) found M=5 matches or beats M=15 on every accuracy metric at
~1/3 the planning cost, so this switches the control loop off its prior
M=15 default (control/README.md's own "open items" flagged this as a live,
not-yet-done option).

Failure handling (settled by explicit interview): on any real-plant
non-convergence (ForgingPlant.step raising RuntimeError), the ENTIRE
rollout is dropped -- no partial credit for hits that succeeded before the
failure, no retry (retrying with unchanged inputs is meaningless -- the FEM
solve is deterministic given state+control). Tallied in a failure count
reported alongside the results, not silently absorbed into the average.

Presented (per-hit, averaged across successful rollouts only): Chamfer,
Hausdorff, and SQP planning wall-clock time, as a results table
(results.json's "summary_presented") plus a histogram per hit per metric.

Recorded but NOT presented (per explicit instruction -- kept in
results.json's "per_rollout" for possible follow-up, not surfaced in the
table/plots): per-hit RMSE against target, the actual applied controls
(d_j_frac, R_j_deg, u_j_mm) for every real step of every rollout, and each
rollout's own true per-hit trajectory's distance to its own hit-5 state (a
free "for-reference" curve -- costs nothing extra since those .vtu files
already exist from data generation; no additional real-plant simulation
needed since the stored trajectory already *is* the open-loop replay of
that rollout's true controls through the real plant).

Usage (single job, run on one H100 -- see
slurm_scripts/submit_closed_loop_test_set_eval.sh):
    python -m applications.Agility_Forge.control.eval_closed_loop_test_set \\
        --out-dir applications/Agility_Forge/control/eval_closed_loop_test_set
"""

import argparse
import json
import os
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
import torch

from applications.Agility_Forge.GNN.data import build_surface_mesh_info, load_examples, split_examples_by_rollout
from applications.Agility_Forge.GNN.model import ForgeGNN
from applications.Agility_Forge.GNN.train import _chamfer_hausdorff_mm
from applications.Agility_Forge.control.mpc import MPCController
from applications.Agility_Forge.control.plant_interface import ForgingPlant, load_default_billet, extract_surface_state

DATASET_DIR = "applications/Agility_Forge/data/dataset_pretraining"
SNAPSHOT_PATH = "applications/Agility_Forge/GNN/mp_sweep/rollout_snapshot_383.json"
CHECKPOINT_PATH = "applications/Agility_Forge/GNN/mp_sweep/checkpoint_mp_5.pt"
MP_STEPS = 5
BAND_WIDTH_FRAC = 0.2  # dataset default, matches mpc.py/run_closed_loop.py convention
N_HITS = 5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_test_rollout_ids():
    """The 77-rollout test split of the pinned 383-rollout snapshot -- the
    same split checkpoint_mp_5.pt was trained/evaluated against (see
    mp_sweep/full_test_set_eval.py's identical construction)."""
    with open(SNAPSHOT_PATH) as f:
        allowed = json.load(f)
    examples, _ = load_examples(DATASET_DIR, allowed_rollouts=allowed)
    _, test_ex = split_examples_by_rollout(examples, train_frac=0.8)
    return sorted({ex.rollout for ex in test_ex})


def _load_true_surface_disp(rollout_id: int, hit_idx: int, mesh_info) -> torch.Tensor:
    """hit_idx=0 -> undeformed.vtu, hit_idx=1..5 -> hit_0{hit_idx}_final.vtu."""
    fname = "undeformed.vtu" if hit_idx == 0 else f"hit_{hit_idx:02d}_final.vtu"
    m = meshio.read(os.path.join(DATASET_DIR, f"rollout_{rollout_id:02d}", fname))
    disp_full = m.point_data["Displacement"].astype(np.float32)
    return torch.tensor(disp_full[mesh_info.full_to_surface], dtype=torch.float32)


def _hit_metrics(rest_pos, x_k, x_ref):
    chamfer, hausdorff = _chamfer_hausdorff_mm(rest_pos + x_ref, rest_pos + x_k)
    rmse = ((x_k - x_ref) ** 2).mean().sqrt().item()
    return chamfer, hausdorff, rmse


def _summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "median": float(np.median(arr)),
            "std": float(arr.std()), "n": int(arr.size)}


def _build_summary(per_hit_chamfer, per_hit_hausdorff, per_hit_time):
    summary = {}
    for k in range(1, N_HITS + 1):
        if not per_hit_chamfer[k]:
            continue
        summary[k] = {
            "chamfer_mm2": _summarize(per_hit_chamfer[k]),
            "hausdorff_mm": _summarize(per_hit_hausdorff[k]),
            "plan_time_s": _summarize(per_hit_time[k]),
        }
    return summary


def _save_results(out_dir, test_rollout_ids, per_rollout_records, failed_rollouts,
                   per_hit_chamfer, per_hit_hausdorff, per_hit_time):
    """Called after every rollout (success or failure), not just at the
    end -- so a scancel'd or crashed job still leaves every rollout's
    results usable, matching run_closed_loop.py's step_callback convention
    for the same reason. Overwrites results.json each time (cheap: this
    dict tops out around a few MB for 77 rollouts)."""
    results = {
        "n_test_rollouts": len(test_rollout_ids),
        "n_succeeded": len(per_rollout_records),
        "n_failed": len(failed_rollouts),
        "failed_rollouts": failed_rollouts,
        "summary_presented": _build_summary(per_hit_chamfer, per_hit_hausdorff, per_hit_time),
        "per_rollout": per_rollout_records,
    }
    tmp_path = os.path.join(out_dir, "results.json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, os.path.join(out_dir, "results.json"))  # atomic swap
    return results


def _plot_histograms(per_hit_values, metric_label, out_path):
    hits = sorted(per_hit_values)
    fig, axes = plt.subplots(1, len(hits), figsize=(4 * len(hits), 4), sharey=False)
    if len(hits) == 1:
        axes = [axes]
    for ax, h in zip(axes, hits):
        vals = per_hit_values[h]
        ax.hist(vals, bins=min(20, max(5, len(vals) // 3)), color="#4C72B0", edgecolor="white")
        ax.set_title(f"hit {h} (n={len(vals)})")
        ax.set_xlabel(metric_label)
        if h == hits[0]:
            ax.set_ylabel("count")
    fig.suptitle(f"{metric_label} per hit, across test-set rollouts")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    mesh_info = build_surface_mesh_info(os.path.join(DATASET_DIR, "rollout_01", "undeformed.vtu"))
    rest_pos = mesh_info.rest_pos.to(args.device)

    model = ForgeGNN(mesh_info, latent_size=128, num_layers=2, message_passing_steps=MP_STEPS)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    mesh, R, H, T_linear_fn = load_default_billet()
    plant = ForgingPlant(mesh, R, H, T_linear_fn)

    test_rollout_ids = load_test_rollout_ids()
    print(f"test rollouts: {len(test_rollout_ids)}", flush=True)

    per_hit_chamfer, per_hit_hausdorff, per_hit_time = (defaultdict(list) for _ in range(3))
    per_rollout_records = {}
    failed_rollouts = []

    t_start = time.time()
    for r in test_rollout_ids:
        print(f"=== rollout {r} ({len(per_rollout_records)}/{len(test_rollout_ids)} done so far, "
              f"{len(failed_rollouts)} failed, {time.time() - t_start:.0f}s elapsed) ===", flush=True)

        target = _load_true_surface_disp(r, N_HITS, mesh_info).to(args.device)

        # Free "for-reference" baseline: this rollout's own true per-hit
        # trajectory's distance to its own final state. No extra real-plant
        # simulation -- these .vtu files already exist from data generation
        # (that trajectory already IS the open-loop replay of this
        # rollout's true controls through the real plant).
        true_traj = {"chamfer_mm2": {}, "hausdorff_mm": {}, "rmse_mm": {}}
        target_cpu = target.cpu()
        for k in range(1, N_HITS + 1):
            true_k = _load_true_surface_disp(r, k, mesh_info)
            c, h, rmse = _hit_metrics(mesh_info.rest_pos, true_k, target_cpu)
            true_traj["chamfer_mm2"][k] = c
            true_traj["hausdorff_mm"][k] = h
            true_traj["rmse_mm"][k] = rmse

        controller = MPCController(model, mesh_info, horizon=N_HITS, target_state=target,
                                    band_width_frac=BAND_WIDTH_FRAC, device=args.device)

        recorded_states = {}

        def step_callback(step_idx, state, hit):
            recorded_states[step_idx + 1] = extract_surface_state(state.sol_u, mesh_info).to(args.device)

        try:
            applied_controls, plan_vs_actual_rmse_mm, plan_time_s, _ = controller.run(
                plant, n_hits=N_HITS, step_callback=step_callback)
        except RuntimeError as e:
            failed_at = len(recorded_states) + 1
            print(f"  FAILED at hit {failed_at} (non-convergence), dropping rollout: {e}", flush=True)
            failed_rollouts.append({"rollout": r, "failed_at_hit": failed_at})
            _save_results(args.out_dir, test_rollout_ids, per_rollout_records, failed_rollouts,
                          per_hit_chamfer, per_hit_hausdorff, per_hit_time)
            continue

        rollout_chamfer, rollout_hausdorff, rollout_rmse = {}, {}, {}
        for k in range(1, N_HITS + 1):
            c, h, rmse = _hit_metrics(rest_pos, recorded_states[k], target)
            rollout_chamfer[k] = c
            rollout_hausdorff[k] = h
            rollout_rmse[k] = rmse
            per_hit_chamfer[k].append(c)
            per_hit_hausdorff[k].append(h)
            per_hit_time[k].append(plan_time_s[k - 1])

        per_rollout_records[r] = {
            "chamfer_mm2": rollout_chamfer,
            "hausdorff_mm": rollout_hausdorff,
            "rmse_mm": rollout_rmse,
            "plan_time_s": {k: plan_time_s[k - 1] for k in range(1, N_HITS + 1)},
            "plan_vs_actual_rmse_mm": {k: plan_vs_actual_rmse_mm[k - 1] for k in range(1, N_HITS + 1)},
            "applied_controls": {
                k: {"d_j_frac": applied_controls[k - 1][0], "R_j_deg": applied_controls[k - 1][1],
                    "u_j_mm": applied_controls[k - 1][2]}
                for k in range(1, N_HITS + 1)
            },
            "true_trajectory_baseline": true_traj,
        }
        print(f"  hit {N_HITS}: chamfer={rollout_chamfer[N_HITS]:.4f}mm^2 "
              f"hausdorff={rollout_hausdorff[N_HITS]:.4f}mm rmse={rollout_rmse[N_HITS]:.4f}mm", flush=True)
        _save_results(args.out_dir, test_rollout_ids, per_rollout_records, failed_rollouts,
                      per_hit_chamfer, per_hit_hausdorff, per_hit_time)

    results = _save_results(args.out_dir, test_rollout_ids, per_rollout_records, failed_rollouts,
                            per_hit_chamfer, per_hit_hausdorff, per_hit_time)
    summary_presented = results["summary_presented"]

    print(f"\n=== summary ({len(per_rollout_records)}/{len(test_rollout_ids)} rollouts succeeded, "
          f"{len(failed_rollouts)} failed) ===")
    for k in range(1, N_HITS + 1):
        if k not in summary_presented:
            print(f"hit {k}: no successful rollouts reached this hit")
            continue
        s = summary_presented[k]
        print(f"hit {k}: chamfer={s['chamfer_mm2']['mean']:.4f}+-{s['chamfer_mm2']['std']:.4f}mm^2  "
              f"hausdorff={s['hausdorff_mm']['mean']:.4f}+-{s['hausdorff_mm']['std']:.4f}mm  "
              f"plan_time={s['plan_time_s']['mean']:.1f}+-{s['plan_time_s']['std']:.1f}s")

    _plot_histograms(per_hit_chamfer, "Chamfer (mm^2)", os.path.join(args.out_dir, "hist_chamfer.png"))
    _plot_histograms(per_hit_hausdorff, "Hausdorff (mm)", os.path.join(args.out_dir, "hist_hausdorff.png"))
    _plot_histograms(per_hit_time, "SQP plan time (s)", os.path.join(args.out_dir, "hist_plan_time.png"))

    print(f"\nSaved: {args.out_dir}/results.json, hist_chamfer.png, hist_hausdorff.png, hist_plan_time.png")


if __name__ == "__main__":
    main()
