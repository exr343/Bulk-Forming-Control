"""Training loop for the MeshGraphNets forging surrogate.

Usage:
    python -m applications.Agility_Forge.GNN.train \\
        --dataset-dir applications/Agility_Forge/data/dataset --epochs 100

Trains one-hit-ahead only (clean teacher-forced inputs, no noise injection,
no multi-step unrolling -- see GNN/README.md's "Training regime" decision).
The loss is per-node MSE in normalized-target space (ForgeGNN.loss); Chamfer
and Hausdorff distance (see _chamfer_hausdorff_mm) are tracked alongside as
reporting diagnostics only, not part of the backpropagated objective -- per-
node correspondence is already known here, so they'd be strictly weaker
supervision than the exact per-node MSE actually optimized. Disable with
--skip-geometry-metrics if the extra O(N^2) nearest-neighbor search becomes
a wall-clock bottleneck at full scale.

Runtime note: at the settled reference sizing (latent=128,
message_passing_steps=15, ~2.33M params), one batch of 4 examples took ~9.4s
on a 4-thread CPU during development -- this is GPU-bound work, not
something to train on a login node. Run on a GPU node per cluster_setup.md.

After training, one held-out test rollout is rolled out *autoregressively*
(the model's own hit-1 prediction feeds hit-2's input, etc., not the true
per-hit x_k) and written to out-dir/eval/ as true/predicted .vtu pairs, plus
a rollout-MSE-per-hit curve in metrics.json -- this is the first real
measurement of the compounding-error behavior GNN/README.md's "Training
regime" section flagged as an open question to revisit once a baseline
exists.
"""

import argparse
import json
import logging
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from applications.Agility_Forge.GNN.data import (
    build_node_features,
    build_surface_mesh_info,
    load_examples,
    save_surface_vtu,
    split_examples_by_rollout,
    ForgeGNNDataset,
)
from applications.Agility_Forge.GNN.model import ForgeGNN

# Standalone logger, deliberately not `from jax_forge import logger` -- same
# rationale as koopman/train.py: stay decoupled from jax_forge's import-time
# side effects (pyfiglet banner, pulling in the whole JAX/FEM stack). This
# module talks to the FEM pipeline only via plain .vtu/manifest.json files.
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(message)s", datefmt="%m-%d %H:%M:%S"))
    logger.addHandler(_handler)
    logger.propagate = False
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--dataset-dir", default="applications/Agility_Forge/data/dataset")
    parser.add_argument("--rollout-allowlist", default=None,
                         help="Path to a JSON file containing a list of rollout ids to restrict "
                              "training to (e.g. reproducing an exact past snapshot when the "
                              "dataset dir has since grown). Default: use every complete rollout.")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Fraction of rollouts used for training.")

    parser.add_argument("--latent-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2, help="Hidden layers per MLP (each MLP: in -> latent x num_layers -> out).")
    parser.add_argument("--message-passing-steps", type=int, default=15)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stop patience (epochs) on test loss.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-path", default="applications/Agility_Forge/GNN/checkpoint.pt")
    parser.add_argument("--out-dir", default="applications/Agility_Forge/GNN/runs",
                         help="Directory for metrics.json, loss_curves.png, and the eval/ .vtu dump.")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--skip-geometry-metrics", action="store_true",
                         help="Skip the per-epoch Chamfer/Hausdorff tracking (see "
                              "_chamfer_hausdorff_mm's docstring). These are O(N^2) "
                              "nearest-neighbor searches over the ~4,355-node surface mesh, "
                              "recomputed for every training/test example every epoch -- real "
                              "overhead on top of the GNN forward/backward pass, untested at "
                              "full-scale training. Skip if wall-clock time becomes a problem; "
                              "NRMSE (the existing per-node diagnostic) is unaffected either way.")
    return parser.parse_args()


def _chamfer_hausdorff_mm(true_pts: torch.Tensor, pred_pts: torch.Tensor):
    """Symmetric Chamfer distance (mm^2, mean squared nearest-neighbor distance,
    summed both directions) and symmetric Hausdorff distance (mm, max of the
    two one-sided max nearest-neighbor distances) between two (N, 3) point
    sets -- same definitions runs/geometry_error_metrics.py uses post-hoc on
    saved .vtu pairs, and the same style of comparison Table 2 in the
    JAX-FORGE paper (Wright et al., CIRP Annals 2026) uses for scanned-vs-
    simulated surface profiles.

    Deliberately NOT part of the backpropagated loss (see ForgeGNN.loss's
    docstring and GNN/README.md's "Loss function & reported metrics"
    section): per-node correspondence between `true_pts` and `pred_pts` is
    already known here (same fixed mesh topology, same node ordering), so
    the exact per-node MSE actually optimized is strictly better-informed
    supervision than an unordered nearest-neighbor match would be -- Chamfer/
    Hausdorff are for the opposite situation (point clouds with unknown
    correspondence). Tracked here purely as reporting diagnostics, the same
    role NRMSE already plays alongside the real training loss."""
    with torch.no_grad():
        d = torch.cdist(true_pts.unsqueeze(0), pred_pts.unsqueeze(0)).squeeze(0)  # (N, N)
        d_true_to_pred = d.min(dim=1).values
        d_pred_to_true = d.min(dim=0).values
        chamfer = (d_true_to_pred ** 2).mean() + (d_pred_to_true ** 2).mean()
        hausdorff = torch.max(d_true_to_pred.max(), d_pred_to_true.max())
    return chamfer.item(), hausdorff.item()


def _find_reference_vtu(dataset_dir: str, manifest: dict) -> str:
    """Any rollout's undeformed.vtu works identically (mesh is shared across
    every rollout -- see GNN/README.md's "Mesh topology" verification).
    Picks the lowest-numbered complete rollout for determinism."""
    rollouts = sorted({r["rollout"] for r in manifest["records"] if r["kind"] == "undeformed"})
    return os.path.join(dataset_dir, f"rollout_{rollouts[0]:02d}", "undeformed.vtu")


def _compute_target_std(loader) -> float:
    """std of the true displacement-delta target, pooled over every node/
    component/example in `loader` -- computed once per split (train, test),
    since the target data itself doesn't change epoch to epoch. Used as the
    fixed denominator for that split's NRMSE (see `_epoch_pass`/`_rollout_eval`
    and GNN/README.md's "Loss function & reported metrics" section for the
    exact definition: NRMSE = RMSE / this value)."""
    s, ss, n = 0.0, 0.0, 0
    for batch in loader:
        t = batch["target_delta"]
        s += t.sum().item()
        ss += (t**2).sum().item()
        n += t.numel()
    mean = s / max(n, 1)
    var = max(ss / max(n, 1) - mean**2, 0.0)
    return var**0.5


def _epoch_pass(model, loader, device, target_std, rest_pos, optimizer=None, track_geometry=True):
    """Returns (avg_loss, nrmse, chamfer_mm2, hausdorff_mm). `avg_loss` is the
    actual optimized objective (per-node MSE in normalized-target space,
    ForgeGNN.loss) -- used for backprop and for early-stopping/checkpoint
    selection. `nrmse`, `chamfer_mm2`, and `hausdorff_mm` are all physical-
    units diagnostics computed from the SAME forward pass, not backpropagated
    -- reported/plotted only. `nrmse` is RMSE of the denormalized prediction
    vs. the true physical target, divided by `target_std` (NRMSE=1 means "no
    better than always predicting the per-component mean displacement
    delta"). `chamfer_mm2`/`hausdorff_mm` are averaged (per-example mean, per
    `_chamfer_hausdorff_mm`) nearest-neighbor geometry error between the
    predicted and true deformed surface -- see that function's docstring for
    why these are diagnostics, not loss terms. When `track_geometry` is
    False, both are returned as 0.0 (see --skip-geometry-metrics)."""
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, n_batches = 0.0, 0
    se_sum, n_elems = 0.0, 0
    chamfer_sum, hausdorff_sum, n_examples = 0.0, 0.0, 0
    for batch in loader:
        node_features = batch["node_features"].to(device)
        x_k = batch["x_k"].to(device)
        target_delta = batch["target_delta"].to(device)
        with torch.set_grad_enabled(train_mode):
            loss, pred_phys = model.loss_and_predict(node_features, x_k, target_delta)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        with torch.no_grad():
            pred_phys = pred_phys.detach()
            se = (pred_phys - target_delta) ** 2
            se_sum += se.sum().item()
            n_elems += se.numel()
            if track_geometry:
                # Chamfer/Hausdorff are shape-comparison metrics -- they need
                # absolute point positions (rest + displacement), not raw
                # displacement vectors. x_k/target_delta/pred_phys are all
                # displacement-only; anchoring by rest_pos was missing here,
                # which silently computed nearest-neighbor distance between
                # *displacement vectors* instead of deformed-surface points --
                # a different, physically meaningless quantity that happened
                # to still produce a plausible-looking number. Bug found via
                # the "how can hit1's Hausdorff be worse than hit2's despite
                # hit1 having much lower RMSE" investigation.
                true_pts = rest_pos + x_k + target_delta
                pred_pts = rest_pos + x_k + pred_phys
                for b in range(true_pts.shape[0]):
                    c, h = _chamfer_hausdorff_mm(true_pts[b], pred_pts[b])
                    chamfer_sum += c
                    hausdorff_sum += h
                    n_examples += 1
    avg_loss = total_loss / max(n_batches, 1)
    rmse = (se_sum / max(n_elems, 1)) ** 0.5
    nrmse = rmse / max(target_std, 1e-8)
    avg_chamfer = chamfer_sum / max(n_examples, 1) if track_geometry else 0.0
    avg_hausdorff = hausdorff_sum / max(n_examples, 1) if track_geometry else 0.0
    return avg_loss, nrmse, avg_chamfer, avg_hausdorff


def _rollout_eval(model, mesh_info, test_examples, dataset_dir, device, target_std):
    """Autoregressive rollout on one held-out rollout: hit 1 is predicted
    from the true x_0 (zero displacement); each subsequent hit is predicted
    from the model's OWN previous prediction, not the true x_k -- this is
    what actually measures the compounding-error behavior flagged as open in
    GNN/README.md, as opposed to the one-hit-ahead teacher-forced loss above.
    Reports physical-units RMSE (mm, directly comparable to the ~0-5mm
    per-hit displacement magnitudes in this dataset), NRMSE (RMSE /
    `target_std`, the same reference scale `_epoch_pass` uses for the test
    split), and Chamfer (mm^2) / Hausdorff (mm) distance -- see
    `_chamfer_hausdorff_mm`'s docstring -- for each hit.
    """
    rollout_id = test_examples[0].rollout
    hits = sorted([ex for ex in test_examples if ex.rollout == rollout_id], key=lambda e: e.hit)

    model.eval()
    n_surface = mesh_info.rest_pos.shape[0]
    x_k_pred = torch.zeros(1, n_surface, 3, device=device)
    rest_pos_dev = mesh_info.rest_pos.to(device)

    per_hit_rmse_mm = []
    per_hit_nrmse = []
    per_hit_chamfer_mm2 = []
    per_hit_hausdorff_mm = []
    per_hit_inputs = []
    eval_records = []
    with torch.no_grad():
        for ex in hits:
            true_next = ForgeGNNDataset(dataset_dir, [ex], mesh_info)._load_surface_displacement(ex.x_next_vtu).to(device)

            node_features_pred = build_node_features(
                mesh_info, x_k_pred[0], ex.d_j_frac, ex.d_j_mm, ex.x_max_band_mm, ex.R_j_deg, ex.u_j_frac,
            ).unsqueeze(0).to(device)
            delta_pred = model.predict_delta(node_features_pred, x_k_pred, accumulate=False)
            x_next_pred = x_k_pred + delta_pred

            rmse_mm = ((x_next_pred[0] - true_next) ** 2).mean().sqrt().item()
            per_hit_rmse_mm.append(rmse_mm)
            per_hit_nrmse.append(rmse_mm / max(target_std, 1e-8))
            # Absolute positions (rest + displacement), not raw displacement
            # vectors -- see _epoch_pass's matching fix for why this matters.
            chamfer_mm2, hausdorff_mm = _chamfer_hausdorff_mm(
                rest_pos_dev + true_next, rest_pos_dev + x_next_pred[0])
            per_hit_chamfer_mm2.append(chamfer_mm2)
            per_hit_hausdorff_mm.append(hausdorff_mm)
            per_hit_inputs.append({
                "d_j_frac": ex.d_j_frac, "d_j_mm": ex.d_j_mm,
                "R_j_deg": ex.R_j_deg,
                "u_j_frac": ex.u_j_frac, "u_j_mm": ex.u_j_mm,
            })
            eval_records.append((ex.hit, true_next.cpu().numpy(), x_next_pred[0].cpu().numpy()))

            x_k_pred = x_next_pred

    return (rollout_id, per_hit_rmse_mm, per_hit_nrmse, per_hit_chamfer_mm2, per_hit_hausdorff_mm,
            per_hit_inputs, eval_records)


def _dump_eval_table(rollout_id, per_hit_rmse_mm, per_hit_nrmse, per_hit_chamfer_mm2,
                      per_hit_hausdorff_mm, per_hit_inputs, out_dir):
    """One row per hit of the held-out autoregressive rollout eval: the
    hit's control inputs (d_j/R_j/u_j) alongside its RMSE/NRMSE/Chamfer/
    Hausdorff error against ground truth. Companion to `_dump_eval_vtu` --
    same rollout, same hits, tabular form for inspection outside ParaView."""
    import csv
    out_path = os.path.join(out_dir, "eval", f"rollout_{rollout_id:02d}", "eval_table.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["rollout", "hit", "d_j_frac", "d_j_mm", "R_j_deg", "u_j_frac", "u_j_mm",
                  "rmse_mm", "nrmse", "chamfer_mm2", "hausdorff_mm"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, inputs in enumerate(per_hit_inputs):
            writer.writerow({
                "rollout": rollout_id, "hit": i + 1,
                **inputs,
                "rmse_mm": per_hit_rmse_mm[i], "nrmse": per_hit_nrmse[i],
                "chamfer_mm2": per_hit_chamfer_mm2[i], "hausdorff_mm": per_hit_hausdorff_mm[i],
            })
    logger.info(f"Eval table (per-hit inputs + Hausdorff/Chamfer) -> {out_path}")


def _dump_eval_vtu(mesh_info, rollout_id, eval_records, out_dir):
    eval_dir = os.path.join(out_dir, "eval", f"rollout_{rollout_id:02d}")
    for hit, true_disp, pred_disp in eval_records:
        save_surface_vtu(mesh_info, true_disp, os.path.join(eval_dir, f"hit_{hit:02d}_true.vtu"))
        save_surface_vtu(mesh_info, pred_disp, os.path.join(eval_dir, f"hit_{hit:02d}_pred.vtu"))
    logger.info(f"Eval .vtu pairs (autoregressive rollout) -> {eval_dir}")


def _plot_loss_curves(history, out_path):
    """Plots NRMSE (RMSE of the denormalized, physical-units prediction vs.
    the true displacement delta, divided by the true target's std) rather
    than the raw normalized-space training loss -- NRMSE is directly
    interpretable (1.0 = no better than predicting the per-component mean;
    0 = perfect) where the training loss's units/scale are not. This is a
    reporting diagnostic computed alongside the loss, not the optimized
    objective itself -- see GNN/README.md's "Loss function & reported
    metrics" section and `_epoch_pass`'s docstring."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["train_nrmse"], label="train NRMSE")
    ax.plot(history["test_nrmse"], label="test NRMSE")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1,
               label="NRMSE = 1 (no-skill baseline)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("NRMSE = RMSE(Δx) / std(true Δx)")
    ax.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_raw_loss_curve(history, out_path):
    """Plots the actual optimized objective -- per-node MSE in normalized-
    target space (ForgeGNN.loss), train vs. test, one point per epoch.
    Companion to `_plot_loss_curves` (which plots NRMSE, a physical-units
    diagnostic, instead): this is the literal loss value early stopping and
    checkpoint selection are computed from, log-scaled on the y-axis since it
    spans ~2.2 (epoch 1) down to ~0.4 (converged) -- a linear axis compresses
    the converged region where the train/test gap is most visible."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["train_loss"], label="train loss")
    ax.plot(history["test_loss"], label="test loss")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (per-node MSE, normalized-target space)")
    ax.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_geometry_curves(history, out_path):
    """Chamfer/Hausdorff companion to _plot_loss_curves -- same reporting-
    diagnostic role (see _chamfer_hausdorff_mm's docstring), plotted
    separately since they're a different unit/scale than NRMSE."""
    fig, (ax_c, ax_h) = plt.subplots(1, 2, figsize=(12, 5))
    ax_c.plot(history["train_chamfer_mm2"], label="train")
    ax_c.plot(history["test_chamfer_mm2"], label="test")
    ax_c.set_xlabel("epoch")
    ax_c.set_ylabel("Chamfer distance [mm^2]")
    ax_c.legend(fontsize=13)

    ax_h.plot(history["train_hausdorff_mm"], label="train")
    ax_h.plot(history["test_hausdorff_mm"], label="test")
    ax_h.set_xlabel("epoch")
    ax_h.set_ylabel("Hausdorff distance [mm]")
    ax_h.legend(fontsize=13)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    allowed_rollouts = None
    if args.rollout_allowlist:
        with open(args.rollout_allowlist) as f:
            allowed_rollouts = json.load(f)
    examples, manifest = load_examples(args.dataset_dir, allowed_rollouts=allowed_rollouts)
    train_ex, test_ex = split_examples_by_rollout(examples, train_frac=args.train_frac)
    n_train_rollouts = len({ex.rollout for ex in train_ex})
    n_test_rollouts = len({ex.rollout for ex in test_ex})
    logger.info(f"Rollouts: {n_train_rollouts} train / {n_test_rollouts} test "
                f"({len(train_ex)} / {len(test_ex)} single-hit training pairs)")

    reference_vtu = _find_reference_vtu(args.dataset_dir, manifest)
    mesh_info = build_surface_mesh_info(reference_vtu)
    logger.info(f"Surface mesh: {mesh_info.rest_pos.shape[0]} nodes, "
                f"{mesh_info.senders.shape[0]} directed edges (from {reference_vtu})")

    train_dataset = ForgeGNNDataset(args.dataset_dir, train_ex, mesh_info)
    test_dataset = ForgeGNNDataset(args.dataset_dir, test_ex, mesh_info)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = ForgeGNN(
        mesh_info, latent_size=args.latent_size, num_layers=args.num_layers,
        message_passing_steps=args.message_passing_steps,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"ForgeGNN | latent_size={args.latent_size} num_layers={args.num_layers} "
                f"message_passing_steps={args.message_passing_steps} | params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Fixed once per split -- the true target data doesn't change epoch to
    # epoch, so its std (the NRMSE denominator) doesn't either. See
    # `_compute_target_std`'s docstring.
    train_target_std = _compute_target_std(train_loader)
    test_target_std = _compute_target_std(test_loader)
    logger.info(f"std(true Δx): train={train_target_std:.4f}mm test={test_target_std:.4f}mm "
                f"(NRMSE denominators)")

    track_geometry = not args.skip_geometry_metrics
    rest_pos_dev = mesh_info.rest_pos.to(device)
    history = {
        "train_loss": [], "test_loss": [], "train_nrmse": [], "test_nrmse": [],
        "train_chamfer_mm2": [], "test_chamfer_mm2": [],
        "train_hausdorff_mm": [], "test_hausdorff_mm": [],
    }
    best_test_loss = float("inf")
    best_test_nrmse = float("inf")
    best_state_dict = None
    best_epoch = 0
    epochs_since_improve = 0

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_nrmse, train_chamfer, train_hausdorff = _epoch_pass(
            model, train_loader, device, train_target_std, rest_pos_dev,
            optimizer=optimizer, track_geometry=track_geometry)
        test_loss, test_nrmse, test_chamfer, test_hausdorff = _epoch_pass(
            model, test_loader, device, test_target_std, rest_pos_dev, track_geometry=track_geometry)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_nrmse"].append(train_nrmse)
        history["test_nrmse"].append(test_nrmse)
        history["train_chamfer_mm2"].append(train_chamfer)
        history["test_chamfer_mm2"].append(test_chamfer)
        history["train_hausdorff_mm"].append(train_hausdorff)
        history["test_hausdorff_mm"].append(test_hausdorff)

        # Model selection is on test_loss (the actual optimized objective,
        # in normalized space) -- NRMSE/Chamfer/Hausdorff are reporting
        # diagnostics, tracked alongside but not used to pick the checkpoint.
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_nrmse = test_nrmse
            best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch % args.print_every == 0 or epoch == 1:
            geom_str = (f" | Chamfer {train_chamfer:.4f}/{test_chamfer:.4f}mm^2 "
                        f"Hausdorff {train_hausdorff:.4f}/{test_hausdorff:.4f}mm" if track_geometry else "")
            logger.info(f"Epoch {epoch:4d} | train loss {train_loss:.4e} (NRMSE {train_nrmse:.3f}) "
                        f"| test loss {test_loss:.4e} (NRMSE {test_nrmse:.3f}){geom_str}")

        if epochs_since_improve >= args.patience:
            logger.info(f"Early stopping at epoch {epoch} (no test loss improvement for "
                        f"{args.patience} epochs). Restoring epoch {best_epoch}'s weights.")
            break

    elapsed = time.time() - t0
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    os.makedirs(os.path.dirname(args.checkpoint_path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, args.checkpoint_path)
    logger.info(f"Checkpoint saved -> {args.checkpoint_path} (best epoch {best_epoch}, "
                f"test loss {best_test_loss:.4e}, test NRMSE {best_test_nrmse:.3f})")

    (rollout_id, per_hit_rmse_mm, per_hit_nrmse, per_hit_chamfer_mm2, per_hit_hausdorff_mm,
     per_hit_inputs, eval_records) = _rollout_eval(
        model, mesh_info, test_ex, args.dataset_dir, device, test_target_std)
    logger.info(f"Autoregressive rollout eval (rollout {rollout_id}), per-hit RMSE (mm) / NRMSE / "
                f"Chamfer (mm^2) / Hausdorff (mm): "
                + ", ".join(f"hit{h}={r:.4f}mm/{n:.3f}/{c:.4f}/{ha:.4f}"
                            for h, (r, n, c, ha) in enumerate(
                                zip(per_hit_rmse_mm, per_hit_nrmse, per_hit_chamfer_mm2, per_hit_hausdorff_mm), 1)))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "history": history,
            "best_epoch": best_epoch,
            "best_test_loss": best_test_loss,
            "best_test_nrmse": best_test_nrmse,
            "train_target_std_mm": train_target_std,
            "test_target_std_mm": test_target_std,
            "elapsed_s": elapsed,
            "n_params": n_params,
            "rollout_eval_rollout_id": rollout_id,
            "rollout_eval_per_hit_rmse_mm": per_hit_rmse_mm,
            "rollout_eval_per_hit_nrmse": per_hit_nrmse,
            "rollout_eval_per_hit_chamfer_mm2": per_hit_chamfer_mm2,
            "rollout_eval_per_hit_hausdorff_mm": per_hit_hausdorff_mm,
        }, f, indent=2)
    _plot_loss_curves(history, os.path.join(args.out_dir, "loss_curves.png"))
    _plot_raw_loss_curve(history, os.path.join(args.out_dir, "raw_loss_curve.png"))
    if track_geometry:
        _plot_geometry_curves(history, os.path.join(args.out_dir, "geometry_error_curves.png"))
    _dump_eval_vtu(mesh_info, rollout_id, eval_records, args.out_dir)
    _dump_eval_table(rollout_id, per_hit_rmse_mm, per_hit_nrmse, per_hit_chamfer_mm2,
                      per_hit_hausdorff_mm, per_hit_inputs, args.out_dir)
    logger.info(f"Metrics + loss curves + eval .vtu + eval table -> {args.out_dir}")

    return model, history


if __name__ == "__main__":
    train(parse_args())
