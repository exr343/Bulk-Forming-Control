"""Finetune a pretrained ForgeGNN checkpoint on the scheduled square-rod
rollouts in data/dataset_finetuning/ (settled by interview):

  - Start: GNN/mp_sweep/checkpoint_mp_5.pt (message_passing_steps=5).
  - Train: every square / square_jitter_seed* dir in dataset_finetuning
    except the test dir (auto-discovered; --train-dirs overrides).
    Test: square_jitter_seed3. bulge_head is not used.
  - Replay (--replay-ratio, default 1.0): each epoch also trains on a fresh
    random sample of pretraining-train hits (the train side of the split
    checkpoint_mp_5.pt was trained on), equal in number to the square
    training hits, to limit forgetting. The first run (square only, 0.0)
    moved pretraining-test NRMSE 0.298 -> 0.402. Forgetting is still
    measured on that split's test side before and after.
  - Normalizers frozen at their pretrained statistics.
  - Same objective as train.py (ForgeGNN per-node MSE in normalized-target
    space), AdamW lr 1e-5 (10x below pretraining), wd 1e-4, batch 4.
  - Early stopping on test loss (seed 3), patience 20, max 300 epochs --
    the same selection rule train.py uses, so seed 3 both selects the epoch
    and reports the score (slightly optimistic, accepted).

Baseline (pretrained) and finetuned models are both evaluated on seed 3
(one-hit-ahead loss/NRMSE and a full 48-hit autoregressive rollout) and on
the pretraining test split, so every number has a before/after pair.

Each dataset dir has its own manifest with rollout id 1, so examples are
re-labelled with distinct rollout ids and their .vtu paths made absolute
(ForgeGNNDataset joins dataset_dir with an absolute path -> the path itself).

Usage:
    python -m applications.Agility_Forge.GNN.finetune
"""

import argparse
import dataclasses
import glob
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from applications.Agility_Forge.GNN.data import (
    CONTROLS,
    ForgeGNNDataset,
    build_surface_mesh_info,
    load_examples,
    sample_no_change,
    split_examples_by_rollout,
)
from applications.Agility_Forge.GNN.model import ForgeGNN
from applications.Agility_Forge.GNN.train import (
    _compute_target_std,
    _dump_eval_table,
    _dump_eval_vtu,
    _epoch_pass,
    _plot_geometry_curves,
    _plot_loss_curves,
    _plot_raw_loss_curve,
    _rollout_eval,
    logger,
)

FT_DIR = "applications/Agility_Forge/data/dataset_finetuning"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init-checkpoint", default="applications/Agility_Forge/GNN/mp_sweep/checkpoint_mp_5.pt")
    p.add_argument("--train-dirs", default=None,
                   help="Comma-separated. Default: every square/square_jitter_seed* dir in dataset_finetuning "
                        "except --test-dir.")
    p.add_argument("--from-scratch-mp", type=int, default=None,
                   help="Joint-training baseline: ignore --init-checkpoint, start from random weights with this many "
                        "message-passing steps (latent 128, 2 layers, as train.py), let the normalizers learn "
                        "their statistics (not frozen), and skip the pretrained baseline evaluation. Pair with "
                        "--replay-ratio -1 and --lr 1e-4 to train on all uniform + square data at once.")
    p.add_argument("--replay-ratio", type=float, default=1.0,
                   help="Pretraining-train hits mixed into each epoch, as a multiple of the square training "
                        "hits (freshly re-sampled every epoch). 0 = square data only; -1 = every "
                        "pretraining-train hit, every epoch.")
    p.add_argument("--test-dir", default=f"{FT_DIR}/square_jitter_seed3")
    p.add_argument("--val-dir", default=None,
                   help="If set, early stopping watches this run instead of --test-dir, so the test run is "
                        "only touched for the before/after scores. Excluded from auto-discovered train dirs.")
    p.add_argument("--pretrain-dataset-dir", default="applications/Agility_Forge/data/dataset_pretraining")
    p.add_argument("--pretrain-allowlist", default="applications/Agility_Forge/GNN/mp_sweep/rollout_snapshot_383.json",
                   help="Rollout snapshot the init checkpoint was trained on (for the forgetting check).")
    p.add_argument("--pretrain-train-frac", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--control", choices=CONTROLS, default=None,
                   help="Default: the init checkpoint's own control (stroke if it predates the option).")
    p.add_argument("--no-change-frac", type=float, default=0.0,
                   help="Fraction of each epoch made of synthetic no-contact hits (gap control only).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="applications/Agility_Forge/GNN/finetune_square")
    p.add_argument("--checkpoint-path",
                   default="applications/Agility_Forge/GNN/finetune_square/checkpoint_mp_5_finetuned_square.pt")
    return p.parse_args()


def load_dirs(dirs, first_rollout_id):
    """Examples from several single-rollout dataset dirs, relabelled with
    distinct rollout ids and absolute .vtu paths."""
    out, rid = [], first_rollout_id
    for d in dirs:
        exs, _ = load_examples(d)
        for ex in exs:
            out.append(dataclasses.replace(
                ex, rollout=rid,
                x_k_vtu=os.path.abspath(os.path.join(d, ex.x_k_vtu)),
                x_next_vtu=os.path.abspath(os.path.join(d, ex.x_next_vtu))))
        logger.info(f"  rollout id {rid} <- {d} ({len(exs)} hits)")
        rid += 1
    return out, rid


def freeze_normalizers(model):
    """OnlineNormalizer only accumulates while num_accumulations <
    max_accumulations, so capping at the current count freezes it."""
    for norm in (model.node_physical_normalizer, model.edge_normalizer, model.output_normalizer):
        norm.max_accumulations = int(norm.num_accumulations.item())


def evaluate(model, loaders, stds, rest_pos, device):
    out = {}
    for name, loader in loaders.items():
        loss, nrmse, ch, ha = _epoch_pass(model, loader, device, stds[name], rest_pos)
        out[name] = {"loss": loss, "nrmse": nrmse, "chamfer_mm2": ch, "hausdorff_mm": ha}
    return out


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    logger.info("Finetuning data:")
    if args.train_dirs:
        train_dirs = args.train_dirs.split(",")
    else:
        train_dirs = sorted(glob.glob(f"{FT_DIR}/square") + glob.glob(f"{FT_DIR}/square_jitter_seed*"))
        held_out = {os.path.abspath(d) for d in (args.test_dir, args.val_dir) if d}
        train_dirs = [d for d in train_dirs if os.path.abspath(d) not in held_out]
    train_ex, next_id = load_dirs(train_dirs, 1)
    test_ex, next_id = load_dirs([args.test_dir], next_id)
    val_ex = load_dirs([args.val_dir], next_id)[0] if args.val_dir else None
    monitor = "val" if val_ex else "test"
    with open(args.pretrain_allowlist) as f:
        allowed = json.load(f)
    pre_ex, _ = load_examples(args.pretrain_dataset_dir, allowed_rollouts=allowed)
    pre_train_ex, pre_test_ex = split_examples_by_rollout(pre_ex, train_frac=args.pretrain_train_frac)
    pre_train_ex, pre_test_ex = [[dataclasses.replace(
        ex, x_k_vtu=os.path.abspath(os.path.join(args.pretrain_dataset_dir, ex.x_k_vtu)),
        x_next_vtu=os.path.abspath(os.path.join(args.pretrain_dataset_dir, ex.x_next_vtu))) for ex in exs]
        for exs in (pre_train_ex, pre_test_ex)]
    n_replay = (len(pre_train_ex) if args.replay_ratio < 0
                else min(int(round(args.replay_ratio * len(train_ex))), len(pre_train_ex)))
    logger.info(f"Replay: {n_replay} of {len(pre_train_ex)} pretraining-train hits re-sampled each epoch")
    logger.info(f"Early stopping on the {monitor} run"
                + (f" ({len(val_ex)} hits)" if val_ex else ""))
    logger.info(f"Train {len(train_ex)} hits | test {len(test_ex)} hits | pretraining test split "
                f"{len({e.rollout for e in pre_test_ex})} rollouts / {len(pre_test_ex)} hits (forgetting check)")

    mesh_info = build_surface_mesh_info(os.path.join(args.test_dir, "rollout_01", "undeformed.vtu"))
    # One dataset over square + all pretraining-train hits (so the .vtu cache
    # persists across epochs); each epoch samples all square hits plus a
    # fresh random n_replay subset of the pretraining ones.
    if args.control is None:
        args.control = ("stroke" if args.from_scratch_mp is not None else
                        torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)["args"].get(
                            "control", "stroke"))
    assert args.no_change_frac == 0.0 or args.control == "gap", "--no-change-frac needs --control gap"
    logger.info(f"Control input: {args.control}")
    ctl = args.control
    train_dataset = ForgeGNNDataset("", train_ex + pre_train_ex, mesh_info, control=ctl)
    base_examples = train_dataset.examples
    replay_rng = np.random.default_rng(args.seed)
    n_real = len(train_ex) + n_replay
    n_no_change = int(round(args.no_change_frac / (1.0 - args.no_change_frac) * n_real))
    if n_no_change:
        logger.info(f"No-change hits: {n_no_change} per epoch ({args.no_change_frac:.0%})")

    def epoch_train_loader():
        idx = list(range(len(train_ex)))
        if n_replay:
            idx += (len(train_ex) + replay_rng.choice(len(pre_train_ex), n_replay, replace=False)).tolist()
        if n_no_change:
            # Drawn from this epoch's real hits (square + replay), appended past the base list.
            pool = [base_examples[i] for i in idx]
            train_dataset.examples = base_examples + sample_no_change(pool, n_no_change, replay_rng)
            idx += list(range(len(base_examples), len(train_dataset.examples)))
        return DataLoader(train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(idx))

    train_loader = epoch_train_loader()
    test_ds = ForgeGNNDataset("", test_ex, mesh_info, control=ctl)
    test_ex = test_ds.examples
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    pre_loader = DataLoader(ForgeGNNDataset("", pre_test_ex, mesh_info, control=ctl), batch_size=args.batch_size,
                            shuffle=False)
    val_loader = (DataLoader(ForgeGNNDataset("", val_ex, mesh_info, control=ctl), batch_size=args.batch_size,
                             shuffle=False) if val_ex else None)
    stds = {"train": _compute_target_std(train_loader), "test": _compute_target_std(test_loader),
            "pretrain_test": _compute_target_std(pre_loader)}
    if val_loader is not None:
        stds["val"] = _compute_target_std(val_loader)
    logger.info(f"std(true Δx) mm: {stds}")

    scratch = args.from_scratch_mp is not None
    if scratch:
        a = {"latent_size": 128, "num_layers": 2, "message_passing_steps": args.from_scratch_mp, "control": ctl}
        model = ForgeGNN(mesh_info, **a).to(device)
        logger.info(f"From scratch (message_passing_steps={args.from_scratch_mp}); normalizers learn online")
    else:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        a = ckpt["args"]
        model = ForgeGNN(mesh_info, latent_size=a["latent_size"], num_layers=a["num_layers"],
                         message_passing_steps=a["message_passing_steps"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        freeze_normalizers(model)
        logger.info(f"Loaded {args.init_checkpoint} (message_passing_steps={a['message_passing_steps']}); "
                    f"normalizers frozen at {int(model.output_normalizer.num_accumulations.item())} accumulations")

    rest_pos = mesh_info.rest_pos.to(device)
    eval_loaders = {"test": test_loader, "pretrain_test": pre_loader}
    if val_loader is not None:
        eval_loaders["val"] = val_loader
    monitor_loader = val_loader if val_loader is not None else test_loader

    baseline, base_roll = None, None
    if not scratch:
        logger.info("Baseline (pretrained) evaluation...")
        baseline = evaluate(model, eval_loaders, stds, rest_pos, device)
        base_roll = _rollout_eval(model, mesh_info, test_ex, "", device, stds["test"], control=ctl)
        logger.info(f"Baseline: {json.dumps(baseline)} | seed3 rollout final-hit RMSE {base_roll[1][-1]:.4f} mm")
        _dump_eval_vtu(mesh_info, base_roll[0], base_roll[6], os.path.join(args.out_dir, "baseline"))
        _dump_eval_table(*base_roll[:6], os.path.join(args.out_dir, "baseline"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    history = {k: [] for k in ["train_loss", "test_loss", "train_nrmse", "test_nrmse", "train_chamfer_mm2",
                               "test_chamfer_mm2", "train_hausdorff_mm", "test_hausdorff_mm"]}
    # history's "test_*" keys hold whichever run early stopping watches (the
    # val run when --val-dir is set), so train.py's plot helpers work unchanged.
    best_loss = baseline[monitor]["loss"] if baseline else float("inf")
    best_epoch, best_state, since = 0, None, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loader = epoch_train_loader() if epoch > 1 else train_loader
        tr = _epoch_pass(model, train_loader, device, stds["train"], rest_pos, optimizer=optimizer)
        te = _epoch_pass(model, monitor_loader, device, stds[monitor], rest_pos)
        for split, vals in (("train", tr), ("test", te)):
            for key, v in zip(["loss", "nrmse", "chamfer_mm2", "hausdorff_mm"], vals):
                history[f"{split}_{key}"].append(v)
        if te[0] < best_loss:
            best_loss, best_epoch, since = te[0], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        logger.info(f"Epoch {epoch:4d} | train loss {tr[0]:.4e} (NRMSE {tr[1]:.3f}) | {monitor} loss {te[0]:.4e} "
                    f"(NRMSE {te[1]:.3f}) | Chamfer {te[2]:.4f}mm^2 Hausdorff {te[3]:.4f}mm")
        if since >= args.patience:
            logger.info(f"Early stopping at epoch {epoch}; restoring epoch {best_epoch}.")
            break
    elapsed = time.time() - t0

    # Epoch 0 = the pretrained weights themselves; if no epoch beat them, keep them.
    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "args": {**a, "finetune": vars(args), "finetune_best_epoch": best_epoch}},
               args.checkpoint_path)
    logger.info(f"Checkpoint -> {args.checkpoint_path} (best epoch {best_epoch}, {monitor} loss {best_loss:.4e})")

    final = evaluate(model, eval_loaders, stds, rest_pos, device)
    ft_roll = _rollout_eval(model, mesh_info, test_ex, "", device, stds["test"], control=ctl)
    logger.info(f"Finetuned: {json.dumps(final)} | seed3 rollout final-hit RMSE {ft_roll[1][-1]:.4f} mm")
    _dump_eval_vtu(mesh_info, ft_roll[0], ft_roll[6], os.path.join(args.out_dir, "finetuned"))
    _dump_eval_table(*ft_roll[:6], os.path.join(args.out_dir, "finetuned"))

    def roll_dict(r):
        return {"per_hit_rmse_mm": r[1], "per_hit_nrmse": r[2], "per_hit_chamfer_mm2": r[3],
                "per_hit_hausdorff_mm": r[4]}

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"baseline": ({**baseline, "seed3_rollout": roll_dict(base_roll)} if baseline else None),
                   "finetuned": {**final, "seed3_rollout": roll_dict(ft_roll)},
                   "best_epoch": best_epoch, "epochs_run": len(history["train_loss"]),
                   "early_stopping_on": monitor,
                   "target_std_mm": stds, "elapsed_s": elapsed, "history": history,
                   "args": vars(args)}, f, indent=2)
    _plot_loss_curves(history, os.path.join(args.out_dir, "loss_curves.png"))
    _plot_raw_loss_curve(history, os.path.join(args.out_dir, "raw_loss_curve.png"))
    _plot_geometry_curves(history, os.path.join(args.out_dir, "geometry_error_curves.png"))
    logger.info(f"Metrics + plots + baseline/finetuned eval dumps -> {args.out_dir}")


if __name__ == "__main__":
    main(parse_args())
