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
    ForgeGNNDataset,
    build_surface_mesh_info,
    load_examples,
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
    p.add_argument("--replay-ratio", type=float, default=1.0,
                   help="Pretraining-train hits mixed into each epoch, as a multiple of the square training "
                        "hits (freshly re-sampled every epoch). 0 = square data only.")
    p.add_argument("--test-dir", default=f"{FT_DIR}/square_jitter_seed3")
    p.add_argument("--pretrain-dataset-dir", default="applications/Agility_Forge/data/dataset_pretraining")
    p.add_argument("--pretrain-allowlist", default="applications/Agility_Forge/GNN/mp_sweep/rollout_snapshot_383.json",
                   help="Rollout snapshot the init checkpoint was trained on (for the forgetting check).")
    p.add_argument("--pretrain-train-frac", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=20)
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
        train_dirs = [d for d in train_dirs if os.path.abspath(d) != os.path.abspath(args.test_dir)]
    train_ex, next_id = load_dirs(train_dirs, 1)
    test_ex, _ = load_dirs([args.test_dir], next_id)
    with open(args.pretrain_allowlist) as f:
        allowed = json.load(f)
    pre_ex, _ = load_examples(args.pretrain_dataset_dir, allowed_rollouts=allowed)
    pre_train_ex, pre_test_ex = split_examples_by_rollout(pre_ex, train_frac=args.pretrain_train_frac)
    pre_train_ex, pre_test_ex = [[dataclasses.replace(
        ex, x_k_vtu=os.path.abspath(os.path.join(args.pretrain_dataset_dir, ex.x_k_vtu)),
        x_next_vtu=os.path.abspath(os.path.join(args.pretrain_dataset_dir, ex.x_next_vtu))) for ex in exs]
        for exs in (pre_train_ex, pre_test_ex)]
    n_replay = int(round(args.replay_ratio * len(train_ex)))
    logger.info(f"Replay: {n_replay} of {len(pre_train_ex)} pretraining-train hits re-sampled each epoch")
    logger.info(f"Train {len(train_ex)} hits | test {len(test_ex)} hits | pretraining test split "
                f"{len({e.rollout for e in pre_test_ex})} rollouts / {len(pre_test_ex)} hits (forgetting check)")

    mesh_info = build_surface_mesh_info(os.path.join(args.test_dir, "rollout_01", "undeformed.vtu"))
    # One dataset over square + all pretraining-train hits (so the .vtu cache
    # persists across epochs); each epoch samples all square hits plus a
    # fresh random n_replay subset of the pretraining ones.
    train_dataset = ForgeGNNDataset("", train_ex + pre_train_ex, mesh_info)
    replay_rng = np.random.default_rng(args.seed)

    def epoch_train_loader():
        idx = list(range(len(train_ex)))
        if n_replay:
            idx += (len(train_ex) + replay_rng.choice(len(pre_train_ex), n_replay, replace=False)).tolist()
        return DataLoader(train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(idx))

    train_loader = epoch_train_loader()
    test_loader = DataLoader(ForgeGNNDataset("", test_ex, mesh_info), batch_size=args.batch_size, shuffle=False)
    pre_loader = DataLoader(ForgeGNNDataset("", pre_test_ex, mesh_info), batch_size=args.batch_size, shuffle=False)
    stds = {"train": _compute_target_std(train_loader), "test": _compute_target_std(test_loader),
            "pretrain_test": _compute_target_std(pre_loader)}
    logger.info(f"std(true Δx) mm: {stds}")

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

    logger.info("Baseline (pretrained) evaluation...")
    baseline = evaluate(model, eval_loaders, stds, rest_pos, device)
    base_roll = _rollout_eval(model, mesh_info, test_ex, "", device, stds["test"])
    logger.info(f"Baseline: {json.dumps(baseline)} | seed3 rollout final-hit RMSE {base_roll[1][-1]:.4f} mm")
    _dump_eval_vtu(mesh_info, base_roll[0], base_roll[6], os.path.join(args.out_dir, "baseline"))
    _dump_eval_table(*base_roll[:6], os.path.join(args.out_dir, "baseline"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    history = {k: [] for k in ["train_loss", "test_loss", "train_nrmse", "test_nrmse", "train_chamfer_mm2",
                               "test_chamfer_mm2", "train_hausdorff_mm", "test_hausdorff_mm"]}
    best_loss, best_epoch, best_state, since = baseline["test"]["loss"], 0, None, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loader = epoch_train_loader() if epoch > 1 else train_loader
        tr = _epoch_pass(model, train_loader, device, stds["train"], rest_pos, optimizer=optimizer)
        te = _epoch_pass(model, test_loader, device, stds["test"], rest_pos)
        for split, vals in (("train", tr), ("test", te)):
            for key, v in zip(["loss", "nrmse", "chamfer_mm2", "hausdorff_mm"], vals):
                history[f"{split}_{key}"].append(v)
        if te[0] < best_loss:
            best_loss, best_epoch, since = te[0], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        logger.info(f"Epoch {epoch:4d} | train loss {tr[0]:.4e} (NRMSE {tr[1]:.3f}) | test loss {te[0]:.4e} "
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
    logger.info(f"Checkpoint -> {args.checkpoint_path} (best epoch {best_epoch}, test loss {best_loss:.4e})")

    final = evaluate(model, eval_loaders, stds, rest_pos, device)
    ft_roll = _rollout_eval(model, mesh_info, test_ex, "", device, stds["test"])
    logger.info(f"Finetuned: {json.dumps(final)} | seed3 rollout final-hit RMSE {ft_roll[1][-1]:.4f} mm")
    _dump_eval_vtu(mesh_info, ft_roll[0], ft_roll[6], os.path.join(args.out_dir, "finetuned"))
    _dump_eval_table(*ft_roll[:6], os.path.join(args.out_dir, "finetuned"))

    def roll_dict(r):
        return {"per_hit_rmse_mm": r[1], "per_hit_nrmse": r[2], "per_hit_chamfer_mm2": r[3],
                "per_hit_hausdorff_mm": r[4]}

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"baseline": {**baseline, "seed3_rollout": roll_dict(base_roll)},
                   "finetuned": {**final, "seed3_rollout": roll_dict(ft_roll)},
                   "best_epoch": best_epoch, "epochs_run": len(history["train_loss"]),
                   "target_std_mm": stds, "elapsed_s": elapsed, "history": history,
                   "args": vars(args)}, f, indent=2)
    _plot_loss_curves(history, os.path.join(args.out_dir, "loss_curves.png"))
    _plot_raw_loss_curve(history, os.path.join(args.out_dir, "raw_loss_curve.png"))
    _plot_geometry_curves(history, os.path.join(args.out_dir, "geometry_error_curves.png"))
    logger.info(f"Metrics + plots + baseline/finetuned eval dumps -> {args.out_dir}")


if __name__ == "__main__":
    main(parse_args())
