"""Training loop for the Koopman Autoencoder.

Usage:
    python -m applications.Agility_Forge.koopman.train \\
        --dataset-dir applications/Agility_Forge/data/dataset_pretraining \\
        --latent-dim 512 --epochs 100

Trained over a window of K+1 = n_hits_per_rollout+1 consecutive states and K
controls per rollout (see koopman/dataset.py), with three loss terms:

    L_id  -- encode -> decode every state in the window (reconstruction).
             Needed here because, unlike the LRAN_LD reference this model is
             adapted from, the decoder is a *learned* linear map (x is not
             concatenated into the latent), so decoding is not exact by
             construction.
    L_fwd -- roll the latent dynamics forward from z_0 under A/B and the
             observed controls, decode each step, compare to the true future
             states. This is the metric that matters most for stage 3's MPC
             loop, which will roll this model forward multiple hits.
    L_lin -- same rollout, but compare predicted latents directly to the
             *encoded* true future states (no decoding) -- keeps the lifting
             map itself consistent with the learned LTI dynamics, not just
             consistent after decoding.

Optionally, L_eig penalizes eigenvalues of A outside the unit circle
(gamma_eig, default off).
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
import torch.nn as nn
from torch.utils.data import DataLoader

from applications.Agility_Forge.koopman.dataset import (
    build_rollout_dataset,
    denormalize_state,
    save_state_vtu,
)
from applications.Agility_Forge.koopman.model import KoopmanAutoencoder

# Standalone logger (deliberately not `from jax_forge import logger`: that
# import triggers jax_forge's pyfiglet banner and pulls in the whole FEM/JAX
# stack, which koopman/README.md explicitly keeps decoupled from -- this
# module talks to jax_forge only via plain NumPy .vtu files on disk).
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(message)s", datefmt="%m-%d %H:%M:%S"))
    logger.addHandler(_handler)
    logger.propagate = False
logger.setLevel(logging.INFO)

LOSS_TERM_EXPLANATIONS = {
    "id": "L_id  (reconstruction): encode -> decode every state in the window vs. the true state.",
    "fwd": "L_fwd (forward prediction): latent rollout, decoded, vs. true future states -- what MPC needs.",
    "lin": "L_lin (latent linearity): latent rollout vs. encoded true future states (no decode).",
    "eig": "L_eig (eigenvalue stability): penalizes |eig(A)| > 1. Off by default (gamma_eig=0).",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--dataset-dir", default="applications/Agility_Forge/data/dataset_pretraining")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Fraction of rollouts used for training.")

    parser.add_argument("--n-pod-modes", type=int, default=75,
                         help="r: dimension of the fixed POD-reduced representation the encoder "
                              "actually receives (state -> POD-project onto this many modes, fit "
                              "on training rollouts only -- see koopman/dataset.py's "
                              "compute_pod_basis -- then lifted to n_z by the Lifting network).")
    parser.add_argument("--latent-dim", type=int, default=250, help="n_z: full dimension of z = Psi(a).")
    parser.add_argument("--block-widths", type=lambda s: [int(w) for w in s.split(",")],
                         default=[250, 250, 250, 250],
                         help="Comma-separated hidden width per residual block, e.g. '250,250,250,250' "
                              "for 4 blocks at constant width. A Linear+activation transition bridges "
                              "each pair of blocks whose widths differ (skipped when they match, so "
                              "constant-width blocks chain directly with no extra layers).")
    parser.add_argument("--layers-per-block", type=int, default=8,
                         help="[LayerNorm -> Linear(width,width) -> activation] layers per residual "
                              "block (each block wraps one skip connection: out = in + F(in)).")
    parser.add_argument("--activation", default="Tanh", choices=["Tanh", "ReLU", "LeakyReLU", "Sigmoid"])
    parser.add_argument("--init-scale", type=float, default=0.99, help="Initial spectral radius of A.")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--gamma-id", type=float, default=1.0)
    parser.add_argument("--gamma-fwd", type=float, default=1.0)
    parser.add_argument("--gamma-lin", type=float, default=1.0)
    parser.add_argument("--gamma-eig", type=float, default=0.0, help="Weight on eigenvalue stability loss (0 = off).")
    parser.add_argument("--patience", type=int, default=30, help="Early-stop patience (epochs) on test L_fwd.")

    parser.add_argument("--seed", type=int, default=0, help="Model init / training seed.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-path", default="applications/Agility_Forge/koopman/checkpoint.pt")
    parser.add_argument("--out-dir", default="applications/Agility_Forge/koopman/runs",
                         help="Directory for metrics.json, loss_curves.png, and the eval/ .vtu dump.")
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def _compute_losses(model, xs, us, gamma_eig, device):
    """xs: list of K+1 tensors (batch, n_x). us: list of K tensors (batch, n_u)."""
    K = len(us)
    xs = [x.to(device) for x in xs]
    us_stacked = torch.stack([u.to(device) for u in us], dim=1)  # (batch, K, n_u)

    z_enc = [model.encode(xs[k]) for k in range(K + 1)]

    loss_id = sum(nn.functional.mse_loss(model.decode(z_enc[k]), xs[k]) for k in range(K + 1)) / (K + 1)

    z_preds = model.rollout(z_enc[0], us_stacked)
    loss_lin = sum(nn.functional.mse_loss(z_preds[k], z_enc[k + 1]) for k in range(K)) / K
    loss_fwd = sum(nn.functional.mse_loss(model.decode(z_preds[k]), xs[k + 1]) for k in range(K)) / K

    if gamma_eig > 0:
        eigs = torch.linalg.eigvals(model.A.weight)
        loss_eig = (eigs.abs() - 1).clamp(min=0).sum()
    else:
        loss_eig = torch.zeros((), device=device)

    return loss_id, loss_fwd, loss_lin, loss_eig


def _epoch_pass(model, dataset, K, gamma_id, gamma_fwd, gamma_lin, gamma_eig, device,
                 optimizer=None, batch_size=None):
    """One pass over `dataset`. Trains (with grad) if optimizer is given, else evaluates (no grad)."""
    train_mode = optimizer is not None
    model.train(train_mode)

    if train_mode:
        loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, drop_last=False)
    else:
        loader = [dataset.tensors]

    totals = {"loss": 0.0, "id": 0.0, "fwd": 0.0, "lin": 0.0, "eig": 0.0}
    n_batches = 0
    for batch in loader:
        xs, us = list(batch[:K + 1]), list(batch[K + 1:])
        with torch.set_grad_enabled(train_mode):
            loss_id, loss_fwd, loss_lin, loss_eig = _compute_losses(model, xs, us, gamma_eig, device)
            loss = gamma_id * loss_id + gamma_fwd * loss_fwd + gamma_lin * loss_lin + gamma_eig * loss_eig

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        totals["loss"] += loss.item()
        totals["id"] += loss_id.item()
        totals["fwd"] += loss_fwd.item()
        totals["lin"] += loss_lin.item()
        totals["eig"] += loss_eig.item()
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


def _dump_eval_vtu(model, test_dataset, info, out_dir, x_scale, device):
    """Roll out the first rollout in the test split from its true x_0, write
    true/predicted .vtu pairs for every step to out_dir/eval/rollout_XX/."""
    K = info["n_hits"]
    n_nodes = info["n_nodes"]
    rollout_idx = info["test_rollout_indices"][0]
    window = test_dataset[0]
    xs_true = list(window[:K + 1])
    us = torch.stack(list(window[K + 1:]), dim=0).unsqueeze(0).to(device)  # (1, K, n_u)

    model.eval()
    with torch.no_grad():
        x0 = xs_true[0].unsqueeze(0).to(device)
        z0 = model.encode(x0)
        z_preds = model.rollout(z0, us)
        x_hat = [x0] + [model.decode(z) for z in z_preds]

    eval_dir = os.path.join(out_dir, "eval", f"rollout_{rollout_idx:02d}")
    for k in range(K + 1):
        true_phys = denormalize_state(xs_true[k].numpy(), x_scale, n_nodes)
        pred_phys = denormalize_state(x_hat[k].squeeze(0).cpu().numpy(), x_scale, n_nodes)
        name = "undeformed" if k == 0 else f"hit_{k:02d}_final"
        save_state_vtu(true_phys, info["template_vtu_path"], os.path.join(eval_dir, f"{name}_true.vtu"))
        save_state_vtu(pred_phys, info["template_vtu_path"], os.path.join(eval_dir, f"{name}_pred.vtu"))
    logger.info(f"Eval .vtu pairs written -> {eval_dir}")


def _plot_loss_curves(history, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for ax, term in zip(axes, ["id", "fwd", "lin"]):
        ax.plot(history[f"train_{term}"], label="train")
        ax.plot(history[f"test_{term}"], label="test")
        ax.set_ylabel(f"L_{term}")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title(LOSS_TERM_EXPLANATIONS[term], fontsize=8)
    axes[-1].set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    logger.info("Loss terms:\n  " + "\n  ".join(LOSS_TERM_EXPLANATIONS.values()))

    train_dataset, test_dataset, info = build_rollout_dataset(
        args.dataset_dir, train_frac=args.train_frac, n_pod_modes=args.n_pod_modes,
    )
    K = info["n_hits"]
    logger.info(f"Rollouts: {info['n_rollouts_train']} train / {info['n_rollouts_test']} test "
                f"(K={K} steps/window, n_x={info['n_x']}, n_u={info['n_u']}, "
                f"r={info['n_pod_modes']} POD modes)")

    model = KoopmanAutoencoder(
        n_x=info["n_x"], n_u=info["n_u"], n_z=args.latent_dim,
        pod_mean=info["pod_mean"], pod_modes=info["pod_modes"],
        block_widths=args.block_widths, layers_per_block=args.layers_per_block,
        act=args.activation, init_scale=args.init_scale,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"KoopmanAutoencoder | n_x={info['n_x']} r={model.r} n_u={info['n_u']} "
                f"n_z={args.latent_dim} block_widths={args.block_widths} "
                f"layers_per_block={args.layers_per_block} | params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    history = {f"{split}_{term}": [] for split in ("train", "test") for term in ("id", "fwd", "lin", "eig")}
    best_test_fwd = float("inf")
    best_state_dict = None
    best_epoch = 0
    epochs_since_improve = 0

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_totals = _epoch_pass(
            model, train_dataset, K, args.gamma_id, args.gamma_fwd, args.gamma_lin, args.gamma_eig, device,
            optimizer=optimizer, batch_size=args.batch_size,
        )
        test_totals = _epoch_pass(
            model, test_dataset, K, args.gamma_id, args.gamma_fwd, args.gamma_lin, args.gamma_eig, device,
        )
        for term in ("id", "fwd", "lin", "eig"):
            history[f"train_{term}"].append(train_totals[term])
            history[f"test_{term}"].append(test_totals[term])

        if test_totals["fwd"] < best_test_fwd:
            best_test_fwd = test_totals["fwd"]
            best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch % args.print_every == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:4d} | train loss {train_totals['loss']:.4e} "
                        f"(id {train_totals['id']:.4e} fwd {train_totals['fwd']:.4e} lin {train_totals['lin']:.4e}) | "
                        f"test loss {test_totals['loss']:.4e} "
                        f"(id {test_totals['id']:.4e} fwd {test_totals['fwd']:.4e} lin {test_totals['lin']:.4e})")

        if epochs_since_improve >= args.patience:
            logger.info(f"Early stopping at epoch {epoch} (no test L_fwd improvement for {args.patience} epochs). "
                        f"Restoring epoch {best_epoch}'s weights.")
            break

    elapsed = time.time() - t0
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    os.makedirs(os.path.dirname(args.checkpoint_path) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "args": vars(args),
        "x_scale": info["x_scale"],
        "n_x": info["n_x"],
        "n_u": info["n_u"],
        "n_nodes": info["n_nodes"],
    }, args.checkpoint_path)
    logger.info(f"Checkpoint saved -> {args.checkpoint_path} (best epoch {best_epoch}, test L_fwd {best_test_fwd:.4e})")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "history": history,
            "best_epoch": best_epoch,
            "best_test_fwd": best_test_fwd,
            "elapsed_s": elapsed,
            "n_params": n_params,
        }, f, indent=2)
    _plot_loss_curves(history, os.path.join(args.out_dir, "loss_curves.png"))
    logger.info(f"Metrics + loss curves -> {args.out_dir}")

    _dump_eval_vtu(model, test_dataset, info, args.out_dir, info["x_scale"], device)

    return model, history


if __name__ == "__main__":
    train(parse_args())
