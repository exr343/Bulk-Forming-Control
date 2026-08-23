"""SVD / POD effective-dimension analysis of the Agility_Forge displacement state.

Usage:
    python -m applications.Agility_Forge.SVD.svd_analysis \\
        --dataset-dir applications/Agility_Forge/data/dataset_snapshot_r25

Stacks every saved state (undeformed + hit_final, every complete rollout) into a
(n_snapshots, n_features) matrix using the same displacement-only state and pooled
[-1,1] normalization koopman/dataset.py fits for Koopman training (fit here on the
full snapshot set -- there is no train/test split, this is exploratory analysis of
the whole dataset's intrinsic dimensionality). Mean-centers the normalized matrix,
takes its economy SVD, and reports/plots the cumulative energy spectrum
(cumsum(S**2)/sum(S**2)) to answer: how many modes actually capture the variance
in how rollouts differ from each other?

See README.md for the full rationale (why normalize-then-center, why rows are
snapshots, why cumulative energy on the centered matrix).
"""

import argparse
import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from applications.Agility_Forge.koopman.dataset import (
    _group_complete_rollouts,
    compute_state_scale,
    load_manifest,
    load_state,
    normalize_state,
)

# Standalone logger, same reasoning as koopman/train.py: avoid `from jax_forge import
# logger`, which pulls in the pyfiglet banner and the whole FEM/JAX stack. This module
# only ever touches jax_forge output via plain NumPy .vtu files on disk.
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
    parser.add_argument("--dataset-dir", default="applications/Agility_Forge/data/dataset_snapshot_r25",
                         help="manifest.json + .vtu snapshot dir. Defaults to the frozen 25-rollout "
                              "snapshot -- data/dataset/ may still be actively (non-atomically) "
                              "appended to by a running generate_dataset.py job; point here "
                              "explicitly once that job is done.")
    parser.add_argument("--out-dir", default="applications/Agility_Forge/SVD/runs")
    parser.add_argument("--thresholds", type=lambda s: [float(t) for t in s.split(",")],
                         default=[0.90, 0.95, 0.99, 0.999],
                         help="Comma-separated cumulative-energy thresholds to report the "
                              "effective dimension at.")
    parser.add_argument("--save-modes", action="store_true",
                         help="Also save the right singular vectors (spatial POD modes) to "
                              "svd_results.npz. Off by default (one mode is n_features floats).")
    return parser.parse_args()


def load_snapshot_matrix(dataset_dir):
    """Every snapshot from every complete rollout under dataset_dir, as rows of a
    (n_snapshots, n_features) matrix, normalized with koopman/dataset.py's pooled
    [-1,1] scale (fit here on the full snapshot set -- no train/test split).

    Returns (X, n_nodes, x_scale, labels).
    """
    meta, records = load_manifest(dataset_dir)
    n_hits = meta["n_hits_per_rollout"]
    complete = _group_complete_rollouts(records, n_hits)
    if not complete:
        raise RuntimeError(f"No complete rollouts (0..{n_hits} hits) found under {dataset_dir}.")

    raw_states, labels, n_nodes = [], [], None
    for rollout_idx, hit_records in complete:
        for rec in hit_records:
            vtu_path = os.path.join(dataset_dir, rec["vtu_path"])
            x = load_state(vtu_path)
            if n_nodes is None:
                n_nodes = x.shape[0] // 3
            raw_states.append(x)
            labels.append(f"rollout_{rollout_idx:02d}_{rec['kind']}_hit{rec['hit']}")

    x_scale = compute_state_scale(raw_states, n_nodes)
    X = np.stack([normalize_state(x, x_scale, n_nodes) for x in raw_states])
    return X, n_nodes, x_scale, labels


def compute_svd(X):
    """Mean-centers rows (subtracts the mean normalized displacement field across all
    snapshots), then economy SVD of the centered matrix.

    X: (n_snapshots, n_features), already normalized to [-1,1].
    Returns (U, S, Vt, mean_row).
    """
    mean_row = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - mean_row, full_matrices=False)
    return U, S, Vt, mean_row


def cumulative_energy(S):
    """cumsum(S**2) / sum(S**2) -- fraction of variance captured by the first r
    singular values/modes, r = 1..len(S)."""
    return np.cumsum(S**2) / np.sum(S**2)


def effective_dimension(cum_energy, thresholds):
    """First mode count r (1-indexed) whose cumulative energy >= threshold, per
    threshold. Returns {threshold: r}."""
    out = {}
    for t in thresholds:
        r = int(np.searchsorted(cum_energy, t) + 1)
        out[t] = min(r, len(cum_energy))
    return out


def plot_cumulative_energy(cum_energy, eff_dims, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    modes = np.arange(1, len(cum_energy) + 1)
    ax.plot(modes, cum_energy, marker="o", ms=3)
    for t, r in eff_dims.items():
        ax.axhline(t, color="gray", ls="--", lw=0.5)
        ax.axvline(r, color="gray", ls="--", lw=0.5)
        ax.annotate(f"{t * 100:.1f}%: r={r}", xy=(r, t),
                    xytext=(5, -10), textcoords="offset points", fontsize=8)
    ax.set_xlabel("number of modes")
    ax.set_ylabel("cumulative energy")
    ax.set_ylim(0, 1.02)
    ax.set_title("POD cumulative energy spectrum")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main(args):
    X, n_nodes, x_scale, labels = load_snapshot_matrix(args.dataset_dir)
    logger.info(f"Snapshot matrix: {X.shape[0]} snapshots x {X.shape[1]} features (n_nodes={n_nodes})")

    U, S, Vt, mean_row = compute_svd(X)
    cum_energy = cumulative_energy(S)
    eff_dims = effective_dimension(cum_energy, args.thresholds)
    for t, r in eff_dims.items():
        logger.info(f"  {t * 100:5.1f}% energy -> {r} modes")

    os.makedirs(args.out_dir, exist_ok=True)

    npz_path = os.path.join(args.out_dir, "svd_results.npz")
    npz_data = {"singular_values": S, "cumulative_energy": cum_energy, "mean_row": mean_row}
    if args.save_modes:
        npz_data["right_modes"] = Vt
    np.savez(npz_path, **npz_data)
    logger.info(f"Singular values / cumulative energy -> {npz_path}")

    json_path = os.path.join(args.out_dir, "effective_dimension.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_snapshots": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "thresholds": {str(t): r for t, r in eff_dims.items()},
        }, f, indent=2)
    logger.info(f"Effective dimension -> {json_path}")

    plot_path = os.path.join(args.out_dir, "cumulative_energy.png")
    plot_cumulative_energy(cum_energy, eff_dims, plot_path)
    logger.info(f"Cumulative energy plot -> {plot_path}")

    return eff_dims


if __name__ == "__main__":
    main(parse_args())
