"""Training loop for the Koopman Autoencoder (stub).

Usage (once implemented):
    python -m applications.Agility_Forge.koopman.train \\
        --dataset-dir applications/Agility_Forge/data/dataset \\
        --latent-dim 32 --epochs 100

TODO: this module is a stub — fill in once koopman/dataset.py and
koopman/model.py are implemented.
"""

import argparse

import torch

from applications.Agility_Forge.koopman.dataset import build_rollout_dataset
from applications.Agility_Forge.koopman.model import KoopmanAutoencoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="applications/Agility_Forge/data/dataset")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-path", default="applications/Agility_Forge/koopman/checkpoint.pt")
    return parser.parse_args()


def train(args):
    raise NotImplementedError(
        "TODO: build_rollout_dataset(args.dataset_dir) -> DataLoader, "
        "construct KoopmanAutoencoder, torch.optim.Adam training loop, "
        "save state_dict to args.checkpoint_path"
    )


if __name__ == "__main__":
    train(parse_args())
