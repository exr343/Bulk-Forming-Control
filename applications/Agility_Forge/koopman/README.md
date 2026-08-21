# Koopman Autoencoder (stage 2) — design notes

Trains a latent linear-dynamics model of the forging state from the rollouts
produced by `applications/Agility_Forge/generate_dataset.py` (stage 1), for use
as the fast predictive model inside the MPC loop (`../control/`, stage 3).

## Data source

`generate_dataset.py` writes `manifest.json` + one `.vtu` per saved state
(`undeformed`, and `hit_final` per hit) under `applications/Agility_Forge/data/dataset/`.
Each `.vtu` holds per-node `Displacement` (3-vec) and `Temperature` (scalar) on
the shared billet mesh; each hit's manifest record carries its control inputs
(`d_j` compression depth, `x_max_band` hit position, `R_j` rotation, `u_j`).

## Open design questions (resolve before filling in the stubs)

- **State vector**: full flattened `[Displacement, Temperature]` per node, or a
  reduced representation (subsampled nodes, PCA/POD pre-projection, surface-only)?
  The full mesh is likely too high-dimensional for a tractable Koopman latent
  space directly.
- **Latent dimension** and **encoder/decoder architecture** (plain MLP vs. a
  mesh-aware architecture, given node connectivity is fixed across rollouts).
- **Form of control-dependent dynamics**: single linear `z_{t+1} = A z_t + B u_t`,
  or something bilinear in `u` if a single hit's effect depends nonlinearly on
  where/how hard it lands.
- **Loss terms**: reconstruction, one-step latent prediction, multi-step rollout
  consistency (important if MPC will roll the model forward over several hits).
- **Train/val split**: by rollout (recommended, avoids leaking a rollout's later
  hits into validation) vs. by individual transition.

## Stack

PyTorch (`torch.nn`), decoupled from JAX-FEM's JAX stack — the boundary between
the plant (`jax_forge`, JAX) and this model (PyTorch) is bridged via plain NumPy
arrays, not direct JAX↔Torch tensor conversion.
