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

## Design decisions (implemented)

Settled through an explicit interview with the user; nothing below is a silent
default. Modeled on `peter-frazier/KAE_for_uniaxial_tensile_test`'s
`LRAN_BLRAN/LRAN_LD`, with one deliberate departure: **the original state does
not appear in the latent** (no `z = [x; Psi(x)]`) — here `z = Psi(x)` only.

- **State vector**: full flattened `[Displacement, Temperature]` over every
  node, `n_x = n_nodes * 4` (no reduction). A comparably-sized FEM dataset in
  the reference codebase fed the same shape directly into an MLP encoder with
  no PCA/POD/surface-only pre-projection, so this follows that precedent.
- **Latent / architecture**: `z = Psi(x)` is a purely learned nonlinear lift
  (`model.Lifting`, an MLP) — since `x` isn't concatenated in, `n_z` *is* the
  full latent dimension. The decoder is a single **learned linear** map
  (`nn.Linear(n_z, n_x)`), not exact and not a nonlinear MLP — all
  nonlinearity lives in the lift; `A`, `B`, and the decoder stay linear
  (closer to EDMD/Koopman-with-a-learned-dictionary). Current defaults:
  `n_z=512`, `alpha=32` (hidden width 512, matched to `n_z` so no layer
  bottlenecks the next), `n_h=8` — a deliberate bet sized for the dataset
  growing substantially via `generate_dataset.py`'s append-mode (see below),
  not for today's dataset size; expect heavy overfitting until more rollouts
  accumulate.
- **Control-dependent dynamics**: linear, `z_{k+1} = A z_k + B u_k`
  (not bilinear). `u_k = [d_j_frac, sin(R_j), cos(R_j)]`, used **raw, with no
  normalization** — in Koopman model fitting the inputs are left untouched,
  only the state is lifted/transformed. `sin`/`cos` encode the hit rotation's
  periodicity so a linear `B` doesn't have to fit a discontinuity at 0/360°.
- **Loss terms**: `L_id + L_fwd + L_lin` (equal weight, `gamma_id=gamma_fwd=
  gamma_lin=1.0`), `+ L_eig` optional (`gamma_eig`, default `0.0`). `L_id` is
  required here (unlike LRAN_LD) because the decoder isn't exact. See
  `train.py`'s module docstring for what each term measures.
- **State normalization**: pooled per physical quantity — one `[-1,1]`
  min-max scale shared across all `Displacement` components (all nodes, all 3
  axes), a separate shared scale across all `Temperature` components. Fit on
  the training split only. Rescaling `ux`/`uy`/`uz` independently would
  distort displacement *direction*, which is physically meaningful here.
- **Train/test split**: by rollout, prefix-style (`rollouts[:n_train]` /
  `rollouts[n_train:]`) — valid because rollouts are already IID (independent
  seeded draws per rollout). One window per rollout, `K = n_hits_per_rollout`
  (the full rollout), to maximize the multi-step horizon supervision stage
  3's MPC will actually need.

`generate_dataset.py` now **appends** to an existing dataset dir across
invocations (continues rollout numbering, merges `manifest.json`, hard-errors
on a config mismatch) rather than wiping it — so `build_rollout_dataset` must
keep working against however many complete rollouts exist at call time, not
assume a fixed total.

## Stack

PyTorch (`torch.nn`), decoupled from JAX-FEM's JAX stack — the boundary between
the plant (`jax_forge`, JAX) and this model (PyTorch) is bridged via plain NumPy
arrays, not direct JAX↔Torch tensor conversion.
