# Koopman Autoencoder (earlier stage-2 approach) — design notes

> **Status**: superseded by the MeshGraphNets-style GNN surrogate in
> `../GNN/` (see its README) — kept here for reference, code still runs, but
> this is no longer the active stage-2 approach.

Trains a latent linear-dynamics model of the forging state from the rollouts
produced by `applications/Agility_Forge/generate_dataset.py` (stage 1), for use
as the fast predictive model inside the MPC loop (`../control/`, stage 3).

## Data source

`generate_dataset.py` writes `manifest.json` + one `.vtu` per saved state
(`undeformed`, and `hit_final` per hit) under `applications/Agility_Forge/data/dataset_pretraining/`.
Each `.vtu` holds per-node `Displacement` (3-vec) and `Temperature` (scalar) on
the shared billet mesh; each hit's manifest record carries its control inputs
(`d_j` compression depth, `x_max_band` hit position, `R_j` rotation, `u_j`).

## Design decisions (implemented)

Settled through an explicit interview with the user; nothing below is a silent
default. Modeled on `peter-frazier/KAE_for_uniaxial_tensile_test`'s
`LRAN_BLRAN/LRAN_LD`, with one deliberate departure: **the original state does
not appear in the latent** (no `z = [x; Psi(x)]`) — here `z = Psi(x)` only.

- **State vector**: flattened `Displacement` only over every node, `n_x =
  n_nodes * 3` (no reduction). Originally `[Displacement, Temperature]`
  (`n_x = n_nodes * 4`) following a comparably-sized FEM dataset in the
  reference codebase that fed the same shape directly into an MLP encoder
  with no PCA/POD/surface-only pre-projection; Temperature was dropped from
  the state at the user's request so the model predicts nodal displacement
  fields only. `generate_dataset.py`/the `.vtu` files still carry Temperature
  (jax_forge's actual thermo-mechanical coupling doesn't change), it's just
  not read into `x` by `koopman/dataset.py` anymore.
- **POD pre-reduction (fixed, not learned)**: before any neural network runs,
  the normalized `n_x`-dim state is projected onto a fixed `r`-dim POD basis
  (`--n-pod-modes`, default `r=75`): `a = (x - pod_mean) @ pod_modes.T`.
  `pod_mean`/`pod_modes` are computed once by `dataset.compute_pod_basis`
  (mean-center, economy SVD, keep the top `r` right singular vectors) fit on
  the **training rollouts' normalized states only** — same train-only-fit
  principle as `x_scale` below, so the basis never sees test data. Motivated
  by `applications/Agility_Forge/SVD/`'s finding that the full 25-rollout,
  150-snapshot dataset needs only ~11/30/71 modes for 90/95/99.9% cumulative
  energy — the raw `n_x=22599`-dim (and previously `n_z=512`-to-2.4B-param
  encoder) representation was wildly overparameterized relative to the
  data's actual intrinsic dimensionality. **Caveat, flagged rather than
  quietly assumed**: fitting `r=75` modes from ~20 training rollouts (~120
  snapshots, mean-centered rank ≤119) uses roughly 63% of every degree of
  freedom the training split has — the higher modes in that basis (roughly
  30-75) are more likely to encode rollout-specific noise than
  generalizable deformation structure than the lower ones are. `r=75` is an
  explicit user choice (between the full-dataset 99%/99.9% thresholds of
  30/71 modes), not a validated-robust one; the `id`/`fwd` loss floor (POD
  truncation error) is a decent live signal of how much this matters — watch
  it, don't just watch `L_fwd`.
- **Latent / architecture**: `z = Psi(a)` is a purely learned nonlinear lift
  of the POD-reduced coefficients `a` (`model.Lifting`), *not* of the raw
  state — `n_z=250` deliberately *exceeds* `r=75` (classical Koopman: lift to
  a *higher*-dimensional space where the dynamics become linear). The
  decoder is a single **learned linear** map back down to `r`
  (`nn.Linear(n_z, r)`), followed by the **fixed** POD reconstruction
  `x_hat = pod_mean + a_hat @ pod_modes` — not exact (the learned half isn't)
  and not a nonlinear MLP; all nonlinearity lives in the lift, `A`, `B`, and
  both linear maps around the fixed POD step stay linear (closer to
  EDMD/Koopman-with-a-learned-dictionary). `Lifting` is `in_proj(r-
  >block_widths[0]) -> residual blocks, one per entry in block_widths ->
  out_proj(block_widths[-1]->n_z)`. Each block wraps `layers_per_block`
  `[LayerNorm -> Linear(width,width) -> activation]` layers in a single skip
  connection (`out = in + F(in)`); a plain Linear+activation *transition* (no
  skip) bridges consecutive blocks whose widths differ -- skipped entirely
  (not even an identity layer) when they match, so constant-width blocks
  chain directly with no extra layers. Current defaults: `r=75`, `n_z=250`,
  `block_widths=[250, 250, 250, 250]` (4 blocks, constant width),
  `layers_per_block=8` (32 hidden layers total, ~2.19M params) — at the
  user's explicit request, replacing the earlier `n_x`-input, `block_widths=
  [10000, 5000, 2500]`, ~2.4B-param design once the SVD/POD analysis showed
  that scale was unjustified by the data. LayerNorm (kept from that earlier
  design, where it was required for stability at width>=2500/16-layer
  blocks) empirically remains stable at this smaller width/depth too.
- **Control-dependent dynamics**: linear, `z_{k+1} = A z_k + B u_k`
  (not bilinear). `u_k = [d_j_frac, sin(R_j), cos(R_j)]`, used **raw, with no
  normalization** — in Koopman model fitting the inputs are left untouched,
  only the state is lifted/transformed. `sin`/`cos` encode the hit rotation's
  periodicity so a linear `B` doesn't have to fit a discontinuity at 0/360°.
- **Loss terms**: `L_id + L_fwd + L_lin` (equal weight, `gamma_id=gamma_fwd=
  gamma_lin=1.0`), `+ L_eig` optional (`gamma_eig`, default `0.0`). `L_id` is
  required here (unlike LRAN_LD) because the decoder isn't exact. See
  `train.py`'s module docstring for what each term measures.
- **State normalization**: one pooled `[-1,1]` min-max scale shared across
  every `Displacement` component (all nodes, all 3 axes) — not per-axis,
  which would distort displacement *direction*, physically meaningful here.
  Fit on the training split only (`x_scale`), applied before the POD step
  above (POD is fit on already-normalized training states, not raw ones).
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
