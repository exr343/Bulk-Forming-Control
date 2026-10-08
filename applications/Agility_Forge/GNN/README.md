# MeshGraphNets surrogate (stage 2) — design notes

Learns a per-hit graph neural network surrogate of the forging state from the
rollouts produced by `applications/Agility_Forge/generate_dataset.py` (stage 1),
scoped initially to a reduced 2-parameter actuation space, for eventual use as
the fast predictive model inside the MPC loop (`../control/`, stage 3).

An earlier Koopman Autoencoder surrogate (`../koopman/`) was explored first for
stage 2; this module is now the active stage-2 approach. `../koopman/README.md`
is kept as reference for that design, but it's no longer part of the current
plan, and the rest of this document is written to stand on its own.

**Status**: implemented and validated against the real dataset (shapes,
gradient flow, normalizer accumulation, and a full CLI run all checked — see
"PyTorch implementation" below). Now trained against the full **3D actuation
space** (`d_j`/`R_j`/`u_j` all independently randomized — see "Data source"
below for how `u_j` was added as a 4th control-broadcast dim); the
checkpoint referenced throughout this doc as `runs/` is the earlier 2D-only
run and is preserved as `runs_2dim/` for reference, superseded by
`checkpoint_3dim.pt`/`runs_3dim/`. A message-passing-depth sweep
(`mp_sweep/`, 1/5/15/45/135 steps) found accuracy saturates at ~5 steps with
no benefit from going deeper — see "Message-passing depth sweep" below. The
surrogate is also now wired into a real MPC control loop (`../control/`) —
see that module's README for the closed-loop results.

## Vendored reference code

`applications/Agility_Forge/GNN/old/` holds an unmodified, wholesale copy of
[google-deepmind/deepmind-research's meshgraphnets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)
(`core_model.py`, `common.py`, `normalization.py`, `cfd_model.py`/`cloth_model.py`
as the two reference task setups, `dataset.py`, `run_model.py`, plus its own
README) — cloned via a sparse git checkout, kept for reference only (**TF1 +
Sonnet + graph_nets**, `tf.Session`-based, not runnable against this project's
data as-is; moved into `old/` since it isn't part of the hot-forging project
itself). The Agility_Forge-specific PyTorch port lives one level up, directly
in `GNN/`, as `normalizer.py`/`data.py`/`model.py`/`train.py` (distinct
filenames from the vendored `normalization.py`/`dataset.py` specifically to
avoid colliding with them — the vendored files are untouched).

## PyTorch implementation

- `normalizer.py` — `OnlineNormalizer`: PyTorch port of `normalization.py`'s
  running-statistics accumulator (buffers, not TF variables, so it moves with
  `.to(device)` and (de)serializes via `state_dict()` automatically). Matches
  the reference's semantics exactly (per-component mean/std, gated by
  `training` and `max_accumulations`) — see "Normalization" below for the
  reasoning.
- `data.py` — `build_surface_mesh_info`: extracts the exterior-boundary
  surface (any rollout's `undeformed.vtu` — mesh is identical across all of
  them) and the static `is_bottom_face`/`is_pin_node`/`is_lateral_wall` flags,
  matching `boundary_conditions.py`'s own node-selection logic exactly (same
  tolerances/thresholds — see "Per-node feature vector" below for exactly how).
  `load_examples`/`split_examples_by_rollout`: builds one `HitExample` per hit
  across every complete rollout in `manifest.json`, by-rollout prefix
  train/test split. `ForgeGNNDataset`: one item = one hit's raw (unnormalized)
  10-dim node features + `x_k` + target `Δx`, all restricted to the surface
  node subset; `.vtu` reads are cached per-path since consecutive hits share
  files (a hit's post-state is the next hit's pre-state). `build_node_features`
  is factored out so `train.py`'s autoregressive rollout eval can reuse the
  exact same feature construction with a model-predicted `x_k` instead of the
  ground-truth one. Written as an independent minimal loader, not an import of
  `koopman/dataset.py`.
- `model.py` — `MLP`/`GraphNetBlock`/`EncodeProcessDecode`: direct PyTorch
  port of `core_model.py`'s structure (same MLP shape, same per-block
  edge-then-node update with residual adds, same independently-weighted
  blocks — no shared/recurrent processor weights). Batching is a dense
  leading batch dimension `(B, N, ...)`/`(B, E, ...)` with unbatched
  `senders`/`receivers` index tensors, **not** the reference's disjoint-graph
  segment-sum batching — a deliberate simplification, exactly equivalent
  given every example shares one fixed topology (verified — see "Mesh
  topology" below), not a partial port. `ForgeGNN`: the Agility-Forge-specific
  wrapper (analogous to `cfd_model.Model`/`cloth_model.Model`) owning the
  fixed-mesh buffers, the three `OnlineNormalizer`s, edge-feature
  construction from `x_k`, and `loss()`/`loss_and_predict()`/`predict_delta()`.
  Confirmed by direct test: real-data forward+backward pass produces nonzero
  gradients on 100% of parameters, normalizer accumulation counts match
  `batch_size × node_count` / `× edge_count` exactly, and `eval()` correctly
  freezes accumulation. At the settled reference sizing (latent=128, 15
  message-passing steps), parameter count measured at exactly 2,333,571 —
  matching this doc's earlier hand-computed ≈2.33M estimate.
- `train.py` — training loop (standalone logger decoupled from `jax_forge`,
  argparse CLI, early stopping on test loss, `metrics.json` + loss-curve
  plot). One-hit-ahead teacher-forced loss only, per the settled "Training
  regime" decision. After training, one held-out rollout is rolled out
  **autoregressively** (the model's own hit-1 prediction feeds hit-2's input,
  not the true per-hit `x_k`) and dumped as true/predicted `.vtu` pairs plus
  a per-hit RMSE/NRMSE curve in `metrics.json`. See "Loss function & reported
  metrics" below for exactly what is optimized vs. what is only reported.
- `finetune.py` — finetunes a pretrained checkpoint on the scheduled
  square-rod runs in `data/dataset_finetuning/`, reusing `train.py`'s loss,
  metrics and early stopping. Evaluates the pretrained and finetuned models
  on the test run and on the pretraining test split (forgetting check). See
  "Finetuning on square-rod data" below.

## Data source

**3D actuation space — done.** Sections below describing the per-node
feature vector and control broadcast reflect the current, 3D-actuation
design: `d_j`, `R_j`, **and `u_j`** (strike depth) are all independently
randomized per hit, `u_j ~ Uniform(0.5, 2.0)`mm — range justified in
`generate_dataset.py`'s module docstring against the JAX-FORGE paper's
reported forging depths (0.62mm, 1.63mm) for this same billet geometry.
`u_j` was added as a 4th control-broadcast dimension: `u_j_frac =
(u_j_mm - min)/(max - min)`, raw/unnormalized, following `d_j_frac`'s
existing convention exactly rather than routing it through the
`OnlineNormalizer` — settled by explicit interview. `NODE_IN_DIM` moved
10 → 11 (`N_CONTROL_DIMS` 3 → 4) accordingly; see `model.py`/`data.py`. The
earlier 2D-actuation dataset (`u_j` fixed at 2.5mm, later found to be an
undocumented placeholder value — *deeper* than either of the JAX-FORGE
paper's reported real forging depths, not a validated setting) is preserved
at `data/backup/dataset_2dim` and the model trained on it at `runs_2dim/`, kept for
reference but superseded by the 3D-actuation dataset (`data/dataset_pretraining`) and
`checkpoint_3dim.pt`/`runs_3dim/`.

**Known bug, fixed**: `_chamfer_hausdorff_mm` in `train.py` was originally
called with raw displacement vectors instead of absolute positions
(`rest_pos + displacement`) in both `_epoch_pass` and `_rollout_eval` — this
silently computed nearest-neighbor distance in a physically meaningless
"displacement space" rather than actual deformed-surface shape. Found via a
non-monotonic Hausdorff value across autoregressive rollout hits that
couldn't be explained by the worst-case-vs-average distinction alone; fixed
in both call sites, verified against an independent `scipy.cKDTree`
recomputation. All Chamfer/Hausdorff numbers in this doc and in `mp_sweep/`
post-date the fix.

Reuses `generate_dataset.py`'s `manifest.json` + one `.vtu` per saved state
directly: each `.vtu` holds per-node `Displacement` (3-vec) and `Temperature`
(scalar) on the shared, topologically fixed billet mesh (same connectivity
across every hit and every rollout — no remeshing). Each hit's manifest
record carries its control inputs (`d_j` compression-band axial position,
`x_max_band`, `R_j` rotation, `u_j` compression depth), already in both
fractional and absolute-mm form.

**For the 2D-actuation dataset this document otherwise describes**: no
dataset regeneration was required to scope training to a 2D control space,
since `generate_dataset.py` already randomized `d_j` (axial band position →
`x_min_band`) and `R_j` (rotation → `rotation_euler_x`) per hit, while
treating `u_j` as a fixed CLI-level argument shared by the whole dataset
(`compression_displacement_mm: 2.5`, unchanged across all `generation_runs`),
never sampled.

**Actual dataset scale, 2D-actuation** (`data/backup/dataset_2dim`, `runs_2dim/`'s
training run): `n_hits_per_rollout=5`, 338 rollouts with all 5 hits complete
→ **1,690 single-step training pairs**. This is the number that grounds the
model-sizing decision in "Network shape" below (kept at this size for the
3D-actuation training too, rather than scaled up — see "Message-passing
depth sweep").

**Actual dataset scale, 3D-actuation** (`data/dataset_pretraining`, `checkpoint_3dim.pt`
and the `mp_sweep/` comparison): 400 complete rollouts total, of which a
fixed **383-rollout snapshot** (306 train / 77 test, by-rollout prefix
split) is used for every model in this doc and in `mp_sweep/` — pinned via
`mp_sweep/rollout_snapshot_383.json` and `load_examples`'s
`allowed_rollouts` param, specifically so dataset growth after
`checkpoint_3dim.pt` was trained doesn't confound later comparisons (the
dataset kept growing via a background gap-fill job while this work was in
progress) → **1,915 single-step training pairs** (1,530 train / 385 test).

## Temperature: not an input or output, but not absent from the physics either

The GNN never sees `Temperature` and never predicts it — but that does **not**
mean temperature has no effect on the training data. `generate_dataset.py`
solves the **genuinely coupled** thermo-mechanical PDE, not a mechanical-only
approximation, and that coupling is exactly what shows up implicitly in the
`Displacement` field the GNN trains on:

- `main.py`/`generate_dataset.py` build `ThermalMechanical(mesh=[mesh, mesh],
  vec=[3, 1], ...)` — 3 mechanical displacement DOFs *and* 1 temperature DOF
  per node, solved jointly at every Newton step of every hit, not
  sequentially or approximately.
- Flow stress in the J2 finite-strain plasticity constitutive model
  (`lib/constitutive.py`) is temperature-dependent, so the resulting
  `Displacement` field genuinely reflects whatever thermal state the billet
  was actually in during that hit.
- Between hits, `main.py` runs an explicit **thermal-state relaxation**: a
  10-step temperature ramp (`T_old` blended from the post-hit cooled state
  toward a reheated state), with a full mechanical re-equilibration solve at
  *each* ramp step. This exists specifically because resetting temperature in
  one shot produces a dt-independent Newton-solver stall (per `main.py`'s own
  module docstring) — i.e. temperature's coupling to the mechanical solve is
  strong enough that the solver itself requires careful handling of it, not
  an afterthought.
- `generate_dataset.py` writes **both** fields to every `.vtu`
  (`point_infos=[("Displacement", ...), ("Temperature", ...)]`) — `data.py`
  simply chooses not to read the `Temperature` array back in.

So the net effect is: temperature's influence on the mechanical response is
real, solved for, and baked into the ground-truth displacement data — but the
surrogate has no visibility into it at inference time. It must implicitly
infer whatever temperature-driven pattern exists purely from the correlation
between (rest position, accumulated displacement, and the two control values)
and the resulting displacement, with no direct signal of the actual thermal
state. This is the concrete mechanism behind the "known risk" already flagged
under "State fields" below: if a hit's real thermal history ever differs from
what was typical in training for that control setting, the surrogate has no
way to know.

## Graph construction: surface mesh only, not the full volumetric mesh

The GNN's node set is the billet's **exterior boundary surface**
(`mesh_container.py`'s `MeshContainer.extract_faces()`/`get_surface()`, which
extracts every triangle belonging to exactly one tet — i.e. the full boundary,
both end caps *and* the lateral cylindrical wall, not just the lateral-wall
subset `boundary_conditions.py`'s `on_outer_surface_np` checks for contact) —
not the full volumetric tet mesh. Rationale: the eventual real-world
measurement of a forged part (what stage 3's control loop is ultimately
validated against) is exterior shape, not internal nodal displacement, so
training and predicting on interior nodes the physical process never actually
measures is unnecessary. This also means mesh edges can be built directly with
the vendored reference's own `common.triangles_to_edges()` (built for surface
triangles) rather than writing a custom tet-edge extractor.

**Exact reduction, measured against the real mesh**: 7,533 total volumetric
nodes → 4,355 kept on the surface, **3,178 interior nodes (42.2%) omitted
entirely** — never fed into the graph, never predicted. 8,706 surface
triangles, 13,059 unique undirected mesh edges (26,118 directed).

## Design decisions (settled)

Settled through an explicit interview with the user; nothing below is a silent
default.

- **Strike-depth input: the relative stroke `u_j`, as JAX-FORGE defines it
  (settled 2026-09-27; the absolute-gap alternative is a closed dead end).**
  `u_j` is how far each die travels over the hit, starting from the bar's
  outermost point in the die band on its own side, measured along the press
  direction on the current deformed surface
  (`lib/boundary_conditions.build_cylinder_press_bcs`). It is the input every
  training set and every MPC run up to experiment 8 used. An absolute
  half-gap input (`--control gap` in `GNN/data.py`, `control="gap"` in
  `control/mpc.py`) was tried in experiments 9-11 and rejected: equal
  accuracy on the held-out square run, but on real MPC trajectories it
  over-predicted stretch about twice as much (0.44 vs 0.20 mm per hit) and
  drifted 2-3x faster, and every gap-based MPC variant ended worse than the
  open loop. See `progress/control_reports/2026-09-27_cost_e9_gap_control/`
  and `2026-09-27_cost_e10_e11_gap_diagnosis/`. The gap code is left in
  place (off by default) only so those results stay reproducible; don't
  build on it.

- **Step granularity**: one GNN forward pass predicts one full hit-to-hit
  state transition, not a small solver sub-step the way the reference
  MeshGraphNets paper's rollouts are usually framed. `generate_dataset.py`
  only saves one `.vtu` per hit (not per adaptive-dt solver substep), so a
  "step" in the available data is already a large plastic-strain jump —
  chosen over regenerating substep-resolution data or predicting cumulative
  state directly from the rest configuration, to reuse the existing dataset
  as-is. Tradeoff, accepted explicitly: this is a harder per-step regression
  target than the paper's small-dt formulation and will likely need a
  deeper/wider processor than the reference configs. If performance is poor,
  more (or finer-grained) data generation is the fallback, not a silent
  redesign.
- **Graph structure**: mesh edges only, from the fixed surface-mesh
  connectivity described above (same mesh reused across every hit/rollout —
  no remeshing). No world-space / dynamic-proximity edge set is built, unlike
  the reference `cloth_model.py`'s obstacle-collision handling — see
  "Actuation/contact encoding" below for why (the press is analytic, not a
  meshed body).
- **Output parameterization**: predict a per-node displacement delta
  (`x_{k+1} - x_k`, the "velocity"/output convention from the reference
  MeshGraphNets papers), decoded and added onto the current displacement to
  get the next state — not a direct next-state regression.
- **State fields**: displacement only (`x` = 3D displacement per node);
  Temperature is present in every `.vtu` but is not part of the predicted
  state (see "Temperature" above for exactly how its effect still reaches the
  training data implicitly). Known risk, flagged rather than silently
  assumed: flow stress in the underlying constitutive model is
  temperature-dependent, so a displacement-only surrogate may drift as die
  position/orientation change local heating across hits; revisit if
  displacement-only accuracy proves insufficient.
- **Training regime**: clean one-hit-ahead training only — ground-truth
  inputs, no input-noise injection, no multi-step/full-rollout unrolling.
  Chosen over the reference papers' noise-injection-for-rollout-stability
  trick and over a full-rollout training loss, since with only ~5 hits per
  rollout and each hit already a large discrete jump (not the paper's small
  `dt`), it's not yet clear the noise-injection trick transfers as-is;
  deferred as a known limitation to revisit now that a baseline exists and
  its compounding-error behavior at rollout time has been measured (see
  `runs/metrics.json`'s `rollout_eval_per_hit_nrmse`).
- **Loss masking**: resolved (not left open) — `is_bottom_face`/`is_pin_node`
  nodes are **not** masked out of the training loss. Initially framed as
  analogous to the reference code's `OBSTACLE`/`HANDLE` exclusion, but that
  analogy doesn't hold: the reference's excluded nodes have their *entire*
  future position externally prescribed (fully known, not predicted at all).
  `bottom_face`/`pin_node` are only *partially* constrained (`bottom_face`
  fixes just x; `pin_node` fixes just y/z) — the rest of each node's
  displacement is still free and physically driven, so masking the whole node
  would discard real supervision on its unconstrained components for
  negligible benefit (a handful of nodes, and the constrained components are
  trivially easy — deterministically 0 — so leaving them in adds no
  meaningful noise either).
- **Train/test split**: by-rollout prefix split (`rollouts[:n_train]` /
  `rollouts[n_train:]`) rather than a random shuffle-with-fixed-seed — valid
  since rollouts are IID (independent random `d_j`/`R_j` draws). Considered
  and rejected: a shuffle would guard against theoretical drift between the
  12 separate generation-run batches, but nothing in the pipeline suggests
  such drift exists, since every other config (`n_hits`, band width,
  compression displacement) is enforced constant across runs — only the
  random seed differs.

## Normalization

Settled through explicit review, following the vendored reference's own
`normalization.py` conventions:

- **Online accumulator, not precompute-once**: normalization statistics
  (mean/std) are accumulated online during training (running sum /
  sum-of-squares / count, updated each training step, frozen after
  `max_accumulations`) rather than fit once upfront from the training split.
  Needs porting to PyTorch as a stateful module (registered buffers updated
  in-place during training-mode forward passes). Consequence, accepted rather
  than overlooked: with only ~1,690 training pairs (far fewer than the
  datasets the reference's `max_accumulations=10**6` default was sized for),
  the accumulator keeps refining for the entire training run rather than
  converging and freezing partway through — a mild "moving target" effect on
  normalized inputs/targets during early training. This is very likely the
  cause of two sharp, brief loss spikes observed early in the first real
  training run (see `runs/metrics.json`) — not confirmed, but the timing is
  consistent with normalizer statistics still shifting substantially at that
  point in training.
- **Independent per-axis (per-component) scaling, not one pooled scale
  across all axes**: each component of a vector-valued field (`x_k`, `Δx`,
  and both edge relative-position features) gets its own
  independently-accumulated mean/std. This does not preserve a displacement
  vector's geometric direction the way a single shared scale across all
  three axes would (each axis can end up rescaled by a different factor);
  accepted since this matches the reference `Normalizer`'s own convention
  (which normalizes e.g. a velocity vector's components independently) and
  the forging geometry's axes plausibly do have different statistics anyway
  (axial vs. radial displacement are physically different phenomena here).
- **Input state and output target get independently-fit scales**: `x_k`
  (accumulated displacement, grows across a rollout — small at hit 1,
  largest by hit 5) and `Δx` (this hit's increment, a bounded per-hit delta
  likely to have a more consistent range since compression depth is fixed
  every hit) are normalized with two separate accumulators — one scale
  sized for the wider input range would poorly resolve the narrower output
  range, or vice versa.
- **What does *not* get normalized**: the three binary node-type flags
  (`is_bottom_face`, `is_pin_node`, `is_lateral_wall`) and the raw control
  broadcast (`[d_j_frac, sin(R_j), cos(R_j)]`) are left unnormalized — already
  well-scaled 0/1 indicators and an already-bounded control encoding,
  respectively. Three normalizers exist in total: one over the four
  non-flag/non-control node dims (signed distance to band + `x_k`), one over
  all 8 edge dims, one over the 3 `Δx` output dims.

## Per-node feature vector (11 dims, fed into the encoder)

The three node-type flags below are **not arbitrary categories** — each is
read directly off a specific Dirichlet boundary condition in
`boundary_conditions.py`, using that file's own selection logic (same
location functions, same tolerances) rather than an independently-invented
rule:

| # | Feature | Dims | Derived from (in `boundary_conditions.py`) | Why it's needed |
|---|---|---|---|---|
| 1 | `is_bottom_face` | 1 | The `bottom_face` Dirichlet location function: `x == x_bottom` (the billet's minimum x-coordinate), which fixes that face's x-displacement to exactly 0 — an axial symmetry-style constraint. | Flags an externally *imposed* displacement, not a free plastic response the network needs to learn. |
| 2 | `is_pin_node` | 1 | The `pin_node` selection: the single mesh point nearest `(x_bottom, y_min, 0)`, whose y/z displacement is fixed to remove residual rigid-body rotation. | A numerical anchor (see "Graph construction" diagram in the PDF version of this doc), not a physically special node — flagging it tells the network its motion is constrained for bookkeeping, not material response. Kept as an **independent** flag rather than folded into a mutually-exclusive one-hot with `is_bottom_face`, because `pin_node`'s location is chosen near the bottom face and the two conditions can co-occur on the same node. |
| 3 | `is_lateral_wall` | 1 | The same radius-tolerance check `on_outer_surface_np` uses to find contact-candidate nodes (`abs(radius - R0) <= tol_r`, `tol_r = 0.25*(2*R0/5)`) — 1 on the curved wall, 0 on either flat end. | Meaningful precisely because the graph is surface-only (every node is "on the surface"): contact/bulging only ever happens on the wall, never the caps, and this distinction isn't otherwise recoverable, since MeshGraphNets deliberately excludes absolute node position from node features (translation invariance; only *relative* position flows through edges). |
| 4 | signed distance to this hit's contact band | 1 | See formula below — computed from this node's rest x-position and the *current hit's* `d_j`/`x_max_band` (which vary hit to hit, unlike flags 1-3 which are fixed forever). | The one indispensable feature: the two control scalars (`d_j`, `R_j`) are broadcast identically to every node, so without this, nothing in the graph tells the network *which* nodes are physically near the press this hit. Must stay a differentiable function of `d_j` (see "Known future item" below), not a precomputed/cached constant. |
| 5-7 | current displacement `x_k` | 3 | N/A (state, not a BC) | `(ux, uy, uz)` at the start of this hit — the state being updated. |
| 8-11 | control broadcast | 4 | N/A (control input) | `[d_j_frac, sin(R_j), cos(R_j), u_j_frac]` for this hit, identical on every node. Rotation is periodic (0° and 360° are the same physical orientation); encoding it as a raw scalar would create a false discontinuity at the wrap-around point that a network has to learn around, while `sin`/`cos` make that periodicity exact by construction. `u_j_frac` added for the 3D-actuation expansion, following `d_j_frac`'s raw/unnormalized convention (see "Data source" above). |

**Exact formula for feature 4** (`data.py`'s `build_node_features`):

```
band_center_mm = 0.5 * (d_j_mm + x_max_band_mm)
dist_to_band    = rest_x_mm - band_center_mm
```

where `d_j_mm` and `x_max_band_mm` are read directly from that hit's
`manifest.json` record (already in absolute mm — no unit conversion needed).
Worked example from the real dataset (rollout 1, hit 1): `d_j_mm=48.84`,
`x_max_band_mm=68.14` → `band_center_mm=58.49`. A surface node at the axial
end (`rest_x=-5.0mm`) gets `dist_to_band=-63.49mm`; a node at the opposite end
(`rest_x=96.52mm`) gets `dist_to_band=+38.03mm`.

Features 1-3 and 8-10 are fed to the encoder raw, unnormalized; features 4
and 5-7 pass through the online node normalizer described above.

## Per-edge feature vector (8 dims, mesh edges only)

| # | Feature | Dims | What it is |
|---|---|---|---|
| 1-3 | relative rest position | 3 | `mesh_pos[sender] - mesh_pos[receiver]`, undeformed configuration |
| 4 | its norm | 1 | rest-frame edge length |
| 5-7 | relative current position | 3 | `(mesh_pos + x_k)[sender] - (mesh_pos + x_k)[receiver]`, deformed configuration |
| 8 | its norm | 1 | current-frame edge length — comparing this to #4 is effectively a local strain signal |

Chosen over the CFD-style rest-frame-only convention (`cfd_model.py`, 4-dim)
because forging is a large-plastic-strain structural problem, much closer to
`cloth_model.py`'s regime (which uses this same 8-dim rest+current
construction) than to an incompressible, non-deforming fluid domain.

## Network shape (encoder → processor → decoder)

![ForgeGNN architecture: encoder, processor, decoder](architecture_diagram.png)

- **Encoder** — two separate MLPs, one per graph component:
  - Node encoder: `10 → 128 → 128 → 128`, + LayerNorm. Takes the 10-dim raw
    node feature vector (table above) for every node and produces a 128-dim
    node **latent**.
  - Edge encoder: `8 → 128 → 128 → 128`, + LayerNorm. Same idea for the
    8-dim edge feature vector, producing a 128-dim edge **latent**.
  - Every MLP in this document has the same internal shape convention:
    `num_layers=2` hidden layers of width `latent_size=128` (ReLU between
    them), then one more Linear layer to the stated output width, with no
    activation after that last layer.
- **Processor** — 15 copies of a message-passing block (`GraphNetBlock`),
  **each with its own independent weights** (not a shared/recurrent block —
  15× the parameters of one block). Each block does, in order:
  1. **Edge update**: gather each edge's sender and receiver node latents,
     concatenate with that edge's own current latent (`128+128+128=384`),
     run the edge-update MLP (`384→128→128→128`, +LayerNorm) to get a
     candidate new edge latent.
  2. **Residual add**: the candidate is added to the edge's latent from
     *before* this block (`new = candidate + old`), not used to replace it
     outright — this is what lets gradients and information flow cleanly
     through all 15 blocks.
  3. **Aggregate**: sum every edge's (just-updated) latent into its receiver
     node (`torch.zeros(...).index_add_(1, receivers, edge_latents)` — one
     sum per node, over however many edges point at it).
  4. **Node update**: concatenate each node's own current latent with that
     summed incoming-edge value (`128+128=256`), run the node-update MLP
     (`256→128→128→128`, +LayerNorm) to get a candidate new node latent.
  5. **Residual add**: same pattern as step 2, for nodes.
  The resulting updated node and edge latents become the input to the next
  of the 15 blocks; after the last block, only the final node latents
  continue on to the decoder (the final edge latents are discarded).
- **Decoder** — one more MLP, `128 → 128 → 128 → 3`, **no LayerNorm** (the
  only MLP in the network without one, matching the reference — the output
  needs to be able to take any real value, not a normalized one). Applied to
  every node's final latent, producing the predicted (still-normalized)
  displacement delta.
- **Sizing decision**: reference MeshGraphNets defaults (`latent_size=128,
  num_layers=2, message_passing_steps=15`) kept as the starting point, **not**
  scaled down. Total ≈2.33M parameters (encoder ≈69K, processor ≈2.23M,
  decoder ≈33K) — the 1,690-single-step-pair, 338-rollout dataset (each pair
  supervising ~4,355 surface nodes at once) justifies this scale; to be
  adjusted only if train/test curves actually show over/under-fitting.

## Loss function & reported metrics

These are two different quantities, computed from the same forward pass but
used for entirely different purposes — this distinction matters and was a
source of confusion previously, so it's stated explicitly:

**Training objective (what is actually backpropagated)** — `ForgeGNN.loss()`:
per-node squared error, summed over the 3 displacement components and
averaged over every node and every example in the batch, computed in
*normalized* space:

```
loss = mean over (node, example) of  Σ_c (Δx̂_norm,c − Δx_norm,c)²
```

where `Δx̂_norm` is the raw decoder output (no denormalization) and
`Δx_norm` is the true displacement delta passed through the *same* output
`OnlineNormalizer`. Comparing in normalized space (not physical mm) matches
the reference MeshGraphNets convention and keeps the loss on the scale the
network's own LayerNorms already operate at. **This is the only quantity
used for backpropagation and for early-stopping/checkpoint selection**
(`best_test_loss`) — it is never replaced by, or trained on, NRMSE.

**Reported/plotted diagnostic (NRMSE)** — added because the training loss
above has no intuitive physical scale (it's in normalized units that depend
on the normalizer's current, still-refining statistics). This metric is
**purely a reporting tool**: not backpropagated, not used for early stopping
or checkpoint selection, computed from the same forward pass at essentially
no extra cost:

```
NRMSE = RMSE(Δx̂_phys, Δx_phys) / std(Δx_phys)
```

where `Δx̂_phys` is the *denormalized* (physical-units, mm) prediction,
`Δx_phys` is the true physical displacement delta, RMSE is pooled over every
node/component/example in a split, and `std(Δx_phys)` is computed **once**
per split (train, test) — since the true target data doesn't change epoch to
epoch, neither does its std. `NRMSE = 1` means the model does no better than
always predicting the per-component mean displacement delta for that split;
`0` is perfect; lower is better. `train.py`'s `loss_curves.png` plots NRMSE
(with a dashed reference line at 1.0), not the raw training loss, for exactly
this reason. The autoregressive rollout eval reports the same NRMSE
definition per hit (using the test split's std as a fixed reference scale),
alongside plain physical RMSE in mm.

## MPC compatibility of the contact-band feature — resolved

Previously an open/deferred item; stage 3 (`../control/`) is now built, and
both sub-questions below are settled:

1. **Implementation**: `build_node_features`'s control inputs (`d_j_frac`,
   `R_j_deg`, `u_j_frac`, and `d_j_mm`/`x_max_band_mm` feeding the
   distance-to-band formula) originally broke autograd — they were baked
   into a fresh `torch.tensor(...)` each call, detaching them from any
   upstream graph. Fixed to use `torch.as_tensor` + `torch.stack`/
   `torch.deg2rad` instead, so a grad-tracking tensor passed in from the MPC
   planner flows through unchanged, while the existing plain-float training
   path is unaffected (verified both paths explicitly). This is what makes
   the SQP planner below actually able to compute `∂cost/∂u`.
2. **Solver**: not CEM/gradient-free as this section originally speculated
   — `control/mpc.py` uses **single-shooting SQP** (`scipy.optimize.minimize`,
   `method="SLSQP"`, which does BFGS-approximated Hessian updates
   internally). "Single-shooting" resolved the real blocker this section
   didn't anticipate: the control/README.md original framing assumed
   *multi*-shooting (state `x_k` as an explicit decision variable, tied to
   the dynamics via equality constraints) — intractable here since `x_k` is
   the full ~13,065-dim surface displacement field. Single-shooting instead
   treats only the controls (`u_0..u_{N-1}`, 3 per horizon step) as decision
   variables, rolling the GNN forward to get `x_k` rather than optimizing it
   directly — dynamics are satisfied by construction, no equality
   constraints needed at all. See `../control/README.md` for the full
   closed-loop results.

## Stack

PyTorch, ported from the vendored TF1/Sonnet reference — matching this
repo's existing PyTorch-based surrogate-training tooling and `environment.yml`
(no TensorFlow dependency in this repo). `data.py` reads the same
`manifest.json`/`.vtu` format the rest of this repo uses, via its own
independent minimal loader.

## Message-passing depth sweep (`mp_sweep/`)

Trained 5 models (`message_passing_steps` = 1, 5, 15, 45, 135; everything
else at reference defaults) on the identical 383-rollout snapshot
(`rollout_snapshot_383.json`), then evaluated Chamfer/Hausdorff per forging
step, averaged across all 77 test rollouts (`full_test_set_eval.py`) —
against an undeformed-billet no-skill baseline (`undeformed_baseline_eval.py`)
for context. Full results, tables, and discussion are in the session that
produced them; summary:

- **Chamfer (average error) is flat across every depth from 1 to 135** —
  differences are within rollout-to-rollout noise. All five models beat the
  undeformed-baseline by 7-8x, so the surrogate is learning real structure;
  depth just isn't what limits average accuracy.
- **Hausdorff (worst-case error) tells a different story**: M=1 degrades
  sharply hit-over-hit (0.48mm -> 1.50mm by hit 5, autoregressive), roughly
  2x worse than every deeper model by hit 5. M=5 through M=135 all sit
  tightly together (~0.78-0.82mm) — accuracy saturates at M=5, and going
  deeper buys nothing further on either metric.
- **Likely reason**: the network doesn't need message passing to find "where
  the press is" — that's handed to it directly via the signed
  distance-to-band feature (per-node feature #4 above). What's left for
  message passing is short-range local mechanical coupling (load transfer
  between neighboring elements), not long-range propagation — consistent
  with saturating after only a handful of hops. A few worst-case nodes
  (plausibly geometric transition regions needing >1 hop of context) likely
  drive the M=1 Hausdorff gap without being numerous enough to move Chamfer.
- **Cost scales linearly and steeply with depth** — gradient-computation
  time (one forward+backward through a 5-step single-shooting horizon, the
  literal cost of one SQP iteration): 0.57s (M=1) -> 1.2s (M=5) -> 3.3s
  (M=15) -> 10.2s (M=45) -> **31.0s (M=135)**. M=135 additionally **OOM'd at
  48GB** and needed 180GB to complete one gradient call — a real feasibility
  limit for MPC, not just a speed cost. Given zero accuracy benefit beyond
  M=5, there's a real case for switching the MPC-facing model from the
  current M=15 default to M=5 (a real, measured full SQP solve: 3.28 min at
  M=5 vs. 9-16 min at M=15).

## Finetuning on square-rod data (`finetune.py`, `finetune_square/`)

`checkpoint_mp_5.pt` finetuned on scheduled square-rod rollouts from
`data/dataset_finetuning/` (`generate_square_rollout.py`: a 48-hit
simple_square-inspired schedule plus jittered variants). Same per-node MSE
loss, AdamW lr 1e-5, normalizers frozen at their pretrained statistics,
early stopping on the test run (patience 20). Test run: `square_jitter_seed3`
(held out of training, but it also selects the early-stopping epoch, so its
scores are slightly optimistic). "Hit 48" is the end of a full 48-hit
autoregressive rollout from the undeformed billet. The last column is the
forgetting check: one-hit-ahead NRMSE on the 77-rollout pretraining test
split.

| Model | Chamfer, hit 48 (mm²) | Hausdorff, hit 48 (mm) | RMSE, hit 48 (mm) | Pretraining-test NRMSE |
|---|---|---|---|---|
| Pretrained (`checkpoint_mp_5.pt`) | 23.79 | 17.46 | 4.39 | 0.298 |
| Finetune 1: square + seeds 1-2 (144 hits), no replay | 0.93 | 2.57 | 0.53 | 0.402 |
| **Finetune 2: square + seeds 1-2, 4-9 (432 hits) + equal pretraining replay** | **0.91** | **2.11** | **0.52** | **0.298** |

Finetune 2 (`--replay-ratio 1.0`: each epoch adds 432 freshly drawn
pretraining-train hits; best epoch 116 of 136, 20 min on an L40S) is the
current checkpoint, `finetune_square/checkpoint_mp_5_finetuned_square.pt`.
It matches or beats finetune 1 on the square and shows no forgetting.
Finetune 1's checkpoint was deleted; the table row is its only record.
`finetune_square/` is wiped at the start of every
`submit_gnn_finetune_square.sh` run, so record results here.

## Running

```bash
python -m applications.Agility_Forge.GNN.train \
    --dataset-dir applications/Agility_Forge/data/dataset_pretraining --epochs 100
```

Run on a GPU node (`cluster_setup.md`), not a login node — see `train.py`'s
docstring for the measured CPU-vs-GPU-bound runtime at full reference sizing,
and `applications/Agility_Forge/slurm_scripts/submit_gnn_train.sh` for the
actual SLURM job used for the first real training run.

## Coil / spatial-die GNN (`coil.py`, `coil_sweep/`) — settled 2026-10-04

For the new simulator (12.7 mm die fixed in space, coil reheat every 6 hits). The old pipeline above is untouched; `coil.py` is self-contained (reuses `EncodeProcessDecode`, `OnlineNormalizer`, the surface mesh, and the Chamfer/Hausdorff metric).

- **Inputs (16 per node):** node type (3), reheat flag, die push (3: the displacement the die would impose on the node, from the current shape and the stroke; soft 0.25 mm footprint edge for MPC gradients), distance to the die centre, displacement (3), coil temperature, temperature change at a reheat (/300), sin and cos of the angle, stroke fraction. No absolute die position. Mesh edges only (no world edges: no self-contact in the data or the simulator).
- **Temperature:** the coil formula's value from the latest reheat, carried with the metal; never predicted. Old-simulator data uses its starting profile.
- **Reheat step:** the same network with the reheat flag set, dT = new − old coil temperature, and no die inputs; it predicts the reheat's shape change. A fit from coil temperatures alone did no better than "no change" (0.118 vs 0.115 mm), so the report compares the trained step against "no change".
- **Stages:** (1) old random hits from scratch, lr 1e-4, normalizers learn; (2) old finished square runs + an equal number of old random hits each epoch, lr 1e-5; (3) new 5-pass runs (`dataset_die12_coil_5pass`, keep-the-hotter reheat; pass 5 half-gap 5.0–5.3 mm across the runs) as whole cycles (reheat + up to 6 hits chained from the true pre-reheat shape), plus an equal number of old random hits as one-hit examples, half with strokes under 1 mm (the new runs have none under 1.4 mm), lr 1e-5. Split: train square + seeds 1, 2, 4–8; early stop seed 9; test seed 3. Runs that stopped early contribute the hits they reached.
- **Sweep and report:** M = 1–10 (`slurm_scripts/submit_gnn_coil_stage12.sh`, `_stage3.sh`, `_summary.sh`); per M: cycle-chain Hausdorff/Chamfer on seed 3 (with the reheat step and with "no change"), the reheat step's error vs "no change", the short-stroke check (old random test hits under 1 mm, before vs after stage 3), and gradient time for a reheat + 6-hit plan. Outputs in `coil_sweep/mp_M/` and `coil_sweep/summary.json`.

### Temperature in the nodal state (`coil_T.py`) — 2026-10-07

Same pipeline as `coil.py`, but the state per surface node is displacement + temperature. Each hit step predicts the change of both (4 outputs; in the loss temperature counts like one displacement component). The temperature input is the real surface temperature at the step's start (in practice an IR camera at each scan); old-simulator hits start from their starting profile. A reheat's temperature is the simulator's own rule (hotter of current and coil profile, checked exact to 0.000 C), so only its shape change is learned. Cycle chains feed predicted temperature forward with predicted shape. Why: the bookkept coil temperature of `coil.py` reached >= 1090 C on 96% of the square section from cycle 3 on (real mean ~1000–1045 C) and could not show the die chill (the 2nd hit at a station starts ~180–215 C colder under the die than that input says). Reports add temperature error (overall and under the die). Sweep: `coil_T_sweep_test_square/` (plain square held out, validation seed 9, M = 1–10); `coil.py`'s sweeps are the frozen-temperature baseline.

### Deep ensemble for the MPC's uncertainty penalty — 2026-10-08

Five `coil_T.py` models, M = 5, plain square held out, identical except `--seed` (initial weights and data shuffling): seed 0 is `coil_T_sweep_test_square/mp_5/stage3.pt`; seeds 1–4 are trained through all three stages from scratch (stage 1 not reused, to keep the members diverse) into `coil_T_ensemble_test_square/seed_k/mp_5/` by `slurm_scripts/submit_gnn_coilT_ensemble.sh`. The MPC uses the members' spread as a penalty (`control/README.md`, "Ensemble uncertainty penalty"). Before using it, `ensemble_calibration.py` checks that the spread means something: per hit of a cycle chain (each member from the true cycle start, as the MPC forecasts), the spread (sum over nodes of the members' y,z variance, mm²) against the members' mean's y,z squared error (mm²), on the held-out plain square run and on the MPC runs' own hits; Pearson (raw and log-log), Spearman, median error/spread, and `spread_vs_error.png`, written to `coil_T_ensemble_test_square/calibration/`. Seed-only ensembles of GNNs can collapse to near-identical predictions (Vieira et al. 2026, arXiv 2605.22593); if the correlation is weak, the fallback is bootstrapped members.
