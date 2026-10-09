# MPC control loop (stage 3)

Uses the trained GNN surrogate (`../GNN/`, stage 2) as the fast predictive
model inside an MPC loop, and `plant_interface.ForgingPlant` (the real
Agility_Forge/`jax_forge` simulation) as the ground-truth plant each control
step is actually applied to and validated against. An earlier Koopman
Autoencoder surrogate (`../koopman/`) was explored for this role first; the
GNN surrogate is the active model throughout everything below.

**Status**: implemented and validated against the real plant, not just the
surrogate. `plant_interface.ForgingPlant` mirrors `generate_dataset.py`'s
per-hit loop exactly (same BC rebuild, same thermal relaxation ramp — see
"Design decisions" below for why that had to be settled explicitly, not
assumed). `mpc.MPCController` implements single-shooting SQP planning
(`plan()`) and the full receding-horizon closed loop against the real plant
(`run()`).

## Design decisions (settled)

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

- **Shooting method: single-shooting, not multi-shooting.** The literal
  problem statement (`x_{k+1}=f(x_k,u_k)` as a constraint, cost over `x_k`)
  reads like multi-shooting notation — `x_k` as an explicit decision
  variable, tied to the dynamics via an equality constraint. That's
  intractable here: `x_k` is the full ~13,065-dim surface displacement
  field, so a 5-step horizon would mean ~65,000+ decision variables and
  as many nonlinear equality constraints. Single-shooting instead makes
  the *only* decision variables the controls (`u_0..u_{N-1}`, 3 per step —
  `d_j_frac`, `R_j_deg`, `u_j_frac`); `x_k` is computed by literally rolling
  the GNN forward from `x_0`, so the dynamics constraint is satisfied by
  construction and never appears in the solver at all. For a horizon this
  short (~5 hits), this is standard practice, not a simplification of
  the real problem.
- **Solver: SQP via `scipy.optimize.minimize(method="SLSQP")`.** Not the
  QP-per-step approach `GNN/README.md` originally floated — that only works
  for the koopman model's *linear* latent dynamics; a GNN processor (many
  rounds of nonlinear message passing) is not affine in the control
  regardless of solver choice. SLSQP does exactly the requested
  BFGS-approximated-Hessian SQP internally (sequential QP subproblems,
  quasi-Newton Hessian of the Lagrangian) — no hand-rolled Hessian update
  needed. Gradients are exact, not finite-difference: `_rollout_cost`'s
  entire single-shooting rollout (all horizon steps chained into one graph)
  gets one `torch.autograd` `.backward()` call per SQP iteration. This
  required a real prerequisite fix — `GNN/data.py`'s `build_node_features`
  originally broke autograd for the control inputs (constructed a fresh,
  non-differentiable tensor from plain Python floats each call); fixed via
  `torch.as_tensor`/`torch.stack` so a grad-tracking tensor flows through
  unchanged while the existing plain-float training path is untouched.
- **Cost function**: `sum_k ||x_k - x_ref||_Q^2` over the horizon, `Q =
  identity` over the full surface field (no per-node/per-axis weighting —
  a default, not a considered choice; worth revisiting if certain regions
  should matter more than others). `x_ref` is a **fixed target for every
  step** (a real rollout's true final-hit geometry, from the held-out test
  set) — there's no time-varying intermediate reference, so this
  deliberately pulls early states toward the final shape too, which shows
  up in practice as several controls landing exactly on their bounds
  (see "Validation results" below).
- **Control bounds**: `d_j_frac in [0.02, 0.78]` and `R_j_deg in [0, 360)`
  taken directly from what `generate_dataset.py`'s sampler actually used
  (edge_margin_frac=0.02, band_width_frac=0.2 defaults; `R_j_deg` periodic —
  optimizing a plain bounded scalar rather than `(sin,cos)` + unit-circle
  constraint is a deliberate simplification, fine unless the true optimum
  sits at the wraparound). `u_j_frac in [0, 1]`, mapping to physical
  `u_j_mm in [0.0, 2.0]` (`MPCController`'s `u_j_mm_min`/`u_j_mm_max`,
  default `mpc.U_J_MM_MIN`/`U_J_MM_MAX`) — **not** the training data's
  literal 0.5-2.0mm sampled range. An extended-horizon pilot
  (`control/results/before_2026-09-26_fixes/rollout_415_10hits/`) showed the 0.5mm floor gives
  receding-horizon MPC no way to choose "no hit": once already near target,
  every remaining required hit is forced to apply real plastic deformation,
  which plateaued Chamfer and made Hausdorff creep upward hit-over-hit
  rather than staying flat. Lowering the floor to 0.0 let the optimizer
  choose a genuine near-zero stroke instead, and Hausdorff stayed flat.
  Accepted tradeoff: `u_j` in `[0, 0.5)`mm is outside what
  `generate_dataset.py` ever sampled, so the GNN is extrapolating there.
- **Thermal relaxation ramp between real hits: included, not skipped.**
  `plant_interface.ForgingPlant.step()` mirrors `generate_dataset.py`'s
  10-step reheat-and-reequilibrate ramp exactly. This was a real fork: the
  GNN was trained exclusively on data generated *with* this ramp, so
  skipping it would put every real state the MPC re-plans from
  systematically out-of-distribution relative to training.
- **Failure handling: fail-fast, no retry.** `ForgingPlant.step` raises on
  solver non-convergence, matching `generate_dataset.py`'s own convention —
  a silent retry-with-different-control could mask a real problem with the
  planned control rather than surfacing it.
- **Warm-starting**: `run()`'s receding-horizon loop seeds each re-plan's
  SQP initial guess from the previous plan's un-applied tail (shifted by
  one), rather than a fresh generic guess every step — cheap to add, real
  reduction in iterations needed per step since each successive plan is a
  shorter version of a similar problem.

## Validation results

**Open-loop** (`validate_open_loop.py`): a single SQP-planned 5-step control
sequence, computed once against the GNN surrogate targeting a real held-out
rollout's true hit-5 geometry, applied to the real plant with no
re-planning. Real-plant result: **0.1536mm RMSE** against the target — the
GNN's own internal prediction for this plan was 0.152mm, i.e. the real
result matched the surrogate's prediction almost exactly (0.002mm gap). By
comparison, replaying the target rollout's *actual real controls* through
the GNN scored 0.627mm — the SQP-found plan did meaningfully better than
just repeating the true trajectory, evaluated through the same surrogate.

**Closed-loop** (`run_closed_loop.py`, `MPCController.run()`): re-plans from
the plant's *true* post-hit state before every real hit (horizon shrinking
5→4→3→2→1). Run against 5 independent held-out targets (`results/before_2026-09-26_fixes/rollout_
{415,416,417,418,419}/`, each definitionally never seen by
`checkpoint_3dim.pt`'s training — completed by a later dataset-generation
job after that checkpoint had already been trained):

| target rollout | final RMSE | Chamfer (mm²) | Hausdorff (mm) |
|---|---|---|---|
| 415 | 0.166mm | 0.165 | 0.795 |
| 416 | 0.196mm | 0.229 | 0.834 |
| 417 | 0.268mm | 0.378 | 1.218 |
| 418 | 0.189mm | 0.207 | 1.014 |
| 419 | 0.201mm | 0.241 | 1.168 |

All 5 in the same range (0.17-0.27mm RMSE) — no target is dramatically
easier or harder than the others, though 417/418/419's larger Hausdorff
values (>1mm, vs. 415/416's <0.85mm) suggest some targets leave a more
localized worst-case mismatch than others even at similar average error.

**Caveat, stated plainly**: none of this is proof the surrogate's plans are
*better forging schedules* in some general sense — the cost function
literally optimizes the GNN's own predictions, so a plan that beats the
true trajectory *through the GNN* isn't automatically beating it in
reality. What the open-loop result actually shows is that, for this one
case, the surrogate's planning signal transferred to the real plant with
very little degradation — encouraging, but a single data point, not a
general guarantee.

## Per-step output for animation

Each `results/before_2026-09-26_fixes/rollout_{id}/` folder holds `target.vtu` (the true target
geometry) plus `step_00.vtu` (undeformed) through `step_05.vtu` (state after
each real hit) as surface-mesh `.vtu` files, saved incrementally via
`MPCController.run()`'s `step_callback` — so a crash/timeout partway through
a multi-hour closed-loop run still leaves usable partial output. Intended
for stringing together into an animation of the controlled forging sequence
against its target.

## Folder layout and 2026-09-26 fixes

Scripts live at the top of `control/`; every run's output lives in
`results/<run>/` (moved there 2026-09-26 from top-level `mpc_*` folders).

Two planner bugs were fixed on 2026-09-26. Results made before then should be
read with them in mind:

- **Stroke mismatch.** Once `U_J_MM_MIN` was lowered to 0.0, the MPC sent the
  plant `u_mm = 2 * u_j_frac` but fed the GNN the same `u_j_frac`, which the
  GNN reads on its 0.5-2.0mm training scale. Frac 0.5 meant 1.0mm to the plant
  but 1.25mm to the GNN. Now every GNN call goes through `mpc.gnn_u_frac()`.
  Affected: `results/before_2026-09-26_fixes/prism_target*`, `results/before_2026-09-26_fixes/rollout_415_chamfer_hausdorff`,
  `results/before_2026-09-26_fixes/closed_loop_test_set`.
- **SLSQP scaling.** The raw cost is ~1e6 (squared mm over ~4.4k nodes) with
  gradients ~1e5, and R_j spans 0-360 against 0-1 for the other controls.
  SLSQP returned its initial guess unchanged while reporting success. `plan()`
  now optimizes controls rescaled to [0, 1] against a cost normalized by its
  initial value; returned controls and `result.fun` are unchanged in units.
  This probably explains earlier "do nothing" plans (e.g. the prism target's
  near-zero strokes pinned at d_j = 0.78). Every run before the fix is affected.

`MPCController` also takes `d_j_bounds` (e.g. a floor at the initial
temperature profile's 800 C point, d_j = 0.1837).

`eval_square_target.py` runs closed-loop MPC toward the square run's final
geometry (`data/dataset_finetuning/square`, hit 48) with the finetuned GNN:
horizon always 10, exactly 50 hits, stroke 0-2mm, d_j >= 0.1837. After every
hit it checkpoints the plant state and rewrites `results.json` and the plots
(error_vs_hit.png, controls_{station,angle,stroke}.png), so a resubmitted job
resumes from the last completed hit. Output: `results/real_simulator/e0_original_cost/`.

## Open items

- **Finetuned surrogate not yet used here.** `GNN/finetune_square/checkpoint_mp_5_finetuned_square.pt`
  (finetuned on scheduled square-rod rollouts with pretraining replay; see
  `GNN/README.md`) predicts a full 48-hit square-forming rollout far better
  than `checkpoint_mp_5.pt` (final-hit Hausdorff 2.1mm vs. 17.5mm). Only
  `eval_square_target.py` uses it; the other eval scripts still default to the
  un-finetuned checkpoints.
- **`U_J_MM_MAX = 2.0` is a surrogate-validity bound, not a die limit.** The
  square-rod rollouts show the 2mm per-hit stroke cap binding often (9-26 of
  44-48 hits per run), so those runs end short of their planned sizes.

- **Cost function's Q and same-target-at-every-step choices are defaults,
  not settled decisions** — worth reconsidering (a terminal-only cost, or a
  reduced/weighted Q) given the bound-pinning pattern observed across every
  closed-loop run so far.
- **Message-passing depth**: `MPCController` currently uses `checkpoint_
  3dim.pt` (M=15). `../GNN/README.md`'s depth sweep found M=5 matches or
  beats M=15 on every accuracy metric at a fraction of the planning cost
  (3.28 min vs. 9-16 min per full SQP solve) — switching the control loop's
  default model is a live option, not yet done.
- **Only 5 closed-loop targets tested**, all drawn from the same held-out
  snapshot region; no systematic study of which targets are easy/hard yet.

## MPC for the new simulator (`mpc_coil.py`) — built 2026-10-04, not yet run

For the 12.7 mm die fixed in space with a coil reheat every 6 hits (a reheat keeps the hotter of current and coil temperature), using the coil GNN (`../GNN/coil.py`). The old `mpc.py` is unchanged.

- **Cycle:** scan → plan 6 hits → reheat → apply all 6 hits to the simulator → scan. Single-shooting SQP (SLSQP), 18 variables: die centre, angle, stroke per hit.
- **Coil and temperature:** the coil sits at the average of the 6 planned die centres, so moving a hit moves the coil. The GNN's temperature input is bookkept, never measured: cycle 1 gets the coil profile; each later reheat keeps the hotter of the bookkept temperature and the coil profile on the scanned shape (smooth max and smooth plateau edge while planning, exact when committed).
- **Forecast:** GNN reheat step (none on cycle 1), then 6 GNN hit steps with the cycle's temperatures fixed.
- **Cost:** cross-section (y, z) distance of every surface node to the ideal square target, summed over the 6 forecast shapes (as in e14).
- **Bounds:** die centre 31.58 mm to the scanned free end − 6.35 − 3 mm; angle 0–180°; stroke 0.5–2 mm.
- **Stop:** every width in the target's square region within 10.6 + 0.2 mm, or 13 cycles.
- **Output:** the data runs' manifest format (so MPC-visited hits can be used for GNN training), the GNN's forecast per cycle, and the forecast error per hit. Checkpoints after every hit and reheat. `slurm_scripts/submit_mpc_coil.sh`.
- **Tested:** end to end with a fake simulator and a stand-in GNN (stage 2, M = 2); not yet on the real simulator or with a trained stage-3 GNN.

**Update 2026-10-08 (temperature in the GNN state, `GNN/coil_T.py`):** the scan also reads the real surface temperature (an IR camera in practice) instead of the bookkept estimate; the forecast applies the reheat rule (hotter of measured temperature and coil profile; cycle 1: the coil profile) and then 6 GNN hits that predict shape AND temperature, the temperature carried hit to hit; cost unchanged. Also decided: every die centre within ±11 mm of the coil centre (linear SLSQP constraints; the data's cycles spread ±10.7 mm), angles 0–180° kept, cap 20 cycles, stop check from the first station (31.58 mm; the x ≈ 27 mm spot next to the taper never got below 11–13 mm in any data run), direct linear solver in the simulator. Planning test on saved states (CPU): cost −5% (fresh billet) and −11% (after hit 40), coil-zone limit active, angles chosen within 6° of 0/90° though free; ~3 min per plan on CPU. First run: M = 5, job 3984018 → `results/coil_mpc_tstate_M5/`.

**Update 2026-10-08 (multi-start planner):** random shooting + top-K refinement (RoboCraft-style): N = 100 uniform plans scored by the GNN in batches without gradients, the K = 5 best each refined by the unchanged SLSQP, best kept; the ±11 mm coil-zone constraint was removed. Report: `../progress/control_reports/2026-10-08_multistart_planner/`.

### Ensemble uncertainty penalty — 2026-10-08, not yet run

Run 1 failed because the plans exploited GNN errors in states and actions the GNN had not seen. To make the planner pay for that without copying the schedule, `mpc_coil.py` can plan with a deep ensemble (Lakshminarayanan et al. 2017) as in PETS (Chua et al. 2018), with a variance penalty as in MOPO (Yu et al. 2020).

- **Members:** 5 copies of the temperature-state GNN (`../GNN/coil_T.py`, M = 5, plain square held out) differing only in `--seed`: seed 0 is `GNN/coil_T_sweep_test_square/mp_5/stage3.pt`, seeds 1–4 go into `GNN/coil_T_ensemble_test_square/seed_k/` (`slurm_scripts/submit_gnn_coilT_ensemble.sh`, all three stages each).
- **Forecast:** each member rolls the 6-hit plan forward on its own from the same scan (its own shape and temperature hit to hit; the same reheat rule).
- **Cost:** J = Σ_k cross_section(x̄_k, target) + λ Σ_k s_k, where x̄_k is the members' mean shape after hit k and s_k is the sum over surface nodes of the across-member variance (1/M) of the y and z displacement. Both terms are mm², so λ (`--var-weight`) is dimensionless; λ = 1 equals the members' average own cost. Used in the screening and in the SLSQP refinement (gradients through all members).
- **Flags:** `--checkpoints a.pt b.pt ...` and `--var-weight λ`; `--checkpoint` still works as a 1-member ensemble, and with λ = 0 it reproduces the previous planner bit for bit (checked on two saved states: same screening costs, plans and final costs).
- **Logged:** per cycle in `manifest.json` (`cycles[].cost_terms`): the chosen plan's cross-section term, spread term, λ and number of members; per-hit forecast spread (`forecast_spread_mm2`) and y,z squared error of the mean (`forecast_yz_sq_err_mm2`); members' forecasts in `rollout_01/cycle_NN_forecast_members.npy` (the mean is in `cycle_NN_forecast.npy` as before).
- **Check first:** `../GNN/ensemble_calibration.py` (spread vs the mean's actual error per hit, on the held-out plain square and on the MPC runs' own hits). If the spread barely correlates with the error, the members have collapsed; the fallback is bootstrapping (each member on its own resample of the training hits).
- **Coil zone again (optional):** `--coil-zone-mm 11` restores the first run's constraint (every die centre within 11 mm of the coil centre, linear constraints in SLSQP; screening samples ignore it); off by default. With one checkpoint, λ = 0 and `--n-samples 1` it is the first run's planner: cycle 1's starting cost matches run 1 exactly and the final cost within 0.2% (CPU vs run 1's GPU; SLSQP magnifies float differences into a plan ~2 mm / 4° different).
- **λ sweep:** `slurm_scripts/submit_mpc_coil_ensemble.sh`, λ ∈ {0, 0.1, 1, 10} → `results/coil_mpc_tstate_M5_ens_lam<λ>/`, otherwise the multi-start run's settings. λ = 0 is the ensemble-mean-only baseline (gain from averaging vs gain from the penalty).
