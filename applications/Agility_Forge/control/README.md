# MPC control loop (stage 3) — design notes

Uses the trained Koopman Autoencoder (`../koopman/`, stage 2) as the fast
predictive model inside an MPC loop, and `plant_interface.ForgingPlant` (the
real Agility_Forge/`jax_forge` simulation) as the ground-truth plant each
control step is actually applied to and validated against.

## Open design questions (resolve before filling in the stubs)

- **Control horizon**: how many hits ahead to plan over. Longer horizons need
  the Koopman model's multi-step rollout to stay accurate (see the koopman
  README's multi-step-consistency loss question).
- **Solver**: with linear latent dynamics, MPC reduces to a QP per step — via
  `cvxpy`, a custom projected-gradient loop in `torch`, or another QP library.
  Not yet chosen / added as a dependency.
- **Control bounds**: physical limits on `d_j` (compression depth), `x_max_band`
  (hit position), `R_j` (rotation) — should mirror what `generate_dataset.py`'s
  rollout sampler already treats as valid ranges, so MPC never plans a hit
  stage 1 never demonstrated.
- **Cost function / target state**: what the controller is actually driving
  toward (final displacement/temperature profile? a specific target shape?) —
  needs to be defined before `MPCController.__init__`'s target_state makes sense.
- **Failure handling**: `ForgingPlant.step` can fail to converge (see its
  docstring) — the MPC loop needs a fallback (re-plan with a smaller/different
  control, or abort) rather than assuming every planned hit succeeds.

## Validation loop

Because every planned control is applied to the *real* plant (not just
predicted), `MPCController.run`'s plan-vs-actual mismatch at each step is a
direct, ongoing measurement of Koopman surrogate quality — worth logging
explicitly, not just the final control performance.
