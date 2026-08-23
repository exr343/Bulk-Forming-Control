# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

This repo is **Agility Forge**: a differentiable thermo-mechanical simulation of multi-hit rotational forging, built for a CIRP 2027 submission, plus (in progress) a Koopman Autoencoder surrogate and an MPC control loop around it. See [README.md](README.md) for the pipeline overview.

It started as a clone of [JAX-FEM](https://github.com/deepmodeling/jax-fem) plus its full example-application gallery, docs, and test suite; those were stripped out (they weren't dependencies of this project) leaving only the forging-specific code. `jax_forge/` — the FEM core Agility_Forge actually runs on — is a project-specific fork of JAX-FEM (originally named `jax_fem_checkpoint/`, renamed once the unmodified upstream copy it was disambiguated against was removed). If you're looking for the general-purpose JAX-FEM library/example gallery, it isn't here; this repo is scoped to forging + the surrogate/control pipeline only.

## Environment setup

Conda env defined in `environment.yml` (Python 3.10, JAX, PyTorch, PETSc via `petsc4py`, FEniCS, gmsh, meshio):

```bash
conda env create -f environment.yml
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # required for petsc4py
```

`jax`/`jaxlib`/`torch` are pinned in `environment.yml` (`jax==0.6.2`, `jaxlib==0.6.2`, `torch==2.11.0+cu128`).

**GPU driver gotcha**: this cluster's GPU nodes run NVIDIA driver branch R570 (checked across V100/A100/P100 nodes), which caps CUDA compatibility at **12.8** — CUDA 13.0 requires driver ≥580, which isn't available anywhere on this cluster (it's not a specific-node/GPU-model issue; every node sampled was on R570). A bare `pip install torch` defaults to the newest build (currently a CUDA 13.x wheel) and silently falls back to CPU with a `torch.cuda.is_available() == False` + a driver-version warning — install from PyTorch's CUDA 12.8 wheel index instead:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### On the cluster (see `cluster_setup.md`)

```bash
srun -p gpu --gres=gpu:1 -c 4 --mem=32G -t 04:00:00 --pty bash -l
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Batch dataset generation runs go through `applications/Agility_Forge/submit_dataset_generation.sh` (`sbatch submit_dataset_generation.sh`, must be submitted with CWD at repo root — it `cd`s there itself). Logs land in `slurm_logs/`, relative to wherever `sbatch` was invoked from (repo root if following the instruction above, `applications/Agility_Forge/slurm_logs/` if submitted from inside that directory instead) — both locations are git-ignored (SLURM logs are transient run output, not source; can grow to tens of MB per run).

## Common commands

Run everything as a module from the repo root (not as a script from inside subdirectories), since imports are absolute (`jax_forge...`, `applications...`).

```bash
# Agility Forge: single multi-hit forging run (fixed hit schedule)
python -m applications.Agility_Forge.main

# Agility Forge: randomized hit schedule
python -m applications.Agility_Forge.main_random_hits

# Agility Forge: dataset generation (many rollouts, for training data).
# Appends to an existing dataset dir rather than replacing it -- rerunning
# adds more rollouts instead of discarding what's there. Omit --seed for a
# fresh (non-reproducible) draw each run; pass one for a reproducible run.
python -m applications.Agility_Forge.generate_dataset --n-rollouts 25 --n-hits 5

# Koopman Autoencoder training (stage 2 — implemented, proof-of-concept)
python -m applications.Agility_Forge.koopman.train --dataset-dir applications/Agility_Forge/data/dataset --epochs 100
```

There is no lint/format config in this repo and no test suite yet (the upstream benchmark tests that shipped with JAX-FEM tested the now-removed unmodified library, not `jax_forge` or Agility_Forge — there's no existing test coverage for this project's own code).

## Architecture

### FEM core (`jax_forge/`)

- `generate_mesh.py` — `Mesh` container and mesh-generation helpers (`rectangle_mesh`, meshio interop, `get_meshio_cell_type`).
- `basis.py` / `basis_new.py` — reference-element shape functions/gradients per element type and quadrature.
- `fe.py` / `fe_new.py` — `FiniteElement`: one FE variable (mesh + vec + dim + ele_type + Dirichlet BC info); assembles per-variable shape data.
- `problem.py` / `problem_new.py` — `Problem`: composes one or more `FiniteElement`s (`self.fes`) into a (possibly coupled multi-physics) weak form. Problems subclass `Problem` and override `get_tensor_map` / `get_mass_map` / `custom_init` / a `universal_kernel` for the weak form, plus `set_params`/`set_timestep` for time-dependent/parametric runs.
- `solver.py` / `solver_latest.py` — Newton/nonlinear solve driver (`solver(problem, ...)`); `solver_latest.py` is the one Agility_Forge's time stepper actually calls. Supports JAX, scipy, and PETSc (`petsc4py`) linear solves, with optional AMGX (`pyamgx`, optional dependency) and adjoint/differentiable solves.
- `mma.py` — Method of Moving Asymptotes (topology-optimization-style problems; not currently used by Agility_Forge).
- `experimental/` — newer/unstable features (custom JVPs, adjoint checkpoint-to-disk, sparse utilities, JIT strategy helpers), not part of a stable API.

The `_new`/`_latest` files (`fe_new.py`, `problem_new.py`, `basis_new.py`, `solver_latest.py`) are further-diverged variants layered on top of the base files during this project's development — check which one a given Agility_Forge module actually imports (`grep` for `jax_forge\.` in `applications/Agility_Forge/`) before assuming `fe.py`/`problem.py`/`solver.py` are the live code path.

### `applications/Agility_Forge/` — stage 1: forging simulation & data generation

- `hit_config.py` — `Hit` dataclass: parameters for one forging hit (band position, compression displacement, rotation, duration).
- `lib/constitutive.py` — `ThermalMechanical(Problem)`: coupled thermo-mechanical `jax_forge` problem — temperature-dependent J2 finite-strain plasticity with F-bar, plus the coupled weak form and stress/internal-variable post-processing. Material parameters come from `material_data.py`.
- `lib/boundary_conditions.py` — `build_cylinder_press_bcs(...)`: constructs Dirichlet BCs for the rotating cylindrical press/platen contact (updated-Lagrangian, re-detected from the current deformed surface each step) plus convective BCs; `refresh_problem_surface_integrals(...)` re-applies them to a `Problem` after BCs change mid-run.
- `lib/time_stepper.py` — `AutomaticTimeStepperTM`: adaptive-dt Newton time-stepping loop around `jax_forge.solver_latest.solver`, with retry-on-failure, extrapolated initial guesses, VTK output, and step-history plotting. `seed_state(...)` primes/resets the extrapolation history — must be called whenever the solution is externally reset (e.g., between hits) to avoid a doubled first-step initial guess.
- `mesh_container.py` — `MeshContainer`: loads/converts stock geometry (Open3D mesh → JSON → jax-fem `Mesh`).
- `main.py` — driver for a fixed multi-hit rotational forging schedule; handles inter-hit thermal-state relaxation (ramped `T_old` blend + re-equilibration, since resetting temperature in one shot produces a dt-independent Newton stall) and rebuilds contact BCs on the post-relaxation surface each hit.
- `main_random_hits.py` — same driver logic with randomized hit schedules.
- `generate_dataset.py` — runs many randomized rollouts (`run_dataset_generation`, CLI via `--n-rollouts/--n-hits/--seed`) and writes a manifest (`ManifestWriter`) plus one `.vtu` per saved state (per-node `Displacement` + `Temperature`) — this is the training data source for stage 2. This is what `submit_dataset_generation.sh` invokes on the cluster. **Appends** to an existing dataset dir across invocations rather than wiping it: rollout numbering continues from the highest existing `rollout_XX`, `manifest.json` merges (records accumulate, `generation_runs` tracks one provenance entry — seed/n_rollouts/timestamp — per invocation), and a run whose `--n-hits`/`--band-width-frac`/`--compression-displacement`/`--total-time` doesn't match the dataset's existing config is rejected rather than silently mixed in. `--seed` defaults to fresh OS entropy (not a fixed value) so repeated runs add *new* rollouts.
- `data/` — `msh/` (mesh assets, a real input — hardcoded path, don't move), `dataset/` (stage 1's output, see above), `backup/` (leftover diagnostic/VTK output from `main.py`/`main_random_hits.py` runs, moved aside rather than deleted); regenerated/cleaned at the start of each `main.py`/`main_random_hits.py` run (git-ignored). Don't write into `dataset/` while `generate_dataset.py` has a job actively running against it (manifest.json is rewritten, non-atomically, after every saved record).

`open3d` is an optional import in `main.py`/`main_random_hits.py`/`generate_dataset.py` — mesh-conversion functions are disabled with a warning if it's not installed.

### `applications/Agility_Forge/koopman/` — stage 2: Koopman Autoencoder (implemented, proof-of-concept — not yet run against the full dataset)

Modeled on `peter-frazier/KAE_for_uniaxial_tensile_test`'s `LRAN_BLRAN/LRAN_LD`, with one deliberate departure: the original state does **not** appear in the latent (`z = Psi(x)`, not LRAN_LD's `z = [x; Psi(x)]`). Full rationale for every design decision (state representation, control encoding, normalization, loss terms, architecture sizing) is in `koopman/README.md`'s "Design decisions (implemented)" section — read that before changing any of these files.

- `dataset.py` — loads `manifest.json` + `.vtu` rollouts. State is the flattened `Displacement` field per node (`n_x = n_nodes*3`, no reduction) — Temperature is present in every `.vtu` (jax_forge's coupling is unchanged) but is not read into the state; control is `[d_j_frac, sin(R_j), cos(R_j)]` used **raw** (Koopman convention: only the state is normalized/lifted, not the input); state normalization is pooled per physical quantity (one `[-1,1]` scale across all `Displacement` components) so per-axis rescaling doesn't distort displacement direction, fit on the training split only. `compute_pod_basis` additionally fits a fixed (non-learned) rank-`r` POD basis (mean-center + economy SVD of the normalized training states) that `build_rollout_dataset(..., n_pod_modes=r)` returns via `info["pod_mean"]`/`info["pod_modes"]` for the model to consume — see `applications/Agility_Forge/SVD/` for the analysis that motivated this. One training window per rollout (`K = n_hits_per_rollout`), prefix train/test split by rollout index.
- `model.py` — `KoopmanAutoencoder`: pipeline is state -> POD (fixed) -> lift (learned) -> linear dynamics -> unlift (learned) -> POD^-1 (fixed). `encode`: POD-project the `n_x`-dim normalized state onto the fixed `r`-dim basis (`pod_mean`/`pod_modes`, registered buffers — not learned), then `Lifting` (`z = Psi(a)`, `n_z > r` deliberately — Koopman theory lifts to a *higher*-dim space) lifts the `r`-dim POD coefficients to the `n_z`-dim latent. `Lifting` is `in_proj(r->block_widths[0])` -> one residual block per entry in `block_widths` -> `out_proj(block_widths[-1]->n_z)`; each block wraps `layers_per_block` `[LayerNorm -> Linear(width,width) -> activation]` layers in one skip connection (`out = in + F(in)`), with a plain Linear+activation transition (no skip) bridging blocks whose widths differ — skipped (no extra layer at all) when consecutive widths match. `decode`: learned linear `n_z -> r` (`nn.Linear`, not exact), then the *fixed* POD reconstruction `pod_mean + a_hat @ pod_modes` back to `n_x`. `A`/`B` linear dynamics (`z_{k+1} = A z_k + B u_k`) operate on the `n_z`-dim latent, unchanged in kind. Default `r=75, n_z=250, block_widths=[250,250,250,250], layers_per_block=8` (32 hidden layers, ~2.19M learned params) — replaced an earlier `n_x`-input, ~2.4B-param design once `applications/Agility_Forge/SVD/`'s POD energy analysis showed that scale wasn't justified by the data (~11/30/71 modes for 90/95/99.9% energy on the full 25-rollout dataset). LayerNorm (needed for stability in the earlier wider/deeper version) is kept and still empirically stable at this smaller scale.
- `train.py` — `L_id + L_fwd + L_lin` (+ optional `L_eig`, off by default) loss, early stopping on test `L_fwd`, per-epoch train-vs-test loss-curve plot + `metrics.json`, true-vs-predicted `.vtu` dump for one held-out rollout. Default architecture as above (`--n-pod-modes 75 --latent-dim 250 --block-widths 250,250,250,250 --layers-per-block 8`) — sized to what the SVD/POD analysis actually justified, not (unlike the prior default) a bet sized for a much larger future dataset; still expect some overfitting risk at 20-25 training rollouts even at this much smaller scale. Deliberately imports its own stdlib logger rather than `from jax_forge import logger` — `jax_forge/__init__.py` has import-time side effects (prints a `pyfiglet` banner, pulls in the JAX/FEM stack) that this module is meant to stay decoupled from.
- `applications/Agility_Forge/SVD/` — standalone (no PyTorch/JAX) POD/SVD analysis of the same displacement state, run via `python -m applications.Agility_Forge.SVD.svd_analysis`. Computes cumulative energy spectrum + effective-dimension thresholds; motivated the `koopman/` module's shift from a huge nonlinear encoder to a small one behind a fixed POD pre-reduction. See its own `README.md` for the exact convention (normalize-then-center, rows-as-snapshots).

### `applications/Agility_Forge/control/` — stage 3: MPC loop (scaffolded, not implemented)

`plant_interface.py` (`ForgingPlant`: wraps `ThermalMechanical`/`AutomaticTimeStepperTM` behind a `step(state, control) -> next_state` interface) and `mpc.py` (`MPCController`: receding-horizon loop using the Koopman model for planning, the real plant for ground truth) are stubs. See `control/README.md` for open questions (horizon, QP solver choice, control bounds, cost function).

### `applications/Example/`

`CIRP/` and `Fine_mesh/` — small earlier prototype scripts for this same forging project (also import `jax_forge`), kept alongside `Agility_Forge` but not part of its module structure.

## Notes

- `jax_enable_x64` is turned on explicitly in files that need double precision (e.g. `applications/Agility_Forge/main.py`, `jax_forge/solver.py`) — don't assume it's on globally.
- `data/`, `output/`, VTK dirs, and other generated artifacts are git-ignored; treat anything under an app's `data/` directory as regenerable, not source.
- `Agility_Forge_Random_Hits_Report.pdf` and `old/` are git-ignored (proprietary report artifacts) — don't add or reference them from tracked files.
