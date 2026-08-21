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

`jax`/`jaxlib`/`torch` were only recently added to `environment.yml` and are currently **unpinned** — pin them to whatever's actually active in `jax-fem-env` (`python -c "import jax; print(jax.__version__)"`) from a real cluster session, since this file was edited without access to a live environment.

### On the cluster (see `cluster_setup.md`)

```bash
srun -p gpu --gres=gpu:1 -c 4 --mem=32G -t 04:00:00 --pty bash -l
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Batch dataset generation runs go through `applications/Agility_Forge/submit_dataset_generation.sh` (`sbatch submit_dataset_generation.sh`, must be submitted with CWD at repo root — it `cd`s there itself). Logs land in `applications/Agility_Forge/slurm_logs/`.

## Common commands

Run everything as a module from the repo root (not as a script from inside subdirectories), since imports are absolute (`jax_forge...`, `applications...`).

```bash
# Agility Forge: single multi-hit forging run (fixed hit schedule)
python -m applications.Agility_Forge.main

# Agility Forge: randomized hit schedule
python -m applications.Agility_Forge.main_random_hits

# Agility Forge: dataset generation (many rollouts, for training data)
python -m applications.Agility_Forge.generate_dataset --n-rollouts 10 --n-hits 10 --seed 42

# Koopman Autoencoder training (stage 2 — scaffolded, not yet implemented)
python -m applications.Agility_Forge.koopman.train --dataset-dir applications/Agility_Forge/data/dataset
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
- `generate_dataset.py` — runs many randomized rollouts (`run_dataset_generation`, CLI via `--n-rollouts/--n-hits/--seed`) and writes a manifest (`ManifestWriter`) plus one `.vtu` per saved state (per-node `Displacement` + `Temperature`) — this is the training data source for stage 2. This is what `submit_dataset_generation.sh` invokes on the cluster.
- `data/` — mesh assets, VTK output, per-hit step/scale diagnostic plots; regenerated/cleaned at the start of each run (git-ignored).

`open3d` is an optional import in `main.py`/`main_random_hits.py`/`generate_dataset.py` — mesh-conversion functions are disabled with a warning if it's not installed.

### `applications/Agility_Forge/koopman/` — stage 2: Koopman Autoencoder (scaffolded, not implemented)

`dataset.py` (loads `manifest.json` + `.vtu` rollouts into training tensors), `model.py` (`KoopmanAutoencoder`: encoder/decoder + latent linear dynamics), `train.py` (PyTorch training loop) are stubs — signatures and docstrings only, every body raises `NotImplementedError`. See `koopman/README.md` for the open design questions (state-vector representation, latent dimension, loss terms) that need resolving before filling these in.

### `applications/Agility_Forge/control/` — stage 3: MPC loop (scaffolded, not implemented)

`plant_interface.py` (`ForgingPlant`: wraps `ThermalMechanical`/`AutomaticTimeStepperTM` behind a `step(state, control) -> next_state` interface) and `mpc.py` (`MPCController`: receding-horizon loop using the Koopman model for planning, the real plant for ground truth) are stubs. See `control/README.md` for open questions (horizon, QP solver choice, control bounds, cost function).

### `applications/Example/`

`CIRP/` and `Fine_mesh/` — small earlier prototype scripts for this same forging project (also import `jax_forge`), kept alongside `Agility_Forge` but not part of its module structure.

## Notes

- `jax_enable_x64` is turned on explicitly in files that need double precision (e.g. `applications/Agility_Forge/main.py`, `jax_forge/solver.py`) — don't assume it's on globally.
- `data/`, `output/`, VTK dirs, and other generated artifacts are git-ignored; treat anything under an app's `data/` directory as regenerable, not source.
- `Agility_Forge_Random_Hits_Report.pdf` and `old/` are git-ignored (proprietary report artifacts) — don't add or reference them from tracked files.
