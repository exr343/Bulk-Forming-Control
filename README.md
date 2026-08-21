# Agility Forge

A differentiable, GPU-accelerated simulation of multi-hit rotational forging, built for a CIRP 2027 submission. This repo is being developed into a 3-stage pipeline:

1. **Data generation** (done) — `applications/Agility_Forge/generate_dataset.py` runs randomized multi-hit forging rollouts against the forging simulation and records the resulting state trajectories.
2. **Koopman Autoencoder training** (scaffolded, not yet implemented) — `applications/Agility_Forge/koopman/` will learn a latent linear-dynamics surrogate of the forging state from stage-1 rollouts.
3. **MPC control loop** (scaffolded, not yet implemented) — `applications/Agility_Forge/control/` will use the trained Koopman model as the fast predictive model inside a receding-horizon controller, validating every planned control against the real simulation as the plant.

## The forging simulation

The FEM core, `jax_forge/`, is a fork of [JAX-FEM](https://github.com/deepmodeling/jax-fem) (a differentiable finite element package built on [JAX](https://github.com/google/jax)) that's been extended for this project's constitutive model and solver. `applications/Agility_Forge/` builds a coupled thermo-mechanical (J2 finite-strain plasticity, temperature-dependent) simulation of a cylindrical billet under a rotating multi-hit press on top of it — see [applications/Agility_Forge/main.py](applications/Agility_Forge/main.py) for the driver and [CLAUDE.md](CLAUDE.md) for the full architecture.

## Setup

```bash
conda env create -f environment.yml
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # required for petsc4py
```

On the HPC cluster this project runs on, see [cluster_setup.md](cluster_setup.md) for the SLURM/module-load workflow.

## Running

```bash
# Single multi-hit forging run (fixed hit schedule)
python -m applications.Agility_Forge.main

# Randomized hit schedule
python -m applications.Agility_Forge.main_random_hits

# Dataset generation for stages 2/3
python -m applications.Agility_Forge.generate_dataset --n-rollouts 10 --n-hits 10 --seed 42
```

## License

`jax_forge/` is a derivative of JAX-FEM and this repo is licensed under the GNU General Public License v3 — see [LICENSE](LICENSE).

If you use the underlying FEM library, consider citing the original JAX-FEM paper:

```bibtex
@article{xue2023jax,
  title={JAX-FEM: A differentiable GPU-accelerated 3D finite element solver for automatic inverse design and mechanistic data science},
  author={Xue, Tianju and Liao, Shuheng and Gan, Zhengtao and Park, Chanwook and Xie, Xiaoyu and Liu, Wing Kam and Cao, Jian},
  journal={Computer Physics Communications},
  pages={108802},
  year={2023},
  publisher={Elsevier}
}
```
